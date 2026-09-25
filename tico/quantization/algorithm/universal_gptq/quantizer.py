"""
Universal GPTQ Quantizer for PyTorch Models.

This module implements a model-agnostic GPTQ (Gradient-based Post-Training Quantization)
algorithm that works with any PyTorch model composed of standard layers (Linear, Conv1d,
Conv2d, Conv3d, ConvTranspose2d) without requiring model-specific wrapper classes.

The quantizer uses a frontier-based execution strategy with a state machine approach:
1. COLLECT: Accumulate Hessian information from calibration inputs.
2. CACHE: Return cached outputs if available; otherwise compute and cache.
3. COMPUTE: Normal forward pass after quantization.

Key Features:
- Model-agnostic: Works with any PyTorch architecture
- Layer-by-layer quantization with Hessian-based error minimization
- Memory-efficient caching on CPU to save GPU memory
- Support for per-channel and per-tensor quantization
- Configurable bit-width with module-level overrides
- Sensitivity-aware MSE optimization (optional)

Example:
    from tico.quantization.config.gptq import UniversalGPTQConfig
    from tico.quantization.algorithm.universal_gptq.quantizer import UniversalGPTQQuantizer

    config = UniversalGPTQConfig(weight_bits=8, percdamp=0.01)
    quantizer = UniversalGPTQQuantizer(config)

    model = quantizer.prepare(model)
    # Run calibration data through model
    model = quantizer.convert(model)
"""

import re
import types
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from tico.quantization.algorithm.gptq.gptq import GPTQ
from tico.quantization.algorithm.gptq.quant import Quantizer
from tico.quantization.config.gptq import UniversalGPTQConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer
from tico.utils.utils import move_to_device, Stack
from tqdm.auto import tqdm


__all__ = [
    "UniversalGPTQQuantizer",
    "UniversalGPTQConfig",
]


_QUANTIZABLE_LAYER_TYPES: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose2d,
)


def move_to_cpu(obj) -> Any:
    """
    Move a tensor or nested structure of tensors to CPU.

    This is a convenience wrapper around `move_to_device` that always moves to CPU.
    Used to cache model inputs/outputs during GPTQ quantization to save GPU memory.

    Parameters:
        obj: A tensor or nested structure (tuple, list, dict) containing tensors.

    Returns:
        The same structure with all tensors moved to CPU.
    """
    return move_to_device(obj, "cpu")


def infer_module_device(model: nn.Module) -> torch.device | None:
    """
    Return the device of the first parameter in the model.

    Parameters:
        model: Target model.

    Returns:
        The device where the model currently resides.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None


def infer_object_device(obj: Any) -> torch.device | None:
    """
    Infer the device of a tensor or nested structure of tensors.

    This function recursively traverses tuples, lists, dicts, and objects
    with __dict__ attributes to determine the device where the tensors reside.
    All tensors in the structure must be on the same device.

    Parameters:
        obj: A tensor or nested structure (tuple, list, dict, or object)
             containing tensors.

    Returns:
        The device where the tensors reside, or None if obj is None or
        contains no tensors.

    Raises:
        AssertionError: If tensors within the same structure are on different devices.
    """
    if obj is None:
        return None

    if isinstance(obj, torch.Tensor):
        return obj.device

    if isinstance(obj, (str, bytes, bytearray)):
        return None

    devices: set[torch.device]

    if isinstance(obj, Mapping):
        if not obj:
            return None
        devices = {
            device
            for device in (infer_object_device(x) for x in obj.values())
            if device is not None
        }
        if devices:
            assert (
                len(devices) == 1
            ), f"The object is assumed to be located on a single device. Got {devices}."
            return next(iter(devices))
        else:
            return None

    if isinstance(obj, Iterable):
        if not obj:
            return None
        devices = {
            device
            for device in (infer_object_device(x) for x in obj)
            if device is not None
        }
        if devices:
            assert (
                len(devices) == 1
            ), f"The object is assumed to be located on a single device. Got {devices}."
            return next(iter(devices))
        else:
            return None

    if hasattr(obj, "__dict__"):
        return infer_object_device(obj.__dict__)

    return None


@register_quantizer(UniversalGPTQConfig)
class UniversalGPTQQuantizer(BaseQuantizer):
    """
    Universal GPTQ quantizer for PyTorch models.

    This quantizer implements a model-agnostic GPTQ algorithm that
    quantizes weights layer-by-layer using Hessian-based error minimization.
    It supports any model composed of standard PyTorch layers
    (Linear, Conv1d, Conv2d, Conv3d, ConvTranspose2d).

    The quantization workflow follows three stages:

    1. **Prepare**: Replace the model's forward method to cache calibration inputs.
    2. **Calibrate**: Run calibration data through the model to collect inputs.
    3. **Convert**: Apply GPTQ quantization layer-by-layer using cached data.

    Example usage:
        ```python
        from tico.quantization.config.gptq import UniversalGPTQConfig

        config = UniversalGPTQConfig(weight_bits=8, percdamp=0.01)
        quantizer = UniversalGPTQQuantizer(config)

        # Prepare
        model = quantizer.prepare(model)

        # Calibrate (cache inputs)
        for batch in calibration_data:
            model(batch)

        # Convert (apply GPTQ)
        model = quantizer.convert(model)

        # Model now has 'quantizers' attribute with per-layer quantizers
        ```
    """

    def __init__(self, config: UniversalGPTQConfig):
        """
        Initialize the GPTQ quantizer.

        Parameters:
            config: GPTQ configuration specifying quantization parameters
                    (weight_bits, percdamp, groupsize, etc.).
        """
        super().__init__(config)
        self._cache_args: list[
            tuple[Any]
        ] = []  # cache_args[i] -> i-th batch of positional arguments
        self._cache_kwargs: list[
            dict[str, Any]
        ] = []  # cache_kwargs[i] -> i-th batch of keyword arguments
        self._orig_model_forward: Optional[Callable[..., Any]] = None

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Any | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        """
        Prepare the model for GPTQ quantization by replacing its forward method.

        This method replaces the model's forward method with a wrapper that caches
        calibration inputs. After calling prepare(), run calibration data through
        the model to collect inputs, then call convert() to apply quantization.

        Parameters:
            model: The PyTorch model to quantize.
            args: Optional positional arguments (not used, kept for API compatibility).
            kwargs: Optional keyword arguments (not used, kept for API compatibility).

        Returns:
            The same model with modified forward method for input caching.

        Raises:
            AssertionError: If prepare() is called twice without convert().
        """
        assert len(self._cache_args) == 0, "prepare() called twice without convert()"
        # Substitute the model's forward method to cache model inputs
        def new_forward(model: nn.Module, *args, **kwargs) -> Any:
            """
            Stores this batch's inputs and kwargs, then raises StopForward to stop computation.
            """
            self._cache_args.append(tuple(move_to_cpu(arg) for arg in args))
            self._cache_kwargs.append({k: move_to_cpu(v) for k, v in kwargs.items()})
            return None

        self._orig_model_forward = model.forward
        model.forward = types.MethodType(new_forward, model)
        return model

    @torch.no_grad()
    def convert(self, model: torch.nn.Module) -> nn.Module:
        """
        Apply GPTQ quantization to the prepared model.

        This method restores the original forward method and applies GPTQ quantization
        layer-by-layer using the cached calibration data. After conversion, the model
        will have a `quantizers` attribute containing per-layer Quantizer objects
        with computed scale and zero-point parameters.

        Parameters:
            model: The prepared model (must have been passed through prepare() first).

        Returns:
            The quantized model with updated weights and a `quantizers` attribute.

        Raises:
            AssertionError: If convert() is called before prepare() or before calibration.
        """
        assert self._orig_model_forward is not None, "convert() called before prepare()"
        assert len(self._cache_args) > 0, "convert() called before calibration"
        model.forward = self._orig_model_forward
        assert type(self.config) is UniversalGPTQConfig
        gptq_quantize(
            model,
            self.config,
            args_dataset=self._cache_args,
            kwargs_dataset=self._cache_kwargs,
        )
        return model


class GPTQ_STATE(Enum):
    """
    State machine for GPTQ layer-by-layer quantization:

    COLLECT (1) ------[finish_collection]-------> CACHE (2) ------------------------[finish_caching]
    |                          |                      |  ^-----------------------------|
    |                          |                      |                                |
    |                          |                      |                                |
    v                          v                      v                                v
    Hessian accumulation   Quantize weights   Return cached if within cache limit   Clear childrens' cache
    StopForward raised                        Compute and cache outputs otherwise
                                              StopForward raised
    """

    COLLECT = 1
    CACHE = 2
    COMPUTE = 3


@dataclass
class GPTQ_Data:
    """
    Internal data structure storing per-module GPTQ state.

    This dataclass is attached as an attribute (`gptq_data`) to each module
    during the quantization process. It tracks the module's state through
    the quantization lifecycle and stores intermediate results.

    Attributes:
        old_forward: The original forward method before wrapping.
        gptq: GPTQ instance for Hessian accumulation (None after quantization).
        quantizer: Quantizer with computed scale/zero-point (None before quantization).
        is_cacheable: Whether the module's outputs are cached during model replay.
        cached_output: Cached outputs for replay (freed after use). cached_output[batch_idx][invocation_idx].
        collected_inputs: Inputs collected for Hessian computation.
        state: Current state in the quantization lifecycle.
        invocation_idx: Current invocation index during 1 batch replay.
        batch_idx: Current batch index during dataset replay.
        total_invocations: Total invocations count during dataset replay.
        weight_device: Device where the module's parameters reside.
        out_device: Device where module output should reside.
    """

    full_module_name: str
    old_forward: Callable
    gptq: GPTQ | None
    quantizer: Quantizer | None
    is_cacheable: bool
    cached_output: list[list[Any]]
    collected_inputs: list[torch.Tensor]
    state: GPTQ_STATE
    invocation_idx: int
    batch_idx: int
    total_invocations: int
    weight_device: torch.device | None
    out_device: torch.device | None


class StopForward(Exception):
    """
    Exception raised to halt forward propagation during GPTQ quantization.

    This exception is used as a control flow mechanism to stop execution at
    specific modules (frontier modules) during the layer-by-layer quantization
    process. When caught, it indicates that the frontier module has been reached
    and should be processed (either Hessian collection or output caching).

    Attributes:
        module: The module where execution should stop.
    """

    def __init__(self, module: nn.Module):
        self.module = module


def get_gptq_data(module: nn.Module) -> GPTQ_Data:
    """
    Retrieve GPTQ data from a module.

    Parameters:
        module: The module to get GPTQ data from.

    Returns:
        The GPTQ_Data attached to the module.

    Raises:
        AssertionError: If the module doesn't have gptq_data or it's not the right type.
    """
    gptq_data: GPTQ_Data = getattr(module, "gptq_data")
    assert gptq_data is not None and type(gptq_data) is GPTQ_Data
    return gptq_data


def has_gptq_data(module: nn.Module) -> bool:
    """
    Check if a module has GPTQ data attached.

    Parameters:
        module: The module to check.

    Returns:
        True if the module has gptq_data attribute, False otherwise.
    """
    return hasattr(module, "gptq_data")


def delete_gptq_data(module: nn.Module) -> None:
    """
    Remove GPTQ data from a module.

    This is called during cleanup to remove the gptq_data attribute
    after quantization is complete.

    Parameters:
        module: The module to clean up.
    """
    if hasattr(module, "gptq_data"):
        delattr(module, "gptq_data")


def wrap_model(
    model: nn.Module,
    full_model_name: str,
    ignored_module_patterns: Iterable[re.Pattern],
    ignored_modules: Iterable[nn.Module],
) -> None:
    """
    Recursively wrap a model's forward methods for GPTQ quantization.

    This function attaches GPTQ_Data to each module in the model hierarchy
    and replaces forward methods with a state-aware wrapper. The wrapper
    handles four states:

    - COLLECT: Accumulate Hessian from inputs, raise StopForward
    - CACHE: If within cached range - return cached outputs without computation.
             Otherwise run original forward, cache output, raise StopForward.
    - COMPUTE: Run original forward without caching

    After wrapping, the model is ready for layer-by-layer quantization
    using the frontier-based execution strategy.

    Parameters:
        model: The model to wrap (modified in-place).
        full_model_name: Hierarchical dot-delimited model name following "grandparent.parent.clild" pattern.
        ignored_module_patterns: Collection of regular expressions to be matched against submodules' full names. If matched a submodule is not quantized.
        ignored_modules: Modules to exclude from GPTQ quantization.
        cache_outputs: Whether to enable modules' outputs caching for performance optimization.
    """

    def old_forward(gptq_data: GPTQ_Data, *args, **kwargs) -> Any:
        args = move_to_device(args, gptq_data.weight_device)
        kwargs = move_to_device(kwargs, gptq_data.weight_device)
        assert type(kwargs) is dict
        return gptq_data.old_forward(*args, **kwargs)

    def new_forward(module: nn.Module, *args, **kwargs) -> Any:
        gptq_data: GPTQ_Data = get_gptq_data(module)
        match gptq_data.state:
            case GPTQ_STATE.COLLECT:
                # Move input to model's weight device for Hessian accumulation
                gptq_data.collected_inputs.append(
                    args[0].data.to(gptq_data.weight_device)
                )
                raise StopForward(module)

            case GPTQ_STATE.CACHE:
                assert gptq_data.is_cacheable

                n_cached_batches: int = len(gptq_data.cached_output)
                if gptq_data.batch_idx >= n_cached_batches:
                    assert gptq_data.batch_idx == n_cached_batches
                    assert gptq_data.invocation_idx == 0
                    # Allocate new list for this batche's invocations
                    gptq_data.cached_output.append([])

                cached_batch_out: list[Any] = gptq_data.cached_output[
                    gptq_data.batch_idx
                ]
                n_cached_invocations: int = len(cached_batch_out)
                if gptq_data.invocation_idx < n_cached_invocations:
                    # Return cached output
                    result: Any = cached_batch_out[gptq_data.invocation_idx]
                    # Move cached output from CPU to model's device
                    if gptq_data.out_device:
                        result = move_to_device(result, gptq_data.out_device)
                    gptq_data.invocation_idx += 1
                    return result
                else:
                    # Compute output and cache it, then raise StopForward
                    assert gptq_data.invocation_idx == n_cached_invocations
                    out: Any = old_forward(gptq_data, *args, **kwargs)
                    if gptq_data.out_device is None:
                        gptq_data.out_device = infer_object_device(out)
                    cached_batch_out.append(move_to_cpu(out))
                    gptq_data.invocation_idx += 1
                    raise StopForward(module)

            case GPTQ_STATE.COMPUTE:
                return old_forward(gptq_data, *args, **kwargs)

            case _:
                assert False, "We should never get here"

    not_quantizable: bool = (
        type(model) not in _QUANTIZABLE_LAYER_TYPES
        or any(
            regex.fullmatch(full_model_name) is not None
            for regex in ignored_module_patterns
        )
        or model in ignored_modules
    )

    is_cacheable: bool = getattr(model, "is_cacheable")
    assert is_cacheable is not None
    delattr(model, "is_cacheable")

    total_invocations: int = getattr(model, "total_invocations")
    assert total_invocations is not None
    delattr(model, "total_invocations")

    setattr(
        model,
        "gptq_data",
        GPTQ_Data(
            full_module_name=full_model_name,
            old_forward=model.forward,
            gptq=None,
            quantizer=None,
            is_cacheable=is_cacheable,
            cached_output=[],
            collected_inputs=[],
            state=(
                (GPTQ_STATE.CACHE if is_cacheable else GPTQ_STATE.COMPUTE)
                if not_quantizable
                else GPTQ_STATE.COLLECT
            ),
            invocation_idx=0,
            batch_idx=0,
            total_invocations=total_invocations,
            weight_device=infer_module_device(model),
            out_device=None,
        ),
    )

    model.forward = types.MethodType(new_forward, model)

    child_name: str
    child: nn.Module
    for child_name, child in model.named_children():
        full_child_name = (
            f"{full_model_name}.{child_name}" if full_model_name else child_name
        )
        wrap_model(
            child,
            full_model_name=full_child_name,
            ignored_module_patterns=ignored_module_patterns,
            ignored_modules=ignored_modules,
        )


def unwrap_model(model: nn.Module, retain_gptq_data: bool = False) -> None:
    """
    Restore original module structure after GPTQ quantization.

    This function reverses the wrapping applied by `wrap_model`, restoring
    the original forward methods and removing the `gptq_data` attribute.
    It also clears cached outputs to free memory.

    Parameters:
        model: The wrapped model to unwrap (modified in-place).
        retain_gptq_data: Whether to retain gptq_data.
    """
    for module in model.modules():
        gptq_data: GPTQ_Data = get_gptq_data(module)
        module.forward = gptq_data.old_forward
        if not retain_gptq_data:
            gptq_data.cached_output.clear()
            delete_gptq_data(module)


def collect_quantizers(
    model: nn.Module,
    full_model_name: str,
    quantizers: dict[str, Quantizer],
) -> None:
    """
    Recursively collect quantizers from all quantized modules.

    This function traverses the module hierarchy and collects Quantizer
    objects from modules that have been through GPTQ quantization. The
    quantizers are stored in a dictionary keyed by their full module path.

    Parameters:
        model: The module to collect quantizers from.
        full_model_name: The current module path (used for building keys).
        quantizers: Dictionary to store collected quantizers (modified in-place).
    """
    gptq_data: GPTQ_Data = get_gptq_data(model)

    if gptq_data.quantizer is not None:
        quantizers[full_model_name] = gptq_data.quantizer

    child_name: str
    child: nn.Module
    for child_name, child in model.named_children():
        full_child_name = (
            f"{full_model_name}.{child_name}" if full_model_name else child_name
        )
        collect_quantizers(
            child,
            full_model_name=full_child_name,
            quantizers=quantizers,
        )


def run_model(
    model: nn.Module,
    args_dataset: list[tuple[Any]],
    kwargs_dataset: list[dict[str, Any]],
    show_progress: bool,
) -> set[nn.Module]:
    """
    Run the model on calibration data to identify frontier modules.

    This function executes the model on each calibration batch and catches
    StopForward exceptions to identify which modules stopped execution
    (frontier modules). These are the modules that need processing in the
    current iteration.

    Parameters:
        model: The wrapped model to run.
        args_dataset: List of positional argument tuples for each batch.
        kwargs_dataset: List of keyword argument dicts for each batch.

    Returns:
        Set of frontier modules that raised StopForward during execution.
    """
    model_device = get_gptq_data(model).weight_device
    frontier_submodules: set[nn.Module] = set()
    for args, kwargs in tqdm(
        zip(args_dataset, kwargs_dataset, strict=True),
        desc="(Re)playing model",
        leave=False,
        unit="batch",
        total=len(args_dataset),
        disable=not show_progress,
    ):
        args = move_to_device(args, model_device)
        kwargs = move_to_device(kwargs, model_device)
        assert type(kwargs) is dict
        gptq_data: GPTQ_Data
        try:
            model(*args, **kwargs)
        except StopForward as stop_fwd:
            assert stop_fwd.module is not None
            frontier_submodules.add(stop_fwd.module)
        finally:
            # Update all modules' state after batch completion
            for m in model.modules():
                gptq_data = get_gptq_data(m)

                # For cacheable modules check that all cached outputs for this batch were actually acquired
                assert (
                    not gptq_data.is_cacheable
                    or gptq_data.batch_idx == len(gptq_data.cached_output)
                    or gptq_data.invocation_idx
                    == len(gptq_data.cached_output[gptq_data.batch_idx])
                ), "Not all cached invocations were acquired for this batch"

                # Increment batch counter for modules that were invoked at least once
                if gptq_data.invocation_idx > 0:
                    gptq_data.batch_idx += 1
                    # Reset invocation counter
                    gptq_data.invocation_idx = 0

    # reset batch counter
    for m in model.modules():
        gptq_data = get_gptq_data(m)
        # For cacheable modules check that all cached outputs for entire dataset were actually acquired
        assert (
            not gptq_data.is_cacheable
            or gptq_data.batch_idx == len(gptq_data.cached_output)
            or (
                gptq_data.batch_idx == 0
                and len(gptq_data.cached_output) == 1
                and len(gptq_data.cached_output[0]) == 0
            )
        ), "Not all cached batches were acquired"
        gptq_data.batch_idx = 0

    return frontier_submodules


def resolve_weight_bits(
    gptq_config: UniversalGPTQConfig,
    full_module_name: str,
) -> int:
    """
    Resolve the effective bit-width for a quantized submodule.

    This function checks for bit-width overrides in the following order:
    1. Full module name match (e.g., "model.layers.0.self_attn.o_proj")
    2. Local module name match (e.g., "self_attn.o_proj")
    3. Suffix match (e.g., "o_proj" or "down_proj")

    Parameters:
        gptq_config: GPTQ configuration containing weight_bits and overrides.
        full_module_name: Hierarchical module path (e.g., "model.layers.0.mlp.down_proj").

    Returns:
        The effective bit-width for this module.
    """
    if full_module_name in gptq_config.weight_bits_overrides:
        return gptq_config.weight_bits_overrides[full_module_name]

    local_module_name = full_module_name.split(".")[-1]
    if local_module_name in gptq_config.weight_bits_overrides:
        return gptq_config.weight_bits_overrides[local_module_name]

    suffix_matches = [
        bits
        for pattern, bits in gptq_config.weight_bits_overrides.items()
        if full_module_name.endswith(f".{pattern}")
    ]

    if suffix_matches:
        return suffix_matches[-1]

    return gptq_config.weight_bits


def get_sensitivity(
    sensitivity: dict[str, torch.Tensor] | None,
    full_module_name: str,
) -> torch.Tensor | None:
    """
    Retrieve sensitivity tensor for a specific module.

    Sensitivity tensors contain second-order derivative information used for
    sensitivity-aware MSE optimization during GPTQ quantization.

    Parameters:
        sensitivity: Dictionary mapping module names to sensitivity tensors,
                     or None if sensitivity is not being used.
        full_module_name: Hierarchical module path to look up.

    Returns:
        The sensitivity tensor for this module, or None if not available.
    """
    if (
        sensitivity is not None
        and isinstance(sensitivity, dict)
        and full_module_name in sensitivity
    ):
        return sensitivity[full_module_name]
    else:
        return None


def finish_collection(
    module: nn.Module,
    gptq_config: UniversalGPTQConfig,
) -> None:
    """
    Complete Hessian collection and quantize a module's weights.

    This function is called when a module has finished collecting Hessian
    information from all calibration batches. It configures the quantizer,
    runs GPTQ quantization (fasterquant), and transitions the module to
    the CACHE state.

    Parameters:
        module: The module to quantize (must be in COLLECT state).
        gptq_config: GPTQ configuration with quantization parameters.

    Raises:
        RuntimeError: If the module received no calibration data (empty Hessian).
    """
    assert type(module) in _QUANTIZABLE_LAYER_TYPES
    gptq_data: GPTQ_Data = get_gptq_data(module)
    assert gptq_data.state == GPTQ_STATE.COLLECT
    assert len(gptq_data.collected_inputs) > 0

    if gptq_data.gptq is None:
        gptq_data.gptq = GPTQ(module)
    input: torch.Tensor
    for input in tqdm(
        gptq_data.collected_inputs,
        desc=f"[{gptq_data.full_module_name}] -> Accumulating Hessian",
        leave=False,
        unit="batch",
        disable=not gptq_config.show_progress,
    ):
        gptq_data.gptq.add_batch(
            inp=input,
            out=None,  # out is ignored in GPTQ.add_batch
        )

    # Check if Hessian was actually accumulated
    assert (
        gptq_data.gptq.H is not None and gptq_data.gptq.H.numel() > 0
    ), f"Module received no calibration data"
    gptq_data.collected_inputs.clear()

    # Configure the quantizer before running fasterquant
    gptq_data.gptq.quantizer.configure(
        bits=resolve_weight_bits(
            gptq_config,
            full_module_name=gptq_data.full_module_name,
        ),
        perchannel=gptq_config.perchannel,
        sym=gptq_config.symmetric,
        mse=gptq_config.mse,
        sensitivity=get_sensitivity(
            gptq_config.sensitivity,
            full_module_name=gptq_data.full_module_name,
        ),
    )

    # Quantize weights
    if gptq_config.verbose:
        print(f"[{gptq_data.full_module_name}] -> Quantizing ...")

    gptq_data.gptq.fasterquant(
        percdamp=gptq_config.percdamp,
        groupsize=gptq_config.groupsize,
        actorder=gptq_config.actorder,
        static_groups=gptq_config.static_groups,
        verbose=gptq_config.verbose,
    )
    gptq_data.state = GPTQ_STATE.CACHE if gptq_data.is_cacheable else GPTQ_STATE.COMPUTE
    gptq_data.quantizer = gptq_data.gptq.quantizer
    gptq_data.gptq = None
    assert gptq_data.invocation_idx == 0


def finish_caching(
    module: nn.Module,
    release_children_cache: bool,
    verbose: bool,
) -> None:
    """
    Complete output caching for a particular call site.

    This function is called when a module has cached outputs from all
    calibration batches for a particular call site in the python code.
    It frees children's cached outputs (memory optimization).

    Parameters:
        module: The module to transition (must be in CACHE state).
        release_children_cache: Whether to release child modules cached outputs.
        verbose: Whether to print out the number of released cached outputs.
    """
    gptq_data: GPTQ_Data = get_gptq_data(module)
    assert gptq_data.state == GPTQ_STATE.CACHE
    assert gptq_data.invocation_idx == 0
    if verbose:
        print(
            f"[{gptq_data.full_module_name}] Cached {len(gptq_data.cached_output)} outputs"
        )

    # Free (grand)children's cached outputs
    if release_children_cache:
        for child_name, child in module.named_modules():
            if not child_name:
                continue
            child_gptq_data: GPTQ_Data = get_gptq_data(child)
            if (
                not child_gptq_data.is_cacheable
                or child_gptq_data.state != GPTQ_STATE.CACHE
            ):
                continue

            assert child_gptq_data.invocation_idx == 0
            assert child_gptq_data.batch_idx == 0
            if child_gptq_data.cached_output:
                if verbose:
                    print(
                        f"[{gptq_data.full_module_name}.{child_name}] Released {len(child_gptq_data.cached_output)} cached outputs"
                    )
                child_gptq_data.cached_output.clear()


def find_multicall_modules(
    model: nn.Module,
    args_dataset: list[tuple[Any]],
    kwargs_dataset: list[dict[str, Any]],
    show_progress: bool,
    verbose: bool,
) -> set[nn.Module]:
    """
    Detect modules that are invoked multiple times within a single forward pass.

    This function identifies modules that participate in circular data dependencies
    (or are simply called multiple times within a single batch) by temporarily wrapping
    each module's forward method to count invocations per batch. Modules with more than
    one invocation per batch are excluded from GPTQ quantization to avoid violations
    of the GPTQ algorithm's assumption that each module receives inputs from
    already-quantized predecessors.

    The detection works as follows:
    1. Wrap each module's forward to count invocations
    2. Run the model on calibration data
    3. Reset invocation count after each batch
    4. Modules with count > 1 in any batch are marked as multiply-invoked
    5. Restore original forward methods

    Modules excluded from GPTQ will be handled by the subsequent PTQ stage.

    Parameters:
        model: The model to analyze for multiply-invoked modules.
        args_dataset: List of positional argument tuples for calibration.
        kwargs_dataset: List of keyword argument dicts for calibration.
        show_progress: Whether to display a progress bar during analysis.
        verbose: Whether to print out found modules' names

    Returns:
        A set of modules that are invoked multiple times per batch.

    Side-effect:
        Creates a new attribute in each model's submodule named 'total_invocations' that
        reflects the total count of calls to the respective submodule across the entire dataset.

    Raises:
        AssertionError: If datasets are empty or have mismatched lengths.
    """
    assert len(args_dataset) > 0, "Empty calibration dataset"
    assert len(args_dataset) == len(kwargs_dataset), "Dataset length mismatch"

    multicall_modules: set[nn.Module] = set()

    def set_add(s: set[Any], x: Any) -> bool:
        l = len(s)
        s.add(x)
        return len(s) > l

    def new_forward(module: nn.Module, *args, **kwargs) -> Any:
        old_forward = getattr(module, "old_forward")
        assert old_forward is not None

        per_batch_invocation_count: int = getattr(
            module, "per_batch_invocation_count", 0
        )
        per_batch_invocation_count += 1
        setattr(module, "per_batch_invocation_count", per_batch_invocation_count)

        total_invocations: int = getattr(module, "total_invocations", 0)
        total_invocations += 1
        setattr(module, "total_invocations", total_invocations)

        if per_batch_invocation_count > 1:
            if set_add(multicall_modules, module) and verbose:
                full_module_name: str = getattr(module, "full_module_name")
                print(f"[MULTI-CALL MODULE] {full_module_name}")

        return old_forward(*args, **kwargs)

    for name, m in model.named_modules():
        assert not hasattr(m, "old_forward")
        assert not hasattr(m, "per_batch_invocation_count")
        assert not hasattr(m, "total_invocations")
        assert not hasattr(m, "full_module_name")
        setattr(m, "old_forward", m.forward)
        setattr(m, "per_batch_invocation_count", 0)
        setattr(m, "total_invocations", 0)
        setattr(m, "full_module_name", name)
        m.forward = types.MethodType(new_forward, m)

    # Infer model device from first parameter
    model_device = infer_module_device(model)

    try:
        for args, kwargs in tqdm(
            zip(args_dataset, kwargs_dataset, strict=True),
            desc="Detecting multi-call modules",
            leave=False,
            unit="batch",
            total=len(args_dataset),
            disable=not show_progress,
        ):
            args = move_to_device(args, model_device)
            kwargs = move_to_device(kwargs, model_device)
            assert type(kwargs) is dict
            model(*args, **kwargs)
            for m in model.modules():
                setattr(m, "per_batch_invocation_count", 0)
    finally:
        for m in model.modules():
            old_forward = getattr(m, "old_forward")
            assert old_forward is not None
            m.forward = old_forward
            delattr(m, "per_batch_invocation_count")
            delattr(m, "old_forward")
            delattr(m, "full_module_name")

    if verbose:
        print(f"[MULTI-CALL MODULE] {len(multicall_modules)} multi-call modules found")

    return multicall_modules


def find_caller_and_callee_modules(
    model: nn.Module,
    caller_criterion: Callable[[nn.Module], bool],
    callee_criterion: Callable[[nn.Module], bool],
    relate_criterion: Callable[[nn.Module, nn.Module], bool],
    args_dataset: list[tuple[Any]],
    kwargs_dataset: list[dict[str, Any]],
    show_progress: bool,
    verbose: bool,
    description: str,
) -> dict[nn.Module, set[nn.Module]]:
    """
    Runs specified model on a specified dataset to find the pairs of modules that satisfy the following criteria:
    1. Caller module satisfies condition defined by 'caller_criterion(caller)' function.
    2. Callee module satisfies condition defined by 'callee_criterion(callee)' function.
    3. Caller module's forward method calls (directly or indirectly) callee's forward method.
    4. Caller and callee are related to each other according to `relate_criterion(caller, callee)` function (e.g. callee is a direct child of caller).
    The same caller module may call any number of different callee modules.

    Parameters:
        model: The PyTorch model to be analyzed.
        caller_criterion: Function returning True iff specified module is a caller.
        callee_criterion: Function returning True iff specified module is a callee.
        args_dataset: List of positional arguments.
        kwargs_dataset: List of keyword arguments.
        show_progress: Whether to display a progress bar during analysis.
        verbose: Whether to print out found modules' names.

    Returns:
        dict[nn.Module, set[nn.Module]]: Dictionary with caller modules as keys and callee modules as values.
    """
    assert len(args_dataset) > 0, "Empty calibration dataset"
    assert len(args_dataset) == len(kwargs_dataset), "Dataset length mismatch"

    result: dict[nn.Module, set[nn.Module]] = defaultdict(set)
    call_stack: Stack[nn.Module] = Stack()

    def new_forward(module: nn.Module, *args, **kwargs) -> Any:
        old_forward = getattr(module, "old_forward")
        assert old_forward is not None

        if callee_criterion(module):
            for caller in call_stack:
                if relate_criterion(caller, module):
                    result[caller].add(module)

        if caller_criterion(module):
            call_stack.push(module)
            try:
                return old_forward(*args, **kwargs)
            finally:
                call_stack.pop()
        else:
            return old_forward(*args, **kwargs)

    for name, m in model.named_modules():
        assert not hasattr(m, "old_forward")
        assert not hasattr(m, "full_module_name")
        setattr(m, "old_forward", m.forward)
        setattr(m, "full_module_name", name)
        m.forward = types.MethodType(new_forward, m)

    model_device = infer_module_device(model)

    try:
        for args, kwargs in tqdm(
            zip(args_dataset, kwargs_dataset, strict=True),
            desc=description,
            leave=False,
            unit="batch",
            total=len(args_dataset),
            disable=not show_progress,
        ):
            args = move_to_device(args, model_device)
            kwargs = move_to_device(kwargs, model_device)
            assert type(kwargs) is dict
            model(*args, **kwargs)
            assert len(call_stack) == 0

        if verbose:
            caller: nn.Module
            callees: set[nn.Module]
            for caller, callees in result.items():
                caller_name: str = getattr(caller, "full_module_name")
                callee_names: Iterable[str] = (
                    getattr(c, "full_module_name") for c in callees
                )
                print(f"[INFO] {caller_name} CALLS {', '.join(callee_names)}")
    finally:
        for m in model.modules():
            old_forward = getattr(m, "old_forward")
            assert old_forward is not None
            m.forward = old_forward
            delattr(m, "old_forward")
            delattr(m, "full_module_name")

    return result


def gptq_quantize(
    model: nn.Module,
    gptq_config: UniversalGPTQConfig,
    args_dataset: list[tuple[Any]],
    kwargs_dataset: list[dict[str, Any]],
) -> None:
    """
    Apply GPTQ quantization to a model layer-by-layer.

    This is the main driver function for the universal GPTQ algorithm. It
    implements a frontier-based execution strategy where modules are processed
    in topological order:

    1. Wrap the model with GPTQ state tracking
    2. Iteratively run the model to find frontier modules
    3. For each frontier module:
       - COLLECT state: Quantize weights using accumulated Hessian
       - CACHE state: Cache outputs for downstream replay
    4. Collect all quantizers and attach to model
    5. Unwrap the model (restore original forward methods)

    The function ensures proper cleanup via try/finally even if errors occur.

    Parameters:
        model: The PyTorch model to quantize (modified in-place).
        gptq_config: GPTQ configuration with quantization parameters.
        args_dataset: List of cached positional arguments from calibration.
        kwargs_dataset: List of cached keyword arguments from calibration.

    Raises:
        AssertionError: If the calibration dataset is empty or mismatched.
    """
    assert len(args_dataset) > 0, "Empty calibration dataset"
    assert len(args_dataset) == len(kwargs_dataset), "Dataset length mismatch"

    if not has_gptq_data(model):
        cacheable_module_patterns: list[re.Pattern] = [
            re.compile(pattern) for pattern in gptq_config.cacheable_modules
        ]

        # For each module determine if it is cacheable
        for full_module_name, m in model.named_modules():
            is_cacheable: bool = bool(full_module_name) and any(
                regex.fullmatch(full_module_name) is not None
                for regex in cacheable_module_patterns
            )
            assert not hasattr(m, "is_cacheable")
            setattr(m, "is_cacheable", is_cacheable)

        # Detect cacheable modules that call other cacheable modules
        cacheable_modules_callers_to_callees: dict[nn.Module, set[nn.Module]]
        if gptq_config.allow_calls_between_cacheable_modules:
            cacheable_modules_callers_to_callees = find_caller_and_callee_modules(
                model=model,
                caller_criterion=lambda m: getattr(m, "is_cacheable", False),
                callee_criterion=lambda m: getattr(m, "is_cacheable", False),
                relate_criterion=lambda caller, callee: callee not in caller.modules(),
                args_dataset=args_dataset,
                kwargs_dataset=kwargs_dataset,
                show_progress=gptq_config.show_progress,
                verbose=gptq_config.verbose,
                description="Detecting cacheable modules calling other cacheable modules",
            )
        else:
            cacheable_modules_callers_to_callees = find_caller_and_callee_modules(
                model=model,
                caller_criterion=lambda m: getattr(m, "is_cacheable", False),
                callee_criterion=lambda m: getattr(m, "is_cacheable", False),
                relate_criterion=lambda caller, callee: True,
                args_dataset=args_dataset,
                kwargs_dataset=kwargs_dataset,
                show_progress=gptq_config.show_progress,
                verbose=gptq_config.verbose,
                description="Detecting cacheable modules calling other cacheable modules",
            )

        if cacheable_modules_callers_to_callees:
            raise RuntimeError(
                "Cacheable modules calling other cacheable modules detected"
            )

        # Detect modules that are called multiple times in a single input batch
        multicall_modules: set[nn.Module] = (
            find_multicall_modules(
                model,
                args_dataset=args_dataset,
                kwargs_dataset=kwargs_dataset,
                show_progress=gptq_config.show_progress,
                verbose=gptq_config.verbose,
            )
            if gptq_config.ignore_multi_call_modules
            else set()
        )

        wrap_model(
            model,
            full_model_name="",
            ignored_module_patterns=(
                [re.compile("lm_head.*")] if not gptq_config.quantize_lm_head else []
            ),
            ignored_modules=multicall_modules,
        )

    gptq_data: GPTQ_Data
    try:
        while True:
            # Run model to do either of the following:
            # - collect inputs of frontier submodules to accumulate their Hessians
            # - cache the outputs of frontier submodules
            frontier_submodules: set[nn.Module] = run_model(
                model,
                args_dataset,
                kwargs_dataset,
                show_progress=gptq_config.show_progress,
            )

            if not frontier_submodules:
                break

            if len(frontier_submodules) > 1 and gptq_config.verbose:
                print(f"[INFO] Hit {len(frontier_submodules)} frontier modules")

            # 1. Sort frontier modules into 3 lists: cacheable, ready to quantize, not ready to quantize
            modules_to_cache: list[nn.Module] = []
            modules_ready_to_quantize: list[nn.Module] = []
            modules_not_ready_to_quantize: list[nn.Module] = []
            for frontier_submodule in frontier_submodules:
                gptq_data = get_gptq_data(frontier_submodule)
                match gptq_data.state:
                    case GPTQ_STATE.CACHE:
                        modules_to_cache.append(frontier_submodule)
                    case GPTQ_STATE.COLLECT:
                        if (
                            len(gptq_data.collected_inputs)
                            == gptq_data.total_invocations
                        ):
                            modules_ready_to_quantize.append(frontier_submodule)
                        else:
                            modules_not_ready_to_quantize.append(frontier_submodule)
                    case _:
                        assert False, "We should never get here"

            # 2. If there any cacheable modules were hit
            if modules_to_cache:
                # 2.1. Process cacheable modules in proprity order
                for module_to_cache in modules_to_cache:
                    finish_caching(
                        module_to_cache,
                        release_children_cache=gptq_config.allow_calls_between_cacheable_modules,
                        verbose=gptq_config.verbose,
                    )

                # 2.2. Process quantizable modules that have collected their inputs
                for module_ready_to_quantize in modules_ready_to_quantize:
                    finish_collection(module_ready_to_quantize, gptq_config)

                # 2.3.
                for module_not_ready_to_quantize in modules_not_ready_to_quantize:
                    gptq_data = get_gptq_data(module_not_ready_to_quantize)
                    gptq_data.collected_inputs.clear()

            # 3. No cacheable modules were hit
            else:
                # 3.1. If there are any modules ready to be quantized
                if modules_ready_to_quantize:
                    # 3.1.1. Quantize modules that have callected all required inputs
                    for module_ready_to_quantize in modules_ready_to_quantize:
                        finish_collection(module_ready_to_quantize, gptq_config)

                    # 3.1.2. Reset modules that haven't collected all required inputs (they will start over at the next model replay)
                    for module_not_ready_to_quantize in modules_not_ready_to_quantize:
                        gptq_data = get_gptq_data(module_not_ready_to_quantize)
                        gptq_data.collected_inputs.clear()

                # 3.2. No modules ready to be quantized
                else:
                    if gptq_config.verbose:
                        print(
                            f"[WARNING] No modules have collected all required inputs"
                        )

                    # Pick the module that has the maximum percantage of collected inputs and quantize it,
                    # reset others (they will start over at the next model replay)
                    max_collected_inputs_percentage = 0.0
                    best_module_to_quantize: nn.Module | None = None
                    for module_not_ready_to_quantize in modules_not_ready_to_quantize:
                        gptq_data = get_gptq_data(module_not_ready_to_quantize)
                        collected_inputs_percentage = (
                            len(gptq_data.collected_inputs)
                            / gptq_data.total_invocations
                            * 100.0
                        )
                        if gptq_config.verbose:
                            print(
                                f"[{gptq_data.full_module_name}] collected {len(gptq_data.collected_inputs)} / {gptq_data.total_invocations} ({collected_inputs_percentage:.0f}%) inputs"
                            )
                        assert collected_inputs_percentage > 0.0
                        if (
                            collected_inputs_percentage
                            > max_collected_inputs_percentage
                        ):
                            if best_module_to_quantize:
                                get_gptq_data(
                                    best_module_to_quantize
                                ).collected_inputs.clear()
                            best_module_to_quantize = module_not_ready_to_quantize
                            max_collected_inputs_percentage = (
                                collected_inputs_percentage
                            )
                        else:
                            gptq_data.collected_inputs.clear()
                    assert best_module_to_quantize is not None
                    finish_collection(best_module_to_quantize, gptq_config)

        quantizers: dict[str, Quantizer] = {}
        collect_quantizers(
            model,
            full_model_name="",
            quantizers=quantizers,
        )
        setattr(model, "quantizers", quantizers)
    finally:
        unwrap_model(model, retain_gptq_data=gptq_config.debug_mode)

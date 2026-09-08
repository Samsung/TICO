"""
Universal GPTQ Quantizer for PyTorch Models.

This module implements a model-agnostic GPTQ (Gradient-based Post-Training Quantization)
algorithm that works with any PyTorch model composed of standard layers (Linear, Conv1d,
Conv2d, Conv3d, ConvTranspose2d) without requiring model-specific wrapper classes.

The quantizer uses a frontier-based execution strategy with a state machine approach:
1. COLLECT: Accumulate Hessian information from calibration inputs
2. CACHE: Cache module outputs for downstream replay
3. RETURN_CACHED: Return cached outputs without computation
4. COMPUTE: Normal forward pass after quantization

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
from tico.utils.utils import move_to_device
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

    COLLECT (1) --[finish_collection]--> CACHE (2) --[finish_caching]--> RETURN_CACHED (3) --> COMPUTE (4)
         |                                    |                              |                   |
         | Hessian accumulation               | Output caching               | Output            | Output cache
         | StopForward raised                 | StopForward raised           | cached            | cleared
         v                                    v                              v                   v
    (quantize weights)                  (ready to return cached)         (return cached)       (compute as usual)
    """

    COLLECT = 1
    CACHE = 2
    RETURN_CACHED = 3
    COMPUTE = 4


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
        cached_output: List of cached outputs for replay (freed after use).
        state: Current state in the quantization lifecycle.
        invocation_idx: Current invocation index during replay.
        num_invocations: Total number of invocations (set after caching phase).
        weight_device: Device where the module's parameters reside.
        out_device: Device where module output should reside.
    """

    full_module_name: str
    old_forward: Callable
    gptq: GPTQ | None
    quantizer: Quantizer | None
    cached_output: list[Any]
    state: GPTQ_STATE
    invocation_idx: int
    num_invocations: int
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
    ignored_module_patterns: Iterable[re.Pattern] = [],
) -> None:
    """
    Recursively wrap a model's forward methods for GPTQ quantization.

    This function attaches GPTQ_Data to each module in the model hierarchy
    and replaces forward methods with a state-aware wrapper. The wrapper
    handles four states:

    - COLLECT: Accumulate Hessian from inputs, raise StopForward
    - CACHE: Run original forward, cache output, raise StopForward
    - RETURN_CACHED: Return cached outputs without computation
    - COMPUTE: Run original forward without caching

    After wrapping, the model is ready for layer-by-layer quantization
    using the frontier-based execution strategy.

    Parameters:
        model: The model to wrap (modified in-place).
        full_model_name: Hierarchical dot-delimited model name following "grandparent.parent.clild" pattern.
        ignored_module_patterns: Collection of regular expressions to be matched against submodules' full names. If matched a submodule is not quantized.
    """

    def new_forward(module: nn.Module, *args, **kwargs) -> Any:
        gptq_data: GPTQ_Data = get_gptq_data(module)
        match gptq_data.state:
            case GPTQ_STATE.COLLECT:
                if gptq_data.gptq is None:
                    gptq_data.gptq = GPTQ(module)
                gptq_data.gptq.add_batch(
                    # Move input to model's weight device for Hessian accumulation
                    inp=args[0].data.to(gptq_data.weight_device),
                    out=None,  # out is ignored in GPTQ.add_batch
                )
                raise StopForward(module)

            case GPTQ_STATE.CACHE:
                args = move_to_device(args, gptq_data.weight_device)
                kwargs = move_to_device(kwargs, gptq_data.weight_device)
                assert type(kwargs) is dict
                out: Any = gptq_data.old_forward(*args, **kwargs)
                if gptq_data.out_device is None:
                    gptq_data.out_device = infer_object_device(out)
                # Cache on CPU to save GPU memory
                gptq_data.cached_output.append(move_to_cpu(out))
                gptq_data.invocation_idx += 1
                raise StopForward(module)

            case GPTQ_STATE.RETURN_CACHED:
                result: Any = gptq_data.cached_output[gptq_data.invocation_idx]
                gptq_data.invocation_idx = (
                    gptq_data.invocation_idx + 1
                ) % gptq_data.num_invocations
                # Move cached output back to model's device
                if gptq_data.out_device:
                    result = move_to_device(result, gptq_data.out_device)
                return result

            case GPTQ_STATE.COMPUTE:
                args = move_to_device(args, gptq_data.weight_device)
                kwargs = move_to_device(kwargs, gptq_data.weight_device)
                assert type(kwargs) is dict
                return gptq_data.old_forward(*args, **kwargs)

            case _:
                assert False  # we should never get here

    not_quantizable: bool = type(model) not in _QUANTIZABLE_LAYER_TYPES or any(
        regex.match(full_model_name) is not None for regex in ignored_module_patterns
    )

    setattr(
        model,
        "gptq_data",
        GPTQ_Data(
            full_module_name=full_model_name,
            old_forward=model.forward,
            gptq=None,
            quantizer=None,
            cached_output=[],
            state=GPTQ_STATE.CACHE if not_quantizable else GPTQ_STATE.COLLECT,
            invocation_idx=0,
            num_invocations=0,
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
        )


def unwrap_model(model: nn.Module) -> None:
    """
    Restore original module structure after GPTQ quantization.

    This function reverses the wrapping applied by `wrap_model`, restoring
    the original forward methods and removing the `gptq_data` attribute.
    It also clears cached outputs to free memory.

    Parameters:
        model: The wrapped model to unwrap (modified in-place).
    """
    gptq_data: GPTQ_Data = get_gptq_data(model)

    model.forward = gptq_data.old_forward

    # Immediately free the cache contents
    gptq_data.cached_output.clear()
    delete_gptq_data(model)

    child: nn.Module
    for child in model.children():
        unwrap_model(child)


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
    gptq_data: GPTQ_Data = get_gptq_data(model)
    frontier_submodules: set[nn.Module] = set()
    for args, kwargs in tqdm(
        zip(args_dataset, kwargs_dataset, strict=True),
        desc=f"(Re)playing model",
        leave=False,
        unit="batch",
        disable=not show_progress,
    ):
        args = move_to_device(args, gptq_data.weight_device)
        kwargs = move_to_device(kwargs, gptq_data.weight_device)
        assert type(kwargs) is dict
        try:
            model(*args, **kwargs)
        except StopForward as stop_fwd:
            assert stop_fwd.module is not None
            frontier_submodules.add(stop_fwd.module)
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
    gptq_data: GPTQ_Data = get_gptq_data(module)
    assert gptq_data.gptq is not None

    # Check if Hessian was actually accumulated
    if gptq_data.gptq.H is None or gptq_data.gptq.H.numel() == 0:
        raise RuntimeError(
            f"Module {type(module).__name__} received no calibration data"
        )

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
    gptq_data.state = GPTQ_STATE.CACHE
    gptq_data.quantizer = gptq_data.gptq.quantizer
    gptq_data.gptq = None
    gptq_data.invocation_idx = 0


def finish_caching(module: nn.Module) -> None:
    """
    Complete output caching and transition a module to RETURN_CACHED state.

    This function is called when a module has cached outputs from all
    calibration batches. It transitions the module to RETURN_CACHED state
    and frees children's cached outputs (memory optimization).

    Parameters:
        module: The module to transition (must be in CACHE state).
    """
    gptq_data: GPTQ_Data = get_gptq_data(module)
    gptq_data.state = GPTQ_STATE.RETURN_CACHED
    gptq_data.num_invocations = gptq_data.invocation_idx
    gptq_data.invocation_idx = 0

    # Free children's cached outputs
    for child in module.children():
        child_gptq_data = get_gptq_data(child)
        child_gptq_data.cached_output.clear()
        child_gptq_data.state = GPTQ_STATE.COMPUTE


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
        wrap_model(
            model,
            full_model_name="",
            ignored_module_patterns=[re.compile("lm_head.*")]
            if not gptq_config.quantize_lm_head
            else [],
        )

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

            frontier_submodule: nn.Module
            for frontier_submodule in frontier_submodules:
                gptq_data: GPTQ_Data = get_gptq_data(frontier_submodule)
                match gptq_data.state:
                    case GPTQ_STATE.COLLECT:
                        assert type(frontier_submodule) in _QUANTIZABLE_LAYER_TYPES
                        finish_collection(frontier_submodule, gptq_config)

                    case GPTQ_STATE.CACHE:
                        finish_caching(frontier_submodule)

                    case _:
                        assert False  # we should never get here

        quantizers: dict[str, Quantizer] = {}
        collect_quantizers(
            model,
            full_model_name="",
            quantizers=quantizers,
        )
        setattr(model, "quantizers", quantizers)
    finally:
        unwrap_model(model)

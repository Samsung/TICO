# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

from tico.quantization.evaluation.metric import MetricCalculator
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.utils.version import package_version_is_at_least


# Type aliases (for more descriptive type hints)
ArgName = str
ModuleOutput = Any
ModuleName = str


@dataclass(frozen=True)
class TensorStatistics:
    """Statistical summary of a tensor's elements."""

    mean: float
    """Mean value of all tensor elements."""

    min: float
    """Minimum value among all tensor elements."""

    max: float
    """Maximum value among all tensor elements."""

    stddev: float
    """Standard deviation of all tensor elements."""

    def _asdict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class DifferenceStatistics(TensorStatistics):
    peir: float
    """PEIR (Peak Error To Interval Ratio)."""


class ModuleInputOutput(NamedTuple):
    """
    Captured input/output data for a single module during forward pass.

    This named tuple stores all information about a module's execution,
    including its inputs, keyword arguments, and output.
    """

    module: torch.nn.Module
    """The module instance that was executed."""

    module_name: ModuleName | None
    """(Optional) Fully qualified name of the module in the model hierarchy."""

    inputs: tuple[torch.Tensor, ...]
    """Positional arguments passed to the module's forward method."""

    kwargs: dict[str, Any]
    """Keyword arguments passed to the module's forward method."""

    output: ModuleOutput
    """Output returned by the module's forward method."""

    quantization: Iterable[ObserverBase] | None = None
    """(Optional) Quantization parameters associated with the module"""

    def as_serializable(
        self,
        include_tensor_content: bool = False,
        include_type: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the module input/output data to a JSON-serializable dictionary.

        Args:
            include_tensor_content: If True, include actual tensor elements
                (for small tensors or "interesting" modules).
            include_type: If True, include 'type' field in the output dictionary.

        Returns:
            Dictionary containing module_name, module_type, inputs, kwargs, and output,
            all in JSON-serializable format.
        """
        data: dict[str, Any] = {
            "module_name": self.module_name,
            "module_type": type(self.module).__name__,
            "inputs": model_output_to_serializable(
                self.inputs,
                include_tensor_content=include_tensor_content,
                include_type=include_type,
            ),
            "kwargs": {
                k: model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in self.kwargs.items()
            },
            "output": model_output_to_serializable(
                self.output,
                include_tensor_content=include_tensor_content,
                include_type=include_type,
            ),
        }
        if self.quantization is not None:
            data["quantization"] = model_output_to_serializable(
                self.quantization,
                include_tensor_content=include_tensor_content,
                include_type=include_type,
            )
        return data


def get_tensor_statistics(x: torch.Tensor) -> TensorStatistics:
    """
    Compute statistical summary of a tensor's elements.

    Args:
        x: Input tensor.

    Returns:
        TensorStatistics containing mean, min, max, and stddev.
    """
    x = x.to(torch.float)
    return TensorStatistics(
        mean=torch.mean(x).item(),
        min=torch.min(x).item(),
        max=torch.max(x).item(),
        stddev=torch.std(x, correction=0).item(),
    )


def detach_tensors(x: ModuleOutput) -> ModuleOutput:
    """
    Recursively detach and clone tensors in a module output.

    Creates detached copies of tensors to prevent gradient tracking and
    preserve tensor values for later comparison.

    Args:
        x: Module output (tensor, ModelOutput, dict, or iterable).

    Returns:
        A copy of the input with all tensors detached and cloned.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone()

    data: dict[str, Any] | Iterable

    if isinstance(x, ModelOutput):
        data = {str(k): detach_tensors(v) for k, v in x.__dict__.items()}
        return x.__class__(**data)

    if isinstance(x, dict):
        data = {k: detach_tensors(v) for k, v in x.items()}
        return data

    if isinstance(x, str):
        return x

    if isinstance(x, Iterable):
        data = (detach_tensors(i) for i in x)
        # For iterables like lists/tuples, we need to convert the generator to the appropriate type
        if hasattr(x, "__class__"):
            try:
                return x.__class__(data)  # type: ignore[call-arg]
            except TypeError:
                # If the constructor doesn't accept a generator, convert to list first
                return x.__class__(list(data))  # type: ignore[call-arg]
        return list(data)

    return x


def model_output_to_serializable(
    x: ModuleOutput,
    include_tensor_content: bool = False,
    include_type: bool = True,
) -> str | Number | dict[str, Any]:
    """
    Convert a module output to a JSON-serializable format.

    Recursively converts tensors, ModelOutput objects, named tuples,
    dicts, and iterables to dictionaries with type information.

    Args:
        x: The output value to convert.
        include_tensor_content: If True, include actual tensor elements
            (for small tensors or "interesting" modules).
        include_type: If True, include 'type' field in the output dictionary.

    Returns:
        A JSON-serializable representation of the output.
    """
    data: dict[str, Any] = {}
    if include_type:
        data["type"] = x.__class__.__name__

    if isinstance(x, torch.Tensor):
        data["dtype"] = str(x.dtype)
        data["shape"] = str(x.shape)
        if x.numel() > 1:
            data["statistics"] = get_tensor_statistics(x)._asdict()
        if x.numel() <= 1 or include_tensor_content:
            data["value"] = x.tolist()
        return data

    if isinstance(x, ModelOutput):
        data.update(
            {
                k: model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in x.__dict__.items()
            }
        )
        return data

    if isinstance(x, AffineObserverBase):
        data.update(
            {
                "name": x.name,
                "dtype": str(x.dtype),
                "qscheme": str(x.qscheme),
                "channel_axis": str(x.channel_axis),
                "min_val": model_output_to_serializable(
                    x.min_val,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                ),
                "max_val": model_output_to_serializable(
                    x.max_val,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                ),
                "scale": model_output_to_serializable(
                    x._cached_scale,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                ),
                "zp": model_output_to_serializable(
                    x._cached_zp,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                ),
            }
        )
        return data

    # NamedTuple
    if hasattr(x, "_asdict"):
        data.update(
            {
                str(k): model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in x._asdict().items()
            }
        )
        return data

    if isinstance(x, dict):
        data.update(
            {
                str(k): model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in x.items()
            }
        )
        return data

    if isinstance(x, str) or isinstance(x, Number):
        return x

    if isinstance(x, Iterable):
        data.update(
            {
                str(i): model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for i, v in enumerate(x)
            }
        )
        return data

    return str(x)


def build_fqn_map(root: torch.nn.Module) -> dict[torch.nn.Module, str]:
    """
    Return {module_object: full_qualified_name} without touching the modules.
    """
    return {m: n for n, m in root.named_modules()}


@contextmanager
def module_hook(
    hook: Callable[
        [torch.nn.Module, tuple[torch.Tensor, ...], dict[str, Any], Any], Any
    ]
):
    """
    Context manager for registering a global forward hook on all modules.

    Args:
        hook: Callback function to be called for each module's forward pass.

    Yields:
        None. The hook is automatically removed when exiting the context.
    """
    handle: torch.utils.hooks.RemovableHandle
    if package_version_is_at_least(package_name="torch", required_version="2.6"):
        handle = torch.nn.modules.module.register_module_forward_hook(
            hook, with_kwargs=True, always_call=True
        )
    else:

        def fallback_hook(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: ModuleOutput,
        ) -> Any:
            return hook(module, inputs, {}, output)

        handle = torch.nn.modules.module.register_module_forward_hook(
            fallback_hook, always_call=True
        )

    yield
    handle.remove()


@contextmanager
def full_tensor_printing():
    """
    Context manager for enabling full tensor printing.

    Sets torch print options to 'full' profile to show all tensor elements,
    then restores previous settings on exit.
    """
    prev_options = torch._tensor_str.PRINT_OPTS.__dict__.copy()
    try:
        torch.set_printoptions(profile="full")
        yield
    finally:
        torch.set_printoptions(**prev_options)


def create_tracing_hook(
    print_input_output: bool,
    module_outputs: MutableMapping[ModuleName, ModuleOutput] | None,
    interesting_modules: Iterable[str] = [],
    breakpoint_on_interesting_modules: bool = False,
) -> Callable[[ModuleInputOutput], None]:
    """
    Create a hook function for tracing module inputs/outputs.

    Args:
        print_input_output: If True, print module input/output data.
        module_outputs: Dictionary to store module outputs (keyed by module name).
        interesting_modules: List of module names to inspect in detail.
        breakpoint_on_interesting_modules: If True, break into debugger for interesting modules.

    Returns:
        A hook function suitable for use with trace_model_input_output.
    """

    def hook(data: ModuleInputOutput):
        this_module_is_interesting = (
            data.module_name is not None and data.module_name in interesting_modules
        )
        if print_input_output:
            print(f"\n{'='*80}")
            print(
                json.dumps(
                    data.as_serializable(
                        include_tensor_content=this_module_is_interesting
                    ),
                    indent=4,
                )
            )
            print(f"{'='*80}")

        if module_outputs is not None and data.module_name is not None:
            module_outputs[data.module_name] = data.output

        if this_module_is_interesting and breakpoint_on_interesting_modules:
            breakpoint()

    return hook


def trace_model_input_output(
    model: torch.nn.Module,
    model_inputs: dict[ArgName, torch.Tensor],
    hook: Callable[[ModuleInputOutput], Any],
    skip_ptqwrappers: bool = True,
):
    """
    Run model forward pass and trace all module inputs/outputs.

    Registers a forward hook on all modules, runs the model, and calls
    the provided hook function for each module's execution.

    Args:
        model: The model to trace.
        model_inputs: Input data for the model forward pass.
        hook: Callback function called for each module with ModuleInputOutput data.
        skip_ptqwrappers: If True, skip PTQWrapper modules (default: True).
    """
    module_to_name: dict[torch.nn.Module, ModuleName] | None
    if isinstance(model, QuantModuleBase):
        module_to_name = None
    else:
        module_to_name = build_fqn_map(model)

    def trim_prefix_up_to(s: str, char: str) -> str:
        """
        Remove prefix from a string up to and including the first occurrence of a character.

        Args:
            s: Input string.
            char: Character to search for.

        Returns:
            String with prefix removed, or original string if character not found.
        """
        char_index: int = s.find(char)
        if char_index >= 0:
            return s[char_index + 1 :]
        else:
            return s

    def _hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
        output: ModuleOutput,
    ):
        if isinstance(module, PTQWrapper) and skip_ptqwrappers:
            return

        if not module_to_name and not hasattr(module, "fp_name"):
            return

        module_name: ModuleName
        if module_to_name:
            module_name = module_to_name[module]
        else:
            module_name = module.fp_name
            # fp_name attribute of PTQWrapper is usually None,
            # fp_name is set to a meaningful string for wrapped modules only.
            if module_name is not None:
                # PTQWrapper adds an fp_name "model" to the top-level model,
                # which is why every submodule obtains an additional "model."
                # prefix in its fp_name. We trim that prefix in order to be
                # consistent with usual (unquantized, unwrapped) models.
                module_name = trim_prefix_up_to(module_name, char=".")

        data = ModuleInputOutput(
            module=module,
            module_name=module_name,
            inputs=inputs,
            kwargs=kwargs,
            output=detach_tensors(output),
            quantization=module._all_observers()
            if isinstance(module, QuantModuleBase)
            else None,
        )
        hook(data)

    with module_hook(_hook):
        with torch.no_grad():
            _ = model(**model_inputs)


class DataMismatchError(Exception):
    ...


def compare_outputs(
    lhs: ModuleOutput,
    rhs: ModuleOutput,
    full_tensor_diff: bool = False,
) -> Number | torch.Tensor | DifferenceStatistics | dict[str, Any] | Exception | None:
    """
    Compare two module outputs and compute their difference.

    Recursively compares outputs of various types (tensors, numbers, dicts,
    lists, named tuples, ModelOutput objects) and returns the difference.

    Args:
        lhs: Left-hand side output (from unquantized model).
        rhs: Right-hand side output (from quantized model).
        full_tensor_diff: If True, return full tensor difference instead of statistics.

    Returns:
        The difference between outputs. For tensors, returns statistics or full tensor.
        For containers, returns a dict of differences. Returns None if both are None.

    Raises:
        DataMismatchError: If the types or structure of lhs and rhs don't match.
    """
    if type(lhs) != type(rhs):
        raise DataMismatchError(f"Type mismatch: {type(lhs)} != {type(rhs)}")

    # None
    if lhs is None:
        return None

    # Tensor
    if isinstance(lhs, torch.Tensor):
        if full_tensor_diff:
            return lhs.to(torch.float) - rhs.to(torch.float)
        else:
            abs_delta: torch.Tensor = (lhs - rhs).abs().to(torch.float)
            delta_stats: TensorStatistics = get_tensor_statistics(abs_delta)
            interval = (lhs.max() - lhs.min()).item()
            if interval != 0.0:
                peir = delta_stats.max / interval
                return DifferenceStatistics(**delta_stats._asdict(), peir=peir)
            return delta_stats

    # Number
    if isinstance(lhs, Number):
        return abs(lhs - rhs)

    diff: dict[str, Any] = {}

    # List, Tuple: compare element-wise
    if isinstance(lhs, Sequence):
        if len(lhs) != len(rhs):
            raise DataMismatchError(f"Length mismatch: {len(lhs)} != {len(rhs)}")
        for i, (lhs_val, rhs_val) in enumerate(zip(lhs, rhs)):
            try:
                diff[str(i)] = compare_outputs(lhs_val, rhs_val)
            except Exception as ex:
                diff[str(i)] = ex
        return diff

    # Dict
    if isinstance(lhs, dict):
        lhs_keys = set(lhs.keys())
        rhs_keys = set(rhs.keys())

        if lhs_keys != rhs_keys:
            raise DataMismatchError(
                "Dict key mismatch: "
                f"lhs_keys={sorted(lhs_keys)}, "
                f"rhs_keys={sorted(rhs_keys)}"
            )

        for key in lhs_keys:
            lhs_val = lhs[key]
            rhs_val = rhs[key]
            try:
                diff[key] = compare_outputs(lhs_val, rhs_val)
            except Exception as ex:
                diff[key] = ex
        return diff

    # Arbitrary type: compare by fields
    attr_names = lhs.__dict__.keys()
    for attr_name in attr_names:
        lhs_val = lhs.__dict__[attr_name]
        rhs_val = rhs.__dict__[attr_name]
        try:
            diff[attr_name] = compare_outputs(lhs_val, rhs_val)
        except Exception as ex:
            diff[attr_name] = ex
    return diff


def compare_side_by_side(
    model_outputs_a: Mapping[str, ModuleOutput],
    model_outputs_b: Mapping[str, ModuleOutput],
    interesting_modules: Iterable[str] = [],
    breakpoint_on_interesting_modules: bool = False,
) -> None:
    """
    Compare outputs from two models side-by-side and print differences.

    For similarly named submodules that are present in both models, computes and prints the difference
    between outputs. Useful for comparing quantized vs unquantized model outputs.

    Args:
        model_outputs_a: First model's outputs (keyed by module name).
        model_outputs_b: Second model's outputs (keyed by module name).
        interesting_modules: Modules to inspect in detail (full tensor diff).
        breakpoint_on_interesting_modules: If True, break into debugger for interesting modules.
    """
    common_module_names: set[ModuleName] = (
        model_outputs_a.keys() & model_outputs_b.keys()
    )
    max_module_name_len = (
        max(len(name) for name in common_module_names) if common_module_names else 0
    )
    format_str = f"{{: <{max_module_name_len}}} {{:}}"

    print("-" * 80)
    print(format_str.format("MODULE NAME", "DIFFERENCE"))
    print("-" * 80)

    for module_name in model_outputs_a.keys():
        if module_name not in model_outputs_b:
            continue
        output_a = model_outputs_a[module_name]
        output_b = model_outputs_b[module_name]
        diff: Number | torch.Tensor | dict[str, Any] | Exception | None
        this_module_is_interesting = module_name in interesting_modules
        try:
            diff = compare_outputs(
                output_a, output_b, full_tensor_diff=this_module_is_interesting
            )
        except Exception as ex:
            diff = ex
        print(
            format_str.format(
                module_name,
                model_output_to_serializable(
                    diff,
                    include_tensor_content=this_module_is_interesting,
                    include_type=False,
                ),
            )
        )

        if this_module_is_interesting and breakpoint_on_interesting_modules:
            breakpoint()


def extract_tensor(output: Any) -> Optional[torch.Tensor]:
    """
    Extract the first tensor found inside an arbitrary output structure.

    Supports:
    - torch.Tensor
    - tuple / list
    - dict
    - dataclass / HF ModelOutput-like objects
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            t = extract_tensor(item)
            if t is not None:
                return t
        return None

    if isinstance(output, dict):
        for v in output.values():
            t = extract_tensor(v)
            if t is not None:
                return t
        return None

    # dataclass / ModelOutput-like objects
    if hasattr(output, "__dict__"):
        for v in vars(output).values():
            t = extract_tensor(v)
            if t is not None:
                return t
        return None

    return None


def save_fp_outputs(
    model: torch.nn.Module,
) -> Tuple[List[torch.utils.hooks.RemovableHandle], Dict[str, torch.Tensor]]:
    """
    Register forward-hooks on every `QuantModuleBase` wrapper itself (not the
    wrapped `module`) and cache its output while the wrapper runs in CALIB mode.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose wrappers are already switched to CALIB mode
        (`enable_calibration()` has been called).

    Returns
    -------
    handles : list[RemovableHandle]
        Hook handles; call `.remove()` on each one to detach the hooks.
    cache : dict[str, torch.Tensor]
        Mapping "wrapper-name → cached FP32 activation" captured from the first
        forward pass. Keys default to `wrapper.fp_name`; if that attribute is
        `None`, the `id(wrapper)` string is used instead.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _save(name: str):
        def hook(_, __, out):
            tensor = extract_tensor(out)
            if tensor is None:
                print(f"[{name}] no tensor found in output type {type(out)}")
                return
            cache[name] = tensor.detach()

        return hook

    for m in model.modules():
        if isinstance(m, QuantModuleBase):
            name = m.fp_name or str(id(m))
            handles.append(m.register_forward_hook(_save(name)))

    return handles, cache


def compare_layer_outputs(
    model: torch.nn.Module,
    cache: Dict[str, torch.Tensor],
    *,
    metrics: Optional[List[str]] = None,
    custom_metrics: Optional[Dict[str, Callable]] = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    collect: bool = False,
):
    """
    Register forward-hooks on every `QuantModuleBase` wrapper to compare its
    QUANT-mode output to the FP32 reference saved by `save_fp_outputs()`.

    Each hook prints a per-layer diff report:

        ✓  layer_name  max=1.23e-02  mean=8.45e-04     (within tolerance)
        ⚠️ layer_name  max=3.07e+00  mean=5.12e-01     (exceeds tolerance)

    Parameters
    ----------
    model : torch.nn.Module
        The model whose wrappers are now in QUANT mode
        (`freeze_qparams()` has been called).
    cache : dict[str, torch.Tensor]
        The reference activations captured during CALIB mode.
    metrics
        Metrics to compute. Defaults to `["diff"]`. Add `peir` to print PEIR.
    custom_metrics
        Optional user metric functions. Same signature as built-ins.
    rtol, atol : float, optional
        Relative / absolute tolerances used to flag large deviations
        (similar to `torch.allclose` semantics).
    collect : bool, optional
        • False (default) → print one-line report per layer, return `None`
        • True            → suppress printing, return a nested dict
                                {layer_name -> {metric -> value}}

    Returns
    -------
    handles
        Hook handles; call `.remove()` once diffing is complete.
    results
        Only if *collect* is True.
    """
    metrics = metrics or ["diff"]
    calc = MetricCalculator(custom_metrics)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    results: Dict[
        str, Dict[str, float]
    ] = {}  # Dict[layer_name, Dict[metric_name, value]]

    def _cmp(name: str):
        ref = cache.get(name)

        def hook(_, __, out):
            if ref is None:
                if not collect:
                    print(f"[{name}]  no cached reference")
                return
            out = extract_tensor(out)
            if out is None:
                print(f"[{name}] no tensor found in output type {type(out)}")
                return

            # Compute all requested metrics
            res = calc.compute([ref], [out], metrics)  # lists with length-1 tensors
            res = {k: v[0] for k, v in res.items()}  # flatten

            if collect:
                results[name] = res  # type: ignore[assignment]
                return

            # Pretty print ------------------------------------------------ #
            diff_val = res.get("diff") or res.get("max_abs_diff")
            thresh = atol + rtol * ref.abs().max().item()
            flag = "⚠️" if (diff_val is not None and diff_val > thresh) else "✓"  # type: ignore[operator]

            pieces = [f"{flag} {name:45s}"]
            for key, val in res.items():
                pieces.append(f"{key}={val:<7.4}")
            print("  ".join(pieces))

        return hook

    for m in model.modules():
        if isinstance(m, PTQWrapper):
            # skip the internal fp module inside the wrapper
            continue
        if isinstance(m, QuantModuleBase):
            lname = m.fp_name or str(id(m))
            handles.append(m.register_forward_hook(_cmp(lname)))

    if collect:
        return handles, results
    return handles

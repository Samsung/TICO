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

from typing import Callable, Dict, List, Optional, Tuple

import torch

from tico.experimental.quantization.evaluation.metric import MetricCalculator
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)


def build_fqn_map(root: torch.nn.Module) -> dict[torch.nn.Module, str]:
    """
    Return {module_object: full_qualified_name} without touching the modules.
    """
    return {m: n for n, m in root.named_modules()}


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
        def hook(_, __, out: torch.Tensor | Tuple):
            if isinstance(out, tuple):
                out = out[0]
            assert isinstance(out, torch.Tensor)
            cache[name] = out.detach()

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
            if isinstance(out, tuple):
                out = out[0]
            assert isinstance(out, torch.Tensor)

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

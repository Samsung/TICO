# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import Any

import torch

from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.linear_folding import (
    reused_weight_qparam_scale_multiplier,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


def find_gptq_quantizers(
    root: torch.nn.Module,
) -> tuple[torch.nn.Module | None, dict[str, Any] | None]:
    """Find GPTQ quantizers attached to a model or one of common wrapper owners."""
    candidates = [root]
    if hasattr(root, "wrapped"):
        candidates.append(root.wrapped)
        if hasattr(root.wrapped, "module"):
            candidates.append(root.wrapped.module)
        if hasattr(root.wrapped, "wrapped"):
            candidates.append(root.wrapped.wrapped)

    for owner in candidates:
        quantizers = getattr(owner, "quantizers", None)
        if isinstance(quantizers, dict):
            return owner, quantizers
    return None, None


def gptq_zero_point_for_observer(
    quantizer: Any,
    observer: Any,
    *,
    module_name: str,
) -> torch.Tensor:
    """Map a GPTQ zero point into the integer domain of a PTQ weight observer.

    The GPTQ solver quantizes on the unsigned grid ``[0, maxq]`` with
    ``maxq = 2**bits - 1`` and keeps its zero point in that domain. Affine
    observers quantize on ``[dtype.qmin, dtype.qmax]``. Both grids describe
    the same values when they have the same number of levels, and the
    observer zero point is then ``zero + dtype.qmin`` (a no-op for unsigned
    dtypes). A level-count mismatch means the GPTQ weights were rounded on a
    different grid than the observer will use, so it is rejected instead of
    being silently loaded.
    """
    maxq = int(quantizer.maxq)
    dtype = observer.dtype
    levels = int(dtype.qmax) - int(dtype.qmin)
    if maxq != levels:
        raise RuntimeError(
            f"GPTQ quantizer for {module_name!r} uses {maxq + 1} integer levels "
            f"but the PTQ weight observer dtype {dtype} has {levels + 1}. The GPTQ "
            "bit width and the PTQ weight dtype must agree; adjust weight_bits / "
            "weight_bits_overrides or the PTQ weight spec."
        )
    zero = quantizer.zero
    if int(dtype.qmin) != 0:
        zero = zero + int(dtype.qmin)
    return zero


def inject_gptq_qparams(
    root: torch.nn.Module,
    gptq_quantizers: dict[str, Any],
    weight_obs_name: str = "weight",
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """Inject compatible GPTQ scale and zero-point values into PTQ observers.

    A matched quantizer must quantize on a grid with the same number of
    integer levels as the observer dtype (see ``gptq_zero_point_for_observer``);
    the zero point is mapped into the observer dtype domain before loading.
    """
    seen: set[str] = set()
    missed: list[str] = []

    for module in root.modules():
        if not isinstance(module, QuantModuleBase):
            continue
        if module.fp_name is None:
            continue

        obs = module.get_observer(weight_obs_name)
        if obs is None:
            continue

        quantizer = gptq_quantizers.get(module.fp_name)
        if quantizer is None:
            missed.append(module.fp_name)
            continue

        scale_multiplier = reused_weight_qparam_scale_multiplier(module)
        scale = quantizer.scale
        if scale_multiplier != 1.0:
            scale = scale * scale_multiplier

        assert isinstance(obs, AffineObserverBase)
        zero = gptq_zero_point_for_observer(quantizer, obs, module_name=module.fp_name)
        obs.load_qparams(scale, zero, lock=True)
        seen.add(module.fp_name)

    unused = set(gptq_quantizers.keys()) - seen

    if verbose:
        print("\n[GPTQ → PTQ injection summary]")
        print(f"  matched : {len(seen)}")
        print(f"  missed  : {len(missed)}")
        print(f"  unused  : {len(unused)}")
        for title, items in [
            ("missed modules", missed),
            ("unused GPTQ entries", sorted(unused)),
        ]:
            if not items:
                continue
            print(f"\n  {title}:")
            for name in list(items)[:10]:
                print(f"    - {name}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")

    return {"matched": len(seen), "missed": len(missed), "unused": len(unused)}


def clear_gptq_quantizers(root: torch.nn.Module) -> None:
    """Drop attached GPTQ quantizer dictionaries after PTQ qparam injection."""
    candidates = [root]
    if hasattr(root, "wrapped"):
        candidates.append(root.wrapped)
        if hasattr(root.wrapped, "module"):
            candidates.append(root.wrapped.module)
        if hasattr(root.wrapped, "wrapped"):
            candidates.append(root.wrapped.wrapped)

    for owner in candidates:
        if hasattr(owner, "quantizers"):
            delattr(owner, "quantizers")

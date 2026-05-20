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


def inject_gptq_qparams(
    root: torch.nn.Module,
    gptq_quantizers: dict[str, Any],
    weight_obs_name: str = "weight",
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """Inject GPTQ scale / zero-point into PTQ weight observers."""
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

        assert isinstance(obs, AffineObserverBase)
        obs.load_qparams(quantizer.scale, quantizer.zero, lock=True)
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

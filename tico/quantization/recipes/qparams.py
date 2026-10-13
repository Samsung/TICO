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

import math
from typing import Any

import torch

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.linear_folding import (
    reused_weight_qparam_scale_multiplier,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


def find_gptq_quantizers(
    root: torch.nn.Module,
) -> tuple[torch.nn.Module | None, dict[str, Any] | None]:
    """Find GPTQ quantizers attached to a model or one of common wrapper owners."""
    candidates: list[Any] = [root]
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


def _validated_gptq_qparams(
    module: QuantModuleBase,
    obs: ObserverBase,
    quantizer: Any,
) -> tuple[AffineObserverBase, torch.Tensor, torch.Tensor]:
    """Validate standard per-tensor/per-output-channel GPTQ affine metadata.

    GPTQ quantizers do not retain a complete grouped-quantization mapping, so
    these checks do not establish compatibility for grouped GPTQ artifacts.
    """
    maxq = getattr(quantizer, "maxq", None)
    source = f"GPTQ maxq={maxq!r}"
    dtype = getattr(obs, "dtype", None)
    target = f"PTQ dtype={dtype!r}, observer={type(obs).__name__}"

    def incompatible(reason: str) -> ValueError:
        return ValueError(
            f"Incompatible GPTQ -> PTQ weight qparams for {module.fp_name!r}: "
            f"{source}; {target}. {reason}"
        )

    if not isinstance(obs, AffineObserverBase):
        raise incompatible("GPTQ qparam reuse requires an affine weight observer.")
    if (
        not isinstance(dtype, DType)
        or isinstance(dtype.bits, bool)
        or not isinstance(dtype.bits, int)
        or dtype.bits <= 0
        or not isinstance(dtype.signed, bool)
    ):
        raise incompatible("PTQ weight dtype must have a positive integer bit width.")
    target = (
        f"PTQ {dtype} range=[{dtype.qmin}, {dtype.qmax}], "
        f"qscheme={obs.qscheme}, axis={obs.channel_axis}"
    )

    if isinstance(maxq, torch.Tensor):
        if maxq.numel() != 1 or maxq.dtype == torch.bool or maxq.is_complex():
            raise incompatible("GPTQ maxq must be a finite positive integer scalar.")
        maxq = maxq.item()
    if (
        isinstance(maxq, bool)
        or not isinstance(maxq, (int, float))
        or not math.isfinite(maxq)
        or maxq <= 0
        or int(maxq) != maxq
    ):
        raise incompatible("GPTQ maxq must be a finite positive integer scalar.")
    maxq = int(maxq)
    if maxq & (maxq + 1):
        raise incompatible("GPTQ maxq must describe a full unsigned 2**bits - 1 range.")
    source = f"GPTQ {maxq.bit_length()}-bit unsigned range=[0, {maxq}] (maxq={maxq})"
    if dtype.signed or dtype.qmin != 0 or dtype.qmax != maxq:
        raise incompatible(
            "Weight bit width and integer range must match exactly; "
            "qparam reuse does not convert bit widths or signedness."
        )
    if obs.qscheme not in {QScheme.PER_CHANNEL_ASYMM, QScheme.PER_TENSOR_ASYMM}:
        raise incompatible("Unsigned PTQ weights require an asymmetric affine qscheme.")

    perchannel = getattr(quantizer, "perchannel", None)
    if not isinstance(perchannel, bool):
        raise incompatible("GPTQ perchannel metadata must be a boolean.")
    owner = getattr(module, "module", module)
    if isinstance(owner, torch.nn.Linear):
        source_channels, channel_axis = owner.out_features, 0
    elif isinstance(owner, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        source_channels, channel_axis = owner.out_channels, 0
    elif isinstance(owner, torch.nn.ConvTranspose2d):
        # GPTQ uses an output-major Conv2d view; PTQ observes the original weight.
        source_channels, channel_axis = owner.out_channels, 1
    else:
        raise incompatible(
            f"Cannot establish the GPTQ weight channel mapping for {type(owner).__name__}."
        )
    weight = owner.weight
    if not isinstance(weight, torch.Tensor) or weight.ndim <= channel_axis:
        raise incompatible("The wrapped weight has no valid output-channel dimension.")

    scale = getattr(quantizer, "scale", None)
    zero = getattr(quantizer, "zero", None)
    for name, value in (("scale", scale), ("zero-point", zero)):
        if (
            not isinstance(value, torch.Tensor)
            or value.numel() == 0
            or value.is_complex()
            or value.dtype == torch.bool
        ):
            raise incompatible(f"GPTQ {name} must be a nonempty real tensor.")
        valid_counts = {source_channels} if perchannel else {1, source_channels}
        if value.numel() not in valid_counts or any(
            dim != 1 for dim in value.shape[1:]
        ):
            raise incompatible(
                f"GPTQ {name} shape {tuple(value.shape)} does not map to "
                f"{source_channels} output channels (perchannel={perchannel})."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise incompatible(f"GPTQ {name} must contain only finite values.")
    assert isinstance(scale, torch.Tensor) and isinstance(zero, torch.Tensor)
    if not scale.is_floating_point() or not bool((scale > 0).all().item()):
        raise incompatible("GPTQ scales must be floating-point and strictly positive.")
    if not bool(((zero >= 0) & (zero <= maxq)).all().item()) or (
        zero.is_floating_point() and not bool((zero == zero.round()).all().item())
    ):
        raise incompatible(
            "GPTQ zero-points must be integral and within the shared range."
        )

    uniform = bool((scale == scale.reshape(-1)[0]).all().item()) and bool(
        (zero == zero.reshape(-1)[0]).all().item()
    )
    if not perchannel and not uniform:
        raise incompatible(
            "GPTQ per-tensor metadata requires identical repeated qparams."
        )
    if obs.qscheme.is_per_channel():
        if (
            isinstance(obs.channel_axis, bool)
            or not isinstance(obs.channel_axis, int)
            or obs.channel_axis != channel_axis
        ):
            raise incompatible(
                f"PTQ weight channel axis must be {channel_axis} for {type(owner).__name__}."
            )
        if scale.dtype != torch.float32:
            raise incompatible(
                "PTQ per-channel fake quantization requires float32 scales."
            )
        target_channels = weight.shape[channel_axis]
        if target_channels <= 0:
            raise incompatible("PTQ weight output-channel dimension must be nonempty.")
        if scale.numel() != target_channels or zero.numel() != target_channels:
            if not uniform:
                raise incompatible(
                    f"GPTQ qparams cannot map to {target_channels} PTQ weight channels."
                )
            scale = scale.reshape(-1)[0].expand(target_channels)
            zero = zero.reshape(-1)[0].expand(target_channels)
    else:
        if obs.channel_axis is not None or not uniform:
            raise incompatible(
                "PTQ per-tensor weights require one identical GPTQ grid."
            )
        scale = scale.reshape(-1)[0]
        zero = zero.reshape(-1)[0]

    try:
        scale_multiplier = reused_weight_qparam_scale_multiplier(module)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise incompatible(str(exc)) from exc
    if isinstance(scale_multiplier, torch.Tensor):
        # Per-output-channel fold, e.g. Qwen3-VL vision fused QKV where only
        # the K rows absorbed the attention-score scaling.
        if scale_multiplier.numel() != scale.numel():
            raise incompatible(
                f"Folded weight scale multiplier has {scale_multiplier.numel()} "
                f"entries, but the GPTQ scale has {scale.numel()}."
            )
        multiplier = scale_multiplier.to(device=scale.device, dtype=scale.dtype)
        if not bool(torch.isfinite(multiplier).all().item()) or not bool(
            (multiplier > 0).all().item()
        ):
            raise incompatible(
                "Folded weight scale multiplier must be finite and positive."
            )
        scale = (scale.reshape(-1) * multiplier.reshape(-1)).reshape(scale.shape)
    else:
        if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
            raise incompatible(
                "Folded weight scale multiplier must be finite and positive."
            )
        if scale_multiplier != 1.0:
            scale = scale * scale_multiplier
    if not bool(torch.isfinite(scale).all().item()) or not bool(
        (scale > 0).all().item()
    ):
        raise incompatible(
            "Folded GPTQ scales must remain finite and strictly positive."
        )
    return obs, scale, zero


def inject_gptq_qparams(
    root: torch.nn.Module,
    gptq_quantizers: dict[str, Any],
    weight_obs_name: str = "weight",
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """Validate every matched GPTQ grid, then inject and lock its PTQ qparams."""
    seen: set[str] = set()
    missed: list[str] = []
    pending: list[tuple[AffineObserverBase, torch.Tensor, torch.Tensor]] = []

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

        pending.append(_validated_gptq_qparams(module, obs, quantizer))
        seen.add(module.fp_name)

    # A later invalid pair (including an unsupported fold) must not leave earlier
    # observers locked with only part of the requested handoff applied.
    for obs, scale, zero in pending:
        obs.load_qparams(scale, zero, lock=True)

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
    candidates: list[Any] = [root]
    if hasattr(root, "wrapped"):
        candidates.append(root.wrapped)
        if hasattr(root.wrapped, "module"):
            candidates.append(root.wrapped.module)
        if hasattr(root.wrapped, "wrapped"):
            candidates.append(root.wrapped.wrapped)

    for owner in candidates:
        if hasattr(owner, "quantizers"):
            delattr(owner, "quantizers")

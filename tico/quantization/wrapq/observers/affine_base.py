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

import math
from typing import Optional, Tuple

import torch

from tico.quantization.wrapq.dtypes import DType, UINT8
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.qscheme import QScheme


class AffineObserverBase(ObserverBase):
    """Base for affine observers (min/max → scale/zp)."""

    min_val: torch.Tensor
    max_val: torch.Tensor
    _cached_scale: torch.Tensor
    _cached_zp: torch.Tensor

    def __init__(
        self,
        *,
        name: str,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_ASYMM,
        channel_axis: Optional[int] = None,
    ):
        super().__init__(
            name=name, dtype=dtype, qscheme=qscheme, channel_axis=channel_axis
        )

        # Register internal statistics as buffers so they:
        #  - move correctly with `model.to(device)`
        #  - are included in state_dict (if persistent)
        #  - follow PyTorch module semantics
        #
        # Shapes may later expand (e.g. per-channel),
        # but the buffers themselves remain tracked.
        self.register_buffer("min_val", torch.tensor(math.inf))
        self.register_buffer("max_val", torch.tensor(-math.inf))

        # Cached quantization parameters.
        # Marked as non-persistent since they can be recomputed.
        self.register_buffer("_cached_scale", torch.tensor([]), persistent=False)
        self.register_buffer(
            "_cached_zp", torch.tensor([], dtype=torch.int), persistent=False
        )

        self.reset()

    def reset(self) -> None:
        """
        Reset running min/max and drop cached qparams.

        Do NOT reassign new tensors here.
        Updating buffers in-place ensures that device and dtype
        tracking remains correct.
        """
        assert isinstance(self.min_val, torch.Tensor)
        assert isinstance(self.max_val, torch.Tensor)
        self.min_val.fill_(math.inf)
        self.max_val.fill_(-math.inf)
        # Clear cached qparams while keeping buffer registration intact
        self._cached_scale = self._cached_scale.new_empty((0,))  # type: ignore[has-type]
        self._cached_zp = self._cached_zp.new_empty((0,), dtype=torch.int)  # type: ignore[has-type]

    def load_qparams(self, scale: torch.Tensor, zp: torch.Tensor, *, lock: bool = True):
        """
        Inject externally computed qparams and optionally lock the observer.

        When locked, subsequent `collect()` calls are ignored.

        Scale is forced to float32 because ``fake_quantize_per_*_affine``
        requires a float32 scale.  When ``double_precision=True`` is used
        during GPTQ calibration, observers may collect float64 min/max
        statistics, which would produce a float64 scale and cause a dtype
        mismatch at inference time.
        """
        self._cached_scale = scale.detach()
        self._cached_zp = zp.to(torch.int)
        if lock:
            self.enabled = False

    @property
    def has_qparams(self) -> bool:
        return self._cached_scale.numel() != 0

    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.min_val, torch.Tensor)
        assert isinstance(self.max_val, torch.Tensor)
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        # Cast min/max to float32 before computing qparams. When
        # ``double_precision=True`` is used during GPTQ calibration,
        # observers may collect float64 min/max statistics. The scale
        # must be float32 for ``fake_quantize_per_*_affine`` to work
        # correctly at inference time.
        min_val = self.min_val
        max_val = self.max_val
        rng = max_val - min_val
        eps = 1e-12

        if self.qscheme.is_symmetric():
            max_abs = torch.maximum(max_val.abs(), min_val.abs())
            scale = torch.clamp(max_abs, min=eps) / qmax
            zp = torch.zeros_like(scale, dtype=torch.int)
            self._cached_scale, self._cached_zp = scale, zp
            return scale, zp

        if (self.channel_axis is None) and torch.all(rng.abs() < 1e-8):
            C = min_val
            if torch.allclose(C, torch.zeros_like(C)):
                scale = torch.ones_like(C)
                zp = torch.zeros_like(C, dtype=torch.int)
            elif (C > 0).all():
                scale = torch.clamp(C, min=eps)
                zp = torch.zeros_like(C, dtype=torch.int)
            else:
                scale = torch.clamp(C.abs(), min=eps)
                zp = torch.full_like(C, qmax, dtype=torch.int)
        else:
            # Force the range to include 0
            rng = torch.where(0 < min_val, max_val, rng)
            rng = torch.where(0 > max_val, -min_val, rng)

            scale = torch.clamp(rng, min=eps) / (qmax - qmin)
            zp = (
                torch.round(qmin - min_val / scale).clamp(qmin, qmax).to(torch.int)
            )

        self._cached_scale, self._cached_zp = scale, zp
        return scale, zp

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_qparams:
            raise RuntimeError(
                "Call compute_qparams()/freeze_qparams() or load_qparams() first."
            )

        scale = self._cached_scale.to(x.device)
        zp = self._cached_zp.to(x.device, dtype=torch.int)

        orig_dtype = x.dtype

        # When input is float64 (double_precision mode during GPTQ),
        # use a custom float64 fake_quant to avoid precision loss from
        # casting to float32. The standard torch.fake_quantize_per_*_affine
        # casts input to float32, which introduces batch-size-dependent
        # rounding differences in the GPTQ re-forward pass.
        #
        # Keep both activations AND scale in float64 during GPTQ calibration.
        # After GPTQ, cast quantizers and model to float32 for evaluation.
        if orig_dtype == torch.float64:
            qmin = self.dtype.qmin
            qmax = self.dtype.qmax
            scale_f64 = scale.to(torch.float64)
            if self.channel_axis is None:
                # Per-tensor affine fake quant in float64
                x_q = torch.round(x / scale_f64 + zp).clamp(qmin, qmax)
                return (x_q - zp) * scale_f64
            else:
                # Per-channel affine fake quant in float64
                # Reshape scale for broadcasting along channel_axis
                shape = [1] * x.dim()
                shape[self.channel_axis] = -1
                scale_b = scale_f64.reshape(shape)
                zp_b = zp.reshape(shape)
                x_q = torch.round(x / scale_b + zp_b).clamp(qmin, qmax)
                return (x_q - zp_b) * scale_b

        if self.channel_axis is None:
            return torch.fake_quantize_per_tensor_affine(
                x.float(),
                scale=scale,
                zero_point=zp,
                quant_min=self.dtype.qmin,
                quant_max=self.dtype.qmax,
            )
        else:
            return torch.fake_quantize_per_channel_affine(
                x.float(),
                scale=scale,
                zero_point=zp,
                axis=self.channel_axis,
                quant_min=self.dtype.qmin,
                quant_max=self.dtype.qmax,
            )



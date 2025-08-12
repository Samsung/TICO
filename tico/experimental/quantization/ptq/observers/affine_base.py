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

from tico.experimental.quantization.ptq.dtypes import DType, UINT8
from tico.experimental.quantization.ptq.observers.base import ObserverBase
from tico.experimental.quantization.ptq.qscheme import QScheme


class AffineObserverBase(ObserverBase):
    """Base for affine observers (min/max → scale/zp)."""

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

    def reset(self) -> None:
        """
        Reset running min/max and drop cached qparams.
        """
        self.min_val: torch.Tensor = torch.tensor(math.inf)
        self.max_val: torch.Tensor = torch.tensor(-math.inf)
        if hasattr(self, "_cached_scale"):
            del self._cached_scale
        if hasattr(self, "_cached_zp"):
            del self._cached_zp

    def load_qparams(self, scale: torch.Tensor, zp: torch.Tensor, *, lock: bool = True):
        """
        Inject externally computed qparams and optionally lock the observer.

        When locked, subsequent `collect()` calls are ignored.
        """
        self._cached_scale = scale.detach()
        self._cached_zp = zp.to(torch.int)
        if lock:
            self.enabled = False

    @property
    def has_qparams(self) -> bool:
        return hasattr(self, "_cached_scale")

    def compute_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        rng = self.max_val - self.min_val
        eps = 1e-12

        if self.qscheme.is_symmetric():
            max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs())
            scale = torch.clamp(max_abs, min=eps) / qmax
            zp = torch.zeros_like(scale, dtype=torch.int)
            self._cached_scale, self._cached_zp = scale, zp
            return scale, zp

        if self.channel_axis is None:
            if torch.all(rng.abs() < 1e-8):
                C = self.min_val
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
                scale = torch.clamp(rng, min=eps) / (qmax - qmin)
                zp = (
                    torch.round(qmin - self.min_val / scale)
                    .clamp(qmin, qmax)
                    .to(torch.int)
                )
        else:
            scale = torch.clamp(rng, min=eps) / (qmax - qmin)
            zp = (
                torch.round(qmin - self.min_val / scale).clamp(qmin, qmax).to(torch.int)
            )

        self._cached_scale, self._cached_zp = scale, zp
        return scale, zp

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        if not self.has_qparams:
            raise RuntimeError(
                "Call compute_qparams()/freeze_qparams() or load_qparams() first."
            )
        scale, zp = self._cached_scale, self._cached_zp
        if self.channel_axis is None:
            return torch.fake_quantize_per_tensor_affine(
                x,
                scale=scale,
                zero_point=zp,
                quant_min=self.dtype.qmin,
                quant_max=self.dtype.qmax,
            )
        else:
            return torch.fake_quantize_per_channel_affine(
                x,
                scale=scale,
                zero_point=zp,
                axis=self.channel_axis,
                quant_min=self.dtype.qmin,
                quant_max=self.dtype.qmax,
            )

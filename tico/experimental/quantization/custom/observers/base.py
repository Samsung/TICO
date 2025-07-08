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
from typing import Optional

import torch

from tico.experimental.quantization.custom.dtypes import DType, UINT8
from tico.experimental.quantization.custom.qscheme import QScheme
from tico.experimental.quantization.custom.utils import reduce_except


class ObserverBase:
    """
    Tracks activation/weight ranges and converts them into (scale, zero-point).
    """

    def __init__(
        self,
        *,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_AFFINE,
        channel_axis: Optional[int] = None,  # None → per-tensor
    ):
        self.dtype = dtype
        self.qscheme = qscheme
        self.channel_axis = channel_axis if qscheme.is_per_channel() else None
        self.enabled = True
        self.reset()

    # --------------------------------------------------------------------- #
    #                      Stats collection & utilities                     #
    # --------------------------------------------------------------------- #
    def reset(self) -> None:
        self.min_val: torch.Tensor = torch.tensor(math.inf)
        self.max_val: torch.Tensor = torch.tensor(-math.inf)

    @torch.no_grad()
    def collect(self, x: torch.Tensor) -> None:
        """
        Update running statistics with a new batch of data.
        """
        if not self.enabled:
            return

        if self.channel_axis is None:
            # Per-tensor: reduce to scalar
            self.min_val = torch.minimum(self.min_val, x.min())
            self.max_val = torch.maximum(self.max_val, x.max())
        else:
            # Per-channel: track a vector of size C
            mins, maxs = reduce_except(x, self.channel_axis)
            self.min_val = torch.minimum(self.min_val, mins)
            self.max_val = torch.maximum(self.max_val, maxs)

    # --------------------------------------------------------------------- #
    #                     Quantization parameter helpers                    #
    # --------------------------------------------------------------------- #
    def compute_qparams(self):
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        rng = self.max_val - self.min_val
        # scalar path
        if torch.all(rng.abs() < 1e-8):
            C = self.min_val
            if torch.allclose(C, torch.zeros_like(C)):
                scale = torch.ones_like(C)
                zp = torch.zeros_like(C)
            elif (C > 0).all():
                scale = C / 1.0
                zp = torch.zeros_like(C)
            else:
                # C < 0
                scale = C.abs() / 1.0
                zp = torch.full_like(C, qmax)
        else:
            # normal path
            eps = 1e-12
            scale = torch.clamp(rng, min=eps) / qmax - qmin
            zp = torch.round(qmin - self.min_val / scale).clamp(qmin, qmax)
        # cache
        self._cached_scale = scale
        self._cached_zp = zp.to(torch.int)
        return self._cached_scale, self._cached_zp

    @property
    def has_qparms(self) -> bool:
        return hasattr(self, "_cached_scale")

    def fake_quant(self, x: torch.Tensor):
        if not self.has_qparms:
            raise RuntimeError("compute_qparams() must be called before fake_quant().")
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

    # String repr helps debugging
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(dtype={str(self.dtype)}, "
            f"qscheme={str(self.qscheme)}, enabled={self.enabled})"
        )

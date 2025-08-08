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
from abc import ABC, abstractmethod
from typing import Optional

import torch

from tico.experimental.quantization.ptq.dtypes import DType, UINT8
from tico.experimental.quantization.ptq.qscheme import QScheme


class ObserverBase(ABC):
    """
    Abstract base class for observers that tracks activation/weight ranges
     and converts them into (scale, zero-point).
    """

    def __init__(
        self,
        *,
        name: str,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_ASYMM,
        channel_axis: Optional[int] = None,  # None → per-tensor
    ):
        self.name = name
        self.dtype = dtype
        self.qscheme = qscheme
        self.channel_axis = channel_axis if qscheme.is_per_channel() else None
        self.enabled = True
        self.reset()

    def reset(self) -> None:
        """
        Reset running statistics to initial (±inf) sentinels.
        """
        self.min_val: torch.Tensor = torch.tensor(math.inf)
        self.max_val: torch.Tensor = torch.tensor(-math.inf)
        # Clear cached qparams if any
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

    @abstractmethod
    def collect(self, x: torch.Tensor) -> None:
        """
        Update running statistics with a new batch of data.

        Must be implemented by subclasses (e.g., MinMax, EMA, Histogram, MSE).
        """
        raise NotImplementedError

    def compute_qparams(self):
        """
        Compute and cache `(scale, zero_point)` according to
        the selected `qscheme`.

        * Symmetric: zero-point = 0, scale = max(|min|, |max|) / qmax
        * Asymmetric: standard affine mapping based on min / max
        """
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        rng = self.max_val - self.min_val
        eps = 1e-12

        # ── Symmetric branch ───────────────────────────────
        if self.qscheme.is_symmetric():
            max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs())
            scale = (
                torch.clamp(max_abs, min=eps) / qmax
            )  # qmax = 2^{bits-1}-1 for signed dtypes
            zp = torch.zeros_like(scale, dtype=torch.int)  # zero-point is always 0
            self._cached_scale, self._cached_zp = scale, zp
            return scale, zp

        # ── Asymmetric branch ──────────────────────────────
        if self.channel_axis is None:  # per-tensor
            # Degenerate case: nearly constant tensor
            if torch.all(rng.abs() < 1e-8):
                C = self.min_val
                if torch.allclose(C, torch.zeros_like(C)):
                    scale = torch.ones_like(C)
                    zp = torch.zeros_like(C, dtype=torch.int)
                elif (C > 0).all():
                    scale = torch.clamp(C, min=eps)
                    zp = torch.zeros_like(C, dtype=torch.int)
                else:  # C < 0
                    scale = torch.clamp(C.abs(), min=eps)
                    zp = torch.full_like(C, qmax, dtype=torch.int)
            else:
                scale = torch.clamp(rng, min=eps) / (qmax - qmin)
                zp = (
                    torch.round(qmin - self.min_val / scale)
                    .clamp(qmin, qmax)
                    .to(torch.int)
                )
        else:  # per-channel
            scale = torch.clamp(rng, min=eps) / (qmax - qmin)
            zp = (
                torch.round(qmin - self.min_val / scale).clamp(qmin, qmax).to(torch.int)
            )

        self._cached_scale, self._cached_zp = scale, zp
        return scale, zp

    @property
    def has_qparams(self) -> bool:
        """
        Return True if qparams have been computed or injected.
        """
        return hasattr(self, "_cached_scale")

    def fake_quant(self, x: torch.Tensor):
        if not self.has_qparams:
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
            f"{self.__class__.__name__}(name={self.name}, dtype={str(self.dtype)}, "
            f"qscheme={str(self.qscheme)}, channel_axis={self.channel_axis}, enabled={self.enabled})"
        )

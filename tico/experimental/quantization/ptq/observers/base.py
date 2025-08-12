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

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

from tico.experimental.quantization.ptq.dtypes import DType, UINT8
from tico.experimental.quantization.ptq.qscheme import QScheme


class ObserverBase(ABC):
    """
    Minimal abstract base for all observers/quantizers.

    Subclasses must implement:
      - reset()
      - collect(x)
      - fake_quant(x)
      - compute_qparams(): optional in practice for some observers (e.g., MX),
        but still part of the interface; those can return None.
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

    @abstractmethod
    def reset(self) -> None:
        """Clear any running statistics or cached params."""
        raise NotImplementedError

    def collect(self, x: torch.Tensor) -> None:
        """
        Update running statistics with a new batch of data.

        This base implementation guards on `enabled` and then calls `_update_stats(x)`.
        Subclasses should implement `_update_stats(x)` instead of overriding `collect`.
        """
        if not self.enabled:
            return
        self._update_stats(x)

    @abstractmethod
    def _update_stats(self, x: torch.Tensor) -> None:
        """
        Update running statistics (min/max, hist, mse buffers, ...).

        Must be implemented by subclasses (e.g., MinMax, EMA, Histogram, MSE).
        """
        raise NotImplementedError

    @abstractmethod
    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the observer's quantization.
        Implementations may or may not rely on qparams.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_qparams(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute and (if applicable) cache quantization params.
        Affine observers typically return (scale, zero_point).
        Observers that do not use qparams (e.g., MX) may return None.
        """
        raise NotImplementedError

    # String repr helps debugging
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name}, dtype={str(self.dtype)}, "
            f"qscheme={str(self.qscheme)}, channel_axis={self.channel_axis}, enabled={self.enabled})"
        )

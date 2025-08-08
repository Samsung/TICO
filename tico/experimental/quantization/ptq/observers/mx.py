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

import torch

from tico.experimental.quantization.ptq.observers.base import ObserverBase
from tico.utils.mx.mx_ops import quantize_mx


class MXObserver(ObserverBase):
    """MX (micro-scaling) observer: no min/max, no affine qparams."""

    def __init__(
        self,
        *,
        name: str,
        elem_format: str = "int8",
        axis: int = 0,
        shared_exp_method: str = "max",
        round: str = "nearest",
        **base_kwargs,
    ):
        super().__init__(name=name, **base_kwargs)
        self.elem_format = elem_format
        self.axis = axis
        self.shared_exp_method = shared_exp_method
        self.round = round

    def reset(self) -> None:
        # No state to reset
        return

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        # No stats required
        return None

    def compute_qparams(self):
        # MX path does not produce affine qparams; keep interface contract.
        return None

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_mx(
            x,
            elem_format=self.elem_format,
            axis=self.axis,
            shared_exp_method=self.shared_exp_method,
            round=self.round,
        )

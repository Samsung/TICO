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

from tico.quantization.wrapq.observers.base import FrozenObserverState, ObserverBase
from tico.utils.mx.formats import ElemFormat, RoundingMode
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
        self._reset_frozen_state()

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        # No stats required
        return None

    def _policy(self) -> tuple:
        return super()._policy() + (
            self.elem_format,
            self.axis,
            self.shared_exp_method,
            self.round,
        )

    def _state_kind(self) -> str:
        return "mx"

    def derive_candidate(self) -> FrozenObserverState:
        self._check_frozen_policy()
        state = FrozenObserverState(
            "mx",
            self._policy(),
            "stateless",
            fake_quant_enabled=self.fake_quant_enabled,
            observed_shape=self._observed_shape,
        )
        self.validate_candidate(state)
        return state

    def validate_candidate(self, state: FrozenObserverState) -> None:
        super().validate_candidate(state)
        if (
            state.kind != "mx"
            or state.provenance != "stateless"
            or state.scale is not None
            or self.elem_format not in ElemFormat.__members__
            or self.round not in RoundingMode.string_enums()
            or self.shared_exp_method not in ("max", "none")
            or not isinstance(self.axis, int)
        ):
            raise RuntimeError(f"Observer {self.name!r}: invalid dynamic MX policy.")
        if state.observed_shape is not None and not -len(
            state.observed_shape
        ) <= self.axis < len(state.observed_shape):
            raise RuntimeError(
                f"Observer {self.name!r}: invalid MX axis for observed rank."
            )

    def compute_qparams(self):
        self.install_candidate(self.derive_candidate())
        return None

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        self._check_frozen_policy()
        if not self.fake_quant_enabled:
            return x
        return quantize_mx(
            x,
            elem_format=self.elem_format,
            axis=self.axis,
            shared_exp_method=self.shared_exp_method,
            round=self.round,
        )

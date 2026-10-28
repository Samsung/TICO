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

"""Explicit stateless float policy, without an affine sentinel grid."""

import torch

from tico.quantization.wrapq.observers.base import FrozenObserverState, ObserverBase


class IdentityObserver(ObserverBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = False

    def reset(self) -> None:
        self._reset_frozen_state()

    def _update_stats(self, x: torch.Tensor) -> None:
        return

    def derive_candidate(self) -> FrozenObserverState:
        self._check_frozen_policy()
        return FrozenObserverState(
            "identity",
            self._policy(),
            "stateless",
            fake_quant_enabled=self.fake_quant_enabled,
            observed_shape=self._observed_shape,
        )

    def validate_candidate(self, state: FrozenObserverState) -> None:
        super().validate_candidate(state)
        if (
            state.kind != "identity"
            or state.provenance != "stateless"
            or state.scale is not None
        ):
            raise RuntimeError(
                f"Observer {self.name!r}: invalid stateless float policy."
            )

    def _state_kind(self) -> str:
        return "identity"

    def compute_qparams(self):
        self.install_candidate(self.derive_candidate())
        return None

    def fake_quant(self, x: torch.Tensor):
        self._check_frozen_policy()
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

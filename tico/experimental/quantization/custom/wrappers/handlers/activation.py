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

from torch import Tensor

from tico.experimental.quantization.custom.wrappers.handlers.base import BaseHandler
from tico.experimental.quantization.custom.wrappers.mode import Mode


class ActivationHandler(BaseHandler):
    """Handles stateless activations."""

    def forward(self, x: Tensor, *, mode: Mode) -> Tensor:
        out = self.module(x)

        if mode is Mode.CALIB:
            self.act_obs.collect(out.detach())
        elif mode is Mode.QUANT:
            out = self.act_obs.fake_quant(out)

        return out

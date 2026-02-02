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

from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("torch.nn.Embedding")
class QuantEmbedding(QuantModuleBase):
    """Per-channel weight fake-quant, eager-output activation fake-quant."""

    def __init__(
        self,
        fp: nn.Embedding,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.weight_obs = self._make_obs(
            "weight",
            qscheme=QScheme.PER_CHANNEL_ASYMM,  # tensorwise quantization breaks the model
            channel_axis=0,  # weight ~ (vocab_size, inner_dim) so that weight_scales ~ (1, vocab_size)
        )
        self.act_out_obs = self._make_obs("act_out")
        self.module = fp

    def enable_calibration(self) -> None:
        super().enable_calibration()
        # immediately capture the fixed weight range
        self.weight_obs.collect(self.module.weight)

    def forward(self, x: torch.Tensor):

        # x is supposed to be in int64 form so no quantization of activations is needed
        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.weight_obs.fake_quant(w)

        y = torch.nn.functional.embedding(
            x,
            w,
            self.module.padding_idx,
            self.module.max_norm,
            self.module.norm_type,
            self.module.scale_grad_by_freq,
            self.module.sparse,
        )

        return self._fq(y, self.act_out_obs)

    def _all_observers(self):
        return (self.act_out_obs, self.weight_obs)

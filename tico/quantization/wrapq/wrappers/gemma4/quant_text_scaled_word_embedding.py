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

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.gemma4.modeling_gemma4.Gemma4TextScaledWordEmbedding"
)
class QuantGemma4TextScaledWordEmbedding(QuantModuleBase):
    """PTQ wrapper for Gemma4 text embeddings with static scale multiplication."""

    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.obs_weight = self._make_obs("weight")
        self.obs_act_out = self._make_obs("act_out")

    def enable_calibration(self) -> None:
        """Enable calibration and collect the static embedding weight range."""
        super().enable_calibration()
        self.obs_weight.collect(self.module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return scaled token embeddings."""
        weight = self.module.weight
        if self._mode is Mode.QUANT:
            weight = self.obs_weight.fake_quant(weight)
        hidden_states = F.embedding(
            input_ids,
            weight,
            padding_idx=self.module.padding_idx,
            max_norm=self.module.max_norm,
            norm_type=self.module.norm_type,
            scale_grad_by_freq=self.module.scale_grad_by_freq,
            sparse=self.module.sparse,
        )
        scale = getattr(self.module, "embed_scale", None)
        if scale is not None:
            hidden_states = hidden_states * scale.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        return self._fq(hidden_states, self.obs_act_out)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_weight, self.obs_act_out)

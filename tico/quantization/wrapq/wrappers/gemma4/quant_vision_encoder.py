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

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionEncoder")
class QuantGemma4VisionEncoder(QuantModuleBase):
    """PTQ wrapper skeleton for the Gemma4 vision encoder."""

    def __init__(
        self,
        fp_encoder: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_encoder
        self.config = fp_encoder.config
        self.rotary_emb = fp_encoder.rotary_emb
        self.layers = nn.ModuleList(
            [
                PTQWrapper(
                    layer,
                    qcfg=qcfg.child("layers").child(str(i)) if qcfg else None,
                    fp_name=join_name(fp_name, f"layers.{i}"),
                )
                for i, layer in enumerate(fp_encoder.layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run the vision encoder with static inputs."""
        position_embeddings = self.rotary_emb(inputs_embeds, pixel_position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                **kwargs,
            )
        return hidden_states

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return ()

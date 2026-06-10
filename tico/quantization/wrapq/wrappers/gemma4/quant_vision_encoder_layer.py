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


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionEncoderLayer")
class QuantGemma4VisionEncoderLayer(QuantModuleBase):
    """PTQ wrapper skeleton for one Gemma4 vision encoder layer."""

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.self_attn = PTQWrapper(
            fp_layer.self_attn,
            qcfg=qcfg.child("self_attn") if qcfg else None,
            fp_name=join_name(fp_name, "self_attn"),
        )
        self.mlp = PTQWrapper(
            fp_layer.mlp,
            qcfg=qcfg.child("mlp") if qcfg else None,
            fp_name=join_name(fp_name, "mlp"),
        )
        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=qcfg.child("input_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "input_layernorm"),
        )
        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=qcfg.child("post_attention_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "post_attention_layernorm"),
        )
        self.pre_feedforward_layernorm = PTQWrapper(
            fp_layer.pre_feedforward_layernorm,
            qcfg=qcfg.child("pre_feedforward_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "pre_feedforward_layernorm"),
        )
        self.post_feedforward_layernorm = PTQWrapper(
            fp_layer.post_feedforward_layernorm,
            qcfg=qcfg.child("post_feedforward_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "post_feedforward_layernorm"),
        )

        self.obs_attn_residual_out = self._make_obs("attn_residual_out")
        self.obs_mlp_residual_out = self._make_obs("mlp_residual_out")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run one vision encoder layer."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_attn_residual_out)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return self._fq(residual + hidden_states, self.obs_mlp_residual_out)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_attn_residual_out, self.obs_mlp_residual_out)

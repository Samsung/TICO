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

from collections import UserDict
from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextModel")
class QuantGemma4TextModel(QuantModuleBase):
    """PTQ wrapper skeleton for the Gemma4 E2B text model."""

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        assert_gemma4_e2b_no_moe(fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config
        self.unique_layer_types = set(self.config.layer_types)
        self.hidden_size_per_layer_input = getattr(
            fp_model, "hidden_size_per_layer_input", None
        )

        self.embed_tokens = PTQWrapper(
            fp_model.embed_tokens,
            qcfg=qcfg.child("embed_tokens") if qcfg else None,
            fp_name=join_name(fp_name, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                PTQWrapper(
                    layer,
                    qcfg=qcfg.child("layers").child(str(i)) if qcfg else None,
                    fp_name=join_name(fp_name, f"layers.{i}"),
                )
                for i, layer in enumerate(fp_model.layers)
            ]
        )
        self.norm = PTQWrapper(
            fp_model.norm,
            qcfg=qcfg.child("norm") if qcfg else None,
            fp_name=join_name(fp_name, "norm"),
        )
        self.rotary_emb = fp_model.rotary_emb

        self.embed_tokens_per_layer: Optional[nn.Module] = None
        self.per_layer_model_projection: Optional[nn.Module] = None
        self.per_layer_projection_norm: Optional[nn.Module] = None
        self.per_layer_input_scale = 1.0
        self.per_layer_model_projection_scale = 1.0
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = PTQWrapper(
                fp_model.embed_tokens_per_layer,
                qcfg=qcfg.child("embed_tokens_per_layer") if qcfg else None,
                fp_name=join_name(fp_name, "embed_tokens_per_layer"),
            )
            self.per_layer_model_projection = PTQWrapper(
                fp_model.per_layer_model_projection,
                qcfg=qcfg.child("per_layer_model_projection") if qcfg else None,
                fp_name=join_name(fp_name, "per_layer_model_projection"),
            )
            self.per_layer_projection_norm = PTQWrapper(
                fp_model.per_layer_projection_norm,
                qcfg=qcfg.child("per_layer_projection_norm") if qcfg else None,
                fp_name=join_name(fp_name, "per_layer_projection_norm"),
            )
            self.per_layer_input_scale = fp_model.per_layer_input_scale
            self.per_layer_model_projection_scale = (
                fp_model.per_layer_model_projection_scale
            )

        self.obs_inputs_embeds = self._make_obs("inputs_embeds")
        self.obs_per_layer_inputs = self._make_obs("per_layer_inputs")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask=None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """Run the wrapped text model.

        TODO: Replace HF mask creation with CPU-provided static mask mapping in
        the static runtime path. This method remains HF-compatible for wrapper
        smoke tests and calibration.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self._fq(inputs_embeds, self.obs_inputs_embeds)

        if self.hidden_size_per_layer_input and per_layer_inputs is None:
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds)
        if per_layer_inputs is not None:
            per_layer_inputs = self._fq(per_layer_inputs, self.obs_per_layer_inputs)

        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        if not isinstance(attention_mask, dict):
            raise NotImplementedError(
                "QuantGemma4TextModel expects static attention mask mapping for the first implementation."
            )

        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }

        hidden_states = inputs_embeds
        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())
        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                shared_key_value=shared_kv_states.get(layer_type),
                position_embeddings=position_embeddings[layer_type],
                attention_mask=attention_mask[layer_type],
                past_key_value=None,
                use_cache=False,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def project_per_layer_inputs(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Compute the context-aware per-layer input projection for Gemma4 PLE."""
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Per-layer input projection is not enabled for this Gemma4 config."
            )
        per_layer_model_projection = self.per_layer_model_projection
        per_layer_projection_norm = self.per_layer_projection_norm
        if per_layer_model_projection is None or per_layer_projection_norm is None:
            raise RuntimeError("Gemma4 PLE projection modules are not initialized.")
        per_layer_projection = (
            per_layer_model_projection(inputs_embeds)
            * self.per_layer_model_projection_scale
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = per_layer_projection_norm(per_layer_projection)
        return per_layer_projection

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_inputs_embeds, self.obs_per_layer_inputs)

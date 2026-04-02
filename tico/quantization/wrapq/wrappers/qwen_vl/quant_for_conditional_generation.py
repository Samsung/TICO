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

from typing import Iterable, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLForConditionalGeneration",
)
class QuantQwen3VLForConditionalGeneration(QuantModuleBase):
    def __init__(
        self,
        model_fp: nn.Module,  # This will be an instance of Qwen3VLForConditionalGeneration
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # Store reference to original model for accessing its attributes
        self.wrapped = model_fp

        # Wrap self.model_fp.model (an instance of Qwen3VLModel)
        model_cfg = qcfg.child("model") if qcfg else None
        # Use type: ignore for the model attribute since we know it's a Module
        self.model = PTQWrapper(
            model_fp.model, qcfg=model_cfg, fp_name=f"{fp_name}.model"  # type: ignore[arg-type]
        )

        # Wrap self.model_fp.lm_head (an instance of nn.Linear)
        lm_head_cfg = qcfg.child("lm_head") if qcfg else None
        # Use type: ignore for the lm_head attribute since we know it's a Module
        self.lm_head = PTQWrapper(
            model_fp.lm_head, qcfg=lm_head_cfg, fp_name=f"{fp_name}.lm_head"  # type: ignore[arg-type]
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Get config from wrapped model
        # Type ignore is needed because Pylance infers self.wrapped incorrectly
        config = self.wrapped.config  # type: ignore[attr-defined]

        output_attentions = config.output_attentions  # type: ignore[attr-defined]
        output_hidden_states = config.output_hidden_states  # type: ignore[attr-defined]
        return_dict = config.use_return_dict  # type: ignore[attr-defined]

        # Call the wrapped model to get hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Get loss function from wrapped model
            loss_fct = self.wrapped.loss_function  # type: ignore[attr-defined]
            loss = loss_fct(
                logits=logits,
                labels=labels,
                vocab_size=config.vocab_size,  # type: ignore[attr-defined]
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _all_observers(self) -> Iterable:
        # Recursively return observers from subcomponents
        yield from self.model._all_observers()
        yield from self.lm_head._all_observers()

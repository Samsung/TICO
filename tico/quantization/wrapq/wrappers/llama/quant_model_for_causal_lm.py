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

from typing import List, Optional, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.processing_utils import Unpack

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


def fix_inputs(config, pad_token_id, input_ids):
    pads = torch.full(
        (
            input_ids.shape[0],
            config.max_position_embeddings - input_ids.shape[1],
        ),
        fill_value=pad_token_id,
        device=input_ids.device,
    )

    return torch.cat((input_ids, pads), dim=1)


@try_register("transformers.models.llama.modeling_llama.LlamaForCausalLM")
class QuantLlamaForCausalLM(QuantModuleBase):
    def __init__(
        self,
        model_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.__dict__.update(model_fp.__dict__)  # for quantizers at least

        # ----- child configs (hierarchical override) -------------------
        model_cfg = qcfg.child("model") if qcfg else None
        lm_head_cfg = qcfg.child("lm_head") if qcfg else None

        ## ----- wrap model/lm_head -------------------------------
        assert hasattr(model_fp, "model") and isinstance(
            model_fp.model, torch.nn.Module
        )
        assert hasattr(model_fp, "lm_head") and isinstance(
            model_fp.lm_head, torch.nn.Module
        )

        self.model = PTQWrapper(
            model_fp.model, qcfg=model_cfg, fp_name=f"{fp_name}.model"
        )

        self.lm_head = PTQWrapper(
            model_fp.lm_head, qcfg=lm_head_cfg, fp_name=f"{fp_name}.lm_head"
        )
        self.config = model_fp.config
        self.loss_function = model_fp.loss_function

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        orig_len = input_ids.shape[-1]
        pad_id = (
            self.config.pad_token_id
            if hasattr(self.config, "pad_token_id")
            else self.config.eos_token_id
        )

        input_ids = fix_inputs(self.config, pad_id, input_ids)
        if labels is not None:
            labels = fix_inputs(self.config, pad_id, labels)

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits[..., :orig_len, :]
        if labels is not None:
            labels = labels[..., :orig_len]

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
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

    def _all_observers(self):
        # recurse into children that are QuantModuleBase
        for m in (self.model, self.lm_head):
            yield from m._all_observers()

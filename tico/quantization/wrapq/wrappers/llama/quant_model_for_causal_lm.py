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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    QuantLlamaForCausalLMDecodeExportAdapter,
    QuantLlamaForCausalLMPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

LegacyCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


@try_register(
    "transformers.models.llama.modeling_llama.LlamaForCausalLM",
    "tico.quantization.algorithm.spinquant.spin_llama.SpinLlamaForCausalLM",
)
class QuantLlamaForCausalLM(QuantModuleBase, GenerationMixin):
    _is_stateful = False

    def __init__(
        self,
        model_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.quantizers = getattr(model_fp, "quantizers", None)

        # ----- child configs (hierarchical override) -------------------
        model_cfg = qcfg.child("model") if qcfg else None
        lm_head_cfg = qcfg.child("lm_head") if qcfg else None
        rotate_lm_head_cfg = qcfg.child("rotate_lm_head") if qcfg else None

        ## ----- wrap model/lm_head -------------------------------
        assert hasattr(model_fp, "model") and isinstance(
            model_fp.model, torch.nn.Module
        )
        assert hasattr(model_fp, "lm_head") and isinstance(
            model_fp.lm_head, torch.nn.Module
        )

        self.model = PTQWrapper(
            model_fp.model, qcfg=model_cfg, fp_name=join_name(fp_name, "model")
        )

        self.lm_head = PTQWrapper(
            model_fp.lm_head, qcfg=lm_head_cfg, fp_name=join_name(fp_name, "lm_head")
        )

        # `rotate_lm_head` exists only for SpinQuant-style custom models.
        # For a standard LlamaForCausalLM, skip creating the wrapper and
        # bypass it during forward.
        self.rotate_lm_head = None
        if hasattr(model_fp, "rotate_lm_head") and isinstance(
            model_fp.rotate_lm_head, torch.nn.Module
        ):
            self.rotate_lm_head = PTQWrapper(
                model_fp.rotate_lm_head,
                rotate_lm_head_cfg,
                fp_name=join_name(fp_name, "rotate_lm_head"),
            )

        self.config = model_fp.config
        self.generation_config = getattr(model_fp, "generation_config", None)
        self.main_input_name = getattr(model_fp, "main_input_name", "input_ids")
        self.loss_function = model_fp.loss_function
        self.device = model_fp.device

    def tie_weights(self):
        pass

    def get_input_embeddings(self):
        return self.model.wrapped.embed_tokens

    def set_input_embeddings(self, value):
        self.model.wrapped.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache | LegacyCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        # Decode step: only feed the newest token ids.
        past_seen_tokens = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_seen_tokens = past_key_values.get_seq_length()
            else:
                past_seen_tokens = int(past_key_values[0][0].shape[2])

        if past_seen_tokens > 0:
            input_ids = input_ids[:, -1:]

        # Standard HF-style position_ids fallback from attention_mask.
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            position_ids = position_ids[:, -input_ids.shape[1] :]

        # Keep cache_position aligned with the current decoding step.
        if cache_position is None:
            past_seen_tokens = 0
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    past_seen_tokens = past_key_values.get_seq_length()
                else:
                    past_seen_tokens = int(past_key_values[0][0].shape[2])

            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_ids.shape[1],
                device=input_ids.device,
            )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "cache_position": cache_position,
        }

        # HF convention: inputs_embeds are only used on the first step.
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs.pop("input_ids", None)

        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: Optional[Cache | LegacyCache],
        beam_idx: torch.LongTensor,
    ):
        if past_key_values is None:
            return past_key_values

        # Cache classes may implement their own reorder logic depending on the
        # transformers version. Use it if present.
        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "reorder_cache"):
                return past_key_values.reorder_cache(beam_idx)
            return past_key_values

        reordered = []
        for layer_past in past_key_values:
            past_k, past_v = layer_past
            reordered.append(
                (
                    past_k.index_select(0, beam_idx),
                    past_v.index_select(0, beam_idx),
                )
            )
        return tuple(reordered)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | LegacyCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
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

        hidden_states = (
            outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        )

        # Apply the SpinQuant rotation only when the source model provides it.
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)

        if isinstance(logits_to_keep, int):
            if logits_to_keep > 0:
                slice_indices = slice(-logits_to_keep, None)
            else:
                slice_indices = slice(None)
        else:
            slice_indices = logits_to_keep

        logits = self.lm_head(hidden_states[:, slice_indices, :])

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
        if self.rotate_lm_head is not None:
            yield from self.rotate_lm_head._all_observers()

    def as_export_module(
        self, mode: ExportMode = "prefill", return_kv: bool = True
    ) -> nn.Module:
        if mode == "prefill":
            return QuantLlamaForCausalLMPrefillExportAdapter(self, return_kv=return_kv)
        elif mode == "decode":
            return QuantLlamaForCausalLMDecodeExportAdapter(self)
        raise ValueError(f"Unsupported export mode: {mode!r}")

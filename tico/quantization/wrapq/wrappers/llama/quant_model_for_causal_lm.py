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

from typing import cast, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaForCausalLM


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
        model_fp = cast("LlamaForCausalLM", model_fp)
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

        self._generation_model_class = type(model_fp)
        self.config = model_fp.config
        self.generation_config = model_fp.generation_config
        self.main_input_name = getattr(model_fp, "main_input_name", "input_ids")
        self.loss_function = model_fp.loss_function
        self.device = model_fp.device

    @property
    def _generation_source_class(self):
        """Read legacy full-model checkpoints without retaining a second model."""
        source_class = getattr(self, "_generation_model_class", None)
        if source_class is None:
            from transformers.models.llama.modeling_llama import LlamaForCausalLM

            source_class = LlamaForCausalLM
        return source_class

    def get_experts_implementation(self):
        """Use the source HF model's configuration snapshot contract."""
        return self._generation_source_class.get_experts_implementation(self)

    def set_experts_implementation(self, experts_implementation):
        """Preserve HF backend validation and dense-model setter semantics."""
        return self._generation_source_class.set_experts_implementation(
            self, experts_implementation
        )

    def _check_and_adjust_experts_implementation(self, experts_implementation):
        """Validate through HF without retaining the original FP model."""
        return self._generation_source_class.get_correct_experts_implementation(
            self, experts_implementation
        )

    def _can_set_experts_implementation(self):
        """Query the source model class, whose dense/MoE contract is unchanged."""
        return self._generation_source_class._can_set_experts_implementation()

    def _grouped_mm_can_dispatch(self):
        """Preserve HF's capability check for an explicitly requested backend."""
        return self._generation_source_class._grouped_mm_can_dispatch(self)

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
        input_ids: torch.Tensor,
        next_sequence_length: int | None = None,
        past_key_values: Optional[Cache | LegacyCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_first_iteration: bool | None = False,
        cache_position: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        # Select the unprocessed tokens supplied by the generation loop.
        past_seen_tokens = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_seen_tokens = past_key_values.get_seq_length()
            else:
                past_seen_tokens = int(past_key_values[0][0].shape[2])

        if next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]
        elif past_seen_tokens > 0:
            input_ids = input_ids[:, -1:]

        use_inputs_embeds = inputs_embeds is not None and (
            is_first_iteration or past_seen_tokens == 0
        )
        if use_inputs_embeds and inputs_embeds is not None:
            if next_sequence_length is not None:
                inputs_embeds = inputs_embeds[:, -next_sequence_length:]
            sequence_length = inputs_embeds.shape[1]
        else:
            sequence_length = input_ids.shape[1]

        # Standard HF-style position_ids fallback from attention_mask.
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        if position_ids is not None:
            position_ids = position_ids[..., -sequence_length:]

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
                past_seen_tokens + sequence_length,
                device=input_ids.device,
            )

        else:
            cache_position = cache_position[-sequence_length:]

        model_inputs = {key: value for key, value in kwargs.items() if key != "labels"}
        model_inputs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "cache_position": cache_position,
            }
        )

        # HF convention: inputs_embeds are only used on the first step.
        if use_inputs_embeds:
            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs.pop("input_ids", None)

        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: Optional[Cache | LegacyCache],
        beam_idx: torch.Tensor,
    ):
        if past_key_values is None:
            return past_key_values

        # Cache classes may implement their own reorder logic depending on the
        # transformers version. Use it if present.
        if isinstance(past_key_values, Cache):
            if hasattr(past_key_values, "reorder_cache"):
                return past_key_values.reorder_cache(cast(torch.LongTensor, beam_idx))
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
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | LegacyCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
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
                slice_indices: slice | torch.Tensor = slice(-logits_to_keep, None)
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
        # This wrapper owns no observers directly.
        return ()

    def as_export_module(
        self, mode: ExportMode = "prefill", return_kv: bool = True
    ) -> nn.Module:
        if mode == "prefill":
            return QuantLlamaForCausalLMPrefillExportAdapter(self, return_kv=return_kv)
        elif mode == "decode":
            return QuantLlamaForCausalLMDecodeExportAdapter(self)
        raise ValueError(f"Unsupported export mode: {mode!r}")

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

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from transformers.generation.utils import GenerationMixin

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM")
class QuantGemma4ForCausalLM(QuantModuleBase, GenerationMixin):
    """PTQ wrapper for Gemma4 text-only causal LM."""

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

        self.model = PTQWrapper(
            fp_model.model,
            qcfg=qcfg.child("model") if qcfg else None,
            fp_name=join_name(fp_name, "model"),
        )
        self.lm_head = PTQWrapper(
            fp_model.lm_head,
            qcfg=qcfg.child("lm_head") if qcfg else None,
            fp_name=join_name(fp_name, "lm_head"),
        )

        # Observers for the logit softcapping path.
        self.obs_logit_softcapping_div = self._make_obs("logit_softcapping_div")
        self.obs_logit_softcapping_tanh = self._make_obs("logit_softcapping_tanh")
        self.obs_logits = self._make_obs("logits")

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Run the wrapped causal LM model.

        Mirrors ``Gemma4ForCausalLM.forward`` including logit softcapping.
        Fake-quantization observers are inserted after the ``tanh`` and on
        the final logits so that the export path carries correct qparam
        metadata.

        When ``labels`` is provided, the cross-entropy loss is computed via
        the original model's ``loss_function`` and a
        ``Gemma4CausalLMOutputWithPast`` is returned, matching the HF
        contract expected by evaluation utilities (e.g. perplexity).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs
        )
        # Match the original's logits_to_keep handling: int → slice, tensor → index.
        if isinstance(logits_to_keep, int) and logits_to_keep:
            slice_indices = slice(-logits_to_keep, None)
        elif isinstance(logits_to_keep, torch.Tensor):
            slice_indices = logits_to_keep
        else:
            slice_indices = slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = self._apply_logit_softcapping(logits)

        loss = None
        if labels is not None:
            loss = self.module.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4CausalLMOutputWithPast,
        )

        return Gemma4CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logit softcapping with fake-quantization observers.

        Mirrors the original ``Gemma4ForCausalLM`` softcapping:
        ``logits = tanh(logits / softcap) * softcap``.

        Three observers are inserted so that every graph node in the
        softcapping chain carries quantization parameter metadata:
        - ``obs_logit_softcapping_div``  — after the division
        - ``obs_logit_softcapping_tanh`` — after the tanh
        - ``obs_logits``                 — on the final logits
        """
        final_logit_softcapping = self.config.final_logit_softcapping
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = self._fq(logits, self.obs_logit_softcapping_div)
            logits = torch.tanh(logits)
            logits = self._fq(logits, self.obs_logit_softcapping_tanh)
            logits = logits * final_logit_softcapping

        logits = self._fq(logits, self.obs_logits)
        return logits

    # --- Generation support --------------------------------------------------
    # Do NOT override ``generate()``: ``GenerationMixin.generate()`` calls our
    # quantized ``forward()`` so fake-quantization is active during decoding.

    main_input_name = "input_ids"
    _is_stateful = True
    _supports_cache_class = True

    @property
    def device(self):
        """Return the device for generation."""
        return self.module.device

    @property
    def generation_config(self):
        """Return the generation config."""
        return self.module.generation_config

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        """Prepare inputs for generation step.

        Delegates to the original model's implementation so the
        fake-quantized ``forward()`` receives correctly prepared inputs.
        """
        return self.module.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            labels=labels,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

    def tie_weights(self):
        pass

    def can_generate(self) -> bool:
        return True

    def get_output_embeddings(self):
        return self.module.get_output_embeddings()

    def get_input_embeddings(self):
        return self.module.get_input_embeddings()

    def get_experts_implementation(self):
        return self.module.get_experts_implementation()

    def set_experts_implementation(self, experts_implementation):
        return self.module.set_experts_implementation(experts_implementation)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
            self.obs_logit_softcapping_div,
            self.obs_logit_softcapping_tanh,
            self.obs_logits,
        )

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

"""PTQ wrapper for the Gemma4 assistant draft-one causal LM."""

from typing import Any, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from transformers.generation.utils import GenerationMixin

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_backbone import (
    QuantGemma4AssistantBackbone,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_masked_embedder import (
    QuantGemma4AssistantMaskedEmbedder,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    extract_assistant_text_config,
    HF_GEMMA4_ASSISTANT_CLASS_PATH,
    validate_gemma4_assistant_architecture,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


LayerKV = Tuple[torch.Tensor, torch.Tensor]


@try_register(HF_GEMMA4_ASSISTANT_CLASS_PATH)
class QuantGemma4AssistantForCausalLM(QuantModuleBase, GenerationMixin):
    """PTQ wrapper for ``Gemma4AssistantForCausalLM`` (MTP draft model).

    The assistant consumes a concatenated target embedding/hidden-state input
    plus target shared KV states, and produces the projected state and draft
    logits for one position. It never uses ``model.embed_tokens`` in its core
    compute, never projects K/V, and never allocates its own KV cache.

    ``lm_head`` is the single quantized source of truth for the tied
    embedding/LM-head weight. The ordered sparse head reads the (fake-)
    quantized weight from that wrapper instead of quantizing a second copy.
    """

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        validate_gemma4_assistant_architecture(fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config
        self.text_config = extract_assistant_text_config(fp_model)
        self.use_ordered_embeddings = bool(
            getattr(self.config, "use_ordered_embeddings", False)
        )
        self.backbone_hidden_size = int(self.config.backbone_hidden_size)
        self.hidden_size = int(self.text_config.hidden_size)
        self.vocab_size = int(self.text_config.vocab_size)
        self._validate_tied_lm_head(fp_model)

        self.pre_projection = PTQWrapper(
            fp_model.pre_projection,
            qcfg=qcfg.child("pre_projection") if qcfg else None,
            fp_name=join_name(fp_name, "pre_projection"),
        )
        self.model = QuantGemma4AssistantBackbone(
            fp_model.model,
            qcfg=qcfg.child("model") if qcfg else None,
            fp_name=join_name(fp_name, "model"),
        )
        self.post_projection = PTQWrapper(
            fp_model.post_projection,
            qcfg=qcfg.child("post_projection") if qcfg else None,
            fp_name=join_name(fp_name, "post_projection"),
        )
        self.lm_head = PTQWrapper(
            fp_model.lm_head,
            qcfg=qcfg.child("lm_head") if qcfg else None,
            fp_name=join_name(fp_name, "lm_head"),
        )

        self.masked_embedding: Optional[QuantGemma4AssistantMaskedEmbedder] = None
        if self.use_ordered_embeddings:
            self.masked_embedding = QuantGemma4AssistantMaskedEmbedder(
                fp_model.masked_embedding,
                qcfg=qcfg.child("masked_embedding") if qcfg else None,
                fp_name=join_name(fp_name, "masked_embedding"),
            )

    @staticmethod
    def _validate_tied_lm_head(fp_model: nn.Module) -> None:
        """Reject a broken tied-weight layout before quantization starts."""
        if not bool(getattr(fp_model.config, "tie_word_embeddings", False)):
            return
        embed_tokens = getattr(fp_model.model, "embed_tokens", None)
        if embed_tokens is None:
            return
        if fp_model.lm_head.weight.data_ptr() != embed_tokens.weight.data_ptr():
            raise ValueError(
                "Gemma4 assistant config declares tie_word_embeddings=True, "
                "but lm_head.weight and model.embed_tokens.weight are not the "
                "same tensor. Quantizing two copies of a tied weight is not "
                "supported."
            )

    # --- Mask handling -------------------------------------------------------

    def _bounded_mask(
        self,
        mask: Any,
        *,
        batch_size: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize one HF or explicit mask to a bounded additive tensor.

        Masked entries use ``PTQConfig.attention_mask_fill_value`` instead of
        the dtype minimum so affine activation observers keep usable ranges.
        """
        fill = float(self.qcfg.attention_mask_fill_value)
        if mask is None:
            return torch.zeros(batch_size, 1, q_len, kv_len, device=device, dtype=dtype)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(
                "Gemma4 assistant masks must have rank 3 or 4, "
                f"got shape={tuple(mask.shape)}."
            )
        if mask.size(-1) != kv_len:
            raise ValueError(
                "Gemma4 assistant mask key length does not match shared KV: "
                f"mask_k={mask.size(-1)}, kv_len={kv_len}."
            )
        mask = mask.to(device=device)
        if mask.dtype is torch.bool:
            return torch.zeros(mask.shape, device=device, dtype=dtype).masked_fill(
                ~mask, fill
            )
        mask = mask.to(dtype=dtype)
        return torch.where(
            mask < 0,
            torch.full_like(mask, fill),
            torch.zeros_like(mask),
        )

    def _create_attention_masks(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Any,
        shared_kv_states: Mapping[str, LayerKV],
    ) -> dict[str, torch.Tensor]:
        """Build bounded per-layer-type masks with HF assistant semantics."""
        batch_size, q_len = inputs_embeds.shape[:2]
        missing = [
            layer_type
            for layer_type in self.model.unique_layer_types
            if shared_kv_states.get(layer_type) is None
        ]
        if missing:
            raise ValueError(f"shared_kv_states is missing layer types: {missing}.")
        kv_lens = {
            layer_type: int(shared_kv_states[layer_type][0].shape[2])
            for layer_type in self.model.unique_layer_types
        }

        if isinstance(attention_mask, Mapping):
            raw = attention_mask
        else:
            raw = self.module.create_attention_masks(
                inputs_embeds, attention_mask, shared_kv_states
            )

        return {
            layer_type: self._bounded_mask(
                raw.get(layer_type),
                batch_size=batch_size,
                q_len=q_len,
                kv_len=kv_lens[layer_type],
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            for layer_type in self.model.unique_layer_types
        }

    # --- Core forward --------------------------------------------------------

    def _lm_head_weight(self) -> torch.Tensor:
        """Return the single (fake-quantized) tied LM-head weight source."""
        quant_linear = self.lm_head.wrapped
        weight = quant_linear.module.weight
        if self._mode is Mode.QUANT:
            weight = quant_linear.obs_weight.fake_quant(weight)
        return weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # ignored, HF signature parity
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Any = None,
        shared_kv_states: Optional[Mapping[str, LayerKV]] = None,
        use_cache: Optional[bool] = None,  # ignored, HF signature parity
        position_embeddings: Optional[
            Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        **kwargs,
    ):
        """Run one assistant draft step with HF-compatible inputs.

        ``attention_mask`` may be the HF 2D padding mask (eager path) or an
        already-built mapping of per-layer-type additive masks (static path).
        ``position_embeddings`` optionally bypasses eager RoPE construction
        with explicit per-layer-type ``(cos, sin)`` tables.
        """
        if inputs_embeds is None or shared_kv_states is None:
            raise ValueError("inputs_embeds and shared_kv_states cannot be None.")
        if kwargs.pop("output_hidden_states", None) or kwargs.pop(
            "output_attentions", None
        ):
            raise NotImplementedError(
                "QuantGemma4AssistantForCausalLM does not collect per-layer "
                "hidden states or attentions."
            )

        hidden_states = self.pre_projection(inputs_embeds)
        attention_masks = self._create_attention_masks(
            hidden_states, attention_mask, shared_kv_states
        )
        hidden_states = self.model(
            hidden_states,
            attention_masks=attention_masks,
            shared_kv_states=shared_kv_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        projected_state = self.post_projection(hidden_states)

        if self.masked_embedding is not None:
            logits = self.masked_embedding(hidden_states, self._lm_head_weight())
        else:
            logits = self.lm_head(hidden_states)

        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantOutput,
        )

        return Gemma4AssistantOutput(
            last_hidden_state=projected_state,
            logits=logits,
        )

    # --- Generation support --------------------------------------------------
    # ``GenerationMixin`` compatibility keeps the quantized assistant usable
    # as ``target.generate(..., assistant_model=...)`` (via
    # ``Gemma4AssistantGenerationAdapter``). All compute stays in the
    # quantized ``forward()``; only metadata is delegated to the FP module.

    main_input_name = "input_ids"
    _is_stateful = False

    @property
    def device(self):
        """Return the device for generation."""
        return self.module.device

    @property
    def generation_config(self):
        """Return the generation config."""
        return self.module.generation_config

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate input preparation to the original model."""
        return self.module.prepare_inputs_for_generation(*args, **kwargs)

    def tie_weights(self):
        pass

    def can_generate(self) -> bool:
        return True

    def _supports_logits_to_keep(self) -> bool:
        return False

    def get_output_embeddings(self):
        return self.module.get_output_embeddings()

    def get_input_embeddings(self):
        return self.module.get_input_embeddings()

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return ()

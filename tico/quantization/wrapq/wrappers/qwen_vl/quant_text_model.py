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

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutputWithPast

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel")
class QuantQwen3VLTextModel(QuantModuleBase):
    """
    Quant-aware drop-in replacement for the Qwen3-VL language model text backbone
    (the `language_model` sub-module inside `Qwen3VLModel`).

    Computes shared RoPE position embeddings and a static causal mask once per
    forward pass, then passes them to every decoder layer so they are quantized
    exactly once rather than independently in each layer.

    RoPE note
    ---------
    Position embeddings are computed by calling the model's own `rotary_emb` module
    (Qwen3VLTextRotaryEmbedding) directly in `forward()`, rather than using a
    simplified 1D pre-computed table.  This is important for accuracy because
    Qwen3-VL uses MRoPE: `inv_freq` is split into three sections (temporal, height,
    width) via `mrope_section`, which means a naively-computed 1D cos/sin table
    would differ from the model's actual position encodings and accumulate error
    across all decoder layers.

    Known PEIR limitation
    ---------------------
    PEIR measured on this full-backbone wrapper will be higher than for single-layer
    wrappers (e.g. QuantQwen3VLTextAttention) because quantization errors from all
    decoder layers compound multiplicatively.  This is expected behavior and does not
    reflect task-level accuracy degradation, which should be measured via downstream
    metrics (e.g. perplexity).  Additionally, the static causal mask does not account
    for padding tokens in the calibration data, which can inflate PEIR further when
    padded sequences are used.
    """

    def __init__(
        self,
        model_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # ----- child configs ------------------------------------------------
        embed_cfg = qcfg.child("embed_tokens") if qcfg else None
        norm_cfg = qcfg.child("norm") if qcfg else None
        layers_cfg = qcfg.child("layers") if qcfg else None

        # ----- assertions ---------------------------------------------------
        assert hasattr(model_fp, "embed_tokens") and isinstance(
            model_fp.embed_tokens, nn.Module
        )
        assert hasattr(model_fp, "norm") and isinstance(model_fp.norm, nn.Module)
        assert hasattr(model_fp, "layers") and isinstance(
            model_fp.layers, nn.ModuleList
        )

        # ----- wrap children ------------------------------------------------
        self.embed_tokens = PTQWrapper(
            model_fp.embed_tokens, embed_cfg, fp_name=f"{fp_name}.embed_tokens"
        )
        self.norm = PTQWrapper(model_fp.norm, norm_cfg, fp_name=f"{fp_name}.norm")

        new_list = nn.ModuleList()
        for idx, layer in enumerate(model_fp.layers):
            child_scope = f"{idx}"
            child_cfg = (
                layers_cfg.child(child_scope) if layers_cfg is not None else None
            )
            new_list.append(
                PTQWrapper(layer, child_cfg, fp_name=child_scope)
            )
        self.layers = new_list

        # ----- local observers ----------------------------------------------
        self.obs_causal_mask = self._make_obs("causal_mask")
        self.obs_cos = self._make_obs("cos")
        self.obs_sin = self._make_obs("sin")

        self.config = model_fp.config

        # ----- static buffers: causal mask template -------------------------
        assert isinstance(self.config.max_position_embeddings, int)
        max_seq = self.config.max_position_embeddings
        mask = torch.full((1, 1, max_seq, max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

        # ----- rotary embedding module ---------------------------------------
        # Store the model's actual rotary_emb (Qwen3VLTextRotaryEmbedding) and
        # call it dynamically in forward() to get exact MRoPE position embeddings.
        # Using a simplified 1D pre-computed table would mis-assign frequencies
        # across mrope_section boundaries, producing wrong cos/sin and degrading
        # calibration accuracy significantly.
        rotary = getattr(model_fp, "rotary_emb", None)
        assert rotary is not None, (
            "Qwen3VLTextModel must have a `rotary_emb` attribute"
        )
        self.rotary_emb = rotary

    def _slice_causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        assert isinstance(self.causal_mask_template, torch.Tensor)
        return self.causal_mask_template[..., :seq_len, :seq_len].to(device)

    def get_attention_mask_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        L = hidden_states.size(1)
        return self._slice_causal(L, hidden_states.device)

    def get_position_embeddings_for(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Delegate to the model's actual Qwen3VLTextRotaryEmbedding so that
        # MRoPE frequencies are split correctly by mrope_section.
        S = hidden_states.size(1)
        position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0)
        return self.rotary_emb(hidden_states, position_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Pre-compute shared causal mask and RoPE (quantized once, shared across layers)
        #
        # The static causal part is quantized once and shared across all decoder layers
        # (suitable for on-device baking as a constant).
        #
        # If a 2D padding mask (attention_mask: [B, L], 1=real token, 0=padding) is
        # provided, a padding correction is applied additively on top of the quantized
        # causal mask.  This ensures that calibration statistics are not inflated by
        # activations at padding positions.  The padding correction is NOT quantized
        # separately — it is applied as a float addend at runtime.
        causal_mask = self.get_attention_mask_for(hidden_states)
        causal_mask = causal_mask.squeeze(0)
        causal_mask = self._fq(causal_mask, self.obs_causal_mask)

        if attention_mask is not None and attention_mask.ndim == 2:
            # attention_mask: [B, L], 1 = real token, 0 = padding
            # Build additive correction: 0 for real, -120 for padding → [B, 1, L]
            pad_corr = (1.0 - attention_mask.to(dtype=hidden_states.dtype,
                                                 device=hidden_states.device))
            pad_corr = pad_corr * float("-120")
            pad_corr = pad_corr.unsqueeze(1)  # [B, 1, L]
            causal_mask = causal_mask + pad_corr  # [B, L, L] (broadcast)

        position_embeddings = self.get_position_embeddings_for(hidden_states)
        cos, sin = position_embeddings
        position_embeddings = (
            self._fq(cos, self.obs_cos),
            self._fq(sin, self.obs_sin),
        )

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

            if hasattr(decoder_layer, "wrapped") and hasattr(
                decoder_layer.wrapped, "return_type"
            ):
                if decoder_layer.wrapped.return_type == "tuple":
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs
            else:
                hidden_states = (
                    layer_outputs[0]
                    if isinstance(layer_outputs, tuple)
                    else layer_outputs
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _all_observers(self):
        yield from (self.obs_causal_mask, self.obs_cos, self.obs_sin)
        for m in (self.embed_tokens, self.norm):
            yield from m._all_observers()
        for m in self.layers:
            yield from m._all_observers()

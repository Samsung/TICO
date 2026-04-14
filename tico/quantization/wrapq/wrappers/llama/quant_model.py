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

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaModel",
    "tico.quantization.algorithm.spinquant.spin_llama.SpinLlamaModel",
)
class QuantLlamaModel(QuantModuleBase):
    def __init__(
        self,
        model_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # ----- child configs (hierarchical override) -------------------
        embed_cfg = qcfg.child("embed_tokens") if qcfg else None
        rotate_embed_cfg = qcfg.child("rotate_embedding") if qcfg else None
        norm_cfg = qcfg.child("norm") if qcfg else None
        layers_cfg = qcfg.child("layers") if qcfg else None

        # ----- wrap children -------------------------------
        assert hasattr(model_fp, "embed_tokens") and isinstance(
            model_fp.embed_tokens, torch.nn.Module
        )
        assert hasattr(model_fp, "norm") and isinstance(model_fp.norm, torch.nn.Module)
        assert hasattr(model_fp, "layers") and isinstance(
            model_fp.layers, torch.nn.ModuleList
        )

        self.embed_tokens = PTQWrapper(
            model_fp.embed_tokens, embed_cfg, fp_name=f"{fp_name}.embed_tokens"
        )

        self.norm = PTQWrapper(model_fp.norm, norm_cfg, fp_name=f"{fp_name}.norm")

        # `rotate_embedding` exists only for SpinQuant-style custom models.
        # For a standard LlamaModel, skip creating the wrapper and bypass it
        # during forward.
        self.rotate_embedding = None
        if hasattr(model_fp, "rotate_embedding") and isinstance(
            model_fp.rotate_embedding, torch.nn.Module
        ):
            self.rotate_embedding = PTQWrapper(
                model_fp.rotate_embedding,
                rotate_embed_cfg,
                fp_name=f"{fp_name}.rotate_embedding",
            )

        new_list = nn.ModuleList()
        for idx, layer in enumerate(model_fp.layers):
            child_scope = f"{fp_name}.layers.{idx}"
            child_cfg = layers_cfg.child(child_scope) if layers_cfg is not None else None  # type: ignore[union-attr]
            new_list.append(
                PTQWrapper(
                    layer,
                    child_cfg,
                    fp_name=child_scope,
                )
            )
        self.obs_causal_mask = self._make_obs("causal_mask")
        self.obs_cos = self._make_obs("cos")
        self.obs_sin = self._make_obs("sin")

        self.layers = new_list  # type: ignore[union-attr]
        self.config = model_fp.config
        # Static causal mask template ---------------------------------------
        assert isinstance(self.config.max_position_embeddings, int)
        max_seq = self.config.max_position_embeddings
        mask = torch.full((1, 1, max_seq, max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

        # Static RoPE (position_embeddings) templates ------------------------
        cfg = self.config
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )

        # 1) inv_freq, scaling
        rotary = getattr(model_fp, "rotary_emb", None)
        assert rotary is not None
        if hasattr(rotary, "inv_freq"):
            inv_freq = rotary.inv_freq.detach().float()
            attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))
        else:
            rope_params = getattr(cfg, "rope_parameters", None)
            if (
                rope_params is not None
                and isinstance(rope_params, dict)
                and "rope_theta" in rope_params
            ):
                base = float(rope_params["rope_theta"])
            else:
                base = float(getattr(cfg, "rope_theta", 10000.0))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            attn_scaling = 1.0

        # 2) Create cos/sin: [max_seq, head_dim]
        pos = torch.arange(
            max_seq, dtype=torch.float32, device=inv_freq.device
        )  # [max_seq]
        freqs = torch.outer(pos, inv_freq)  # [max_seq, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq, head_dim]
        cos_t = emb.cos() * attn_scaling
        sin_t = emb.sin() * attn_scaling
        half_dim = head_dim // 2
        sin_t[..., :half_dim] = -sin_t[..., :half_dim]
        cos_t = cos_t.unsqueeze(0)  # [1, max_seq, head_dim]
        sin_t = sin_t.unsqueeze(0)  # [1, max_seq, head_dim]

        self.register_buffer("rope_cos_template", cos_t, persistent=False)
        self.register_buffer("rope_sin_template", sin_t, persistent=False)

    def _slice_rope(
        self,
        *,
        start: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.rope_cos_template, torch.Tensor)
        assert isinstance(self.rope_sin_template, torch.Tensor)
        end = start + seq_len
        cos = self.rope_cos_template[:, start:end, :].to(device=device, dtype=dtype)
        sin = self.rope_sin_template[:, start:end, :].to(device=device, dtype=dtype)
        return cos, sin

    def _normalize_position_embeddings(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            q_len = hidden_states.size(1)
            past_len = 0 if past_key_value is None else int(past_key_value[0].shape[2])
            cos, sin = self._slice_rope(
                start=past_len,
                seq_len=q_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            cos, sin = position_embeddings
        return self._fq(cos, self.obs_cos), self._fq(sin, self.obs_sin)

    def _normalize_attention_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return an additive attention mask usable with per-head logits.

        Supported cases:
        - None: build a causal mask slice.
        - bool mask: convert to additive mask using 0 / -120.
        - additive mask: use as-is.
        """
        seq_len = hidden_states.size(1)
        past_len = (
            0
            if (past_key_value is None or past_key_value[0] is None)
            else int(past_key_value[0].shape[2])
        )

        if attention_mask is None:
            assert isinstance(self.causal_mask_template, torch.Tensor)
            mask = self.causal_mask_template[
                ..., past_len : past_len + seq_len, : past_len + seq_len
            ].to(device)
            return mask.squeeze(0)

        if attention_mask.dtype == torch.bool or attention_mask.dtype == torch.int64:
            if attention_mask.dtype == torch.int64:
                attention_mask = attention_mask == 1  # convert to bool
            mask = self.causal_mask_template[
                ..., past_len : past_len + seq_len, : past_len + seq_len
            ].to(
                device
            )  # so for q_len == 1 mask will be the last row of causal_mask_template
            # only padding which is assumed to change causal_mask
            additive = torch.zeros_like(attention_mask, dtype=torch.float32)
            additive = additive.masked_fill(~attention_mask, float("-120"))
            mask = torch.max(
                torch.tensor(float("-120")).to(device), additive + mask
            )  # so -120-120->-120, -120+0->-120, 0-120->-120, 0+0->0
            return mask.squeeze(0)

        return attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = []

        present_key_values = []
        hidden_states = inputs_embeds

        # Apply the SpinQuant rotation only when the source model provides it.
        if self.rotate_embedding is not None:
            hidden_states = self.rotate_embedding(hidden_states)

        past_key_value = None  # sample kv-cache to infer past_seq_len
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                if past_key_values.layers[0].keys is not None:
                    past_key_value = (
                        past_key_values.layers[0].keys,
                        past_key_values.layers[0].values,
                    )
            elif len(past_key_values) > 0:
                past_key_value = past_key_values[0]

        # create position_embeddings and causal_mask to be shared across all the decoder layers
        causal_mask = self._normalize_attention_mask(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            device=hidden_states.device,
        )
        causal_mask = self._fq(causal_mask, self.obs_causal_mask)

        position_embeddings = self._normalize_position_embeddings(
            hidden_states=hidden_states,
            position_embeddings=None,
            past_key_value=past_key_value,
        )
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            if use_cache is True:
                if isinstance(past_key_values, DynamicCache):
                    past_key_value = (
                        past_key_values.layers[idx].keys,
                        past_key_values.layers[idx].values,
                    )
                else:
                    past_key_value = (
                        past_key_values[idx] if idx < len(past_key_values) else None
                    )
            else:
                past_key_value = None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            if decoder_layer.wrapped.return_type == "tuple":
                hidden_states = layer_outputs[0]
            elif use_cache is True:
                hidden_states = layer_outputs[0]
                if isinstance(past_key_values, DynamicCache):
                    past_key_values.update(
                        layer_outputs[1][0], layer_outputs[1][1], layer_idx=idx
                    )
                else:
                    present_key_values.append(
                        (layer_outputs[1][0], layer_outputs[1][1])
                    )
            else:
                hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # type: ignore[operator]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=(
                past_key_values
                if use_cache and isinstance(past_key_values, DynamicCache)
                else present_key_values
            ),
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _all_observers(self):
        # Recurse into children that are QuantModuleBase
        yield from (self.obs_causal_mask, self.obs_cos, self.obs_sin)

        for m in (self.embed_tokens, self.norm):
            yield from m._all_observers()

        if self.rotate_embedding is not None:
            yield from self.rotate_embedding._all_observers()

        for m in self.layers:
            yield from m._all_observers()

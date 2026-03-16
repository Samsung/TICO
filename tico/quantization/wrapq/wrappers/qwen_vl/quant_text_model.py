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

import types
from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register
from transformers.cache_utils import Cache


def apply_interleaved_mrope(self, freqs, mrope_section):
    """
    Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity.

    Args:
        freqs: (3, bs, seq_len, head_dim // 2)
        mrope_section: (3,)

    Returns:
        freqs_t: (bs, seq_len, head_dim // 2)

    Design Note:
        This implementation is using slice_copy, index_select, and cat
        to avoid yet unsupported slice_scatter with step=3 operation and
        to avoid unsupported in-place operator index_put.default.
    """
    # Start with T dimension (will keep some, replace some)
    freqs_t_base = freqs[0]

    # For each dimension (H, W), extract frequency bands to be interleaved
    h_w_bands = []

    for dim, offset in enumerate((1, 2), start=1):  # H, W dimensions
        length = mrope_section[dim] * 3
        indices = torch.arange(offset, length, 3, device=freqs.device)

        # Select frequency bands from H/W dimensions
        # freqs[dim] has shape (bs, seq_len, head_dim//2)
        # index_select on last dim: (bs, seq_len, num_selected)
        freqs_bands = freqs[dim].index_select(dim=-1, index=indices)
        h_w_bands.append(freqs_bands)

    # Now we need to build the interleaved output
    # Original T dimension indices range from 0 to (head_dim // 2 - 1)
    # We want to replace specific indices with H/W bands

    # The interleaving pattern: T0, H1, W2, T3, H4, W5, T6, H7, W8, ...
    # - Positions where i % 3 == 0: T dimension (unchanged from freqs[0])
    # - Positions where i % 3 == 1: H dimension (overwritten from freqs[1])
    # - Positions where i % 3 == 2: W dimension (overwritten from freqs[2])
    # After mrope_section[dim] * 3 positions, H/W bands are exhausted and
    # fallback to T values for all remaining mod 1 and mod 2 positions.

    # Build the output by slicing and concatenating
    # Strategy: Slice T dimension into chunks, insert H/W bands, concatenate

    chunks = []
    pos = 0

    # Total length in the last dimension
    total_len = freqs_t_base.shape[-1]

    for i in range(total_len):
        # Determine which dimension this position belongs to
        # Pattern: T, H, W, T, H, W, T, H, W, ... (repeating cycle)
        mod = i % 3

        if mod == 0:
            # T dimension position - take from T
            # Slice just this index from T
            chunk = freqs_t_base[..., i : i + 1]
            chunks.append(chunk)
        elif mod == 1:
            # H dimension position - take from H
            # Calculate which band this is
            band_idx = (i - 1) // 3
            if band_idx < h_w_bands[0].shape[-1]:
                chunk = h_w_bands[0][..., band_idx : band_idx + 1]
                chunks.append(chunk)
            else:
                # Fallback to T if out of bounds
                chunk = freqs_t_base[..., i : i + 1]
                chunks.append(chunk)
        else:  # mod == 2
            # W dimension position - take from W
            band_idx = (i - 2) // 3
            if band_idx < h_w_bands[1].shape[-1]:
                chunk = h_w_bands[1][..., band_idx : band_idx + 1]
                chunks.append(chunk)
            else:
                # Fallback to T if out of bounds
                chunk = freqs_t_base[..., i : i + 1]
                chunks.append(chunk)

    # Concatenate all chunks
    freqs_t = torch.cat(chunks, dim=-1)

    return freqs_t


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextModel",
)
class QuantQwen3VLTextModel(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLTextModel module.

    This is the text model for Qwen3VL, containing:
    - Embedding layer (embed_tokens)
    - Multiple decoder layers (layers)
    - Final normalization layer (norm)
    - Rotary position embedding (rotary_emb) - NOT wrapped
    """

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        self.module = fp_model
        self.config = fp_model.config

        assert hasattr(fp_model, "embed_tokens")
        assert hasattr(fp_model, "layers")
        assert hasattr(fp_model, "norm")
        assert hasattr(fp_model, "rotary_emb")

        # --- Wrap submodules via PTQWrapper ----------------------------------
        embed_tokens_cfg = qcfg.child("embed_tokens") if qcfg else None
        layers_cfg = qcfg.child("layers") if qcfg else None
        norm_cfg = qcfg.child("norm") if qcfg else None

        self.embed_tokens = PTQWrapper(
            fp_model.embed_tokens,
            qcfg=embed_tokens_cfg,
            fp_name=f"{fp_name}.embed_tokens",
        )

        # Wrap each decoder layer
        self.layers = nn.ModuleList()
        for idx, layer in enumerate(fp_model.layers):
            layer_cfg = layers_cfg.child(str(idx)) if layers_cfg else None
            wrapped_layer = PTQWrapper(
                layer,
                qcfg=layer_cfg,
                fp_name=f"{fp_name}.layers.{idx}",
            )
            self.layers.append(wrapped_layer)

        self.norm = PTQWrapper(
            fp_model.norm,
            qcfg=norm_cfg,
            fp_name=f"{fp_name}.norm",
        )

        # rotary_emb
        self.rotary_emb = fp_model.rotary_emb
        # The original Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope emits `slice_scatter with step > 1`.
        # Replace it with pure functional implementation using basic tensor slicing, index_select, and cat.
        self.rotary_emb.apply_interleaved_mrope = types.MethodType(
            apply_interleaved_mrope, self.rotary_emb
        )

        # ----- static buffers: causal mask template ---------------------------
        assert isinstance(self.config.max_position_embeddings, int)
        max_seq = self.config.max_position_embeddings
        mask = torch.full((1, 1, max_seq, max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

        # --- Observers for floating-point tensors -----------------------------
        mk = self._make_obs
        self.obs_inputs_embeds = mk("inputs_embeds")
        self.obs_attention_mask = mk("attention_mask")
        self.obs_local_this = mk("local_this")
        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        self.obs_deepstack_visual_embeds = []
        for layer_idx in range(len(self.layers)):
            obs_name = f"deepstack_visual_embeds_{layer_idx}"
            obs = mk(obs_name)
            self.obs_deepstack_visual_embeds.append(obs)
            self.add_module(obs_name, obs)

    def _slice_causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        assert isinstance(self.causal_mask_template, torch.Tensor)
        return self.causal_mask_template[..., :seq_len, :seq_len].to(device)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        # args for deepstack
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ):
        """
        Forward pass with fake quantization.

        Args:
            input_ids: Token indices (LongTensor, not quantized)
            attention_mask: Attention mask (may be int, not quantized)
            position_ids: Position indices (LongTensor, not quantized)
            past_key_values: Cached key-value pairs for attention (optional)
            inputs_embeds: Pre-computed input embeddings (optional)
            use_cache: Whether to use key-value caching (optional)
            cache_position: Cache position indices (LongTensor, not quantized)
            visual_pos_masks: Mask indicating visual positions (may be int/bool, not quantized)
            deepstack_visual_embeds: Visual feature embeddings to inject (list of float tensors)
            **kwargs: Additional keyword arguments

        Returns:
            BaseModelOutputWithPast with last_hidden_state and past_key_values
        """
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=self.config)

        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = self._fq(inputs_embeds, self.obs_inputs_embeds)

        # Handle cache_position (integer tensor, not quantized)
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Handle position_ids (integer tensor, not quantized)
        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        # Build causal mask if not provided (or provided as bool)
        if attention_mask is None or attention_mask.dtype == torch.bool:
            attention_mask = self._slice_causal(
                inputs_embeds.shape[1], inputs_embeds.device
            )
            # from transformers.masking_utils import create_causal_mask
            # attention_mask = create_causal_mask(
            #    config=self.config,
            #    input_embeds=inputs_embeds,
            #    attention_mask=attention_mask,
            #    cache_position=cache_position,
            #    past_key_values=past_key_values,
            #    position_ids=text_position_ids,
            # )
        if torch.is_floating_point(attention_mask):
            attention_mask = self._fq(attention_mask, self.obs_attention_mask)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # rotary_emb returns (cos, sin) which are float tensors and need quantization
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        position_embeddings = (
            self._fq(cos, self.obs_cos),
            self._fq(sin, self.obs_sin),
        )

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            # deepstack_visual_embeds are float tensors and need quantization
            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                deepstack_visual_embeds[layer_idx] = self._fq(
                    deepstack_visual_embeds[layer_idx],  # type: ignore[index]
                    self.obs_deepstack_visual_embeds[layer_idx],
                )
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],  # type: ignore[index]
                )

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        """
        Process and inject visual features via DeepStack.

        visual_pos_masks: May be int/bool (not quantized)
        visual_embeds: Float tensor (needs quantization)
        """
        # Move tensors to correct device/dtype
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        local_this = self._fq(local_this, self.obs_local_this)
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module."""
        yield from (
            self.obs_inputs_embeds,
            self.obs_attention_mask,
            self.obs_local_this,
            self.obs_cos,
            self.obs_sin,
        )
        yield from self.obs_deepstack_visual_embeds

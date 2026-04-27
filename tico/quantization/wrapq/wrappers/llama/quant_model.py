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
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


LegacyCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


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
            model_fp.embed_tokens, embed_cfg, fp_name=join_name(fp_name, "embed_tokens")
        )

        self.norm = PTQWrapper(
            model_fp.norm, norm_cfg, fp_name=join_name(fp_name, "norm")
        )

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
                fp_name=join_name(fp_name, "rotate_embedding"),
            )

        new_list = nn.ModuleList()
        for idx, layer in enumerate(model_fp.layers):
            child_scope = f"{idx}"  # qcfg scope
            child_fp_name = join_name(fp_name, f"layers.{idx}")
            child_cfg = layers_cfg.child(child_scope) if layers_cfg is not None else None  # type: ignore[union-attr]
            wrapped_layer = PTQWrapper(
                layer,
                child_cfg,
                fp_name=child_fp_name,
            )

            # Generation/cache path needs tuple outputs from decoder layers.
            if hasattr(wrapped_layer.wrapped, "return_type"):
                wrapped_layer.wrapped.return_type = "tuple"

            # Pass layer index down so QuantLlamaAttention can update the
            # correct slot in Cache / DynamicCache.
            if hasattr(wrapped_layer.wrapped, "layer_idx"):
                wrapped_layer.wrapped.layer_idx = idx
            if (
                hasattr(wrapped_layer.wrapped, "self_attn")
                and hasattr(wrapped_layer.wrapped.self_attn, "wrapped")
                and hasattr(wrapped_layer.wrapped.self_attn.wrapped, "layer_idx")
            ):
                wrapped_layer.wrapped.self_attn.wrapped.layer_idx = idx

            new_list.append(wrapped_layer)

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

    def _make_empty_cache(self) -> Cache:
        """
        Create an empty HF cache object in a version-tolerant way.

        Returns:
            An empty cache instance.

        Raises:
            RuntimeError: If no compatible cache constructor is available.
        """
        try:
            return DynamicCache(config=self.config)
        except TypeError:
            pass
        except Exception:
            pass

        try:
            return DynamicCache()
        except Exception as e:
            raise RuntimeError(
                f"Failed to create an empty DynamicCache for type={DynamicCache}: {e}"
            ) from e

    def _legacy_cache_seq_length(self, past_key_values: LegacyCache) -> int:
        """
        Infer sequence length from a legacy tuple cache.

        Args:
            past_key_values: Legacy tuple of per-layer key/value tensors.

        Returns:
            Cached sequence length inferred from the first layer.
        """
        if len(past_key_values) == 0:
            return 0

        first_layer = past_key_values[0]
        if first_layer is None or len(first_layer) < 2:
            return 0

        past_k, past_v = first_layer
        if past_k is None or past_v is None:
            return 0

        return int(past_k.shape[2])

    def _cache_seq_length(self, cache: Optional[Cache | LegacyCache]) -> int:
        """
        Return cached sequence length for either HF Cache or legacy tuple cache.

        Args:
            cache: Cache object or legacy tuple cache.

        Returns:
            Cached sequence length. Returns zero when no cache is available.
        """
        if cache is None:
            return 0

        if isinstance(cache, tuple):
            return self._legacy_cache_seq_length(cache)

        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return int(get_seq_length())
            except TypeError:
                try:
                    return int(get_seq_length(0))
                except Exception:
                    pass
            except Exception:
                pass

        key_cache = getattr(cache, "key_cache", None)
        if key_cache is not None and len(key_cache) > 0 and key_cache[0] is not None:
            return int(key_cache[0].shape[2])

        layers = getattr(cache, "layers", None)
        if layers is not None and len(layers) > 0 and layers[0] is not None:
            layer0 = layers[0]

            if isinstance(layer0, tuple) and len(layer0) >= 2 and layer0[0] is not None:
                return int(layer0[0].shape[2])

            for name in ("keys", "key", "k"):
                t = getattr(layer0, name, None)
                if t is not None:
                    return int(t.shape[2])

        return 0

    def _legacy_to_cache(self, past_key_values: LegacyCache) -> Cache:
        """
        Convert a legacy tuple cache into an HF cache object.

        Args:
            past_key_values: Legacy tuple of per-layer key/value tensors.

        Returns:
            Cache object containing the same cached tensors.

        Raises:
            RuntimeError: If no compatible conversion or update path exists.
        """
        from_legacy_cache = getattr(DynamicCache, "from_legacy_cache", None)
        if callable(from_legacy_cache):
            return from_legacy_cache(past_key_values)

        cache = self._make_empty_cache()

        for layer_idx, layer_kv in enumerate(past_key_values):
            if layer_kv is None or len(layer_kv) < 2:
                continue

            past_k, past_v = layer_kv
            if past_k is None or past_v is None:
                continue

            update = getattr(cache, "update", None)
            if not callable(update):
                raise RuntimeError(
                    "Cache does not support update(), so legacy cache cannot be converted."
                )

            update(past_k, past_v, layer_idx, cache_kwargs=None)

        return cache

    def _cache_to_legacy(self, cache: Cache) -> LegacyCache:
        """
        Convert an HF cache object into a legacy tuple cache.

        Args:
            cache: HF cache object.

        Returns:
            Legacy tuple cache.

        Raises:
            RuntimeError: If the cache layout cannot be converted.
        """
        to_legacy_cache = getattr(cache, "to_legacy_cache", None)
        if callable(to_legacy_cache):
            return to_legacy_cache()

        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            out = []
            n = min(len(key_cache), len(value_cache))
            for i in range(n):
                k = key_cache[i]
                v = value_cache[i]
                if k is None or v is None:
                    break
                out.append((k, v))
            return tuple(out)

        layers = getattr(cache, "layers", None)
        if layers is not None:
            out = []
            for layer in layers:
                if layer is None:
                    break

                if isinstance(layer, tuple) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    if k is None or v is None:
                        break
                    out.append((k, v))
                    continue

                found = False
                for k_name, v_name in (
                    ("keys", "values"),
                    ("key", "value"),
                    ("k", "v"),
                ):
                    k = getattr(layer, k_name, None)
                    v = getattr(layer, v_name, None)
                    if k is not None and v is not None:
                        out.append((k, v))
                        found = True
                        break

                if not found:
                    break

            return tuple(out)

        raise RuntimeError(
            f"Cache does not support legacy conversion: type={type(cache)}"
        )

    def _normalize_past_key_values(
        self,
        past_key_values: Optional[Cache | LegacyCache],
        *,
        use_cache: bool,
    ) -> tuple[Optional[Cache], bool]:
        """
        Normalize cache input into an HF cache object.

        Args:
            past_key_values: Input cache in HF or legacy tuple form.
            use_cache: Whether cache should be used in the current forward pass.

        Returns:
            A tuple of:
                - normalized cache object or None
                - whether the output cache should be converted back to legacy form
        """
        if past_key_values is None:
            if use_cache:
                return self._make_empty_cache(), False
            return None, False

        if isinstance(past_key_values, Cache):
            return past_key_values, False

        return self._legacy_to_cache(past_key_values), True

    def _format_output_cache(
        self,
        cache: Optional[Cache],
        *,
        use_cache: bool,
        return_legacy_cache: bool,
    ):
        """
        Format cache output to match the caller's expected cache representation.

        Args:
            cache: Cache object produced during forward.
            use_cache: Whether cache output is enabled.
            return_legacy_cache: Whether output should be converted to legacy form.

        Returns:
            Cache output in HF or legacy form, or None when caching is disabled.
        """
        if not use_cache or cache is None:
            return None

        if return_legacy_cache:
            return self._cache_to_legacy(cache)

        return cache

    def _slice_causal(
        self,
        *,
        q_len: int,
        k_len: int,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        assert isinstance(self.causal_mask_template, torch.Tensor)
        return self.causal_mask_template[..., past_len : past_len + q_len, :k_len].to(
            device
        )

    def get_attention_mask_for(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        *,
        past_len: int,
    ) -> torch.Tensor:
        q_len = hidden_states.size(1)
        k_len = past_len + q_len
        device = hidden_states.device

        if attention_mask is None:
            causal_mask = self._slice_causal(
                q_len=q_len,
                k_len=k_len,
                past_len=past_len,
                device=device,
            )
            return self._fq(causal_mask.squeeze(0), self.obs_causal_mask)

        if attention_mask.dtype in (torch.bool, torch.int64):
            if attention_mask.dtype == torch.int64:
                attention_mask = attention_mask != 0

            causal_mask = self._slice_causal(
                q_len=q_len,
                k_len=k_len,
                past_len=past_len,
                device=device,
            )

            additive = torch.zeros_like(attention_mask, dtype=torch.float32)
            additive = additive.masked_fill(~attention_mask, float("-120"))
            mask = torch.clamp(causal_mask + additive, min=-120.0)
            return self._fq(mask.squeeze(0), self.obs_causal_mask)

        return self._fq(attention_mask, self.obs_causal_mask)

    def get_position_embeddings_for(
        self,
        hidden_states: torch.Tensor,
        *,
        start: int,
    ):
        end = start + hidden_states.size(1)
        cos = self.rope_cos_template[:, start:end, :].to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        sin = self.rope_sin_template[:, start:end, :].to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        return (
            self._fq(cos, self.obs_cos),
            self._fq(sin, self.obs_sin),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache | LegacyCache] = None,
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

        cache, return_legacy_cache = self._normalize_past_key_values(
            past_key_values,
            use_cache=bool(use_cache),
        )

        past_seen_tokens = self._cache_seq_length(cache)

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # Apply the SpinQuant rotation only when the source model provides it.
        if self.rotate_embedding is not None:
            hidden_states = self.rotate_embedding(hidden_states)

        # create position_embeddings and causal_mask to be shared across all the decoder layers
        model_attention_mask = self.get_attention_mask_for(
            hidden_states,
            attention_mask,
            past_len=past_seen_tokens,
        )
        position_embeddings = self.get_position_embeddings_for(
            hidden_states,
            start=past_seen_tokens,
        )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # type: ignore[operator]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=model_attention_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)  # type: ignore[operator]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)  # type: ignore[operator]

        formatted_cache = self._format_output_cache(
            cache,
            use_cache=bool(use_cache),
            return_legacy_cache=return_legacy_cache,
        )

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=formatted_cache,
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

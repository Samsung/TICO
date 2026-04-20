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

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaAttentionDecodeExportAdapter,
    LlamaAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


CacheOutputMode = Literal["present", "delta"]
LayerKV = Tuple[torch.Tensor, torch.Tensor]


@try_register(
    "transformers.models.llama.modeling_llama.LlamaAttention",
    "transformers.models.llama.modeling_llama.LlamaSdpaAttention",
)
class QuantLlamaAttention(QuantModuleBase):
    """
    Unified quantized Llama attention wrapper.

    A single HF-compatible forward is used for both runtime modes:
      - prefill: `past_key_value is None`
      - decode : `past_key_value is not None`

    Export specialization is provided by thin adapter modules.

    Behavior
    --------
    - If `past_key_value` is None, this behaves like a regular prefill step.
    - If `past_key_value` is not None, current K/V are concatenated to the past.
    - If `use_cache=True`, the returned cache format is controlled by
      `cache_output_mode`:
        - "present": return the full present cache
        - "delta": return only the newly produced K/V
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        cfg = fp_attn.config
        self.config = cfg
        self.layer_idx = layer_idx

        assert hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads")
        assert hasattr(cfg, "num_key_value_heads") and hasattr(
            cfg, "max_position_embeddings"
        )
        assert isinstance(cfg.hidden_size, int)
        assert isinstance(cfg.num_attention_heads, int)
        assert isinstance(cfg.num_key_value_heads, int)
        assert isinstance(cfg.max_position_embeddings, int)

        self.head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.kv_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        self.max_seq = cfg.max_position_embeddings

        q_cfg = qcfg.child("q_proj") if qcfg else None
        k_cfg = qcfg.child("k_proj") if qcfg else None
        v_cfg = qcfg.child("v_proj") if qcfg else None
        o_cfg = qcfg.child("o_proj") if qcfg else None

        assert hasattr(fp_attn, "q_proj") and isinstance(
            fp_attn.q_proj, torch.nn.Module
        )
        assert hasattr(fp_attn, "k_proj") and isinstance(
            fp_attn.k_proj, torch.nn.Module
        )
        assert hasattr(fp_attn, "v_proj") and isinstance(
            fp_attn.v_proj, torch.nn.Module
        )
        assert hasattr(fp_attn, "o_proj") and isinstance(
            fp_attn.o_proj, torch.nn.Module
        )
        self.q_proj = PTQWrapper(
            fp_attn.q_proj, qcfg=q_cfg, fp_name=f"{fp_name}.q_proj" if fp_name else None
        )
        self.k_proj = PTQWrapper(
            fp_attn.k_proj, qcfg=k_cfg, fp_name=f"{fp_name}.k_proj" if fp_name else None
        )
        self.v_proj = PTQWrapper(
            fp_attn.v_proj, qcfg=v_cfg, fp_name=f"{fp_name}.v_proj" if fp_name else None
        )
        self.o_proj = PTQWrapper(
            fp_attn.o_proj, qcfg=o_cfg, fp_name=f"{fp_name}.o_proj" if fp_name else None
        )

        # Constant scale (1/√d)
        scale_t = torch.tensor(
            float(getattr(fp_attn, "scaling", self.head_dim**-0.5))
        )

        # merge scale_t to k_proj, (otherwise merge it to q_proj)
        with torch.no_grad():
            lin = self.k_proj.wrapped.module
            lin.weight.mul_(scale_t)
            if lin.bias is not None:
                lin.bias.mul_(scale_t)

        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        # RoPE tables
        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        # rotate_half sub-steps (q)
        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_cat = mk("q_cat")

        # rotate_half sub-steps (k)
        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_cat = mk("k_cat")

        # RoPE combine
        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        # Masking & attention math
        self.obs_attn_mask = mk("attn_mask")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")
        self.obs_attn_weights = mk("attn_weights")
        self.obs_attn_out_h = mk("attn_out_h")

        # kv cache
        self.obs_past_key = mk("past_key")
        self.obs_past_value = mk("past_value")

        # New kv delta``
        self.obs_new_k = mk("new_k")  # (B, n_kv, 1, H)
        self.obs_new_v = mk("new_v")  # (B, n_kv, 1, H)

        # Total KV after concat (used for matmul/attn)
        self.obs_present_key = mk("present_key")  # (B, max_seq, H)
        self.obs_present_value = mk("present_value")  # (B, max_seq, H)

        # Static causal mask template
        mask = torch.full((1, 1, self.max_seq, self.max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

    def _rot(self, t: torch.Tensor, o_x1, o_x2, o_cat):
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)
        return self._fq(torch.cat((x2, x1), dim=-1), o_cat)

    def _apply_rope(
        self,
        t: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        obs_x1,
        obs_x2,
        obs_cat,
        obs_cos,
        obs_sin,
        obs_rot,
    ):
        t_half = self._rot(t, obs_x1, obs_x2, obs_cat)
        t_cos = self._fq(t * cos, obs_cos)
        t_sin = self._fq(t_half * sin, obs_sin)
        return self._fq(t_cos + t_sin, obs_rot)

    def _extract_layer_kv_from_cache(self, cache: Cache) -> Optional[LayerKV]:
        """
        Extract per-layer KV tensors from an HF Cache object across multiple
        transformers cache layouts.

        Returns:
            A tuple `(past_k, past_v)` with shape `(B, n_kv, K, H)` if available,
            otherwise None.
        """
        if self.layer_idx is None:
            raise RuntimeError(
                "layer_idx must be set to extract per-layer KV from an HF Cache."
            )

        layer_idx = self.layer_idx

        # Empty cache should behave like no past KV.
        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                if int(get_seq_length()) == 0:
                    return None
            except TypeError:
                try:
                    if int(get_seq_length(layer_idx)) == 0:
                        return None
                except Exception:
                    pass
            except Exception:
                pass

        # 1) key_cache / value_cache layout
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            if layer_idx >= len(key_cache) or layer_idx >= len(value_cache):
                return None
            past_k = key_cache[layer_idx]
            past_v = value_cache[layer_idx]
            if past_k is None or past_v is None:
                return None
            return past_k, past_v

        # 2) layers[layer_idx] layout
        layers = getattr(cache, "layers", None)
        if layers is not None:
            if layer_idx >= len(layers):
                return None

            layer_cache = layers[layer_idx]
            if layer_cache is None:
                return None

            if isinstance(layer_cache, tuple):
                if len(layer_cache) >= 2:
                    past_k, past_v = layer_cache[0], layer_cache[1]
                    if past_k is not None and past_v is not None:
                        return past_k, past_v

            for k_name, v_name in (
                ("keys", "values"),
                ("key", "value"),
                ("k", "v"),
            ):
                past_k = getattr(layer_cache, k_name, None)
                past_v = getattr(layer_cache, v_name, None)
                if past_k is not None and past_v is not None:
                    return past_k, past_v

        # 3) tuple-like indexing
        try:
            layer_cache = cache[layer_idx]  # type: ignore[index]
        except Exception:
            layer_cache = None

        if layer_cache is not None:
            if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                past_k, past_v = layer_cache[0], layer_cache[1]
                if past_k is not None and past_v is not None:
                    return past_k, past_v

        # 4) fallback: convert whole cache to legacy cache if supported
        to_legacy_cache = getattr(cache, "to_legacy_cache", None)
        if callable(to_legacy_cache):
            try:
                legacy_cache = to_legacy_cache()
                if legacy_cache is None:
                    return None
                if layer_idx >= len(legacy_cache):
                    return None

                layer_cache = legacy_cache[layer_idx]
                if layer_cache is None:
                    return None

                if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                    past_k, past_v = layer_cache[0], layer_cache[1]
                    if past_k is None or past_v is None:
                        return None
                    return past_k, past_v
            except Exception:
                pass

        raise RuntimeError(
            "Unsupported Cache layout. "
            f"type={type(cache)}, layer_idx={layer_idx}, "
            f"has_key_cache={hasattr(cache, 'key_cache')}, "
            f"has_value_cache={hasattr(cache, 'value_cache')}, "
            f"has_layers={hasattr(cache, 'layers')}, "
            f"has_to_legacy_cache={hasattr(cache, 'to_legacy_cache')}"
        )

    def _normalize_past_key_value(
        self,
        past_key_value: Optional[Cache | LayerKV],
    ) -> Optional[LayerKV]:
        """
        Convert an HF cache object into a per-layer legacy KV tuple.

        Args:
            past_key_value: Input cache in HF `Cache` form or legacy tuple form.

        Returns:
            A per-layer tuple `(past_k, past_v)` if cache exists, otherwise None.

        Raises:
            RuntimeError: If a `Cache` object is provided but this module cannot
                extract the current layer's KV tensors.
        """
        if past_key_value is None:
            return None

        if isinstance(past_key_value, tuple):
            past_k, past_v = past_key_value
            if past_k is None or past_v is None:
                return None
            return past_key_value

        return self._extract_layer_kv_from_cache(past_key_value)

    def _get_past_len(
        self,
        past_key_value: Optional[LayerKV],
    ) -> int:
        """
        Return the cached sequence length for the current layer.

        Args:
            past_key_value: Per-layer legacy KV tuple or None.

        Returns:
            Cached key/value length for this layer.
        """
        if past_key_value is None:
            return 0

        past_k, past_v = past_key_value
        if past_k is None or past_v is None:
            return 0

        return int(past_k.shape[2])

    def _build_attention_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[LayerKV],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build an additive attention mask for per-head logits.

        Supported cases:
        - None: build a causal mask slice.
        - bool/int mask: convert to additive mask using 0 / -120.
        - additive mask: use as-is.

        Args:
            hidden_states: Current hidden states with shape `(B, S, D)`.
            attention_mask: Optional attention mask from the caller.
            past_key_value: Per-layer cached KV tuple or None.
            device: Device where the mask should live.

        Returns:
            Additive attention mask with shape broadcastable to `(B, S, K)`.
        """
        q_len = hidden_states.size(1)
        past_len = self._get_past_len(past_key_value)
        k_len = past_len + q_len

        if attention_mask is None:
            assert isinstance(self.causal_mask_template, torch.Tensor)
            mask = self.causal_mask_template[
                ..., past_len : past_len + q_len, :k_len
            ].to(device)
            mask = mask.squeeze(0)
            return self._fq(mask, self.obs_attn_mask)

        if attention_mask.dtype in (torch.bool, torch.int64):
            if attention_mask.dtype == torch.int64:
                attention_mask = attention_mask != 0

            assert isinstance(self.causal_mask_template, torch.Tensor)
            causal_mask = self.causal_mask_template[
                ..., past_len : past_len + q_len, :k_len
            ].to(device)

            additive = torch.zeros_like(attention_mask, dtype=torch.float32)
            additive = additive.masked_fill(~attention_mask, float("-120"))

            # Combine causal mask and padding mask.
            mask = torch.clamp(causal_mask + additive, min=-120.0)
            return self._fq(mask.squeeze(0), self.obs_attn_mask)

        return self._fq(attention_mask, self.obs_attn_mask)

    def _get_past_kv_head(
        self,
        *,
        past_key_value: Optional[LayerKV],
        kv_idx: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract one KV head slice from the normalized past cache.

        Args:
            past_key_value: Per-layer legacy KV tuple or None.
            kv_idx: KV-head index to read.

        Returns:
            A tuple `(past_k_i, past_v_i)` with shape `(B, Kpast, H)` for each
            tensor, or `(None, None)` if no past cache exists.
        """
        if past_key_value is None:
            return None, None

        past_k, past_v = past_key_value
        if past_k is None or past_v is None:
            return None, None

        past_k_i = self._fq(past_k[:, kv_idx, :, :], self.obs_past_key)
        past_v_i = self._fq(past_v[:, kv_idx, :, :], self.obs_past_value)
        return past_k_i, past_v_i

    def _build_present_kv_head(
        self,
        *,
        past_k_i: Optional[torch.Tensor],
        past_v_i: Optional[torch.Tensor],
        new_k_i: torch.Tensor,
        new_v_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the full KV tensors used by attention for one KV head.

        Args:
            past_k_i: Past key states with shape `(B, Kpast, H)` or None.
            past_v_i: Past value states with shape `(B, Kpast, H)` or None.
            new_k_i: New key states with shape `(B, S, H)`.
            new_v_i: New value states with shape `(B, S, H)`.

        Returns:
            A tuple `(present_k_i, present_v_i)` with shape `(B, K, H)`.
        """
        if past_k_i is None:
            present_k_i = self._fq(new_k_i, self.obs_present_key)
            present_v_i = self._fq(new_v_i, self.obs_present_value)
            return present_k_i, present_v_i

        present_k_i = self._fq(
            torch.cat([past_k_i, new_k_i], dim=1),
            self.obs_present_key,
        )
        present_v_i = self._fq(
            torch.cat([past_v_i, new_v_i], dim=1),
            self.obs_present_value,
        )
        return present_k_i, present_v_i

    def _finalize_cache_output(
        self,
        *,
        past_key_value_in: Optional[Cache | LayerKV],
        new_k_parts: list[torch.Tensor],
        new_v_parts: list[torch.Tensor],
        present_k_parts: list[torch.Tensor],
        present_v_parts: list[torch.Tensor],
        cache_output_mode: CacheOutputMode,
    ) -> Cache | LayerKV:
        """
        Finalize the cache object returned by forward.

        Args:
            past_key_value_in: Original input cache in HF `Cache` or legacy form.
            new_k_parts: Per-head new key tensors of shape `(B, S, H)`.
            new_v_parts: Per-head new value tensors of shape `(B, S, H)`.
            present_k_parts: Per-head full key tensors of shape `(B, K, H)`.
            present_v_parts: Per-head full value tensors of shape `(B, K, H)`.
            cache_output_mode: Cache return policy.

        Returns:
            Cache object to return from forward:
            - delta `(new_k, new_v)` if `cache_output_mode == "delta"`
            - updated HF `Cache` if the input cache was an HF cache
            - full present legacy tuple otherwise
        """
        if cache_output_mode not in ("present", "delta"):
            raise ValueError(f"Unsupported cache_output_mode: {cache_output_mode!r}")

        new_k = self._fq(torch.stack(new_k_parts, dim=1), self.obs_new_k)
        new_v = self._fq(torch.stack(new_v_parts, dim=1), self.obs_new_v)

        if cache_output_mode == "delta":
            return new_k, new_v

        if isinstance(past_key_value_in, Cache):
            if self.layer_idx is None:
                raise RuntimeError(
                    "layer_idx must be set to update an HF Cache object."
                )
            past_key_value_in.update(new_k, new_v, self.layer_idx, cache_kwargs=None)
            return past_key_value_in

        present_k = self._fq(torch.stack(present_k_parts, dim=1), self.obs_present_key)
        present_v = self._fq(
            torch.stack(present_v_parts, dim=1), self.obs_present_value
        )
        return present_k, present_v

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache | LayerKV] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        cache_output_mode: CacheOutputMode = "present",
        **kwargs,
    ):
        """
        Run quantized Llama attention.

        Args:
            hidden_states: Input hidden states with shape `(B, S, D)`.
            position_embeddings: Rotary cosine and sine tensors.
            attention_mask: Optional additive or boolean attention mask.
            past_key_value: Optional cache in HF `Cache` or legacy tuple form.
            use_cache: Whether to return cache output.
            cache_position: Unused compatibility placeholder.
            cache_output_mode: Cache return policy.
                - "present": return the full present cache.
                - "delta": return only the newly produced K/V.
            **kwargs: Extra compatibility arguments ignored by this wrapper.

        Returns:
            A tuple of:
                - attention output
                - attention weights
                - optional cache output when `use_cache=True`
        """
        del cache_position, kwargs

        past_key_value_in = past_key_value
        past_key_value = self._normalize_past_key_value(past_key_value)

        hidden = self._fq(hidden_states, self.obs_hidden)
        B, S, _ = hidden.shape
        H = self.head_dim

        q = self.q_proj(hidden).view(B, S, self.num_heads, H)
        k = self.k_proj(hidden).view(B, S, self.num_kv_heads, H)
        v = self.v_proj(hidden).view(B, S, self.num_kv_heads, H)

        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        attn_mask = self._build_attention_mask(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            device=hidden.device,
        )

        attn_weights_parts: list[torch.Tensor] = []
        attn_out_parts: list[torch.Tensor] = []

        new_k_parts: list[torch.Tensor] = []
        new_v_parts: list[torch.Tensor] = []
        present_k_parts: list[torch.Tensor] = []
        present_v_parts: list[torch.Tensor] = []

        for kv_i in range(self.num_kv_heads):
            new_k_i = k[:, :, kv_i, :]
            new_v_i = v[:, :, kv_i, :]

            new_k_i = self._apply_rope(
                new_k_i,
                cos,
                sin,
                self.obs_k_x1,
                self.obs_k_x2,
                self.obs_k_cat,
                self.obs_k_cos,
                self.obs_k_sin,
                self.obs_k_rot,
            )

            past_k_i, past_v_i = self._get_past_kv_head(
                past_key_value=past_key_value,
                kv_idx=kv_i,
            )

            present_k_i, present_v_i = self._build_present_kv_head(
                past_k_i=past_k_i,
                past_v_i=past_v_i,
                new_k_i=new_k_i,
                new_v_i=new_v_i,
            )

            new_k_parts.append(new_k_i)
            new_v_parts.append(new_v_i)

            if cache_output_mode == "present" and not isinstance(
                past_key_value_in, Cache
            ):
                present_k_parts.append(present_k_i)
                present_v_parts.append(present_v_i)

            for rep_i in range(self.kv_rep):
                q_idx = kv_i * self.kv_rep + rep_i
                q_i = q[:, :, q_idx, :]

                q_i = self._apply_rope(
                    q_i,
                    cos,
                    sin,
                    self.obs_q_x1,
                    self.obs_q_x2,
                    self.obs_q_cat,
                    self.obs_q_cos,
                    self.obs_q_sin,
                    self.obs_q_rot,
                )

                logits_i = self._fq(
                    q_i @ present_k_i.transpose(-2, -1),
                    self.obs_logits,
                )

                assert attn_mask.shape[-2:] == logits_i.shape[-2:], (
                    attn_mask.shape,
                    logits_i.shape,
                )

                logits_i = self._fq(logits_i + attn_mask, self.obs_mask_add)

                attn_i = torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(
                    q_i.dtype
                )
                attn_i = self._fq(attn_i, self.obs_softmax)

                out_i = self._fq(attn_i @ present_v_i, self.obs_attn_out)

                attn_weights_parts.append(attn_i)
                attn_out_parts.append(out_i)

        attn_weights = self._fq(
            torch.stack(attn_weights_parts, dim=1),
            self.obs_attn_weights,
        )
        attn_out_h = self._fq(
            torch.stack(attn_out_parts, dim=1),
            self.obs_attn_out_h,
        )

        attn_out = attn_out_h.transpose(1, 2).reshape(B, S, -1)
        out = self.o_proj(attn_out)

        outputs = (out, attn_weights)

        if use_cache:
            cache_out = self._finalize_cache_output(
                past_key_value_in=past_key_value_in,
                new_k_parts=new_k_parts,
                new_v_parts=new_v_parts,
                present_k_parts=present_k_parts,
                present_v_parts=present_v_parts,
                cache_output_mode=cache_output_mode,
            )
            outputs += (cache_out,)  # type: ignore[assignment]

        return outputs

    def _all_observers(self):
        # local first
        yield from (
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_cat,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
            self.obs_attn_mask,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
            self.obs_past_key,
            self.obs_past_value,
            self.obs_new_k,
            self.obs_new_v,
            self.obs_present_key,
            self.obs_present_value,
        )
        # recurse into children that are QuantModuleBase
        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            yield from m._all_observers()

    def as_export_module(
        self, mode: ExportMode = "prefill", *, return_kv: bool = True
    ) -> nn.Module:
        if mode == "prefill":
            return LlamaAttentionPrefillExportAdapter(self, return_kv=return_kv)
        if mode == "decode":
            return LlamaAttentionDecodeExportAdapter(self, return_kv=return_kv)
        raise ValueError(f"Unsupported export mode: {mode!r}")

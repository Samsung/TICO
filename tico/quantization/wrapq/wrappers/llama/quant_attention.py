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

import copy
from typing import cast, Literal, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaAttention


from tico.quantization.config.llama_attention import (
    get_llama_attention_options,
    is_npu_export_attention_options,
)
from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.linear_folding import (
    _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR,
)
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaAttentionDecodeExportAdapter,
    LlamaAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.llama.position import (
    build_causal_mask,
    combine_binary_attention_mask,
    is_binary_attention_mask,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

CacheOutputMode = Literal["present", "delta"]
LayerKV = Tuple[torch.Tensor, torch.Tensor]


def _clone_projection_with_scale(module: nn.Module, scale: torch.Tensor) -> nn.Module:
    """
    Return a deep-copied projection with its affine parameters scaled.

    The attention scale is scalar. Record the weight multiplier so GPTQ qparams
    computed before PTQ prepare can be reused in the folded weight domain.
    Only existing `weight` and `bias` parameters are modified.
    """
    scaled = copy.deepcopy(module)
    with torch.no_grad():
        weight = getattr(scaled, "weight", None)
        if weight is not None:
            weight_scale = scale.to(device=weight.device, dtype=weight.dtype)
            weight.mul_(weight_scale)

            # Compose earlier folds without losing an unsupported-fold marker.
            multiplier = getattr(
                scaled, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR, 1.0
            )
            if multiplier is not None:
                multiplier = multiplier * float(weight_scale.item())
            setattr(scaled, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR, multiplier)

        bias = getattr(scaled, "bias", None)
        if bias is not None:
            bias.mul_(scale.to(device=bias.device, dtype=bias.dtype))
    return scaled


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

    Export specialization is provided by thin adapter modules. Implementation
    details such as projection scale fusion, RoPE sign convention, and attention
    layout are controlled by `PTQConfig.model_args`.

    Behavior
    --------
    - If `past_key_value` is None, this behaves like a regular prefill step.
    - If `past_key_value` is not None, current K/V are concatenated to the past.
    - If `use_cache=True`, the returned cache format is controlled by
      `cache_output_mode`:
        - "present": return the full cache in the persistent `new_*` grid
        - "delta": return only the new K/V in the persistent `new_*` grid
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        layer_idx: Optional[int] = None,
    ):
        fp_attn = cast("LlamaAttention", fp_attn)
        super().__init__(qcfg, fp_name=fp_name)

        self.attn_options = get_llama_attention_options(self.qcfg)

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

        # Constant scale (1/√d)
        self.attn_scale = torch.tensor(
            float(getattr(fp_attn, "scaling", self.head_dim**-0.5))
        )

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

        q_proj_fp: nn.Module = fp_attn.q_proj
        k_proj_fp: nn.Module = fp_attn.k_proj
        if self.attn_options.scale_fusion == "q_proj":
            q_proj_fp = _clone_projection_with_scale(fp_attn.q_proj, self.attn_scale)
        elif self.attn_options.scale_fusion == "k_proj":
            k_proj_fp = _clone_projection_with_scale(fp_attn.k_proj, self.attn_scale)
        elif self.attn_options.scale_fusion != "none":
            raise RuntimeError(
                f"Invalid scale fusion option: {self.attn_options.scale_fusion!r}"
            )

        self.q_proj = PTQWrapper(
            q_proj_fp, qcfg=q_cfg, fp_name=join_name(fp_name, "q_proj")
        )
        self.k_proj = PTQWrapper(
            k_proj_fp,
            qcfg=k_cfg,
            fp_name=join_name(fp_name, "k_proj"),
        )
        self.v_proj = PTQWrapper(
            fp_attn.v_proj, qcfg=v_cfg, fp_name=join_name(fp_name, "v_proj")
        )
        self.o_proj = PTQWrapper(
            fp_attn.o_proj, qcfg=o_cfg, fp_name=join_name(fp_name, "o_proj")
        )

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
        self.obs_scale = mk("scale")
        self.obs_logits_raw = mk("logits_raw")

        # Persistent cache storage domain.
        self.obs_new_k = mk("new_k")  # (B, n_kv, S, H)
        self.obs_new_v = mk("new_v")  # (B, n_kv, S, H)

        # Attention compute domain used by both Concat operands and its output.
        self.obs_present_key = mk("present_key")  # (B, max_seq, H)
        self.obs_present_value = mk("present_value")  # (B, max_seq, H)

    _OBSOLETE_CACHE_CHECKPOINT = (
        "Llama checkpoint contains obsolete past-cache observer state; "
        "reprepare the float model and recalibrate with cache generation "
        "enabled (calibration.decode_steps=0 is supported)."
    )

    def __setstate__(self, state):
        # Recipe checkpoints pickle full modules, bypassing state-dict loading.
        if any(
            name in state.get("_modules", {})
            for name in ("obs_past_key", "obs_past_value")
        ):
            raise RuntimeError(self._OBSOLETE_CACHE_CHECKPOINT)
        super().__setstate__(state)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Reject even non-strict loads: obsolete read-grid statistics cannot
        # establish the shared prefill/decode storage ABI.
        if any(
            key.startswith((prefix + "obs_past_key.", prefix + "obs_past_value."))
            for key in state_dict
        ):
            error_msgs.append(self._OBSOLETE_CACHE_CHECKPOINT)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _rot(self, t: torch.Tensor, o_x1, o_x2, o_cat):
        """
        Apply the profile-selected rotate-half primitive.

        `rope='hf'` computes `(-x2, x1)`. `rope='pre_negated_sin'` assumes
        the first half of the sine table has already been negated and therefore
        computes `(x2, x1)`.
        """
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)

        if self.attn_options.rope == "hf":
            first = -x2
        elif self.attn_options.rope == "pre_negated_sin":
            first = x2
        else:
            raise RuntimeError(f"Invalid RoPE option: {self.attn_options.rope!r}")

        return self._fq(torch.cat((first, x1), dim=-1), o_cat)

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
        """
        Apply rotary position embedding to a Q or K tensor.
        """
        t_half = self._rot(t, obs_x1, obs_x2, obs_cat)
        t_cos = self._fq(t * cos, obs_cos)
        t_sin = self._fq(t_half * sin, obs_sin)
        return self._fq(t_cos + t_sin, obs_rot)

    def _apply_attention_scale_if_needed(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply runtime attention scaling when it was not fused into a projection.
        """
        if self.attn_options.scale_fusion == "none":
            logits = self._fq(logits, self.obs_logits_raw)
            scale = self._fq(self.attn_scale, self.obs_scale)
            return logits * scale
        return logits

    @staticmethod
    def _expand_rope_for_batched(t: torch.Tensor) -> torch.Tensor:
        """
        Convert a RoPE table to a shape broadcastable to `(B, heads, S, H)`.
        """
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() == 3:
            t = t.unsqueeze(1)
        return t

    @staticmethod
    def _expand_attention_mask_for_batched(t: torch.Tensor) -> torch.Tensor:
        """
        Convert an attention mask to a shape broadcastable to batched logits.
        """
        if t.dim() == 3:
            t = t.unsqueeze(1)
        return t

    def _apply_storage_grid(self, key_value: LayerKV) -> LayerKV:
        """Apply the persistent cache grid without collecting duplicate data."""
        if self._mode is not Mode.QUANT:
            return key_value
        return (
            self.obs_new_k.fake_quant(key_value[0]),
            self.obs_new_v.fake_quant(key_value[1]),
        )

    def _apply_compute_grid(self, key_value: LayerKV) -> LayerKV:
        """Apply the attention compute grid to a branch in quantized mode."""
        if self._mode is not Mode.QUANT:
            return key_value
        return (
            self.obs_present_key.fake_quant(key_value[0]),
            self.obs_present_value.fake_quant(key_value[1]),
        )

    def _get_layer_kv_from_cache(
        self,
        cache: Cache,
        *,
        k_obs=None,
        v_obs=None,
        write_back: bool = False,
    ) -> Optional[LayerKV]:
        """
        Extract per-layer KV tensors from an HF Cache object.

        If k_obs/v_obs are provided, the extracted tensors are fake-quantized.
        If write_back=True, the quantized tensors are written back to the cache when
        the cache layout is mutable.
        """
        if self.layer_idx is None:
            raise RuntimeError(
                "layer_idx must be set to extract per-layer KV from an HF Cache."
            )

        layer_idx = self.layer_idx

        # StaticCache exposes its complete allocation. Only occupied slots are
        # past tokens; padding must not enter attention or calibration statistics.
        seq_length = None
        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                seq_length = int(get_seq_length(layer_idx))
            except TypeError:
                seq_length = int(get_seq_length())
            if seq_length == 0:
                return None

        def maybe_quantize(k: torch.Tensor, v: torch.Tensor) -> LayerKV:
            if seq_length is not None and not write_back:
                k = k[..., :seq_length, :]
                v = v[..., :seq_length, :]
            if k_obs is not None:
                k = self._fq(k, k_obs)
            if v_obs is not None:
                v = self._fq(v, v_obs)
            return k, v

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

            past_k, past_v = maybe_quantize(past_k, past_v)

            if write_back:
                key_cache[layer_idx] = past_k
                value_cache[layer_idx] = past_v

            return past_k, past_v

        # 2) layers[layer_idx] layout
        layers = getattr(cache, "layers", None)
        if layers is not None:
            if layer_idx >= len(layers):
                return None

            layer_cache = layers[layer_idx]
            if layer_cache is None:
                return None

            if isinstance(layer_cache, list) and len(layer_cache) >= 2:
                past_k, past_v = layer_cache[0], layer_cache[1]
                if past_k is None or past_v is None:
                    return None

                past_k, past_v = maybe_quantize(past_k, past_v)

                if write_back:
                    layer_cache[0] = past_k
                    layer_cache[1] = past_v

                return past_k, past_v

            if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                past_k, past_v = layer_cache[0], layer_cache[1]
                if past_k is None or past_v is None:
                    return None

                past_k, past_v = maybe_quantize(past_k, past_v)

                # tuple itself is immutable, but some cache implementations may allow
                # replacing the whole layer entry.
                if write_back:
                    try:
                        layers[layer_idx] = (past_k, past_v)
                    except Exception:
                        pass

                return past_k, past_v

            for k_name, v_name in (
                ("keys", "values"),
                ("key", "value"),
                ("k", "v"),
            ):
                past_k = getattr(layer_cache, k_name, None)
                past_v = getattr(layer_cache, v_name, None)

                if past_k is not None and past_v is not None:
                    past_k, past_v = maybe_quantize(past_k, past_v)

                    if write_back:
                        setattr(layer_cache, k_name, past_k)
                        setattr(layer_cache, v_name, past_v)

                    return past_k, past_v

                # The layer object may exist but still be empty.
                if hasattr(layer_cache, k_name) and hasattr(layer_cache, v_name):
                    return None

        # 3) tuple-like indexing
        try:
            layer_cache = cache[layer_idx]  # type: ignore[index]
        except Exception:
            layer_cache = None

        if layer_cache is not None:
            if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                past_k, past_v = layer_cache[0], layer_cache[1]
                if past_k is None or past_v is None:
                    return None

                past_k, past_v = maybe_quantize(past_k, past_v)
                return past_k, past_v

        # 4) fallback: convert whole cache to legacy cache if supported
        to_legacy_cache = getattr(cache, "to_legacy_cache", None)
        if callable(to_legacy_cache):
            try:
                legacy_cache = to_legacy_cache()
                if legacy_cache is None or layer_idx >= len(legacy_cache):
                    return None

                layer_cache = legacy_cache[layer_idx]
                if layer_cache is None:
                    return None

                if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
                    past_k, past_v = layer_cache[0], layer_cache[1]
                    if past_k is None or past_v is None:
                        return None

                    return maybe_quantize(past_k, past_v)
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

        raw_key_value: Optional[LayerKV]
        if isinstance(past_key_value, tuple):
            raw_key_value = past_key_value
        else:
            raw_key_value = self._get_layer_kv_from_cache(past_key_value)

        if raw_key_value is None:
            return None
        past_k, past_v = raw_key_value
        if past_k is None or past_v is None:
            return None

        # The external cache ABI uses new_* storage qparams. CALIB preserves
        # the supplied values; QUANT enforces S here and converts S to C only
        # on the subsequent attention branch.
        return self._apply_storage_grid((past_k, past_v))

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
        - None: build a causal mask for the current query and key lengths.
        - bool/int mask: convert to additive mask using 0 / configured fill value.
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
            mask = build_causal_mask(
                q_len=q_len,
                k_len=k_len,
                past_len=past_len,
                device=device,
                fill_value=float(self.qcfg.attention_mask_fill_value),
            )
            return self._fq(mask, self.obs_attn_mask)

        if is_binary_attention_mask(attention_mask):
            causal_mask = build_causal_mask(
                q_len=q_len,
                k_len=k_len,
                past_len=past_len,
                device=device,
                fill_value=float(self.qcfg.attention_mask_fill_value),
            )
            mask = combine_binary_attention_mask(
                causal_mask,
                attention_mask,
                fill_value=float(self.qcfg.attention_mask_fill_value),
            )
            return self._fq(mask, self.obs_attn_mask)

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

        past_k_i = past_k[:, kv_idx, :, :]
        past_v_i = past_v[:, kv_idx, :, :]
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
        current_k_i, current_v_i = self._apply_compute_grid((new_k_i, new_v_i))

        if past_k_i is None:
            present_k_i = self._fq(current_k_i, self.obs_present_key)
            present_v_i = self._fq(current_v_i, self.obs_present_value)
            return present_k_i, present_v_i

        assert past_v_i is not None
        compute_past_k_i, compute_past_v_i = self._apply_compute_grid(
            (past_k_i, past_v_i)
        )
        present_k_i = self._fq(
            torch.cat([compute_past_k_i, current_k_i], dim=1),
            self.obs_present_key,
        )
        present_v_i = self._fq(
            torch.cat([compute_past_v_i, current_v_i], dim=1),
            self.obs_present_value,
        )
        return present_k_i, present_v_i

    def _build_storage_cache(
        self,
        *,
        past_key_value: Optional[LayerKV],
        new_key_value: LayerKV,
    ) -> LayerKV:
        """Build a full cache tuple in the persistent storage domain."""
        if past_key_value is None:
            return new_key_value

        full_key_value = (
            torch.cat((past_key_value[0], new_key_value[0]), dim=2),
            torch.cat((past_key_value[1], new_key_value[1]), dim=2),
        )
        return self._apply_storage_grid(full_key_value)

    def _finalize_cache_output(
        self,
        *,
        past_key_value_in: Optional[Cache | LayerKV],
        past_key_value: Optional[LayerKV],
        new_k_parts: list[torch.Tensor],
        new_v_parts: list[torch.Tensor],
        cache_output_mode: CacheOutputMode,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Cache | LayerKV:
        """Finalize an unrolled cache output in the persistent storage domain."""
        new_k = self._fq(torch.stack(new_k_parts, dim=1), self.obs_new_k)
        new_v = self._fq(torch.stack(new_v_parts, dim=1), self.obs_new_v)
        return self._finalize_cache_output_batched(
            past_key_value_in=past_key_value_in,
            past_key_value=past_key_value,
            new_k=new_k,
            new_v=new_v,
            cache_output_mode=cache_output_mode,
            cache_position=cache_position,
        )

    def _finalize_cache_output_batched(
        self,
        *,
        past_key_value_in: Optional[Cache | LayerKV],
        past_key_value: Optional[LayerKV],
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cache_output_mode: CacheOutputMode,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Cache | LayerKV:
        """Update HF storage with the S delta, independently of return policy."""
        if cache_output_mode not in ("present", "delta"):
            raise ValueError(f"Unsupported cache_output_mode: {cache_output_mode!r}")

        new_key_value = (new_k, new_v)
        if isinstance(past_key_value_in, Cache):
            if self.layer_idx is None:
                raise RuntimeError(
                    "layer_idx must be set to update an HF Cache object."
                )
            cache_kwargs = (
                {"cache_position": cache_position}
                if cache_position is not None
                else None
            )
            past_key_value_in.update(
                new_k, new_v, self.layer_idx, cache_kwargs=cache_kwargs
            )
            if cache_output_mode == "present":
                return past_key_value_in

        if cache_output_mode == "delta":
            return new_key_value
        return self._build_storage_cache(
            past_key_value=past_key_value,
            new_key_value=new_key_value,
        )

    def _forward_unrolled(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_value: Optional[LayerKV],
        past_key_value_in: Optional[Cache | LayerKV],
        use_cache: Optional[bool],
        cache_output_mode: CacheOutputMode,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """
        Run the NPU-export-friendly per-head unrolled attention path.
        """
        B, S, _, _ = q.shape

        attn_weights_parts: list[torch.Tensor] = []
        attn_out_parts: list[torch.Tensor] = []

        new_k_parts: list[torch.Tensor] = []
        new_v_parts: list[torch.Tensor] = []

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

                logits_i = q_i @ present_k_i.transpose(-2, -1)
                logits_i = self._apply_attention_scale_if_needed(logits_i)
                logits_i = self._fq(logits_i, self.obs_logits)

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
            torch.stack(attn_out_parts, dim=2),
            self.obs_attn_out_h,
        )

        attn_out = attn_out_h.reshape(B, S, -1)
        out = self.o_proj(attn_out)

        outputs = (out, attn_weights)

        if use_cache:
            cache_out = self._finalize_cache_output(
                past_key_value_in=past_key_value_in,
                past_key_value=past_key_value,
                new_k_parts=new_k_parts,
                new_v_parts=new_v_parts,
                cache_output_mode=cache_output_mode,
                cache_position=cache_position,
            )
            outputs += (cache_out,)  # type: ignore[assignment]

        return outputs

    def _forward_batched(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key_value: Optional[LayerKV],
        past_key_value_in: Optional[Cache | LayerKV],
        use_cache: Optional[bool],
        cache_output_mode: CacheOutputMode,
        cache_position: Optional[torch.Tensor] = None,
    ):
        """
        Run a HF-like batched attention path optimized for GPU evaluation.
        """
        B, S, _, _ = q.shape

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self._expand_rope_for_batched(cos)
        sin = self._expand_rope_for_batched(sin)

        q = self._apply_rope(
            q,
            cos,
            sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
        )
        k = self._apply_rope(
            k,
            cos,
            sin,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_cat,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
        )

        current_k, current_v = k, v
        new_k = self._fq(current_k, self.obs_new_k)
        new_v = self._fq(current_v, self.obs_new_v)

        if past_key_value is None:
            present_k = self._fq(current_k, self.obs_present_key)
            present_v = self._fq(current_v, self.obs_present_value)
        else:
            past_k, past_v = past_key_value
            compute_past_k, compute_past_v = self._apply_compute_grid((past_k, past_v))
            compute_current_k, compute_current_v = self._apply_compute_grid(
                (current_k, current_v)
            )
            present_k = self._fq(
                torch.cat([compute_past_k, compute_current_k], dim=2),
                self.obs_present_key,
            )
            present_v = self._fq(
                torch.cat([compute_past_v, compute_current_v], dim=2),
                self.obs_present_value,
            )

        if self.kv_rep != 1:
            present_k_for_attn = present_k.repeat_interleave(self.kv_rep, dim=1)
            present_v_for_attn = present_v.repeat_interleave(self.kv_rep, dim=1)
        else:
            present_k_for_attn = present_k
            present_v_for_attn = present_v

        logits = q @ present_k_for_attn.transpose(-2, -1)
        logits = self._apply_attention_scale_if_needed(logits)
        logits = self._fq(logits, self.obs_logits)

        attn_mask = self._expand_attention_mask_for_batched(attn_mask)
        assert attn_mask.shape[-2:] == logits.shape[-2:], (
            attn_mask.shape,
            logits.shape,
        )

        logits = self._fq(logits + attn_mask, self.obs_mask_add)

        attn_weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self._fq(attn_weights, self.obs_softmax)
        attn_weights = self._fq(attn_weights, self.obs_attn_weights)

        attn_out_h = self._fq(attn_weights @ present_v_for_attn, self.obs_attn_out)
        attn_out_h = self._fq(attn_out_h, self.obs_attn_out_h)

        attn_out = attn_out_h.transpose(1, 2).contiguous().reshape(B, S, -1)
        out = self.o_proj(attn_out)

        outputs = (out, attn_weights)

        if use_cache:
            cache_out = self._finalize_cache_output_batched(
                past_key_value_in=past_key_value_in,
                past_key_value=past_key_value,
                new_k=new_k,
                new_v=new_v,
                cache_output_mode=cache_output_mode,
                cache_position=cache_position,
            )
            outputs += (cache_out,)  # type: ignore[assignment]

        return outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache | LayerKV] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        cache_output_mode: CacheOutputMode = "present",
        **kwargs,
    ):
        """
        Run quantized Llama attention.

        Args:
            cache_output_mode: Cache return policy.
                - "present": return the full cache in the persistent storage grid.
                - "delta": return only the new K/V in the persistent storage grid.

        Returns:
            A tuple of:
                - attention output
                - attention weights
                - optional cache output when `use_cache=True`
        """
        del kwargs

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

        if self.attn_options.layout == "batched":
            return self._forward_batched(
                q=q,
                k=k,
                v=v,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
                past_key_value=past_key_value,
                past_key_value_in=past_key_value_in,
                use_cache=use_cache,
                cache_output_mode=cache_output_mode,
                cache_position=cache_position,
            )

        if self.attn_options.layout == "unrolled":
            return self._forward_unrolled(
                q=q,
                k=k,
                v=v,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
                past_key_value=past_key_value,
                past_key_value_in=past_key_value_in,
                use_cache=use_cache,
                cache_output_mode=cache_output_mode,
                cache_position=cache_position,
            )

        raise RuntimeError(f"Invalid attention layout: {self.attn_options.layout!r}")

    def _all_observers(self):
        """Yield active observers while retaining dormant module attributes."""
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
        )
        # Projection fusion removes runtime scaling and its quantization sites.
        if self.attn_options.scale_fusion == "none":
            yield from (self.obs_logits_raw, self.obs_scale)
        yield from (
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
            self.obs_new_k,
            self.obs_new_v,
            self.obs_present_key,
            self.obs_present_value,
        )

    def as_export_module(
        self,
        mode: ExportMode = "prefill",
        *,
        return_kv: bool = True,
        require_npu_profile: bool = False,
    ) -> nn.Module:
        """
        Return an export adapter for the requested attention mode.

        Parameters
        ----------
        mode : ExportMode
            Export mode, either ``"prefill"`` or ``"decode"``.
        return_kv : bool
            Whether the adapter should return the newly produced KV tensors.
        require_npu_profile : bool
            If True, reject configurations that do not match the NPU-export
            profile. This protects export flows from accidentally using the
            HF-like evaluation graph.
        """
        if require_npu_profile and not is_npu_export_attention_options(
            self.attn_options
        ):
            raise ValueError(
                "NPU export requires execution profile 'npu_export'. "
                "Set PTQConfig.model_args['profile'] "
                "to 'npu_export'."
            )

        if mode == "prefill":
            return LlamaAttentionPrefillExportAdapter(self, return_kv=return_kv)
        if mode == "decode":
            return LlamaAttentionDecodeExportAdapter(self, return_kv=return_kv)
        raise ValueError(f"Unsupported export mode: {mode!r}")

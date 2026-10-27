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
from typing import cast, Iterable, Literal, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention


from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.config.qwen3_vl_attention import (
    get_qwen3_vl_text_attention_options,
    is_npu_export_text_attention_options,
)
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLTextAttentionDecodeExportAdapter,
    Qwen3VLTextAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.registry import try_register

LayerKV = Tuple[torch.Tensor, torch.Tensor]
CacheOutputMode = Literal["present", "delta"]


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention",
)
class QuantQwen3VLTextAttention(QuantModuleBase):
    """
    Quantized Qwen3-VL text attention wrapper with selectable attention layout.

    The default `npu_export` profile uses a per-query-head unrolled graph for
    NPU export. Set `PTQConfig.model_args["profile"]` to
    "reference_eval" or pass `model_args["attention"]` overrides to run a
    Hugging Face-like batched attention graph for experiments.

    ``new_k/new_v`` own persistent KV storage; ``present_key/present_value``
    own attention computation. Current KV branches into both grids before
    storage quantization, and cached storage is converted only on the read.
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        fp_attn = cast("Qwen3VLTextAttention", fp_attn)
        super().__init__(qcfg, fp_name=fp_name)

        obsolete = {"past_key", "past_value"}.intersection(self.qcfg.overrides)
        if obsolete:
            raise ValueError(
                "Qwen text attention has obsolete past_key/past_value overrides: "
                f"{', '.join(sorted(obsolete))}. Configure storage with new_k/new_v "
                "and attention computation with present_key/present_value instead."
            )
        self.attn_options = get_qwen3_vl_text_attention_options(self.qcfg)

        cfg = fp_attn.config
        self.config = cfg
        self.layer_idx = getattr(fp_attn, "layer_idx", None)

        # Head shapes
        assert hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads")
        assert hasattr(cfg, "num_key_value_heads")
        self.head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        self.num_heads = cfg.num_attention_heads
        assert cfg.num_key_value_heads is not None
        self.num_kv_heads = cfg.num_key_value_heads
        self.kv_rep = self.num_heads // self.num_kv_heads

        # Constant scale (1/sqrt(d)).
        self.attn_scale = torch.tensor(
            float(getattr(fp_attn, "scaling", self.head_dim**-0.5))
        )

        # --- Wrap projection + norms via PTQWrapper --------------------------
        q_cfg = qcfg.child("q_proj") if qcfg else None
        k_cfg = qcfg.child("k_proj") if qcfg else None
        v_cfg = qcfg.child("v_proj") if qcfg else None
        o_cfg = qcfg.child("o_proj") if qcfg else None
        qn_cfg = qcfg.child("q_norm") if qcfg else None
        kn_cfg = qcfg.child("k_norm") if qcfg else None

        assert hasattr(fp_attn, "q_proj") and isinstance(fp_attn.q_proj, nn.Module)
        assert hasattr(fp_attn, "k_proj") and isinstance(fp_attn.k_proj, nn.Module)
        assert hasattr(fp_attn, "v_proj") and isinstance(fp_attn.v_proj, nn.Module)
        assert hasattr(fp_attn, "o_proj") and isinstance(fp_attn.o_proj, nn.Module)
        assert hasattr(fp_attn, "q_norm") and isinstance(fp_attn.q_norm, nn.Module)
        assert hasattr(fp_attn, "k_norm") and isinstance(fp_attn.k_norm, nn.Module)

        self.q_proj = PTQWrapper(
            fp_attn.q_proj, qcfg=q_cfg, fp_name=join_name(fp_name, "q_proj")
        )
        self.k_proj = PTQWrapper(
            fp_attn.k_proj, qcfg=k_cfg, fp_name=join_name(fp_name, "k_proj")
        )
        self.v_proj = PTQWrapper(
            fp_attn.v_proj, qcfg=v_cfg, fp_name=join_name(fp_name, "v_proj")
        )
        self.o_proj = PTQWrapper(
            fp_attn.o_proj, qcfg=o_cfg, fp_name=join_name(fp_name, "o_proj")
        )
        self.q_norm = PTQWrapper(
            fp_attn.q_norm, qcfg=qn_cfg, fp_name=join_name(fp_name, "q_norm")
        )
        self.k_norm = PTQWrapper(
            copy.deepcopy(fp_attn.k_norm),
            qcfg=kn_cfg,
            fp_name=join_name(fp_name, "k_norm"),
        )

        if self.attn_options.scale_fusion == "k_norm":
            # Merge the scale into k_norm, otherwise it must be applied to logits.
            with torch.no_grad():
                weight = getattr(self.k_norm.wrapped.module, "weight", None)
                if weight is None:
                    raise RuntimeError(
                        "Qwen3-VL k_norm scale fusion requires a weight parameter."
                    )
                weight.mul_(
                    self.attn_scale.to(device=weight.device, dtype=weight.dtype)
                )
        elif self.attn_options.scale_fusion != "none":
            raise RuntimeError(
                f"Invalid scale fusion option: {self.attn_options.scale_fusion!r}"
            )

        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        # RoPE tables
        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        # rotate_half sub-steps (q)
        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_neg = mk("q_neg")
        self.obs_q_cat = mk("q_cat")

        # rotate_half sub-steps (k)
        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_neg = mk("k_neg")
        self.obs_k_cat = mk("k_cat")

        # RoPE combine
        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        # Masking and attention math
        self.obs_causal_mask = mk("causal_mask")
        self.obs_logits_raw = mk("logits_raw")
        self.obs_scale = mk("scale")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")
        self.obs_attn_weights = mk("attn_weights")
        self.obs_attn_out_h = mk("attn_out_h")

        # Persistent storage (S) and independent attention computation (C).
        # Both prefill output and decode input/delta use the new_* storage ABI.
        self.obs_new_k = mk("new_k")
        self.obs_new_v = mk("new_v")
        self.obs_present_key = mk("present_key")
        self.obs_present_value = mk("present_value")
        self._validate_cache_observers()

    # Old stateless MX observers leave no tensor keys in state_dict.
    # Version the owning attention module so nested loads reject those too.
    _version = 2

    _OBSOLETE_CACHE_CHECKPOINT = (
        "Qwen checkpoint contains obsolete past-cache observer state; "
        "reprepare the float model and recalibrate "
        "(prefill-only calibration is supported)."
    )

    def __setstate__(self, state):
        # Recipe checkpoints pickle modules and bypass state-dict loading.
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
        # Reject non-strict loads too: obsolete P statistics cannot establish
        # the new shared prefill/decode storage ABI.
        if any(
            key.startswith((prefix + "obs_past_key.", prefix + "obs_past_value."))
            for key in state_dict
        ):
            error_msgs.append(self._OBSOLETE_CACHE_CHECKPOINT)
        elif local_metadata.get("version", 0) < self._version:
            error_msgs.append(
                "Qwen checkpoint has an obsolete or missing KV-cache version; "
                "preserve state_dict metadata for new checkpoints, or reprepare "
                "the float model and recalibrate (prefill-only calibration is supported)."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _validate_cache_observers(self) -> None:
        """Require grids that remain valid when the cache sequence grows."""
        for observer in (
            self.obs_new_k,
            self.obs_new_v,
            self.obs_present_key,
            self.obs_present_value,
        ):
            if (
                isinstance(observer, AffineObserverBase)
                and observer.fake_quant_enabled
                and observer.channel_axis is not None
                and observer.channel_axis % 4 == 2
            ):
                raise ValueError(
                    "Qwen KV-cache affine observers cannot use the sequence axis "
                    "(2 or -2) as channel_axis; use a per-tensor or token-local "
                    "new_*/present_* policy."
                )
        for storage, compute in (
            (self.obs_new_k, self.obs_present_key),
            (self.obs_new_v, self.obs_present_value),
        ):
            storage_mx = isinstance(storage, MXObserver) and storage.fake_quant_enabled
            compute_mx = isinstance(compute, MXObserver) and compute.fake_quant_enabled
            if not (storage_mx or compute_mx):
                continue
            if not (storage_mx and compute_mx):
                raise ValueError(
                    "Qwen KV-cache does not support mixed MX/affine or MX/bypassed "
                    "storage and compute grids. Configure matching MX policies for "
                    "new_* and present_*, or use affine cache observers."
                )
            assert isinstance(storage, MXObserver) and isinstance(compute, MXObserver)
            if (
                storage.axis % 4 == 2
                or compute.axis % 4 == 2
                or storage.axis % 4 != compute.axis % 4
                or any(
                    getattr(storage, key) != getattr(compute, key)
                    for key in ("elem_format", "shared_exp_method", "round")
                )
            ):
                raise ValueError(
                    "Qwen KV-cache requires matching token-local MX storage and "
                    "compute policies; new_* and present_* must agree and their "
                    "axis must not be the sequence axis (2 or -2)."
                )

    def _apply_storage_grid(self, key_value: LayerKV) -> LayerKV:
        """Enforce the cache ABI without observing historical values again."""
        if self._mode is not Mode.QUANT:
            return key_value
        return (
            self.obs_new_k.fake_quant(key_value[0]),
            self.obs_new_v.fake_quant(key_value[1]),
        )

    def _apply_compute_grid(self, key_value: LayerKV) -> LayerKV:
        """Convert one attention branch to C before concatenation."""
        if self._mode is not Mode.QUANT:
            return key_value
        return (
            self.obs_present_key.fake_quant(key_value[0]),
            self.obs_present_value.fake_quant(key_value[1]),
        )

    def _rot(self, t: torch.Tensor, o_x1, o_x2, o_neg, o_cat) -> torch.Tensor:
        """Return rotate_half(t) as [-x2, x1] along the last dimension."""
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)
        x2n = self._fq(-x2, o_neg)
        return self._fq(torch.cat((x2n, x1), dim=-1), o_cat)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to query and key states."""
        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)

        q_half = self._rot(
            q, self.obs_q_x1, self.obs_q_x2, self.obs_q_neg, self.obs_q_cat
        )
        q_cos = self._fq(q * cos_u, self.obs_q_cos)
        q_sin = self._fq(q_half * sin_u, self.obs_q_sin)
        q_rot = self._fq(q_cos + q_sin, self.obs_q_rot)

        k_half = self._rot(
            k, self.obs_k_x1, self.obs_k_x2, self.obs_k_neg, self.obs_k_cat
        )
        k_cos = self._fq(k * cos_u, self.obs_k_cos)
        k_sin = self._fq(k_half * sin_u, self.obs_k_sin)
        k_rot = self._fq(k_cos + k_sin, self.obs_k_rot)

        return q_rot, k_rot

    def _apply_attention_scale_if_needed(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply runtime attention scaling when it was not fused into k_norm."""
        if self.attn_options.scale_fusion == "none":
            logits = self._fq(logits, self.obs_logits_raw)
            scale = self.attn_scale.to(device=logits.device, dtype=logits.dtype)
            scale = self._fq(scale, self.obs_scale)
            return logits * scale

        if self.attn_options.scale_fusion == "k_norm":
            return logits

        raise RuntimeError(
            f"Invalid scale fusion option: {self.attn_options.scale_fusion!r}"
        )

    @staticmethod
    def _normalize_attention_mask_shape(
        mask: torch.Tensor,
        *,
        q_len: int,
        k_len: int,
    ) -> torch.Tensor:
        """
        Normalize an attention mask to a shape broadcastable to per-head logits.

        Supported input shapes:
          - (B, K)
          - (B, Q, K)
          - (B, 1, Q, K)

        Returns:
          - (B, 1, 1, K)
          - (B, 1, Q, K)
          - (B, 1, Q, K)

        Per-head masks with H != 1 are rejected because this decomposed
        attention path applies the same mask to every KV-head group.
        """
        if mask.dim() not in (2, 3, 4):
            raise RuntimeError(
                "Unsupported attention_mask rank for Qwen text attention: "
                f"rank={mask.dim()}, shape={tuple(mask.shape)}"
            )

        if mask.size(-1) != k_len:
            if mask.size(-1) > k_len:
                # HF masks may be preallocated or include a longer cache span.
                mask = mask[..., :k_len]
            else:
                raise RuntimeError(
                    "attention_mask key length is shorter than key states: "
                    f"mask_k={mask.size(-1)}, k_len={k_len}, "
                    f"shape={tuple(mask.shape)}"
                )

        if mask.dim() == 2:
            return mask[:, None, None, :]

        if mask.size(-2) not in (1, q_len):
            if mask.size(-2) > q_len:
                # In decode/cache mode, a full QxK mask may be passed while this
                # layer only processes the last q_len query positions.
                mask = mask[..., -q_len:, :]
            else:
                raise RuntimeError(
                    "attention_mask query length is incompatible with query states: "
                    f"mask_q={mask.size(-2)}, q_len={q_len}, "
                    f"shape={tuple(mask.shape)}"
                )

        if mask.dim() == 3:
            return mask[:, None, :, :]

        if mask.size(1) != 1:
            raise RuntimeError(
                "Per-head attention masks are not supported by this decomposed "
                "Qwen text attention wrapper. Expected mask shape (B, 1, Q, K), "
                f"got shape={tuple(mask.shape)}"
            )

        return mask

    def _build_attention_mask(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        k_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build an additive attention mask for logits shaped `(B, heads, Q, K)`.

        Cases:
          - None: use only the causal mask.
          - bool/int: interpret non-zero values as keep values and combine the
            resulting padding mask with the causal mask.
          - floating point: assume the caller already provided an additive mask.
        """
        # Full-model eager execution and staged export already supply additive
        # masks. Reuse them without allocating another Q-by-K mask per layer.
        if attention_mask is not None and torch.is_floating_point(attention_mask):
            attention_mask = self._normalize_attention_mask_shape(
                attention_mask.to(device),
                q_len=q_len,
                k_len=k_len,
            )
            return self._fq(attention_mask, self.obs_causal_mask)

        fill_val = float(self.qcfg.attention_mask_fill_value)
        # Queries follow the cached prefix. Construct only the actual query/key
        # span; the model's native context limit is not a mask allocation size.
        causal_mask = torch.full(
            (1, 1, q_len, k_len), fill_val, device=device, dtype=torch.float32
        ).triu_(diagonal=max(k_len - q_len, 0) + 1)

        if attention_mask is None:
            return self._fq(causal_mask, self.obs_causal_mask)

        attention_mask = attention_mask.to(device)

        if attention_mask.dtype in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            keep_mask = attention_mask
            if keep_mask.dtype != torch.bool:
                keep_mask = keep_mask != 0

            keep_mask = self._normalize_attention_mask_shape(
                keep_mask,
                q_len=q_len,
                k_len=k_len,
            )

            additive_mask = torch.zeros(
                keep_mask.shape,
                dtype=torch.float32,
                device=device,
            )
            additive_mask = additive_mask.masked_fill(~keep_mask, fill_val)

            # OR semantics for two additive masks:
            #   allowed + allowed -> 0
            #   masked by either side -> fill_val
            # Clamp prevents fill_val + fill_val from becoming twice as negative.
            mask = torch.clamp(causal_mask + additive_mask, min=fill_val)
            return self._fq(mask, self.obs_causal_mask)

        raise RuntimeError(
            "Unsupported attention_mask dtype for Qwen text attention: "
            f"dtype={attention_mask.dtype}, shape={tuple(attention_mask.shape)}"
        )

    @staticmethod
    def _is_static_kv_tuple(past_key_values) -> bool:
        """Return True when the input is a per-layer `(key, value)` cache tuple."""
        if not isinstance(past_key_values, (tuple, list)):
            return False
        if len(past_key_values) < 2:
            return False
        first, second = past_key_values[0], past_key_values[1]
        tensor_or_none = (torch.Tensor, type(None))
        return isinstance(first, tensor_or_none) and isinstance(second, tensor_or_none)

    def _normalize_static_past_key_values(
        self,
        past_key_values,
    ) -> Optional[LayerKV]:
        """Validate a static tuple and apply the persistent storage ABI."""
        if not self._is_static_kv_tuple(past_key_values):
            return None

        if len(past_key_values) != 2:
            raise ValueError("A static KV tuple must contain exactly key and value.")

        past_k, past_v = past_key_values[0], past_key_values[1]
        if past_k is None and past_v is None:
            return None
        if past_k is None or past_v is None:
            raise ValueError(
                "A static KV tuple must provide both key and value tensors."
            )

        self._validate_cache_shapes(past_k, past_v)
        if past_k.size(2) == 0:
            return None
        return self._apply_storage_grid((past_k, past_v))

    def require_calibrated_cache(self) -> None:
        """Require the frozen storage capability, retaining Qwen cache policy."""
        self._validate_cache_observers()
        self.require_capability("cache")

    def _validate_cache_shapes(
        self, past_k: torch.Tensor, past_v: torch.Tensor
    ) -> None:
        if (
            past_k.ndim != 4
            or past_v.shape != past_k.shape
            or past_k.size(1) != self.num_kv_heads
            or past_k.size(-1) != self.head_dim
        ):
            raise ValueError(
                "Qwen cache tensors must have matching (batch, KV heads, length, head_dim) "
                f"shapes; got {tuple(past_k.shape)}, {tuple(past_v.shape)}."
            )

    def _extract_layer_kv_from_cache(self, cache) -> Optional[LayerKV]:
        """
        Extract the current layer KV tensors from common HF cache layouts.

        This is a fallback path for cache implementations whose `update()`
        method returns the cache object itself instead of the updated tensors.
        """
        if hasattr(cache, "k") and hasattr(cache, "v"):
            k = getattr(cache, "k")
            v = getattr(cache, "v")
            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                return k, v

        layer_idx = self.layer_idx
        if layer_idx is None:
            return None

        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            if layer_idx < len(key_cache) and layer_idx < len(value_cache):
                k = key_cache[layer_idx]
                v = value_cache[layer_idx]
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    return k, v

        layers = getattr(cache, "layers", None)
        if layers is not None and layer_idx < len(layers):
            layer_cache = layers[layer_idx]
            if isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                k, v = layer_cache[0], layer_cache[1]
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    return k, v
            for k_name, v_name in (
                ("keys", "values"),
                ("key", "value"),
                ("k", "v"),
            ):
                k = getattr(layer_cache, k_name, None)
                v = getattr(layer_cache, v_name, None)
                if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                    return k, v

        return None

    def _normalize_cache_like(self, cache) -> Optional[LayerKV]:
        """Read supported append-only HF storage without mutating the cache."""
        sliding = getattr(cache, "is_sliding", False)
        has_sliding_layers = (
            any(sliding) if isinstance(sliding, (tuple, list)) else bool(sliding)
        )
        if getattr(cache, "is_compileable", False) or has_sliding_layers:
            raise RuntimeError(
                "Qwen eager quantization supports append-only HF caches; "
                "use a static KV tuple with an external mask for fixed buffers."
            )
        previous = self._extract_layer_kv_from_cache(cache)
        if previous is None:
            get_length = getattr(cache, "get_seq_length", None)
            if callable(get_length) and get_length(self.layer_idx) != 0:
                raise RuntimeError(
                    "Unsupported populated HF cache layout: cannot inspect "
                    f"layer {self.layer_idx}; use a supported dynamic cache or static KV tuple."
                )
        if not callable(getattr(cache, "update", None)):
            raise RuntimeError(
                "past_key_values must be a static KV tuple or expose update(). "
                f"Got type={type(cache)!r}."
            )
        if previous is None:
            return None
        self._validate_cache_shapes(*previous)
        if previous[0].size(2) == 0:
            return None
        return self._apply_storage_grid(previous)

    def _update_cache_like(
        self,
        cache,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        *,
        previous_length: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: Optional[torch.Tensor],
    ) -> LayerKV:
        """Append only S-grid deltas to an already validated HF cache."""
        update = getattr(cache, "update", None)
        if not callable(update):
            raise RuntimeError(
                "past_key_values must be a static KV tuple or expose update(). "
                f"Got type={type(cache)!r}."
            )

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        try:
            updated = update(new_k, new_v, self.layer_idx, cache_kwargs)
        except TypeError:
            try:
                updated = update(
                    new_k,
                    new_v,
                    self.layer_idx,
                    cache_kwargs=cache_kwargs,
                )
            except TypeError:
                updated = update(new_k, new_v)

        if isinstance(updated, (tuple, list)) and len(updated) >= 2:
            present_k, present_v = updated[0], updated[1]
            if isinstance(present_k, torch.Tensor) and isinstance(
                present_v, torch.Tensor
            ):
                return self._normalize_updated_cache(
                    present_k,
                    present_v,
                    new_k,
                    new_v,
                    previous_length=previous_length,
                )

        extracted = self._extract_layer_kv_from_cache(cache)
        if extracted is not None:
            return self._normalize_updated_cache(
                *extracted,
                new_k,
                new_v,
                previous_length=previous_length,
            )

        raise RuntimeError(
            "Cache update did not return key/value tensors and the updated cache "
            f"layout could not be inspected. type={type(cache)!r}"
        )

    def _normalize_updated_cache(
        self,
        stored_k: torch.Tensor,
        stored_v: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        *,
        previous_length: int,
    ) -> LayerKV:
        """Validate the updated S cache without replacing it with compute values."""
        self._validate_cache_shapes(stored_k, stored_v)
        if (
            stored_k.size(0) != new_k.size(0)
            or stored_k.size(2) != previous_length + new_k.size(2)
            or new_v.shape != new_k.shape
        ):
            raise RuntimeError(
                "Qwen eager quantization requires an append-only HF cache layout; "
                "use a static KV tuple for fixed-capacity or sliding buffers."
            )
        return stored_k, stored_v

    def _build_present_key_values(
        self,
        *,
        past: Optional[LayerKV],
        raw_k: torch.Tensor,
        raw_v: torch.Tensor,
    ) -> LayerKV:
        """Build Concat(C) from a raw-to-C current branch and an S-to-C read."""
        current_k, current_v = self._apply_compute_grid((raw_k, raw_v))
        if past is None:
            return self._fq(current_k, self.obs_present_key), self._fq(
                current_v, self.obs_present_value
            )
        compute_past_k, compute_past_v = self._apply_compute_grid(past)
        # CALIB observes the full computation once, without padding filtering.
        # QUANT gives each operand and the Concat output the same C grid.
        return self._fq(
            torch.cat((compute_past_k, current_k), dim=2), self.obs_present_key
        ), self._fq(
            torch.cat((compute_past_v, current_v), dim=2), self.obs_present_value
        )

    def _finalize_cache_output(
        self,
        *,
        past_key_values,
        past: Optional[LayerKV],
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_position: Optional[torch.Tensor],
        cache_output_mode: CacheOutputMode,
    ):
        """Return/update persistent S storage, independently of attention C."""
        if cache_output_mode not in ("present", "delta"):
            raise ValueError(f"Unsupported cache_output_mode: {cache_output_mode!r}")

        if past_key_values is not None and not self._is_static_kv_tuple(
            past_key_values
        ):
            self._update_cache_like(
                past_key_values,
                new_k,
                new_v,
                previous_length=past[0].size(2) if past is not None else 0,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
            )
            if cache_output_mode == "present":
                return past_key_values

        if cache_output_mode == "delta" or past is None:
            return new_k, new_v
        return self._apply_storage_grid(
            (
                torch.cat((past[0], new_k), dim=2),
                torch.cat((past[1], new_v), dim=2),
            )
        )

    def _forward_unrolled(
        self,
        *,
        q_rot: torch.Tensor,
        present_k: torch.Tensor,
        present_v: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        q_len: int,
    ):
        """Run the NPU-export-friendly per-query-head attention path."""
        attn_weights_parts: list[torch.Tensor] = []
        attn_out_parts: list[torch.Tensor] = []

        # Remove the singleton head dimension so every attention operation has
        # the same rank and shape on both sides. The mask builder guarantees
        # that the head dimension is one for this decomposed path.
        attn_mask = attention_mask.squeeze(1)

        for kv_i in range(self.num_kv_heads):
            # (B, K, H)
            k_i = present_k[:, kv_i, :, :]
            v_i = present_v[:, kv_i, :, :]

            for rep_i in range(self.kv_rep):
                q_idx = kv_i * self.kv_rep + rep_i
                q_i = q_rot[:, q_idx, :, :]  # (B, S, H)

                # logits: (B, S, K)
                logits_i = q_i @ k_i.transpose(-2, -1)
                logits_i = self._apply_attention_scale_if_needed(logits_i)
                logits_i = self._fq(logits_i, self.obs_logits)

                logits_i = self._fq(logits_i + attn_mask, self.obs_mask_add)

                attn_i = torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(
                    q_i.dtype
                )
                attn_i = self._fq(attn_i, self.obs_softmax)

                # out: (B, S, H)
                out_i = self._fq(attn_i @ v_i, self.obs_attn_out)

                attn_weights_parts.append(attn_i)
                attn_out_parts.append(out_i)

        # Stack heads back: (B, n_h, S, K) / (B, S, n_h, H)
        attn_weights = self._fq(
            torch.stack(attn_weights_parts, dim=1), self.obs_attn_weights
        )
        attn_out_h = self._fq(torch.stack(attn_out_parts, dim=2), self.obs_attn_out_h)

        # Attention output
        attn_out = attn_out_h.reshape(batch_size, q_len, -1)  # (B, S, n_h * H)

        # Final projection
        out = self.o_proj(attn_out)

        return out, attn_weights

    def _forward_batched(
        self,
        *,
        q_rot: torch.Tensor,
        present_k: torch.Tensor,
        present_v: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        q_len: int,
    ):
        """Run a Hugging Face-like batched attention path for reference testing."""
        if self.kv_rep != 1:
            present_k_for_attn = present_k.repeat_interleave(self.kv_rep, dim=1)
            present_v_for_attn = present_v.repeat_interleave(self.kv_rep, dim=1)
        else:
            present_k_for_attn = present_k
            present_v_for_attn = present_v

        logits = q_rot @ present_k_for_attn.transpose(-2, -1)
        logits = self._apply_attention_scale_if_needed(logits)
        logits = self._fq(logits, self.obs_logits)

        logits = self._fq(logits + attention_mask, self.obs_mask_add)

        attn_weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(
            q_rot.dtype
        )
        attn_weights = self._fq(attn_weights, self.obs_softmax)
        attn_weights = self._fq(attn_weights, self.obs_attn_weights)

        attn_out_h = self._fq(
            attn_weights @ present_v_for_attn,
            self.obs_attn_out,
        )
        attn_out_h = self._fq(attn_out_h, self.obs_attn_out_h)

        attn_out = (
            attn_out_h.transpose(1, 2).contiguous().reshape(batch_size, q_len, -1)
        )
        out = self.o_proj(attn_out)

        return out, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        cache_output_mode: CacheOutputMode = "present",
        **kwargs,
    ):
        """
        Run quantized Qwen3-VL text attention.

        Args:
            hidden_states: Input states with shape `(B, S, hidden_size)`.
            position_embeddings: RoPE cosine and sine tensors with shape
                `(B, S, head_dim)`.
            attention_mask: Optional additive or boolean mask.
            past_key_values: Optional per-layer static KV tuple or HF Cache-like object.
            use_cache: Whether to return cache tensors and update an HF cache.
            cache_position: Optional absolute cache positions for HF caches.
            cache_output_mode: Return full present KV tensors or only the new KV
                delta tensors when `use_cache=True`.

        Returns:
            A tuple `(attn_output, attn_weights)` and, when `use_cache=True`, a
            third cache output item.
        """
        del kwargs

        if use_cache or past_key_values is not None:
            self.require_calibrated_cache()

        hidden = self._fq(hidden_states, self.obs_hidden)
        B, S, _ = hidden.shape
        H = self.head_dim

        # Projections
        q = self.q_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # (B, n_h,  S, H)
        k = self.k_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # (B, n_kv, S, H)
        v = self.v_proj(hidden).view(B, S, -1, H).transpose(1, 2)  # (B, n_kv, S, H)

        # Head-dim RMSNorms
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE tables
        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)
        q_rot, k_rot = self._apply_rope(q, k, cos, sin, unsqueeze_dim=1)

        if past_key_values is None or self._is_static_kv_tuple(past_key_values):
            past = self._normalize_static_past_key_values(past_key_values)
        else:
            past = self._normalize_cache_like(past_key_values)

        # Branch from raw KV before either domain rounds it. S is observed even
        # when prefill does not return a cache, making prefill-only PTQ sufficient.
        new_k = self._fq(k_rot, self.obs_new_k)
        new_v = self._fq(v, self.obs_new_v)

        present_k, present_v = self._build_present_key_values(
            past=past,
            raw_k=k_rot,
            raw_v=v,
        )

        # Build additive attention mask.
        attention_mask = self._build_attention_mask(
            attention_mask=attention_mask,
            q_len=q_rot.size(-2),
            k_len=present_k.size(-2),
            device=hidden.device,
        )

        if self.attn_options.layout == "batched":
            outputs = self._forward_batched(
                q_rot=q_rot,
                present_k=present_k,
                present_v=present_v,
                attention_mask=attention_mask,
                batch_size=B,
                q_len=S,
            )

        elif self.attn_options.layout == "unrolled":
            outputs = self._forward_unrolled(
                q_rot=q_rot,
                present_k=present_k,
                present_v=present_v,
                attention_mask=attention_mask,
                batch_size=B,
                q_len=S,
            )

        else:
            raise RuntimeError(
                f"Invalid attention layout: {self.attn_options.layout!r}"
            )

        if use_cache:
            cache_out = self._finalize_cache_output(
                past_key_values=past_key_values,
                past=past,
                new_k=new_k,
                new_v=new_v,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
                cache_output_mode=cache_output_mode,
            )
            return (*outputs, cache_out)
        return outputs

    def _all_observers(self) -> Iterable:
        yield from (
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_neg,
            self.obs_q_cat,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_neg,
            self.obs_k_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
            self.obs_causal_mask,
            self.obs_logits,
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
        if self.attn_options.scale_fusion == "none":
            yield from (self.obs_logits_raw, self.obs_scale)

    def local_observer_groups(self):
        storage = (self.obs_new_k, self.obs_new_v)
        return {
            "required": tuple(
                obs for obs in self._all_observers() if obs not in storage
            ),
            "cache": storage,
        }

    def as_export_module(
        self,
        mode: ExportMode = "prefill",
        *,
        return_kv: bool = True,
        require_npu_profile: bool = False,
    ) -> nn.Module:
        """
        Return an export adapter for the requested Qwen3-VL text attention mode.

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
        if require_npu_profile and not is_npu_export_text_attention_options(
            self.attn_options
        ):
            raise ValueError(
                "NPU export requires execution profile 'npu_export'. "
                "Set PTQConfig.model_args['profile'] "
                "to 'npu_export'."
            )

        if mode == "prefill":
            if return_kv:
                self.require_calibrated_cache()
            return Qwen3VLTextAttentionPrefillExportAdapter(
                self,
                return_kv=return_kv,
            )
        if mode == "decode":
            self.require_calibrated_cache()
            return Qwen3VLTextAttentionDecodeExportAdapter(
                self,
                return_kv=return_kv,
            )
        raise ValueError(f"Unsupported export mode: {mode!r}")

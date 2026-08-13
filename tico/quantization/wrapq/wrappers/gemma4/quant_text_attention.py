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

from collections.abc import MutableMapping
from typing import Any, Iterable, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.gemma4_attention import get_gemma4_text_attention_options
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import get_model_arg, join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


LayerKV = Tuple[torch.Tensor, torch.Tensor]
CacheOutputMode = Literal["delta", "present", "full"]


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention")
class QuantGemma4TextAttention(QuantModuleBase):
    """PTQ wrapper for dense Gemma4 E2B text attention.

    The wrapper keeps projection, RoPE, cache, mask, and shared-KV semantics
    common across two attention layouts:

    - ``batched`` uses a Hugging Face-like batched GQA implementation and is
      intended for reference evaluation.
    - ``unrolled`` emits rank-3 attention matmuls for every query head and avoids
      head-axis broadcasting for static NPU export.

    The dense E2B scope supports regular K/V-owning layers, global layers where
    ``attention_k_eq_v=True``, and shared-KV consumer layers. Dynamic runtime
    orchestration such as cache writes, shared-KV bookkeeping, mask generation,
    and sampling remains outside this module.
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.attn_options = get_gemma4_text_attention_options(self.qcfg)

        self.module = fp_attn
        self.config = fp_attn.config
        self.layer_idx = fp_attn.layer_idx
        self.layer_type = fp_attn.layer_type
        self.is_sliding = bool(getattr(fp_attn, "is_sliding", False))
        self.sliding_window = int(fp_attn.sliding_window) if self.is_sliding else None
        self.is_kv_shared_layer = bool(getattr(fp_attn, "is_kv_shared_layer", False))
        self.store_full_length_kv = bool(
            getattr(fp_attn, "store_full_length_kv", False)
        )
        self.use_alternative_attention = bool(
            getattr(fp_attn, "use_alternative_attention", False)
        )
        self.head_dim = int(fp_attn.head_dim)
        self.num_key_value_groups = int(fp_attn.num_key_value_groups)
        self.scaling = float(getattr(fp_attn, "scaling", 1.0))
        self.attention_dropout = float(
            getattr(fp_attn, "attention_dropout", 0.0) or 0.0
        )
        self.max_seq = self._resolve_max_seq()

        self.q_proj = PTQWrapper(
            fp_attn.q_proj,
            qcfg=qcfg.child("q_proj") if qcfg else None,
            fp_name=join_name(fp_name, "q_proj"),
        )
        self.q_norm = PTQWrapper(
            fp_attn.q_norm,
            qcfg=qcfg.child("q_norm") if qcfg else None,
            fp_name=join_name(fp_name, "q_norm"),
        )
        if not self.is_kv_shared_layer:
            self.k_proj = PTQWrapper(
                fp_attn.k_proj,
                qcfg=qcfg.child("k_proj") if qcfg else None,
                fp_name=join_name(fp_name, "k_proj"),
            )
            self.k_norm = PTQWrapper(
                fp_attn.k_norm,
                qcfg=qcfg.child("k_norm") if qcfg else None,
                fp_name=join_name(fp_name, "k_norm"),
            )
            self.v_norm = PTQWrapper(
                fp_attn.v_norm,
                qcfg=qcfg.child("v_norm") if qcfg else None,
                fp_name=join_name(fp_name, "v_norm"),
            )
            self.v_proj: Optional[PTQWrapper] = None
            if fp_attn.v_proj is not None:
                self.v_proj = PTQWrapper(
                    fp_attn.v_proj,
                    qcfg=qcfg.child("v_proj") if qcfg else None,
                    fp_name=join_name(fp_name, "v_proj"),
                )

        self.o_proj = PTQWrapper(
            fp_attn.o_proj,
            qcfg=qcfg.child("o_proj") if qcfg else None,
            fp_name=join_name(fp_name, "o_proj"),
        )

        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_neg = mk("q_neg")
        self.obs_q_cat = mk("q_cat")

        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_neg = mk("k_neg")
        self.obs_k_cat = mk("k_cat")

        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        self.obs_new_k = mk("new_k")
        self.obs_new_v = mk("new_v")
        self.obs_present_key = mk("present_key")
        self.obs_present_value = mk("present_value")
        self.obs_attn_mask = mk("attn_mask")
        self.obs_logits_raw = mk("logits_raw")
        self.obs_scale = mk("scale")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")
        self.obs_attn_weights = mk("attn_weights")
        self.obs_attn_out_h = mk("attn_out_h")

        mask = torch.full(
            (1, 1, self.max_seq, self.max_seq),
            float(self.qcfg.attention_mask_fill_value),
        )
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

    _DEFAULT_STATIC_MAX_SEQ = 2048

    def _resolve_max_seq(self) -> int:
        """Resolve the causal-mask capacity from configuration and model limits.

        Resolution precedence is ``calibration.seq_len``,
        ``model_args["text"]["max_seq"]``, ``model_args["max_seq"]``, and finally
        ``min(max_position_embeddings, 2048)``. The cap avoids allocating a full
        model-context square mask for a smaller static export profile.
        """
        from tico.quantization.recipes.config import get_by_path

        try:
            calib_seq_len = get_by_path(self.qcfg, "calibration.seq_len")
            if calib_seq_len is not None:
                return int(calib_seq_len)
        except (KeyError, TypeError):
            pass

        configured = get_model_arg(self.qcfg, "text", "max_seq", default=None)
        if configured is None:
            configured = get_model_arg(self.qcfg, "max_seq", default=None)

        model_capacity = int(getattr(self.config, "max_position_embeddings"))
        max_seq = (
            min(model_capacity, self._DEFAULT_STATIC_MAX_SEQ)
            if configured is None
            else int(configured)
        )
        if max_seq <= 0:
            raise ValueError(
                f"Gemma4 attention max_seq must be positive, got {max_seq}."
            )
        if max_seq > model_capacity:
            raise ValueError(
                "Gemma4 attention max_seq exceeds max_position_embeddings: "
                f"max_seq={max_seq}, model_capacity={model_capacity}."
            )
        return max_seq

    @staticmethod
    def _expand_rope_table(table: torch.Tensor) -> torch.Tensor:
        """Return a RoPE table shaped as ``(B, S, 1, H)`` for Gemma4 text."""
        if table.dim() == 2:
            table = table.unsqueeze(0)
        if table.dim() == 3:
            table = table.unsqueeze(2)
        if table.dim() != 4:
            raise RuntimeError(
                "RoPE table must have rank 2, 3, or 4 for Gemma4 text attention, "
                f"got shape={tuple(table.shape)}."
            )
        return table

    def _rot(
        self,
        tensor: torch.Tensor,
        obs_x1,
        obs_x2,
        obs_neg,
        obs_cat,
    ) -> torch.Tensor:
        """Apply Gemma4/HF ``rotate_half`` as ``[-x2, x1]``."""
        x1, x2 = torch.chunk(tensor, 2, dim=-1)
        x1 = self._fq(x1, obs_x1)
        x2 = self._fq(x2, obs_x2)
        neg_x2 = self._fq(-x2, obs_neg)
        return self._fq(torch.cat((neg_x2, x1), dim=-1), obs_cat)

    def _apply_rope(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        obs_x1,
        obs_x2,
        obs_neg,
        obs_cat,
        obs_cos,
        obs_sin,
        obs_rot,
    ) -> torch.Tensor:
        """Apply rotary position embedding before head transposition."""
        cos = self._expand_rope_table(cos)
        sin = self._expand_rope_table(sin)
        half = self._rot(tensor, obs_x1, obs_x2, obs_neg, obs_cat)
        cos_part = self._fq(tensor * cos, obs_cos)
        sin_part = self._fq(half * sin, obs_sin)
        return self._fq(cos_part + sin_part, obs_rot)

    @staticmethod
    def _normalize_attention_mask_shape(
        mask: torch.Tensor,
        *,
        q_len: int,
        k_len: int,
    ) -> torch.Tensor:
        """Normalize a mask so it broadcasts to ``(B, heads, Q, K)``.

        Supported input shapes are ``(B, K)``, ``(B, Q, K)``, and
        ``(B, 1, Q, K)``. Longer preallocated masks are sliced to the active
        query and key ranges.
        """
        if mask.dim() not in (2, 3, 4):
            raise RuntimeError(
                "Unsupported attention_mask rank for Gemma4 text attention: "
                f"rank={mask.dim()}, shape={tuple(mask.shape)}."
            )

        if mask.size(-1) != k_len:
            if mask.size(-1) > k_len:
                mask = mask[..., :k_len]
            else:
                raise RuntimeError(
                    "attention_mask key length is shorter than key states: "
                    f"mask_k={mask.size(-1)}, k_len={k_len}, "
                    f"shape={tuple(mask.shape)}."
                )

        if mask.dim() == 2:
            return mask[:, None, None, :]

        if mask.size(-2) not in (1, q_len):
            if mask.size(-2) > q_len:
                mask = mask[..., -q_len:, :]
            else:
                raise RuntimeError(
                    "attention_mask query length is incompatible with query states: "
                    f"mask_q={mask.size(-2)}, q_len={q_len}, "
                    f"shape={tuple(mask.shape)}."
                )

        if mask.dim() == 3:
            return mask[:, None, :, :]

        if mask.size(1) != 1:
            raise RuntimeError(
                "Per-head attention masks are not supported by the Gemma4 text "
                "attention wrapper. Expected mask shape (B, 1, Q, K), "
                f"got shape={tuple(mask.shape)}."
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
        """Build an additive attention mask for the active query and key range."""
        fill_val = float(self.qcfg.attention_mask_fill_value)
        assert isinstance(self.causal_mask_template, torch.Tensor)

        if k_len > self.causal_mask_template.size(-1):
            raise RuntimeError(
                "Key length exceeds causal mask capacity: "
                f"k_len={k_len}, capacity={self.causal_mask_template.size(-1)}."
            )

        q_start = max(k_len - q_len, 0)
        q_end = q_start + q_len
        if q_end > self.causal_mask_template.size(-2):
            raise RuntimeError(
                "Query range exceeds causal mask capacity: "
                f"q_start={q_start}, q_end={q_end}, "
                f"capacity={self.causal_mask_template.size(-2)}."
            )

        causal_mask = self.causal_mask_template[..., q_start:q_end, :k_len].to(device)

        if attention_mask is None:
            return self._fq(causal_mask, self.obs_attn_mask)

        attention_mask = attention_mask.to(device)
        if attention_mask.dtype in (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            keep_mask = self._normalize_attention_mask_shape(
                attention_mask.bool(),
                q_len=q_len,
                k_len=k_len,
            )
            additive = torch.zeros(
                keep_mask.shape,
                dtype=torch.float32,
                device=device,
            )
            additive = additive.masked_fill(~keep_mask, fill_val)
            mask = torch.clamp(causal_mask + additive, min=fill_val)
            return self._fq(mask, self.obs_attn_mask)

        if torch.is_floating_point(attention_mask):
            attention_mask = self._normalize_attention_mask_shape(
                attention_mask,
                q_len=q_len,
                k_len=k_len,
            )
            return self._fq(attention_mask, self.obs_attn_mask)

        raise RuntimeError(
            "Unsupported attention_mask dtype for Gemma4 text attention: "
            f"dtype={attention_mask.dtype}, shape={tuple(attention_mask.shape)}."
        )

    def _resolve_shared_key_value(
        self,
        shared_key_value: Optional[LayerKV],
        shared_kv_states: Optional[MutableMapping[str, LayerKV]],
    ) -> LayerKV:
        """Return full shared K/V tensors for a shared-KV consumer layer."""
        if shared_key_value is not None:
            return shared_key_value
        if shared_kv_states is not None and self.layer_type in shared_kv_states:
            return shared_kv_states[self.layer_type]
        raise RuntimeError(
            "Gemma4 shared-KV attention requires shared_key_value or "
            f"shared_kv_states[{self.layer_type!r}]."
        )

    def _project_current_key_value(
        self,
        hidden_states: torch.Tensor,
        hidden_shape: tuple[int, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> LayerKV:
        """Project, normalize, and rotate the current non-shared K/V tensors."""
        key_states_raw = self.k_proj(hidden_states).view(hidden_shape)
        if self.v_proj is None:
            value_states = key_states_raw
        else:
            value_states = self.v_proj(hidden_states).view(hidden_shape)

        key_states = self.k_norm(key_states_raw)
        key_states = self._apply_rope(
            key_states,
            cos,
            sin,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_neg,
            self.obs_k_cat,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
        )
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        key_states = self._fq(key_states, self.obs_new_k)
        value_states = self._fq(value_states, self.obs_new_v)
        return key_states, value_states

    def _build_present_key_value(
        self,
        *,
        new_key_value: LayerKV,
        past_key_value: Optional[Any],
    ) -> tuple[LayerKV, Optional[Any]]:
        """Build full K/V tensors for attention and an optional cache object."""
        new_k, new_v = new_key_value
        cache_object_out = None

        if past_key_value is None:
            present_k = self._fq(new_k, self.obs_present_key)
            present_v = self._fq(new_v, self.obs_present_value)
            return (present_k, present_v), cache_object_out

        if isinstance(past_key_value, tuple):
            past_k, past_v = past_key_value
            past_k = self._fq(past_k, self.obs_present_key)
            past_v = self._fq(past_v, self.obs_present_value)
            present_k = self._fq(
                torch.cat([past_k, new_k], dim=2), self.obs_present_key
            )
            present_v = self._fq(
                torch.cat([past_v, new_v], dim=2), self.obs_present_value
            )
            return (present_k, present_v), cache_object_out

        update = getattr(past_key_value, "update", None)
        if callable(update):
            present_k, present_v = update(new_k, new_v, self.layer_idx)
            present_k = self._fq(present_k, self.obs_present_key)
            present_v = self._fq(present_v, self.obs_present_value)
            cache_object_out = past_key_value
            return (present_k, present_v), cache_object_out

        raise RuntimeError(
            "Unsupported past_key_value type for Gemma4 text attention: "
            f"{type(past_key_value).__name__}."
        )

    @staticmethod
    def _normalize_cache_output_mode(cache_output_mode: str) -> CacheOutputMode:
        """Normalize cache-output aliases used by static runtime adapters."""
        if cache_output_mode == "full":
            return "present"
        if cache_output_mode not in ("delta", "present"):
            raise ValueError(f"Unsupported cache_output_mode: {cache_output_mode!r}.")
        return cache_output_mode  # type: ignore[return-value]

    def _cache_output(
        self,
        *,
        mode: CacheOutputMode,
        new_key_value: Optional[LayerKV],
        present_key_value: Optional[LayerKV],
        cache_object_out: Optional[Any],
    ) -> Optional[Any]:
        """Return cache data according to the requested output policy."""
        if new_key_value is None:
            return None
        if mode == "delta":
            return new_key_value
        if cache_object_out is not None:
            return cache_object_out
        return present_key_value

    def _validate_attention_shapes(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[int, int, int, int]:
        """Validate GQA tensor relationships shared by both attention layouts."""
        if query_states.dim() != 4 or key_states.dim() != 4 or value_states.dim() != 4:
            raise RuntimeError(
                "Gemma4 attention expects rank-4 Q/K/V tensors, got "
                f"Q={tuple(query_states.shape)}, K={tuple(key_states.shape)}, "
                f"V={tuple(value_states.shape)}."
            )
        if attention_mask.dim() != 4 or attention_mask.size(1) != 1:
            raise RuntimeError(
                "Gemma4 attention expects a shared-head mask shaped "
                f"(B, 1, Q, K), got {tuple(attention_mask.shape)}."
            )

        batch_size, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = key_states.size(1)
        kv_rep = self.num_key_value_groups
        if num_heads != num_kv_heads * kv_rep:
            raise RuntimeError(
                "Invalid Gemma4 GQA head relationship: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
                f"num_key_value_groups={kv_rep}."
            )
        if value_states.size(1) != num_kv_heads:
            raise RuntimeError(
                "Gemma4 key/value head counts differ: "
                f"key_heads={num_kv_heads}, value_heads={value_states.size(1)}."
            )
        if key_states.size(0) != batch_size or value_states.size(0) != batch_size:
            raise RuntimeError("Gemma4 Q/K/V batch dimensions must match.")
        if key_states.size(-1) != head_dim or value_states.size(-1) != head_dim:
            raise RuntimeError(
                "Gemma4 Q/K/V head dimensions must match: "
                f"query={head_dim}, key={key_states.size(-1)}, "
                f"value={value_states.size(-1)}."
            )
        if key_states.size(-2) != value_states.size(-2):
            raise RuntimeError(
                "Gemma4 key/value sequence lengths must match: "
                f"key={key_states.size(-2)}, value={value_states.size(-2)}."
            )
        if attention_mask.size(0) not in (1, batch_size):
            raise RuntimeError(
                "Gemma4 attention-mask batch dimension is incompatible with Q/K/V: "
                f"mask_batch={attention_mask.size(0)}, batch={batch_size}."
            )
        if attention_mask.size(-2) not in (1, q_len):
            raise RuntimeError(
                "Gemma4 attention-mask query length is incompatible with Q: "
                f"mask_q={attention_mask.size(-2)}, q_len={q_len}."
            )
        if attention_mask.size(-1) != key_states.size(-2):
            raise RuntimeError(
                "Gemma4 attention-mask key length is incompatible with K/V: "
                f"mask_k={attention_mask.size(-1)}, k_len={key_states.size(-2)}."
            )
        return batch_size, num_heads, q_len, num_kv_heads

    def _apply_attention_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Observe logits, apply the Gemma4 attention scale, and observe output."""
        logits = self._fq(logits, self.obs_logits_raw)
        scale = self._fq(
            torch.tensor(
                self.scaling,
                device=logits.device,
                dtype=logits.dtype,
            ),
            self.obs_scale,
        )
        return self._fq(logits * scale, self.obs_logits)

    def _forward_unrolled(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run NPU-friendly attention with one rank-3 graph per query head."""
        batch_size, _, q_len, num_kv_heads = self._validate_attention_shapes(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )
        kv_rep = self.num_key_value_groups
        attn_mask = attention_mask.squeeze(1)

        attn_weights_parts: list[torch.Tensor] = []
        attn_out_parts: list[torch.Tensor] = []

        for kv_idx in range(num_kv_heads):
            key_i = key_states[:, kv_idx, :, :]
            value_i = value_states[:, kv_idx, :, :]

            for rep_idx in range(kv_rep):
                query_idx = kv_idx * kv_rep + rep_idx
                query_i = query_states[:, query_idx, :, :]

                logits_i = query_i @ key_i.transpose(-2, -1)
                logits_i = self._apply_attention_scale(logits_i)
                logits_i = self._fq(logits_i + attn_mask, self.obs_mask_add)

                attn_i = torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(
                    query_i.dtype
                )
                if self.training and self.attention_dropout > 0.0:
                    attn_i = F.dropout(
                        attn_i,
                        p=self.attention_dropout,
                        training=True,
                    )
                attn_i = self._fq(attn_i, self.obs_softmax)

                out_i = self._fq(attn_i @ value_i, self.obs_attn_out)
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
        attn_output = attn_out_h.reshape(batch_size, q_len, -1)
        return self.o_proj(attn_output), attn_weights

    def _forward_batched(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a Hugging Face-like batched GQA path for reference evaluation."""
        batch_size, _, q_len, _ = self._validate_attention_shapes(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )
        kv_rep = self.num_key_value_groups

        if kv_rep != 1:
            key_for_attention = key_states.repeat_interleave(kv_rep, dim=1)
            value_for_attention = value_states.repeat_interleave(kv_rep, dim=1)
        else:
            key_for_attention = key_states
            value_for_attention = value_states

        logits = query_states @ key_for_attention.transpose(-2, -1)
        logits = self._apply_attention_scale(logits)
        logits = self._fq(logits + attention_mask, self.obs_mask_add)

        attn_weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        if self.training and self.attention_dropout > 0.0:
            attn_weights = F.dropout(
                attn_weights,
                p=self.attention_dropout,
                training=True,
            )
        attn_weights = self._fq(attn_weights, self.obs_softmax)
        attn_weights = self._fq(attn_weights, self.obs_attn_weights)

        attn_out_h = self._fq(
            attn_weights @ value_for_attention,
            self.obs_attn_out,
        )
        attn_out_h = self._fq(attn_out_h, self.obs_attn_out_h)
        attn_output = (
            attn_out_h.transpose(1, 2).contiguous().reshape(batch_size, q_len, -1)
        )
        return self.o_proj(attn_output), attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        shared_key_value: Optional[LayerKV] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        cache_output_mode: str = "delta",
        **kwargs,
    ):
        """Run Gemma4 text attention with selectable execution layout.

        Args:
            hidden_states: Input tensor shaped ``(B, S, hidden_size)``.
            position_embeddings: Tuple ``(cos, sin)`` shaped ``(B, S, head_dim)``.
            attention_mask: Optional additive or keep mask broadcastable to
                attention logits.
            shared_key_value: Optional full K/V tensors for shared-KV layers.
            past_key_value: Optional tuple or HF-like cache object for non-shared
                layers. Tuple caches are shaped ``(B, num_kv_heads, K, H)``.
            use_cache: Whether to append cache data to the output tuple.
            cache_output_mode: ``"delta"`` returns newly projected K/V;
                ``"present"`` or ``"full"`` returns full K/V or the updated
                cache object.

        Returns:
            ``(attn_output, attn_weights)`` and an optional third cache item when
            ``use_cache=True``. Shared-KV layers return ``cache=None`` because
            they do not own K/V projection weights.
        """
        past_key_value = kwargs.pop("past_key_values", past_key_value)
        shared_kv_states = kwargs.pop("shared_kv_states", None)
        cache_output_mode_normalized = self._normalize_cache_output_mode(
            cache_output_mode
        )

        hidden = self._fq(hidden_states, self.obs_hidden)
        batch_size, seq_len, _ = hidden.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        query_states = self.q_proj(hidden).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = self._apply_rope(
            query_states,
            cos,
            sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_neg,
            self.obs_q_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
        )
        query_states = query_states.transpose(1, 2)

        new_key_value: Optional[LayerKV]
        cache_object_out: Optional[Any] = None
        if self.is_kv_shared_layer:
            key_states, value_states = self._resolve_shared_key_value(
                shared_key_value,
                shared_kv_states,
            )
            key_states = self._fq(
                key_states.to(device=query_states.device, dtype=query_states.dtype),
                self.obs_present_key,
            )
            value_states = self._fq(
                value_states.to(device=query_states.device, dtype=query_states.dtype),
                self.obs_present_value,
            )
            new_key_value = None
            present_key_value: Optional[LayerKV] = (key_states, value_states)
        else:
            new_key_value = self._project_current_key_value(
                hidden, hidden_shape, cos, sin
            )
            present_key_value, cache_object_out = self._build_present_key_value(
                new_key_value=new_key_value,
                past_key_value=past_key_value,
            )
            key_states, value_states = present_key_value
            if self.store_full_length_kv and shared_kv_states is not None:
                shared_kv_states[self.layer_type] = (key_states, value_states)

        attn_mask = self._build_attention_mask(
            attention_mask=attention_mask,
            q_len=query_states.size(-2),
            k_len=key_states.size(-2),
            device=query_states.device,
        )

        if self.attn_options.layout == "batched":
            attn_output, attn_weights = self._forward_batched(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attn_mask,
            )
        elif self.attn_options.layout == "unrolled":
            attn_output, attn_weights = self._forward_unrolled(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attn_mask,
            )
        else:
            raise RuntimeError(
                f"Invalid Gemma4 attention layout: {self.attn_options.layout!r}."
            )

        if use_cache:
            return (
                attn_output,
                attn_weights,
                self._cache_output(
                    mode=cache_output_mode_normalized,
                    new_key_value=new_key_value,
                    present_key_value=present_key_value,
                    cache_object_out=cache_object_out,
                ),
            )
        return attn_output, attn_weights

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        common = [
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_neg,
            self.obs_q_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
            self.obs_present_key,
            self.obs_present_value,
            self.obs_attn_mask,
            self.obs_logits_raw,
            self.obs_scale,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
        ]
        if not self.is_kv_shared_layer:
            common.extend(
                [
                    self.obs_k_x1,
                    self.obs_k_x2,
                    self.obs_k_neg,
                    self.obs_k_cat,
                    self.obs_k_cos,
                    self.obs_k_sin,
                    self.obs_k_rot,
                    self.obs_new_k,
                    self.obs_new_v,
                ]
            )
        return tuple(common)

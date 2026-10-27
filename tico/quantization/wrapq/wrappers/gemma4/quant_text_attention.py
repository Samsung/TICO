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
from typing import Any, cast, Iterable, Literal, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.gemma4_attention import get_gemma4_text_attention_options
from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.dtypes import INT16, UINT8
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.utils.utils import get_model_arg, join_name
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4TextAttentionDecodeExportAdapter,
    Gemma4TextAttentionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

if TYPE_CHECKING:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention


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

    causal_mask_template: torch.Tensor

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        fp_attn = cast("Gemma4TextAttention", fp_attn)
        super().__init__(qcfg, fp_name=fp_name)
        obsolete = {"past_key", "past_value"}.intersection(self.qcfg.overrides)
        if obsolete:
            raise ValueError(
                "Gemma4 past_key/past_value observer overrides were removed; "
                "configure new_k/new_v for storage and present_key/present_value "
                "for attention computation, then reprepare and recalibrate."
            )
        self._cache_checkpoint_version = 2
        self.attn_options = get_gemma4_text_attention_options(self.qcfg)

        self.module = fp_attn
        self.config = fp_attn.config
        self.layer_idx = fp_attn.layer_idx
        self.layer_type = fp_attn.layer_type
        self.is_sliding = bool(getattr(fp_attn, "is_sliding", False))
        self.sliding_window = (
            int(cast(int, fp_attn.sliding_window)) if self.is_sliding else None
        )
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
        object.__setattr__(self, "_cache_source", None)
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
        """Resolve the export-mask capacity from configuration and model limits.

        Resolution precedence is ``model_args["text"]["max_seq"]``,
        ``model_args["max_seq"]``, and finally
        ``min(max_position_embeddings, 2048)``. The cap avoids allocating a full
        model-context square mask for a smaller static export profile. Eager
        execution is independent of this capacity and the calibration length.
        """
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
        """Rotate halves using the convention of the supplied sine table."""
        x1, x2 = torch.chunk(tensor, 2, dim=-1)
        x1 = self._fq(x1, obs_x1)
        x2 = self._fq(x2, obs_x2)
        if self.attn_options.rope == "hf":
            x2 = self._fq(-x2, obs_neg)
        return self._fq(torch.cat((x2, x1), dim=-1), obs_cat)

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
        static_export: bool = False,
    ) -> torch.Tensor:
        """Build an additive attention mask for the active query and key range."""
        fill_val = float(self.qcfg.attention_mask_fill_value)
        assert isinstance(self.causal_mask_template, torch.Tensor)
        use_static_template = static_export or torch.compiler.is_compiling()
        capacity = (
            self.causal_mask_template.size(-1)
            if use_static_template
            else int(self.config.max_position_embeddings)
        )
        if k_len > capacity:
            raise RuntimeError(
                "Key length exceeds causal mask capacity: "
                f"k_len={k_len}, capacity={capacity}."
            )

        q_start = max(k_len - q_len, 0)
        q_end = q_start + q_len
        if q_end > capacity:
            raise RuntimeError(
                "Query range exceeds causal mask capacity: "
                f"q_start={q_start}, q_end={q_end}, "
                f"capacity={capacity}."
            )

        # Model-level masks already contain causality, padding, and the exact
        # cache offset for full or sliding attention. Avoid constructing a
        # second mask when the caller supplies that complete additive mask.
        if attention_mask is not None and torch.is_floating_point(attention_mask):
            attention_mask = self._normalize_attention_mask_shape(
                attention_mask.to(device),
                q_len=q_len,
                k_len=k_len,
            )
            return self._fq(attention_mask, self.obs_attn_mask)

        if use_static_template:
            causal_mask = self.causal_mask_template[..., q_start:q_end, :k_len].to(
                device
            )
        else:
            query = torch.arange(q_start, q_end, device=device).view(-1, 1)
            key = torch.arange(k_len, device=device).view(1, -1)
            keep = key <= query
            if self.is_sliding:
                keep = keep & (key > query - int(cast(int, self.sliding_window)))
            causal_mask = (
                torch.zeros(
                    (q_len, k_len), dtype=self.causal_mask_template.dtype, device=device
                )
                .masked_fill(~keep, fill_val)
                .unsqueeze(0)
                .unsqueeze(0)
            )

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
        if (
            shared_kv_states is not None
            and self.layer_type is not None
            and self.layer_type in shared_kv_states
        ):
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

        # Branch to storage and attention only after returning these raw tensors.
        # Computing C from the S-quantized delta would introduce double rounding.
        return key_states, value_states

    def bind_cache_source(self, source: "QuantGemma4TextAttention") -> None:
        """Bind a shared consumer to its storage producer without owning it.

        Bind before calibration. Rebinding an already calibrated consumer is
        allowed only when it retains the same source. This reference is outside
        the module tree, so consumer lifecycle/device traversal never visits the
        producer's weights or observers.
        """
        source._validate_cache_observers()
        if getattr(self, "_external_float_cache", False):
            raise ValueError(
                "Gemma4 external float cache and quantized producer binding are mutually exclusive."
            )
        if not self.is_kv_shared_layer or source.is_kv_shared_layer:
            raise ValueError(
                "Gemma4 cache source must bind a shared consumer to a KV producer."
            )
        if self.layer_type != source.layer_type:
            raise ValueError("Gemma4 cache source and consumer layer types must match.")
        current = getattr(self, "_cache_source", None)
        if current is source:
            return
        if self._mode is not Mode.NO_QUANT:
            raise RuntimeError(
                "Bind the Gemma4 cache producer before prepare/calibration; reprepare and recalibrate this consumer."
            )
        object.__setattr__(self, "_cache_source", source)

    def bind_external_float_cache(self) -> None:
        """Declare FP target KV input for the assistant's external cache.

        This is an explicit float storage contract, not an inferred producer.
        Quantized target-cache tensors need a real producer binding instead.
        """
        if (
            not self.is_kv_shared_layer
            or getattr(self, "_cache_source", None) is not None
        ):
            raise ValueError(
                "Gemma4 external float cache requires an unbound shared-KV consumer."
            )
        if getattr(self, "_external_float_cache", False):
            return
        if self._mode is not Mode.NO_QUANT:
            raise RuntimeError(
                "Declare the Gemma4 external float cache before calibration."
            )
        self._external_float_cache = True
        self.obs_cache_storage_key = IdentityObserver(name="cache_storage_key")
        self.obs_cache_storage_value = IdentityObserver(name="cache_storage_value")

    _OBSOLETE_CACHE_CHECKPOINT = (
        "Gemma4 checkpoint uses an incompatible cache quantization contract; "
        "reprepare and recalibrate the original float model with new_k/new_v "
        "storage and present_key/present_value compute policies. Three-domain "
        "past-KV checkpoints cannot be reused (calibration.decode_steps=0 is supported)."
    )

    def __setstate__(self, state):
        # Full-module recipe checkpoints bypass state-dict loading.
        if state.get("_cache_checkpoint_version") != 2 or any(
            name in state.get("_modules", {})
            for name in ("obs_past_key", "obs_past_value")
        ):
            raise RuntimeError(self._OBSOLETE_CACHE_CHECKPOINT)
        super().__setstate__(state)

    def _validate_cache_observers(self) -> None:
        """Reject old pickled wrappers instead of inventing missing statistics."""
        if getattr(self, "_cache_checkpoint_version", None) != 2 or any(
            name in self._modules for name in ("obs_past_key", "obs_past_value")
        ):
            raise RuntimeError(self._OBSOLETE_CACHE_CHECKPOINT)

    def cache_storage_observers(self) -> tuple[ObserverBase, ObserverBase]:
        """Return the actual producer-owned storage boundaries."""
        self._validate_cache_observers()
        source = getattr(self, "_cache_source", None)
        if self.is_kv_shared_layer:
            if getattr(self, "_external_float_cache", False):
                return self.obs_cache_storage_key, self.obs_cache_storage_value
            if source is None:
                raise RuntimeError(
                    "Gemma4 shared-KV quantization/export requires an explicit cache producer; call bind_cache_source(producer) before calibration."
                )
            return source.obs_new_k, source.obs_new_v
        return self.obs_new_k, self.obs_new_v

    @staticmethod
    def _has_cache_statistics(observer: ObserverBase) -> bool:
        if not isinstance(observer, AffineObserverBase) or isinstance(
            observer, IdentityObserver
        ):
            return True
        if bool(
            torch.isposinf(observer.min_val).all()
            and torch.isneginf(observer.max_val).all()
        ):
            return False
        if not bool(
            torch.isfinite(observer.min_val).all()
            and torch.isfinite(observer.max_val).all()
        ):
            raise RuntimeError(
                f"Gemma4 {observer.name} has non-finite calibration statistics; reprepare and recalibrate."
            )
        return True

    @staticmethod
    def _validate_frozen_observer(observer: ObserverBase) -> None:
        if isinstance(observer, IdentityObserver):
            return
        if (
            not isinstance(observer, AffineObserverBase)
            or observer.channel_axis is not None
            or observer.qscheme.is_per_channel()
        ):
            raise ValueError(
                "Gemma4 cache storage and affine compute require per-tensor affine policies; MX storage and per-channel cache conversion are unsupported."
            )
        if not observer.fake_quant_enabled:
            raise ValueError(
                "Gemma4 affine cache boundaries require fake quantization to preserve storage/compute qparams; use an explicit IdentityObserver float policy instead of disabling fake quantization."
            )
        if observer.dtype not in (INT16, UINT8):
            raise ValueError(
                "Gemma4 affine cache conversion currently supports UINT8 and INT16 storage/compute grids only."
            )
        if not observer.has_qparams:
            raise RuntimeError(
                f"Gemma4 {observer.name} storage/compute qparams are not ready; run prefill calibration (or load valid frozen qparams)."
            )
        scale, zp = observer._cached_scale, observer._cached_zp
        if (
            scale.numel() != 1
            or zp.numel() != 1
            or not bool(torch.isfinite(scale).all() and (scale > 0).all())
            or not bool(
                ((zp >= observer.dtype.qmin) & (zp <= observer.dtype.qmax)).all()
            )
        ):
            raise RuntimeError(
                f"Gemma4 {observer.name} has invalid frozen cache qparams; reprepare and recalibrate."
            )

    def _validate_compute_observer(self, observer: ObserverBase) -> None:
        """Allow sequence-block MX only at the complete attention-V boundary."""
        if not isinstance(observer, MXObserver):
            self._validate_frozen_observer(observer)
            return
        if (
            observer is not self.obs_present_value
            or observer.elem_format != "int8"
            or observer.axis not in (-2, 2)
            or observer.shared_exp_method != "max"
            or observer.round != "nearest"
            or not observer.fake_quant_enabled
        ):
            raise ValueError(
                "Gemma4 MX compute supports present_value=MXINT8 with sequence axis=-2, max shared exponent, nearest rounding, and enabled fake quantization only; storage and present_key require per-tensor affine policies."
            )

    def validate_export_cache(self, mode: str) -> None:
        """Validate the storage owner and direct storage-to-compute contract."""
        self._validate_cache_observers()
        if self._mode is not Mode.QUANT:
            return
        storage = self.cache_storage_observers()
        for storage_observer, compute_observer in (
            (storage[0], self.obs_present_key),
            (storage[1], self.obs_present_value),
        ):
            self._validate_frozen_observer(storage_observer)
            self._validate_compute_observer(compute_observer)
            if isinstance(compute_observer, MXObserver):
                if (
                    isinstance(storage_observer, IdentityObserver)
                    or storage_observer.dtype != INT16
                ):
                    raise ValueError(
                        "Gemma4 MX attention V requires per-tensor affine INT16 storage."
                    )
                continue
            group = (storage_observer, compute_observer)
            identities = [isinstance(observer, IdentityObserver) for observer in group]
            float_input_conversion = identities[0] and not identities[1]
            if any(identities) and not all(identities) and not float_input_conversion:
                raise ValueError(
                    "Gemma4 cache conversion supports per-tensor affine grids, all-identity paths, or explicit float storage converted to an affine compute grid; other identity/affine combinations are unsupported."
                )

    def enable_calibration(self) -> None:
        self._validate_cache_observers()
        super().enable_calibration()

    def freeze_qparams(self) -> None:
        """Freeze independently observed storage and compute grids."""
        self._validate_cache_observers()
        self._mode = Mode.QUANT
        cache_observers = (
            self.obs_new_k,
            self.obs_new_v,
            self.obs_present_key,
            self.obs_present_value,
        )
        for observer in self._all_observers():
            observer.enabled = False
            if (
                observer in cache_observers
                and isinstance(observer, AffineObserverBase)
                and observer.has_qparams
            ):
                # A valid externally loaded frozen grid is sufficient, even
                # when this calibration run never executed the boundary.
                self._validate_frozen_observer(observer)
                if observer._qparams_locked or not self._has_cache_statistics(observer):
                    continue
            if observer in cache_observers and not self._has_cache_statistics(observer):
                raise RuntimeError(
                    f"Gemma4 {observer.name} has no calibration statistics; "
                    "run prefill calibration or load valid frozen qparams."
                )
            observer.compute_qparams()
        for child in self._child_quant_modules():
            child.freeze_qparams()

    _CACHE_OBSERVER_NAMES = (
        "new_k",
        "new_v",
        "present_key",
        "present_value",
    )

    def _owned_cache_observer_names(self) -> tuple[str, ...]:
        # A shared consumer's dormant new_* observers never define storage.
        return (
            self._CACHE_OBSERVER_NAMES[2:]
            if self.is_kv_shared_layer
            else self._CACHE_OBSERVER_NAMES
        )

    @staticmethod
    def _observer_grid_policy(
        observer: ObserverBase,
    ) -> tuple[str, str, str, Optional[int], tuple[tuple[str, Any], ...]]:
        # Record calibration options and MX block policy as well as grid type.
        # Statistics and lifecycle flags are saved separately.
        options = tuple(
            (name, getattr(observer, name))
            for name in (
                "fake_quant_enabled",
                "momentum",
                "percentile",
                "lower_percentile",
                "upper_percentile",
                "max_samples",
                "samples_per_batch",
                "seed",
                "elem_format",
                "axis",
                "shared_exp_method",
                "round",
            )
            if hasattr(observer, name)
        )
        return (
            f"{type(observer).__module__}.{type(observer).__qualname__}",
            str(observer.dtype),
            str(observer.qscheme),
            observer.channel_axis,
            options,
        )

    def get_extra_state(self) -> dict[str, Any]:
        """Persist frozen cache grids, whose generic affine cache is transient."""
        self._validate_cache_observers()
        qparams = {}
        policies = {}
        for name in self._owned_cache_observer_names():
            observer = getattr(self, f"obs_{name}")
            policies[name] = self._observer_grid_policy(observer)
            if (
                isinstance(observer, AffineObserverBase)
                and not isinstance(observer, IdentityObserver)
                and observer.has_qparams
            ):
                qparams[name] = (
                    observer._cached_scale,
                    observer._cached_zp,
                    observer._qparams_locked,
                )
        source = getattr(self, "_cache_source", None)
        source_grids = []
        if source is not None or getattr(self, "_external_float_cache", False):
            for observer in self.cache_storage_observers():
                grid = None
                if (
                    isinstance(observer, AffineObserverBase)
                    and not isinstance(observer, IdentityObserver)
                    and observer.has_qparams
                ):
                    grid = (observer._cached_scale, observer._cached_zp)
                source_grids.append((self._observer_grid_policy(observer), grid))
        return {
            "version": 2,
            "frozen": self._mode is Mode.QUANT,
            "qparams": qparams,
            "source_grids": source_grids,
            "external_float_cache": bool(getattr(self, "_external_float_cache", False)),
            "policies": policies,
            "source_layer_idx": None if source is None else source.layer_idx,
            "source_layer_type": None if source is None else source.layer_type,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore frozen grids and validate the constructor's source binding."""
        if state.get("version") != 2:
            raise RuntimeError(self._OBSOLETE_CACHE_CHECKPOINT)
        names = self._owned_cache_observer_names()
        if set(state.get("policies", {})) != set(names) or not set(
            state.get("qparams", {})
        ).issubset(names):
            raise RuntimeError(self._OBSOLETE_CACHE_CHECKPOINT)
        if bool(state.get("external_float_cache", False)) != bool(
            getattr(self, "_external_float_cache", False)
        ):
            raise RuntimeError(
                "Gemma4 checkpoint external float cache contract differs; declare the original cache boundary before loading state_dict."
            )
        source = getattr(self, "_cache_source", None)
        expected = state.get("source_layer_idx")
        if expected != (None if source is None else source.layer_idx) or state.get(
            "source_layer_type"
        ) != (None if source is None else source.layer_type):
            raise RuntimeError(
                "Gemma4 checkpoint cache producer binding differs; bind the original producer before loading state_dict."
            )
        has_source = source is not None or getattr(self, "_external_float_cache", False)
        if len(state.get("source_grids", [])) != (2 if has_source else 0):
            raise RuntimeError(
                "Gemma4 checkpoint cache source grids differ; restore the original "
                "producer or external float boundary, or reprepare and recalibrate."
            )
        if has_source:
            for observer, (policy, grid) in zip(
                self.cache_storage_observers(),
                state.get("source_grids", []),
                strict=True,
            ):
                if policy != self._observer_grid_policy(observer):
                    raise RuntimeError(
                        "Gemma4 checkpoint producer storage policy differs; bind the original producer before loading state_dict."
                    )
                if grid is not None and (
                    not isinstance(observer, AffineObserverBase)
                    or not observer.has_qparams
                    or not torch.equal(observer._cached_scale.cpu(), grid[0].cpu())
                    or not torch.equal(observer._cached_zp.cpu(), grid[1].cpu())
                ):
                    raise RuntimeError(
                        "Gemma4 checkpoint producer storage qparams differ; load the original frozen producer before loading the consumer state_dict."
                    )
                if state.get("frozen", False):
                    if not isinstance(observer, IdentityObserver) and grid is None:
                        raise RuntimeError(
                            "Gemma4 checkpoint lacks frozen producer storage qparams; reprepare and recalibrate."
                        )
                    self._validate_frozen_observer(observer)
        for name in names:
            observer = getattr(self, f"obs_{name}")
            if state.get("policies", {}).get(name) != self._observer_grid_policy(
                observer
            ):
                raise RuntimeError(
                    "Gemma4 checkpoint cache observer policy differs; reprepare and recalibrate."
                )
        for name, (scale, zp, locked) in state.get("qparams", {}).items():
            observer = getattr(self, f"obs_{name}")
            if not isinstance(observer, AffineObserverBase):
                raise RuntimeError(
                    "Gemma4 checkpoint cache observer policy differs; reprepare and recalibrate."
                )
            observer.load_qparams(
                scale.to(observer.min_val.device),
                zp.to(observer.min_val.device),
                lock=locked or bool(state.get("frozen", False)),
            )
        if state.get("frozen", False):
            for name in names:
                observer = getattr(self, f"obs_{name}")
                if isinstance(observer, MXObserver):
                    # MX compute owns block policy, not affine scale/zero-point.
                    self._validate_compute_observer(observer)
                    continue
                if not isinstance(observer, IdentityObserver) and name not in state.get(
                    "qparams", {}
                ):
                    raise RuntimeError(
                        f"Gemma4 checkpoint lacks frozen {name} cache qparams; reprepare and recalibrate."
                    )
                self._validate_frozen_observer(observer)

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
        if prefix + "_extra_state" not in state_dict or any(
            name.startswith((prefix + "obs_past_key.", prefix + "obs_past_value."))
            for name in state_dict
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

    def _validate_cache_tensor_pair(self, key_value: LayerKV) -> None:
        """Validate cache layout before treating any slots as actual past."""
        key, value = key_value
        if (
            key.dim() != 4
            or value.dim() != 4
            or key.shape != value.shape
            or key.size(-1) != self.head_dim
        ):
            raise ValueError(
                "Gemma4 cache key/value must have matching rank-4 (B, KV heads, tokens, head_dim) shapes."
            )

    def _read_cache_key_value(self, key_value: LayerKV) -> LayerKV:
        """Declare the source storage grid without collecting or writing it.

        The caller converts these S values directly to its own C grid. During
        calibration the supplied values pass through for compute observation.
        """
        self._validate_cache_tensor_pair(key_value)
        if self._mode is Mode.QUANT:
            if not torch.compiler.is_compiling():
                self.validate_export_cache("decode")
            storage_k, storage_v = self.cache_storage_observers()
            key_value = (
                storage_k.fake_quant(key_value[0]),
                storage_v.fake_quant(key_value[1]),
            )
        return key_value

    def _cache_to_compute(self, key_value: LayerKV) -> LayerKV:
        """Convert the declared storage domain directly into attention's grid."""
        return self._fq(key_value[0], self.obs_present_key), self._fq(
            key_value[1], self.obs_present_value
        )

    def _cache_object_past(self, cache: Any, q_len: int) -> Optional[LayerKV]:
        """Read active slots before update using supported HF Cache layer APIs.

        Static padding is excluded, and sliding layers expose only the existing
        tokens retained for this query span. No read result is written back.
        """
        from transformers.cache_utils import (
            Cache,
            DynamicLayer,
            DynamicSlidingWindowLayer,
            StaticLayer,
            StaticSlidingWindowLayer,
        )

        if not isinstance(cache, Cache) or getattr(cache, "offloading", False):
            raise RuntimeError(
                "Gemma4 cache input supports tuples and non-offloaded HF DynamicCache/StaticCache only."
            )
        if self.layer_idx >= len(cache.layers):
            return None
        layer = cache.layers[self.layer_idx]
        if not isinstance(
            layer,
            (
                DynamicLayer,
                DynamicSlidingWindowLayer,
                StaticLayer,
                StaticSlidingWindowLayer,
            ),
        ):
            raise RuntimeError(
                f"Unsupported Gemma4 HF cache layer: {type(layer).__name__}."
            )
        seen = int(cache.get_seq_length(self.layer_idx))
        if seen == 0:
            return None
        if (
            isinstance(layer, StaticLayer)
            and not isinstance(layer, StaticSlidingWindowLayer)
            and seen + q_len > layer.max_cache_len
        ):
            raise ValueError(
                "Gemma4 StaticCache capacity exceeded; allocate a larger cache before decode."
            )
        keys, values = layer.keys, layer.values
        if keys is None or values is None:
            raise RuntimeError("Initialized Gemma4 HF cache lacks key/value storage.")
        active = min(seen, keys.size(-2))
        kv_length, _ = cache.get_mask_sizes(q_len, self.layer_idx)
        active = min(active, max(int(kv_length) - q_len, 0))
        if active == 0:
            return None
        if isinstance(
            layer, (DynamicSlidingWindowLayer, StaticSlidingWindowLayer)
        ) and seen >= keys.size(-2):
            return keys[..., -active:, :], values[..., -active:, :]
        return keys[..., :active, :], values[..., :active, :]

    def _build_present_key_value(
        self,
        *,
        new_key_value: LayerKV,
        current_key_value: LayerKV,
        past_key_value: Optional[Any],
        use_cache: bool = True,
    ) -> tuple[LayerKV, LayerKV, Optional[Any]]:
        """Combine independent raw-to-C and storage-to-C attention branches.

        ``new_key_value`` is the S delta; ``current_key_value`` is the raw
        projection. Only the former is appended to persistent storage.
        """
        new_k, new_v = new_key_value
        current_k, current_v = current_key_value
        cache_object_out = None
        if past_key_value is None:
            past = None
        elif isinstance(past_key_value, tuple):
            if len(past_key_value) != 2:
                raise ValueError(
                    "Gemma4 tuple cache must contain exactly key and value."
                )
            self._validate_cache_tensor_pair(past_key_value)
            if past_key_value[0].shape[:2] != new_k.shape[:2]:
                raise ValueError(
                    "Gemma4 cache batch/KV-head dimensions must match the new delta."
                )
            past = past_key_value if past_key_value[0].size(-2) > 0 else None
        else:
            past = self._cache_object_past(past_key_value, new_k.size(-2))

        if past is None:
            storage = new_key_value
            compute = (
                self._fq(current_k, self.obs_present_key),
                self._fq(current_v, self.obs_present_value),
            )
        else:
            past = self._read_cache_key_value(past)
            read_k, read_v = past
            # Affine compute converts both operands to C before concat. Sequence
            # MX blocks can cross past/current, so assemble S values and raw
            # current V first and apply the full-V boundary once below.
            if self._mode is Mode.QUANT:
                read_k = self._fq(read_k, self.obs_present_key)
                current_k = self._fq(current_k, self.obs_present_key)
                if not isinstance(self.obs_present_value, MXObserver):
                    read_v = self._fq(read_v, self.obs_present_value)
                    current_v = self._fq(current_v, self.obs_present_value)
            compute = (
                self._fq(torch.cat((read_k, current_k), dim=-2), self.obs_present_key),
                self._fq(
                    torch.cat((read_v, current_v), dim=-2), self.obs_present_value
                ),
            )
            storage = (
                torch.cat((past[0], new_k), dim=-2),
                torch.cat((past[1], new_v), dim=-2),
            )
        if (
            use_cache
            and past_key_value is not None
            and not isinstance(past_key_value, tuple)
        ):
            # HF storage receives only the producer's new_* delta. In particular,
            # never update it with compute-grid tensors or post-concat fake quant.
            past_key_value.update(new_k, new_v, self.layer_idx)
            cache_object_out = past_key_value
        return compute, storage, cache_object_out

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
        # Keep both logits fake-quant boundaries. Only the identity scale
        # tensor and its multiplication disappear; no Q/K weights are fused.
        if self.scaling == 1.0:
            return self._fq(logits, self.obs_logits)
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
        static_export: bool = False,
        **kwargs,
    ):
        """Run Gemma4 text attention with selectable execution layout.

        Args:
            hidden_states: Input tensor shaped ``(B, S, hidden_size)``.
            position_embeddings: Tuple ``(cos, sin)`` shaped ``(B, S, head_dim)``.
                Sine must already follow ``attn_options.rope``. This wrapper
                does not transform tables or re-rotate shared/cached keys.
            attention_mask: Optional additive or keep mask broadcastable to
                attention logits.
            shared_key_value: Optional full K/V tensors for shared-KV layers.
            past_key_value: Optional tuple or HF-like cache object for non-shared
                layers. Tuple caches are shaped ``(B, num_kv_heads, K, H)``.
            use_cache: Whether to update an HF cache and append cache data to
                the output tuple. Existing cache tensors can be read without
                mutation when this is false.
            cache_output_mode: ``"delta"`` returns newly projected K/V;
                ``"present"`` or ``"full"`` returns full K/V or the updated
                cache object.
            static_export: Enforce the configured static capacity when an
                export adapter runs eagerly for deployment parity checks.

        Returns:
            ``(attn_output, attn_weights)`` and an optional third cache item when
            ``use_cache=True``. Shared-KV layers return ``cache=None`` because
            they do not own K/V projection weights.
        """
        self._validate_cache_observers()
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
            key_states, value_states = self._read_cache_key_value(
                (
                    key_states.to(device=query_states.device, dtype=query_states.dtype),
                    value_states.to(
                        device=query_states.device, dtype=query_states.dtype
                    ),
                )
            )
            key_states, value_states = self._cache_to_compute(
                (key_states, value_states)
            )
            new_key_value = None
            present_key_value: Optional[LayerKV] = (key_states, value_states)
        else:
            current_key_value = self._project_current_key_value(
                hidden, hidden_shape, cos, sin
            )
            new_key_value = (
                self._fq(current_key_value[0], self.obs_new_k),
                self._fq(current_key_value[1], self.obs_new_v),
            )
            (
                present_key_value,
                storage_key_value,
                cache_object_out,
            ) = self._build_present_key_value(
                new_key_value=new_key_value,
                current_key_value=current_key_value,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            key_states, value_states = present_key_value
            if self.store_full_length_kv and shared_kv_states is not None:
                shared_kv_states[self.layer_type] = storage_key_value

        attn_mask = self._build_attention_mask(
            attention_mask=attention_mask,
            q_len=query_states.size(-2),
            k_len=key_states.size(-2),
            device=query_states.device,
            static_export=static_export,
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
                    present_key_value=(
                        storage_key_value if new_key_value is not None else None
                    ),
                    cache_object_out=cache_object_out,
                ),
            )
        return attn_output, attn_weights

    def as_export_module(
        self,
        mode: ExportMode = "prefill",
        *,
        return_kv: bool = True,
        cache_source: Optional["QuantGemma4TextAttention"] = None,
    ) -> nn.Module:
        """Return a static export adapter for the requested attention mode."""
        if cache_source is not None:
            self.bind_cache_source(cache_source)
        self.validate_export_cache(mode)
        if mode == "prefill":
            return Gemma4TextAttentionPrefillExportAdapter(
                self,
                return_kv=return_kv,
            )
        if mode == "decode":
            return Gemma4TextAttentionDecodeExportAdapter(
                self,
                return_kv=return_kv,
            )
        raise ValueError(f"Unsupported Gemma4 export mode: {mode!r}")

    def _all_observers(self) -> Iterable:
        """Return only active observers, retaining dormant module attributes."""
        self._validate_cache_observers()
        common = [
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
            self.obs_present_key,
            self.obs_present_value,
            self.obs_attn_mask,
            self.obs_logits_raw,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
        ]
        if self.attn_options.rope == "hf":
            common.append(self.obs_q_neg)
        if self.scaling != 1.0:
            common.append(self.obs_scale)
        if not self.is_kv_shared_layer:
            common.extend(
                [
                    self.obs_k_x1,
                    self.obs_k_x2,
                    self.obs_k_cat,
                    self.obs_k_cos,
                    self.obs_k_sin,
                    self.obs_k_rot,
                    self.obs_new_k,
                    self.obs_new_v,
                ]
            )
            if self.attn_options.rope == "hf":
                common.append(self.obs_k_neg)
        if getattr(self, "_external_float_cache", False):
            common.extend((self.obs_cache_storage_key, self.obs_cache_storage_value))
        return tuple(common)

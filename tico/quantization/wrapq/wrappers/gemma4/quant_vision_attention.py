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

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionAttention")
class QuantGemma4VisionAttention(QuantModuleBase):
    """PTQ wrapper for Gemma4 E2B vision attention.

    The wrapper mirrors Hugging Face ``Gemma4VisionAttention`` while keeping
    projection, RMSNorm, multidimensional RoPE, mask addition, softmax, and the
    output projection explicit. This makes the module suitable for activation
    observation, fake-quantized PTQ simulation, and later static-shape export.

    Dynamic image preprocessing and mask construction should stay outside this
    wrapper. The wrapper expects already static tensors, especially a fixed patch
    sequence length and fixed 2-D pixel position ids.
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_attn
        self.config = fp_attn.config
        self.layer_idx = fp_attn.layer_idx
        self.layer_type = getattr(fp_attn, "layer_type", None)
        self.head_dim = int(fp_attn.head_dim)
        self.num_heads = int(
            getattr(fp_attn, "num_heads", self.config.num_attention_heads)
        )
        self.num_key_value_heads = int(
            getattr(
                fp_attn,
                "num_key_value_heads",
                self.config.num_key_value_heads,
            )
        )
        self.num_key_value_groups = int(fp_attn.num_key_value_groups)
        self.scaling = float(getattr(fp_attn, "scaling", 1.0))
        self.attention_dropout = float(
            getattr(fp_attn, "attention_dropout", 0.0) or 0.0
        )
        self.is_causal = bool(getattr(fp_attn, "is_causal", False))

        self.q_proj = PTQWrapper(
            fp_attn.q_proj,
            qcfg=qcfg.child("q_proj") if qcfg else None,
            fp_name=join_name(fp_name, "q_proj"),
        )
        self.k_proj = PTQWrapper(
            fp_attn.k_proj,
            qcfg=qcfg.child("k_proj") if qcfg else None,
            fp_name=join_name(fp_name, "k_proj"),
        )
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
        self.q_norm = PTQWrapper(
            fp_attn.q_norm,
            qcfg=qcfg.child("q_norm") if qcfg else None,
            fp_name=join_name(fp_name, "q_norm"),
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

        mk = self._make_obs
        self.obs_hidden = mk("hidden")

        # RoPE tables.
        self.obs_cos = mk("cos")
        self.obs_sin = mk("sin")

        # rotate_half sub-steps for Q.
        self.obs_q_x1 = mk("q_x1")
        self.obs_q_x2 = mk("q_x2")
        self.obs_q_neg = mk("q_neg")
        self.obs_q_cat = mk("q_cat")

        # rotate_half sub-steps for K.
        self.obs_k_x1 = mk("k_x1")
        self.obs_k_x2 = mk("k_x2")
        self.obs_k_neg = mk("k_neg")
        self.obs_k_cat = mk("k_cat")

        # RoPE combine points.
        self.obs_q_cos = mk("q_cos")
        self.obs_q_sin = mk("q_sin")
        self.obs_q_rot = mk("q_rot")
        self.obs_k_cos = mk("k_cos")
        self.obs_k_sin = mk("k_sin")
        self.obs_k_rot = mk("k_rot")

        # Masking and attention math.
        self.obs_attn_mask = mk("attn_mask")
        self.obs_logits_raw = mk("logits_raw")
        self.obs_scale = mk("scale")
        self.obs_logits = mk("logits")
        self.obs_mask_add = mk("mask_add")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")
        self.obs_attn_weights = mk("attn_weights")
        self.obs_attn_out_h = mk("attn_out_h")

    @staticmethod
    def _expand_rope_table(table: torch.Tensor) -> torch.Tensor:
        """Return a RoPE table shaped as ``(B, S, 1, H)`` for vision attention."""
        if table.dim() == 2:
            table = table.unsqueeze(0)
        if table.dim() == 3:
            table = table.unsqueeze(2)
        if table.dim() != 4:
            raise RuntimeError(
                "Gemma4 vision RoPE table must have rank 2, 3, or 4, "
                f"got shape={tuple(table.shape)}."
            )
        return table

    @staticmethod
    def _rope_ndim(position_ids: Optional[torch.Tensor]) -> int:
        """Return the number of spatial RoPE dimensions for pixel position ids."""
        if position_ids is None:
            # Vision attention is 2-D by construction. This fallback keeps export
            # adapters usable when they precompute cos/sin and omit position ids.
            return 2
        if position_ids.dim() < 3:
            raise RuntimeError(
                "Gemma4 vision position_ids must be shaped as ``(B, S, ndim)``, "
                f"got shape={tuple(position_ids.shape)}."
            )
        return int(position_ids.shape[-1])

    def _rot(
        self,
        tensor: torch.Tensor,
        obs_x1,
        obs_x2,
        obs_neg,
        obs_cat,
    ) -> torch.Tensor:
        """Apply Gemma4/Hugging Face ``rotate_half`` as ``[-x2, x1]``."""
        x1, x2 = torch.chunk(tensor, 2, dim=-1)
        x1 = self._fq(x1, obs_x1)
        x2 = self._fq(x2, obs_x2)
        neg_x2 = self._fq(-x2, obs_neg)
        return self._fq(torch.cat((neg_x2, x1), dim=-1), obs_cat)

    def _apply_rope_part(
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
        """Apply one spatial RoPE partition to Q or K before head transposition."""
        half = self._rot(tensor, obs_x1, obs_x2, obs_neg, obs_cat)
        cos_part = self._fq(tensor * cos, obs_cos)
        sin_part = self._fq(half * sin, obs_sin)
        return self._fq(cos_part + sin_part, obs_rot)

    def _apply_multidimensional_rope(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        obs_x1,
        obs_x2,
        obs_neg,
        obs_cat,
        obs_cos,
        obs_sin,
        obs_rot,
    ) -> torch.Tensor:
        """Apply Gemma4 multidimensional RoPE to Q or K.

        ``Gemma4VisionRotaryEmbedding`` builds cos/sin tables by concatenating
        one RoPE table per spatial dimension. The attention input is split using
        the same partitioning rule and each part receives normal 1-D RoPE.
        """
        ndim = self._rope_ndim(position_ids)
        num_input_channels = int(tensor.shape[-1])
        num_rotated_channels_per_dim = 2 * (num_input_channels // (2 * ndim))

        if num_rotated_channels_per_dim <= 0:
            raise ValueError(
                "Invalid Gemma4 vision RoPE configuration: "
                "num_rotated_channels_per_dim must be positive, got "
                f"{num_rotated_channels_per_dim} "
                f"(head_dim={num_input_channels}, ndim={ndim})."
            )
        if num_rotated_channels_per_dim * ndim != num_input_channels:
            raise ValueError(
                "Gemma4 vision head_dim must be divisible by ``2 * ndim`` for "
                "multidimensional RoPE: "
                f"head_dim={num_input_channels}, ndim={ndim}."
            )

        split_sizes = [num_rotated_channels_per_dim] * ndim
        cos = self._expand_rope_table(cos)
        sin = self._expand_rope_table(sin)

        tensor_parts = torch.split(tensor, split_sizes, dim=-1)
        cos_parts = torch.split(cos, split_sizes, dim=-1)
        sin_parts = torch.split(sin, split_sizes, dim=-1)

        rotated_parts = [
            self._apply_rope_part(
                tensor_parts[idx],
                cos_parts[idx],
                sin_parts[idx],
                obs_x1,
                obs_x2,
                obs_neg,
                obs_cat,
                obs_cos,
                obs_sin,
                obs_rot,
            )
            for idx in range(ndim)
        ]
        return torch.cat(rotated_parts, dim=-1)

    @staticmethod
    def _normalize_attention_mask_shape(
        mask: torch.Tensor,
        *,
        q_len: int,
        k_len: int,
    ) -> torch.Tensor:
        """Normalize a vision attention mask to broadcast over ``(B, heads, Q, K)``.

        Supported input shapes are ``(B, K)``, ``(B, Q, K)``, and
        ``(B, 1, Q, K)``. Longer preallocated masks are sliced to the active
        query and key lengths.
        """
        if mask.dim() not in (2, 3, 4):
            raise RuntimeError(
                "Unsupported attention_mask rank for Gemma4 vision attention: "
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
                "Per-head attention masks are not supported by the Gemma4 vision "
                "attention wrapper. Expected mask shape ``(B, 1, Q, K)``, "
                f"got shape={tuple(mask.shape)}."
            )
        return mask

    def _build_attention_mask(
        self,
        *,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an additive attention mask for vision attention logits."""
        if attention_mask is None:
            mask = torch.zeros(
                (batch_size, 1, q_len, k_len),
                device=device,
                dtype=dtype,
            )
            return self._fq(mask, self.obs_attn_mask)

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
                dtype=dtype,
                device=device,
            )
            # Use ``== False`` instead of ``~`` to avoid ``aten::bitwise_not``
            # which is not supported by the Circle conversion pipeline.
            additive = additive.masked_fill(
                keep_mask == False,
                float(self.qcfg.attention_mask_fill_value),
            )
            return self._fq(additive, self.obs_attn_mask)

        if torch.is_floating_point(attention_mask):
            additive = self._normalize_attention_mask_shape(
                attention_mask,
                q_len=q_len,
                k_len=k_len,
            ).to(dtype=dtype)
            return self._fq(additive, self.obs_attn_mask)

        raise RuntimeError(
            "Unsupported attention_mask dtype for Gemma4 vision attention: "
            f"dtype={attention_mask.dtype}."
        )

    def _attention_forward(
        self,
        *,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run grouped-query vision attention without materializing repeated K/V."""
        batch_size, num_heads, q_len, _ = query_states.shape
        num_kv_heads = key_states.size(1)
        kv_rep = self.num_key_value_groups

        attn_weights_parts: list[torch.Tensor] = []
        attn_out_parts: list[torch.Tensor] = []

        for kv_idx in range(num_kv_heads):
            key_i = key_states[:, kv_idx : kv_idx + 1, :, :]
            value_i = value_states[:, kv_idx : kv_idx + 1, :, :]
            head_start = kv_idx * kv_rep
            head_end = min(head_start + kv_rep, num_heads)
            query_i = query_states[:, head_start:head_end, :, :]
            if query_i.size(1) == 0:
                continue

            logits_i = query_i @ key_i.transpose(-2, -1)
            logits_i = self._fq(logits_i, self.obs_logits_raw)
            scale = self._fq(
                torch.tensor(
                    self.scaling,
                    device=logits_i.device,
                    dtype=logits_i.dtype,
                ),
                self.obs_scale,
            )
            logits_i = self._fq(logits_i * scale, self.obs_logits)
            logits_i = self._fq(logits_i + attention_mask, self.obs_mask_add)

            attn_i = torch.softmax(logits_i, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
            if self.training and self.attention_dropout > 0.0:
                attn_i = F.dropout(attn_i, p=self.attention_dropout, training=True)
            attn_i = self._fq(attn_i, self.obs_softmax)

            out_i = self._fq(attn_i @ value_i, self.obs_attn_out)
            attn_weights_parts.append(attn_i)
            attn_out_parts.append(out_i)

        if not attn_out_parts:
            raise RuntimeError("Gemma4 vision attention produced no head outputs.")

        attn_weights = self._fq(
            torch.cat(attn_weights_parts, dim=1),
            self.obs_attn_weights,
        )
        attn_out_h = self._fq(
            torch.cat(attn_out_parts, dim=1),
            self.obs_attn_out_h,
        )
        attn_output = attn_out_h.transpose(1, 2).reshape(batch_size, q_len, -1)
        return self.o_proj(attn_output), attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run Gemma4 vision attention with explicit PTQ observation points.

        Args:
            hidden_states: Input patch states shaped ``(B, S, hidden_size)``.
            position_embeddings: Tuple ``(cos, sin)`` produced by
                ``Gemma4VisionRotaryEmbedding``. Each table is expected to be
                shaped ``(B, S, head_dim)``.
            attention_mask: Optional additive or keep mask broadcastable to
                ``(B, heads, S, S)``.
            position_ids: Pixel position ids shaped ``(B, S, 2)``. Static export
                adapters may omit this argument when they always use 2-D RoPE.

        Returns:
            Tuple ``(attn_output, attn_weights)`` following the Hugging Face
            vision attention contract.
        """
        if position_embeddings is None:
            raise RuntimeError(
                "Gemma4 vision attention requires ``position_embeddings=(cos, sin)``."
            )

        hidden = self._fq(hidden_states, self.obs_hidden)
        batch_size, seq_len, _ = hidden.shape
        query_shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        kv_shape = (batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        query_states = self.q_proj(hidden).view(query_shape)
        query_states = self.q_norm(query_states)
        query_states = self._apply_multidimensional_rope(
            query_states,
            cos,
            sin,
            position_ids,
            self.obs_q_x1,
            self.obs_q_x2,
            self.obs_q_neg,
            self.obs_q_cat,
            self.obs_q_cos,
            self.obs_q_sin,
            self.obs_q_rot,
        )
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden).view(kv_shape)
        key_states = self.k_norm(key_states)
        key_states = self._apply_multidimensional_rope(
            key_states,
            cos,
            sin,
            position_ids,
            self.obs_k_x1,
            self.obs_k_x2,
            self.obs_k_neg,
            self.obs_k_cat,
            self.obs_k_cos,
            self.obs_k_sin,
            self.obs_k_rot,
        )
        key_states = key_states.transpose(1, 2)

        value_states = self.v_proj(hidden).view(kv_shape)
        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        attn_mask = self._build_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            q_len=query_states.size(-2),
            k_len=key_states.size(-2),
            device=query_states.device,
            dtype=query_states.dtype,
        )

        return self._attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attn_mask,
        )

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
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
            self.obs_attn_mask,
            self.obs_logits_raw,
            self.obs_scale,
            self.obs_logits,
            self.obs_mask_add,
            self.obs_softmax,
            self.obs_attn_out,
            self.obs_attn_weights,
            self.obs_attn_out_h,
        )

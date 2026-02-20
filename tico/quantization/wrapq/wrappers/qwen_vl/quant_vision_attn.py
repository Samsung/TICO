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
from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionAttention",
)
class QuantQwen3VLVisionAttention(QuantModuleBase):
    def __init__(
        self,
        attn_fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        cfg = attn_fp.config
        self.config = cfg
        self.num_heads = attn_fp.num_heads
        self.head_dim = attn_fp.head_dim

        # ---- Wrap q k v o projections via PTQWrapper ---------------
        qkv_cfg = qcfg.child("qkv") if qcfg else None
        proj_cfg = qcfg.child("proj") if qcfg else None

        assert hasattr(attn_fp, "qkv") and isinstance(attn_fp.qkv, torch.nn.Module)
        assert hasattr(attn_fp, "proj") and isinstance(attn_fp.proj, torch.nn.Module)

        self.qkv = PTQWrapper(
            copy.deepcopy(attn_fp.qkv), qcfg=qkv_cfg, fp_name=f"{fp_name}.qkv_cfg"
        )
        self.proj = PTQWrapper(attn_fp.proj, qcfg=proj_cfg, fp_name=f"{fp_name}.proj")

        # Let's fold constant scale (1/√d) to k_proj
        scale_t = torch.tensor(
            float(getattr(attn_fp, "scaling", self.head_dim**-0.5))
        )
        with torch.no_grad():
            lin = self.qkv.wrapped.module
            k_offset = lin.weight.shape[0] // 3
            k_size = lin.weight.shape[0] // 3
            lin.weight[k_offset : k_offset + k_size, :].mul_(scale_t)
            if lin.bias is not None:
                lin.bias[k_offset : k_offset + k_size].mul_(scale_t)

        mk = self._make_obs
        self.obs_hidden = mk("hidden")
        self.obs_scaling = mk("scaling")
        self.obs_mul_logits_scale = mk("mul_logits_scale")
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

        # Masking & attention math
        self.obs_logits = mk("logits")
        self.obs_softmax = mk("softmax")
        self.obs_attn_out = mk("attn_out")

    def _rot(self, t, o_x1, o_x2, o_neg, o_cat):
        x1, x2 = torch.chunk(t, 2, dim=-1)
        x1 = self._fq(x1, o_x1)
        x2 = self._fq(x2, o_x2)
        x2n = self._fq(-x2, o_neg)
        return self._fq(torch.cat((x2n, x1), -1), o_cat)

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        q_rot = self._rot(
            q, self.obs_q_x1, self.obs_q_x2, self.obs_q_neg, self.obs_q_cat
        )
        q_cos = self._fq(q * cos, self.obs_q_cos)
        q_sin = self._fq(q_rot * sin, self.obs_q_sin)
        q_embed = self._fq(q_cos + q_sin, self.obs_q_rot)

        k_rot = self._rot(
            k, self.obs_k_x1, self.obs_k_x2, self.obs_k_neg, self.obs_k_cat
        )
        k_cos = self._fq(k * cos, self.obs_k_cos)
        k_sin = self._fq(k_rot * sin, self.obs_k_sin)
        k_embed = self._fq(k_cos + k_sin, self.obs_k_rot)

        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        hidden_states = self._fq(hidden_states, self.obs_hidden)

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings  # type: ignore[misc]
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)
        query_states, key_states = self._apply_rope(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2)
            for tensor in (query_states, key_states, value_states)
        ]
        assert (
            len(splits)
            == 3  # just a single image, so (q, k, v) = (splits[0], splits[1], splits[2])
            and len(splits[0]) == 1
            and len(splits[1]) == 1
            and len(splits[2]) == 1
        )  # so we may proceed without splitted attention
        rep = query_states.size(-3) // key_states.size(-3)
        assert rep == 1  # currently no GQA is supported

        attn_outputs = []
        for n_head in range(self.num_heads):
            k_i = key_states[:, n_head : n_head + 1, :, :]
            v_i = value_states[:, n_head : n_head + 1, :, :]
            q_i = query_states[:, n_head : n_head + 1, :, :]
            logits_i = self._fq(q_i @ k_i.transpose(-2, -1), self.obs_logits)
            # softmax
            attn_i = torch.softmax(logits_i, -1, dtype=torch.float32).to(q_i.dtype)
            attn_i = self._fq(attn_i, self.obs_softmax)
            out_i = self._fq(attn_i @ v_i, self.obs_attn_out)
            attn_outputs.append(out_i)

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()

        attn_output = self.proj(attn_output)
        return attn_output

    def _all_observers(self) -> Iterable:
        yield from (
            self.obs_hidden,
            self.obs_scaling,
            self.obs_mul_logits_scale,
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
            self.obs_logits,
            self.obs_softmax,
            self.obs_attn_out,
        )
        for m in (self.qkv, self.proj):
            yield from m._all_observers()

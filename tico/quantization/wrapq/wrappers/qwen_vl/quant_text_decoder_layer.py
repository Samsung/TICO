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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextDecoderLayer")
class QuantQwen3VLTextDecoderLayer(QuantModuleBase):
    """
    Quant-aware drop-in replacement for HF `Qwen3VLTextDecoderLayer`.

    Attention & MLP blocks are replaced by their quantized counterparts.
    A "static" causal mask and RoPE templates are pre-built in `__init__`
    to avoid dynamic ops inside `forward`.

    Notes
    -----
    - Prefill-only: `use_cache` is not supported because
      `QuantQwen3VLTextAttention` does not return KV cache.
    - `position_embeddings` can be injected from the parent model-level wrapper
      (shared across all layers); if omitted the layer uses its own pre-computed
      templates as fallback.
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        return_type: Optional[str] = None,
    ):
        self.return_type = return_type
        if self.return_type is None:
            import transformers

            v = tuple(map(int, transformers.__version__.split(".")[:2]))
            self.return_type = "tensor" if v >= (4, 54) else "tuple"
        assert self.return_type is not None

        super().__init__(qcfg, fp_name=fp_name)

        # ----- child configs ------------------------------------------------
        attn_cfg = qcfg.child("self_attn") if qcfg else None
        mlp_cfg = qcfg.child("mlp") if qcfg else None
        input_ln_cfg = qcfg.child("input_layernorm") if qcfg else None
        post_attn_ln_cfg = qcfg.child("post_attention_layernorm") if qcfg else None

        # ----- assertions ---------------------------------------------------
        assert hasattr(fp_layer, "self_attn") and isinstance(
            fp_layer.self_attn, nn.Module
        )
        assert hasattr(fp_layer, "mlp") and isinstance(fp_layer.mlp, nn.Module)
        assert hasattr(fp_layer, "input_layernorm") and isinstance(
            fp_layer.input_layernorm, nn.Module
        )
        assert hasattr(fp_layer, "post_attention_layernorm") and isinstance(
            fp_layer.post_attention_layernorm, nn.Module
        )

        # ----- wrap children ------------------------------------------------
        self.self_attn = PTQWrapper(
            fp_layer.self_attn, qcfg=attn_cfg, fp_name=f"{fp_name}.self_attn"
        )
        self.mlp = PTQWrapper(fp_layer.mlp, qcfg=mlp_cfg, fp_name=f"{fp_name}.mlp")
        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=input_ln_cfg,
            fp_name=f"{fp_name}.input_layernorm",
        )
        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=post_attn_ln_cfg,
            fp_name=f"{fp_name}.post_attention_layernorm",
        )

        # ----- local observers ----------------------------------------------
        self.obs_mlp_residual_out = self._make_obs("mlp_residual_out")
        self.obs_causal_mask = self._make_obs("causal_mask")
        self.obs_cos = self._make_obs("cos")
        self.obs_sin = self._make_obs("sin")

        # ----- static buffers: causal mask template -------------------------
        cfg = fp_layer.self_attn.config
        assert hasattr(cfg, "max_position_embeddings")
        max_seq = cfg.max_position_embeddings
        mask = torch.full((1, 1, max_seq, max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

        # ----- static buffers: RoPE templates --------------------------------
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )

        rotary = getattr(fp_layer, "rotary_emb", None)
        if rotary is not None and hasattr(rotary, "inv_freq"):
            inv_freq = rotary.inv_freq.detach().float()
            attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))
        else:
            base = float(getattr(cfg, "rope_theta", 10000.0))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            attn_scaling = 1.0

        pos = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_t = emb.cos() * attn_scaling
        sin_t = emb.sin() * attn_scaling
        half_dim = head_dim // 2
        sin_t[..., :half_dim] = -sin_t[..., :half_dim]
        cos_t = cos_t.unsqueeze(0)  # [1, max_seq, head_dim]
        sin_t = sin_t.unsqueeze(0)  # [1, max_seq, head_dim]

        self.register_buffer("rope_cos_template", cos_t, persistent=False)
        self.register_buffer("rope_sin_template", sin_t, persistent=False)

    def _slice_causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        assert isinstance(self.causal_mask_template, torch.Tensor)
        return self.causal_mask_template[..., :seq_len, :seq_len].to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor] | torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Build causal mask if not provided (or provided as bool)
        if attention_mask is None or attention_mask.dtype == torch.bool:
            L = hidden_states.size(1)
            attention_mask = self._slice_causal(L, hidden_states.device)
            attention_mask = attention_mask.squeeze(0)
            attention_mask = self._fq(attention_mask, self.obs_causal_mask)

        # Build position embeddings if not provided
        if position_embeddings is None:
            position_embeddings = (
                self.rope_cos_template.to(
                    dtype=hidden_states.dtype, device=hidden_states.device
                ),
                self.rope_sin_template.to(
                    dtype=hidden_states.dtype, device=hidden_states.device
                ),
            )
            cos, sin = position_embeddings
            position_embeddings = (
                self._fq(cos, self.obs_cos),
                self._fq(sin, self.obs_sin),
            )

        # Attention block
        # QuantQwen3VLTextAttention returns (out, attn_weights)
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_out[0]

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_mlp_residual_out)

        if self.return_type == "tuple":
            return (hidden_states,)
        elif self.return_type == "tensor":
            return hidden_states
        else:
            raise RuntimeError(f"Invalid return_type: {self.return_type!r}")

    def _all_observers(self):
        yield from (self.obs_causal_mask, self.obs_cos, self.obs_sin)
        yield from self.self_attn._all_observers()
        yield from self.mlp._all_observers()
        yield from self.input_layernorm._all_observers()
        yield from self.post_attention_layernorm._all_observers()
        yield self.obs_mlp_residual_out

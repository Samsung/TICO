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

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextDecoderLayer",
)
class QuantQwen3VLTextDecoderLayer(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLTextDecoderLayer module.

    This is a Transformer decoder layer for text processing, containing:
    - 2 RMSNorm layers (pre-norm architecture)
    - 1 Self-Attention module
    - 1 MLP (Feed-Forward Network)
    - 2 Residual connections
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        assert hasattr(fp_layer, "self_attn")
        assert hasattr(fp_layer, "mlp")
        assert hasattr(fp_layer, "input_layernorm")
        assert hasattr(fp_layer, "post_attention_layernorm")

        # --- Wrap submodules via PTQWrapper ----------------------------------
        self_attn_cfg = qcfg.child("self_attn") if qcfg else None
        mlp_cfg = qcfg.child("mlp") if qcfg else None
        input_layernorm_cfg = qcfg.child("input_layernorm") if qcfg else None
        post_attention_layernorm_cfg = (
            qcfg.child("post_attention_layernorm") if qcfg else None
        )

        self.self_attn = PTQWrapper(
            fp_layer.self_attn,
            qcfg=self_attn_cfg,
            fp_name=join_name(fp_name, "self_attn"),
        )

        self.mlp = PTQWrapper(
            fp_layer.mlp,
            qcfg=mlp_cfg,
            fp_name=join_name(fp_name, "mlp"),
        )

        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=input_layernorm_cfg,
            fp_name=join_name(fp_name, "input_layernorm"),
        )

        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=post_attention_layernorm_cfg,
            fp_name=join_name(fp_name, "post_attention_layernorm"),
        )

        # --- Observers for residual additions ----------------------------------
        mk = self._make_obs
        self.obs_act_in = mk("act_in")
        self.obs_post_attn = mk("post_attn")
        self.obs_act_out = mk("act_out")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: torch.Tensor | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_embeddings: (cos, sin) position embeddings tuple
            attention_mask: Attention mask tensor (optional)
            position_ids: Position indices tensor (optional)
            past_key_values: Cached key-value pairs for attention (optional)
            use_cache: Whether to use key-value caching (optional)
            cache_position: Cache position indices (optional)
            **kwargs: Additional keyword arguments

        Returns:
            Transformed hidden states of shape (batch_size, seq_len, hidden_size)
        """
        # Quantize input activation
        hidden_states = self._fq(hidden_states, self.obs_act_in)

        # Save input for residual connection
        residual = hidden_states

        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Post-attention residual connection
        hidden_states = hidden_states + residual
        hidden_states = self._fq(hidden_states, self.obs_post_attn)

        # Save for MLP residual connection
        residual = hidden_states

        # Pre-MLP normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Feed-Forward Network (MLP)
        hidden_states = self.mlp(hidden_states)

        # Post-MLP residual connection
        hidden_states = hidden_states + residual

        # Quantize output activation
        hidden_states = self._fq(hidden_states, self.obs_act_out)

        return hidden_states

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module."""
        # Local observers for residual connections
        yield from (
            self.obs_act_in,
            self.obs_post_attn,
            self.obs_act_out,
        )

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
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionBlock",
)
class QuantQwen3VLVisionBlock(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLVisionBlock module.

    This is a Transformer encoder block for vision processing, containing:
    - 2 LayerNorm layers (pre-norm architecture)
    - 1 Self-Attention module
    - 1 MLP (Feed-Forward Network)
    - 2 Residual connections
    """

    def __init__(
        self,
        fp_block: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        assert hasattr(fp_block, "norm1") and isinstance(fp_block.norm1, nn.LayerNorm)
        assert hasattr(fp_block, "norm2") and isinstance(fp_block.norm2, nn.LayerNorm)
        assert hasattr(fp_block, "attn")
        assert hasattr(fp_block, "mlp")

        # --- Wrap submodules via PTQWrapper ----------------------------------
        norm1_cfg = qcfg.child("norm1") if qcfg else None
        norm2_cfg = qcfg.child("norm2") if qcfg else None
        attn_cfg = qcfg.child("attn") if qcfg else None
        mlp_cfg = qcfg.child("mlp") if qcfg else None

        self.norm1 = PTQWrapper(
            fp_block.norm1,
            qcfg=norm1_cfg,
            fp_name=f"{fp_name}.norm1",
        )

        self.norm2 = PTQWrapper(
            fp_block.norm2,
            qcfg=norm2_cfg,
            fp_name=f"{fp_name}.norm2",
        )

        self.attn = PTQWrapper(
            fp_block.attn,
            qcfg=attn_cfg,
            fp_name=f"{fp_name}.attn",
        )

        self.mlp = PTQWrapper(
            fp_block.mlp,
            qcfg=mlp_cfg,
            fp_name=f"{fp_name}.mlp",
        )

        # --- Observers for residual additions ----------------------------------
        mk = self._make_obs
        self.obs_input = mk("input")
        self.obs_post_attn = mk("post_attn")
        self.obs_output = mk("output")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        Args:
            hidden_states: Input tensor of shape (num_patches, hidden_size)
            cu_seqlens: Cumulative sequence lengths
            rotary_pos_emb: Rotary positional embeddings (optional)
            position_embeddings: (cos, sin) position embeddings (optional)
            **kwargs: Additional keyword arguments

        Returns:
            Transformed features of shape (num_patches, hidden_size)
        """
        # Quantize input activation
        hidden_states = self._fq(hidden_states, self.obs_input)

        # Save input for residual connection
        residual = hidden_states

        # Pre-attention normalization
        hidden_states = self.norm1(hidden_states)

        # Self-attention
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Post-attention residual connection
        hidden_states = hidden_states + residual
        hidden_states = self._fq(hidden_states, self.obs_post_attn)

        # Save for MLP residual connection
        residual = hidden_states

        # Pre-MLP normalization
        hidden_states = self.norm2(hidden_states)

        # Feed-Forward Network (MLP)
        hidden_states = self.mlp(hidden_states)

        # Post-MLP residual connection
        hidden_states = hidden_states + residual

        # Quantize output activation
        hidden_states = self._fq(hidden_states, self.obs_output)

        return hidden_states

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module and wrapped submodules."""
        # Local observers for residual connections
        yield from (
            self.obs_input,
            self.obs_post_attn,
            self.obs_output,
        )

        # Observers from wrapped submodules
        for module in (self.norm1, self.norm2, self.attn, self.mlp):
            yield from module.wrapped._all_observers()

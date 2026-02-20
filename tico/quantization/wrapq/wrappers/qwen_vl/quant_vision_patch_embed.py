# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
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
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchEmbed",
)
class QuantQwen3VLVisionPatchEmbed(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLVisionPatchEmbed module.

    This module wraps the Conv3d patch embedding layer that converts raw video
    frames into patch embeddings for the vision transformer.
    """

    def __init__(
        self,
        fp_patch_embed: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        self.patch_size = fp_patch_embed.patch_size
        self.temporal_patch_size = fp_patch_embed.temporal_patch_size
        self.in_channels = fp_patch_embed.in_channels
        self.embed_dim = fp_patch_embed.embed_dim

        assert hasattr(fp_patch_embed, "proj") and isinstance(
            fp_patch_embed.proj, nn.Conv3d
        )

        # Wrap the Conv3d projection layer via PTQWrapper
        # This will use QuantConv3d wrapper (registered in the registry)
        proj_cfg = qcfg.child("proj") if qcfg else None
        self.proj = PTQWrapper(
            fp_patch_embed.proj,
            qcfg=proj_cfg,
            fp_name=f"{fp_name}.proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        Args:
            hidden_states: Input tensor of shape (batch_size, channels, depth, height, width)
                            Raw video frames (RGB: channels=3)

        Returns:
            Patch embeddings of shape (batch_size * T' * H' * W', embed_dim)
            Flattened 2D tensor
        """
        # Reshape input to (B*T*H*W, C, temporal_patch_size, patch_size, patch_size)
        # This flattens batch and spatial dimensions into a single sequence dimension
        hidden = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        # Apply Conv3d patch embedding (quantized via PTQWrapper)
        # Output: (B*T*H*W, embed_dim, 1, 1, 1)
        hidden = self.proj(hidden)

        # Reshape output to (B*T*H*W, embed_dim)
        hidden = hidden.view(-1, self.embed_dim)

        return hidden

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module and wrapped submodules."""
        # Observers from wrapped Conv3d layer
        yield from self.proj.wrapped._all_observers()

# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable or agreed to in writing, software
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
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchMerger",
)
class QuantQwen3VLVisionPatchMerger(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLVisionPatchMerger module.

    This module wraps the patch merger that transforms vision features to
    language model input dimensions through a 2-layer MLP structure.
    """

    def __init__(
        self,
        fp_merger: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        self.hidden_size = fp_merger.hidden_size
        self.use_postshuffle_norm = fp_merger.use_postshuffle_norm

        assert hasattr(fp_merger, "norm") and isinstance(fp_merger.norm, nn.LayerNorm)
        assert hasattr(fp_merger, "linear_fc1") and isinstance(
            fp_merger.linear_fc1, nn.Linear
        )
        assert hasattr(fp_merger, "linear_fc2") and isinstance(
            fp_merger.linear_fc2, nn.Linear
        )
        assert hasattr(fp_merger, "act_fn") and isinstance(fp_merger.act_fn, nn.GELU)

        # --- Wrap submodules via PTQWrapper ----------------------------------
        norm_cfg = qcfg.child("norm") if qcfg else None
        fc1_cfg = qcfg.child("linear_fc1") if qcfg else None
        fc2_cfg = qcfg.child("linear_fc2") if qcfg else None
        act_cfg = qcfg.child("act_fn") if qcfg else None

        self.norm = PTQWrapper(
            fp_merger.norm,
            qcfg=norm_cfg,
            fp_name=f"{fp_name}.norm",
        )

        self.linear_fc1 = PTQWrapper(
            fp_merger.linear_fc1,
            qcfg=fc1_cfg,
            fp_name=f"{fp_name}.linear_fc1",
        )

        self.act_fn = PTQWrapper(
            fp_merger.act_fn,
            qcfg=act_cfg,
            fp_name=f"{fp_name}.act_fn",
        )

        self.linear_fc2 = PTQWrapper(
            fp_merger.linear_fc2,
            qcfg=fc2_cfg,
            fp_name=f"{fp_name}.linear_fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        Args:
            x: Input tensor of shape (num_patches, hidden_size)

        Returns:
            Transformed features of shape (num_patches, out_hidden_size)
        """
        # Apply LayerNorm (with optional reshape based on use_postshuffle_norm)
        if self.use_postshuffle_norm:
            # Reshape to (N, hidden_size) before norm
            x = self.norm(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
        else:
            x = x.view(-1, self.hidden_size)

        # Apply first linear layer
        x = self.linear_fc1(x)

        # Apply GELU activation
        x = self.act_fn(x)

        # Apply second linear layer (projection to language model dimension)
        x = self.linear_fc2(x)

        return x

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module and wrapped submodules."""
        # Observers from wrapped submodules
        for module in (self.norm, self.linear_fc1, self.act_fn, self.linear_fc2):
            yield from module.wrapped._all_observers()

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
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchEmbed",
)
class QuantQwen3VLVisionPatchEmbed(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLVisionPatchEmbed.

    Qwen3-VL receives patches that are already flattened by the processor. Its
    Conv3d projection uses a kernel and stride equal to one complete temporal
    patch, so every patch produces exactly one output vector. The projection is
    therefore equivalent to a Linear layer with the Conv3d weight flattened
    along all non-output dimensions.

    The wrapped Linear always receives a rank-3 tensor with an explicit batch
    dimension of one: ``[1, num_patches, patch_dim]``. All leading input
    dimensions are folded into ``num_patches`` and the public output contract
    remains ``[num_patches, embed_dim]``.
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
        self.patch_dim = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

        if not hasattr(fp_patch_embed, "proj") or not isinstance(
            fp_patch_embed.proj, nn.Conv3d
        ):
            raise TypeError(
                "Qwen3VLVisionPatchEmbed.proj must be an nn.Conv3d instance."
            )

        fp_proj = fp_patch_embed.proj
        self._validate_projection(fp_proj)
        linear_proj = self._linearize_projection(fp_proj)

        # Preserve the existing configuration and observer path:
        # patch_embed.proj.{weight,act_in,act_out}.
        proj_cfg = qcfg.child("proj") if qcfg else None
        self.proj = PTQWrapper(
            linear_proj,
            qcfg=proj_cfg,
            fp_name=join_name(fp_name, "proj"),
        )

    def _validate_projection(self, proj: nn.Conv3d) -> None:
        """Validate the full-patch Conv3d geometry required for linearization."""
        expected_kernel = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        checks = {
            "in_channels": (proj.in_channels, self.in_channels),
            "out_channels": (proj.out_channels, self.embed_dim),
            "kernel_size": (tuple(proj.kernel_size), expected_kernel),
            "stride": (tuple(proj.stride), expected_kernel),
            "padding": (tuple(proj.padding), (0, 0, 0)),
            "dilation": (tuple(proj.dilation), (1, 1, 1)),
            "groups": (proj.groups, 1),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "Qwen3-VL patch projection cannot be linearized: "
                + ", ".join(mismatches)
            )

    @staticmethod
    def _linearize_projection(proj: nn.Conv3d) -> nn.Linear:
        """Create a Linear layer that is numerically equivalent to ``proj``."""
        patch_dim = proj.weight[0].numel()
        linear = nn.Linear(
            in_features=patch_dim,
            out_features=proj.out_channels,
            bias=proj.bias is not None,
            device=proj.weight.device,
            dtype=proj.weight.dtype,
        )

        with torch.no_grad():
            linear.weight.copy_(proj.weight.reshape(proj.out_channels, patch_dim))
            if proj.bias is not None:
                assert linear.bias is not None
                linear.bias.copy_(proj.bias)

        linear.weight.requires_grad_(proj.weight.requires_grad)
        if proj.bias is not None:
            assert linear.bias is not None
            linear.bias.requires_grad_(proj.bias.requires_grad)
        linear.train(proj.training)
        return linear

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project processor-flattened Qwen3-VL patches.

        Args:
            hidden_states: Tensor whose last dimension is ``patch_dim``. The
                normal runtime ABI is ``[1, num_patches, patch_dim]``; a 2-D
                ``[num_patches, patch_dim]`` tensor is also accepted.

        Returns:
            Patch embeddings with shape ``[num_patches, embed_dim]``.
        """
        # Fold every leading dimension into the patch sequence while keeping an
        # explicit batch dimension of one for the NPU-facing Linear operation.
        hidden_states = hidden_states.reshape(1, -1, self.patch_dim)

        # Match the original Hugging Face implementation, which casts patch
        # values to the projection weight dtype before applying the projection.
        target_dtype = (
            self.proj.wrapped.module.weight.dtype  # type: ignore[attr-defined]
        )
        hidden_states = hidden_states.to(dtype=target_dtype)

        hidden_states = self.proj(hidden_states)
        return hidden_states.reshape(-1, self.embed_dim)

    def _all_observers(self) -> Iterable:
        """This wrapper owns no observers directly."""
        return ()

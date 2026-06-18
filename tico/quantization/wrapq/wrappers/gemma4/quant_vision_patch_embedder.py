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
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionPatchEmbedder")
class QuantGemma4VisionPatchEmbedder(QuantModuleBase):
    """PTQ wrapper for Gemma4 vision patch embedding with decomposed forward.

    This wrapper quantizes:
    - position_embedding_table (per-tensor symmetric)
    - Scaled pixel values (input activation)
    - Projected hidden states (intermediate activation)
    - Position embeddings (intermediate activation)
    - Final output (output activation)
    """

    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp

        # Store config attributes
        self.hidden_size = fp.hidden_size
        self.patch_size = fp.patch_size
        self.position_embedding_size = fp.position_embedding_size

        self.input_proj = PTQWrapper(
            fp.input_proj,
            qcfg=qcfg.child("input_proj") if qcfg else None,
            fp_name=join_name(fp_name, "input_proj"),
        )

        # Register position_embedding_table as a buffer
        self.register_buffer(
            "position_embedding_table",
            fp.position_embedding_table.clone(),
            persistent=False,
        )

        self.obs_emb_table = self._make_obs(
            "position_embedding_table",
            dtype=DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
        )

        # Observers for activation tensors (dynamic)
        self.obs_act_in = self._make_obs("act_in")
        self.obs_pixel_values_m_0_5 = self._make_obs("pixel_values_m_0_5")
        self.obs_pixel_values = self._make_obs("pixel_values")
        self.obs_hidden_states = self._make_obs("hidden_states")
        self.obs_position_embeddings = self._make_obs("position_embeddings")
        self.obs_output = self._make_obs("output")

    def enable_calibration(self) -> None:
        """Enable calibration and collect static weight ranges."""
        super().enable_calibration()
        # Collect position_embedding_table statistics
        self.obs_emb_table.collect(self.position_embedding_table)

    def _quant_position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quantized 2D patch position embeddings via embedding lookup.

        Args:
            pixel_position_ids: (batch, num_patches, 2) with x,y indices
            padding_positions: (batch, num_patches) boolean mask

        Returns:
            position_embeddings: (batch, num_patches, hidden_size)
        """
        # Clamp only lower bound for valid indexing
        clamped_positions = pixel_position_ids.clamp(min=0)

        # Quantize position embedding table in QUANT mode
        emb_table = self.position_embedding_table
        if self._mode is Mode.QUANT:
            emb_table = self.obs_emb_table.fake_quant(emb_table)

        # Lookup x and y embeddings
        x_emb = F.embedding(clamped_positions[..., 0], emb_table[0])
        y_emb = F.embedding(clamped_positions[..., 1], emb_table[1])

        # Sum x and y embeddings
        position_embeddings = x_emb + y_emb
        position_embeddings = self._fq(
            position_embeddings, self.obs_position_embeddings
        )

        # Apply padding mask (zero out padding positions)
        # Use zeros_like to avoid torch.export lifting issues with torch.tensor()
        position_embeddings = torch.where(
            padding_positions.unsqueeze(-1),
            torch.zeros_like(position_embeddings),
            position_embeddings,
        )

        return position_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run quantized patch projection and positional embedding addition.

        Args:
            pixel_values: (batch, num_patches, 3 * patch_size^2) raw pixel values
            pixel_position_ids: (batch, num_patches, 2) 2D grid coordinates
            padding_positions: (batch, num_patches) boolean padding mask

        Returns:
            hidden_states: (batch, num_patches, hidden_size) patch embeddings
        """
        pixel_values = self._fq(pixel_values, self.obs_act_in)

        # Step 1: Pixel scaling (no normalization, just centering to [-1, 1])
        pixel_values = pixel_values - 0.5
        pixel_values = self._fq(pixel_values, self.obs_pixel_values_m_0_5)
        pixel_values = 2.0 * pixel_values
        pixel_values = self._fq(pixel_values, self.obs_pixel_values)

        # Apply linear projection (no bias)
        hidden_states = self.input_proj(pixel_values)
        hidden_states = self._fq(hidden_states, self.obs_hidden_states)

        # Step 3: Position embeddings
        position_embeddings = self._quant_position_embeddings(
            pixel_position_ids, padding_positions
        )

        # Step 4: Add position embeddings to hidden states
        hidden_states = hidden_states + position_embeddings

        # Step 5: Quantize output
        return self._fq(hidden_states, self.obs_output)

    def _all_observers(self) -> Iterable:
        """Return all observers owned by this wrapper."""
        return (
            self.obs_emb_table,
            self.obs_act_in,
            self.obs_pixel_values_m_0_5,
            self.obs_pixel_values,
            self.obs_hidden_states,
            self.obs_position_embeddings,
            self.obs_output,
        )

    def as_export_module(self, mode: str = "prefill", **kwargs) -> nn.Module:
        """Return self for export (this wrapper is already exportable)."""
        assert self._mode is Mode.QUANT, "Must be in QUANT mode for export"
        if mode != "prefill":
            raise ValueError(
                f"Unsupported Gemma4 VisionPatchEmbedder export mode: {mode!r}"
            )
        return self

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
from tico.quantization.wrapq.utils.linear_folding import fold_input_affine_into_linear
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionPatchEmbedder")
class QuantGemma4VisionPatchEmbedder(QuantModuleBase):
    """PTQ wrapper for Gemma4 vision patch embedding with folded normalization.

    The wrapper folds ``(pixel_values - 0.5) * 2.0`` into ``input_proj``
    before any observers are created. Calibration and weight quantization
    therefore operate on the final linear parameters, and Circle export does
    not need separate Sub or Mul operators for pixel normalization.

    This wrapper quantizes:
    - position_embedding_table (per-tensor symmetric)
    - Raw pixel values (input activation)
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

        folded_input_proj = fold_input_affine_into_linear(
            fp.input_proj,
            scale=2.0,
            shift=-1.0,
        )
        self.input_proj = PTQWrapper(
            folded_input_proj,
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

        # Observers for activation tensors around the folded projection.
        self.obs_act_in = self._make_obs("act_in")
        self.obs_hidden_states = self._make_obs("hidden_states")
        self.obs_position_embeddings = self._make_obs("position_embeddings")
        self.obs_output = self._make_obs("output")

    def enable_calibration(self) -> None:
        """Enable calibration and collect the static embedding-table range."""
        super().enable_calibration()
        self.obs_emb_table.collect(self.position_embedding_table)

    def _project_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project raw flattened pixel patches with the folded linear layer."""
        pixel_values = self._fq(pixel_values, self.obs_act_in)
        hidden_states = self.input_proj(pixel_values)
        return self._fq(hidden_states, self.obs_hidden_states)

    def _lookup_position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Look up fixed-profile 2D position embeddings before activation PTQ."""
        clamped_positions = pixel_position_ids.clamp(min=0)

        emb_table = self.position_embedding_table
        if self._mode is Mode.QUANT:
            emb_table = self.obs_emb_table.fake_quant(emb_table)

        x_emb = F.embedding(clamped_positions[..., 0], emb_table[0])
        y_emb = F.embedding(clamped_positions[..., 1], emb_table[1])
        return x_emb + y_emb

    @staticmethod
    def _zero_padding_positions(
        position_embeddings: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Zero padding slots in a position-embedding tensor."""
        return torch.where(
            padding_positions.unsqueeze(-1),
            torch.zeros_like(position_embeddings),
            position_embeddings,
        )

    def _finalize_position_embeddings(
        self,
        position_embeddings: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply activation PTQ and zero the baked profile's padding slots."""
        position_embeddings = self._fq(
            position_embeddings,
            self.obs_position_embeddings,
        )
        return self._zero_padding_positions(
            position_embeddings,
            padding_positions,
        )

    def _quant_position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute quantized 2D position embeddings for eager execution."""
        position_embeddings = self._lookup_position_embeddings(pixel_position_ids)
        return self._finalize_position_embeddings(
            position_embeddings,
            padding_positions,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run eager patch projection with runtime position coordinates."""
        hidden_states = self._project_pixel_values(pixel_values)
        position_embeddings = self._quant_position_embeddings(
            pixel_position_ids,
            padding_positions,
        )
        return self._fq(hidden_states + position_embeddings, self.obs_output)

    def forward_export(
        self,
        pixel_values: torch.Tensor,
        *,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run static patch projection with a pre-masked positional template."""
        hidden_states = self._project_pixel_values(pixel_values)
        position_embeddings = self._fq(
            position_embeddings,
            self.obs_position_embeddings,
        )
        return self._fq(hidden_states + position_embeddings, self.obs_output)

    def _all_observers(self) -> Iterable:
        """Return all observers owned by this wrapper."""
        return (
            self.obs_emb_table,
            self.obs_act_in,
            self.obs_hidden_states,
            self.obs_position_embeddings,
            self.obs_output,
        )

    def as_export_module(
        self,
        mode: str = "prefill",
        *,
        pixel_position_ids: Optional[torch.Tensor] = None,
        padding_positions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """Build a pixel-values-only adapter for one fixed position profile."""
        if self._mode not in (Mode.NO_QUANT, Mode.QUANT):
            raise RuntimeError(
                "Gemma4 VisionPatchEmbedder export requires NO_QUANT or "
                f"QUANT mode, got {self._mode}."
            )
        if mode != "prefill":
            raise ValueError(
                f"Unsupported Gemma4 VisionPatchEmbedder export mode: {mode!r}"
            )
        if pixel_position_ids is None:
            raise ValueError(
                "Gemma4 VisionPatchEmbedder export requires construction-time "
                "pixel_position_ids."
            )
        if (
            pixel_position_ids.dim() != 3
            or pixel_position_ids.shape[0] != 1
            or pixel_position_ids.shape[-1] != 2
        ):
            raise ValueError(
                "pixel_position_ids must have shape (1, num_patches, 2), "
                f"got {tuple(pixel_position_ids.shape)}."
            )

        expected_padding = (pixel_position_ids == -1).all(dim=-1)
        if padding_positions is None:
            padding_positions = expected_padding
        else:
            padding_positions = padding_positions.to(
                device=expected_padding.device,
                dtype=torch.bool,
            )
            if tuple(padding_positions.shape) != tuple(expected_padding.shape):
                raise ValueError(
                    "padding_positions shape must match pixel_position_ids: "
                    f"expected={tuple(expected_padding.shape)}, "
                    f"actual={tuple(padding_positions.shape)}."
                )
            if not torch.equal(padding_positions, expected_padding):
                raise ValueError(
                    "padding_positions must match the -1 entries in "
                    "pixel_position_ids."
                )

        with torch.no_grad():
            position_embeddings = self._lookup_position_embeddings(pixel_position_ids)
            position_embeddings = self._zero_padding_positions(
                position_embeddings,
                padding_positions,
            ).detach()

        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionPatchEmbedderPrefillExportAdapter,
        )

        return Gemma4VisionPatchEmbedderPrefillExportAdapter(
            self,
            position_embeddings=position_embeddings,
        )

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
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionModel")
class QuantGemma4VisionModel(QuantModuleBase):
    """PTQ wrapper for the Gemma4 vision model.

    This wrapper supports two modes:
    1. Runtime mode (forward): Supports dynamic tensor shapes and conditional branching
       (config.standardize can be True or False). Not exportable.
    2. Export mode (forward_export): Static tensor shapes, no conditional branching.
       Assumes config.standardize=True. Exportable via torch.export.

    The vision model encodes image pixels into visual soft tokens through:
    - Patch embedder: Projects pixels to patch embeddings with position encoding
    - Encoder: Processes embeddings through transformer layers
    - Pooler: Reduces spatial dimension to fixed number of soft tokens
    - Standardization: Applies learned std_bias and std_scale (if enabled)
    """

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config

        # Wrap submodules with PTQWrapper
        # Note: These will use specialized wrappers (QuantGemma4VisionPatchEmbedder,
        # QuantGemma4VisionEncoder, QuantGemma4VisionPooler) if registered
        self.patch_embedder = PTQWrapper(
            fp_model.patch_embedder,
            qcfg=qcfg.child("patch_embedder") if qcfg else None,
            fp_name=join_name(fp_name, "patch_embedder"),
        )
        self.encoder = PTQWrapper(
            fp_model.encoder,
            qcfg=qcfg.child("encoder") if qcfg else None,
            fp_name=join_name(fp_name, "encoder"),
        )
        self.pooler = PTQWrapper(
            fp_model.pooler,
            qcfg=qcfg.child("pooler") if qcfg else None,
            fp_name=join_name(fp_name, "pooler"),
        )

        # Register std_bias and std_scale as buffers if standardize is enabled
        if self.config.standardize:
            self.register_buffer(
                "std_bias",
                fp_model.std_bias.clone()
                if hasattr(fp_model, "std_bias")
                else torch.empty(self.config.hidden_size),
                persistent=False,
            )
            self.register_buffer(
                "std_scale",
                fp_model.std_scale.clone()
                if hasattr(fp_model, "std_scale")
                else torch.empty(self.config.hidden_size),
                persistent=False,
            )

        # Observers
        self.obs_minus_bias = self._make_obs("minus_bias")
        self.obs_strip_padding = self._make_obs("strip_padding")
        self.obs_last_hidden_state = self._make_obs("last_hidden_state")
        self.obs_std_bias = (
            self._make_obs("std_bias") if self.config.standardize else None
        )
        self.obs_std_scale = (
            self._make_obs("std_scale") if self.config.standardize else None
        )

    def enable_calibration(self) -> None:
        """Enable calibration and collect static weight ranges."""
        super().enable_calibration()
        # Collect std_bias and std_scale statistics if standardize is enabled
        if (
            self.config.standardize
            and self.obs_std_bias is not None
            and self.obs_std_scale is not None
        ):
            self.obs_std_bias.collect(self.std_bias)
            self.obs_std_scale.collect(self.std_scale)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        **kwargs,
    ):
        """Run Gemma4 vision model with dynamic shapes and conditional branching.

        This forward method supports:
        - Dynamic output_length computation from pixel_values shape
        - Conditional standardization based on config.standardize

        This method is NOT exportable via torch.export due to dynamic operations.
        Use forward_export() for export.

        Args:
            pixel_values: Image pixels with shape (batch, channels, height, width)
                or list of (1, channels, height, width) for variable sizes.
            pixel_position_ids: Patch positions with shape (batch_size, max_patches, 2).
                Padding patches are indicated by (-1, -1).
            **kwargs: Additional arguments passed to encoder.

        Returns:
            BaseModelOutputWithPast with last_hidden_state containing visual soft tokens.
        """
        from transformers.models.gemma4.modeling_gemma4 import BaseModelOutputWithPast

        # Compute output_length dynamically
        pooling_kernel_size = self.config.pooling_kernel_size
        output_length = pixel_values.shape[-2] // (
            pooling_kernel_size * pooling_kernel_size
        )

        # Create padding mask from pixel_position_ids
        padding_positions = (pixel_position_ids == -1).all(dim=-1)

        # Patch embedder
        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        # Encoder
        output = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,  # encoder expects True=valid
            pixel_position_ids=pixel_position_ids,
            **kwargs,
        )

        # The encoder may return a BaseModelOutputWithPast (HF original) or a
        # plain tensor (QuantGemma4VisionEncoder wrapper).  Handle both cases.
        if isinstance(output, torch.Tensor):
            encoder_hidden = output
        else:
            encoder_hidden = output.last_hidden_state

        # Pooler
        hidden_states, pooler_mask = self.pooler(
            hidden_states=encoder_hidden,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        # Strip padding tokens
        hidden_states = hidden_states[pooler_mask]
        hidden_states = self._fq(hidden_states, self.obs_strip_padding)

        # Standardization (conditional based on config)
        if self.config.standardize:
            std_bias = self.std_bias
            std_scale = self.std_scale
            if self._mode is Mode.QUANT:
                assert self.obs_std_bias is not None
                assert self.obs_std_scale is not None
                std_bias = self.obs_std_bias.fake_quant(std_bias)
                std_scale = self.obs_std_scale.fake_quant(std_scale)
            hidden_states = hidden_states - std_bias.float()
            hidden_states = self._fq(hidden_states, self.obs_minus_bias)
            hidden_states = hidden_states * std_scale.float()

        # Cast to input dtype
        hidden_states = hidden_states.to(inputs_embeds.dtype)

        # Quantize output
        hidden_states = self._fq(hidden_states, self.obs_last_hidden_state)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    def forward_export(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ):
        """Run Gemma4 vision model with static shapes for torch.export.

        This forward method assumes:
        - config.standardize is True (std_bias and std_scale are always applied)
        - output_length is precomputed and fixed
        - No conditional branching
        - as_export_module() has been called to set up export adapters

        This method IS exportable via torch.export.

        Args:
            pixel_values: Image pixels with shape (batch, channels, height, width).
            pixel_position_ids: Patch positions with shape (batch_size, max_patches, 2).

        Returns:
            BaseModelOutputWithPast with last_hidden_state containing visual soft tokens.
        """
        from transformers.models.gemma4.modeling_gemma4 import BaseModelOutputWithPast

        # Create padding mask from pixel_position_ids
        padding_positions = self.padding_positions

        # Patch embedder (use export adapter if available, otherwise original)
        patch_embedder = self.patch_embedder_export
        inputs_embeds = patch_embedder(
            pixel_values, pixel_position_ids, padding_positions
        )

        # Encoder (no export adapter yet — uses original wrapper)
        output = self.encoder(
            inputs_embeds=inputs_embeds,
            # Use ``== False`` instead of ``~`` to avoid ``aten::bitwise_not``
            # which is not supported by the Circle conversion pipeline.
            attention_mask=(padding_positions == False),
            pixel_position_ids=pixel_position_ids,
            return_dict=True,
        )

        # The encoder may return a BaseModelOutputWithPast (HF original) or a
        # plain tensor (QuantGemma4VisionEncoder wrapper).  Handle both cases.
        encoder_hidden = output

        # Pooler (use export adapter if available, otherwise original)
        pooler = self.pooler_export
        hidden_states, pooler_mask = pooler(
            hidden_states=encoder_hidden,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=self.output_length,
        )

        # Strip padding tokens
        hidden_states = hidden_states[pooler_mask]
        hidden_states = self._fq(hidden_states, self.obs_strip_padding)

        # Standardization (always applied in export mode)
        assert self.obs_std_bias is not None
        assert self.obs_std_scale is not None
        std_bias = self.obs_std_bias.fake_quant(self.std_bias)
        std_scale = self.obs_std_scale.fake_quant(self.std_scale)
        hidden_states = hidden_states - std_bias.float()
        hidden_states = self._fq(hidden_states, self.obs_minus_bias)
        hidden_states = hidden_states * std_scale.float()

        # Cast to input dtype
        hidden_states = hidden_states.to(inputs_embeds.dtype)

        # Quantize output
        hidden_states = self._fq(hidden_states, self.obs_last_hidden_state)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states)

    def as_export_module(
        self,
        mode: str = "prefill",
        *,
        pixel_position_ids: torch.Tensor,
        **kwargs,
    ) -> nn.Module:
        """Prepare the model for torch.export by precomputing static tensors.

        This method:
        1. Asserts that config.standardize is True (required for export)
        2. Asserts that the model is in QUANT mode
        3. Recursively converts submodules to their export adapters
        4. Registers output_length as a buffer for static export

        Submodule export adapters are stored as separate attributes
        (e.g. ``patch_embedder_export``, ``pooler_export``) so that the
        original wrapper attributes are not mutated.  ``forward_export()``
        uses these export adapter attributes when they exist.

        Args:
            mode: Export mode (only "prefill" is supported).
            pixel_position_ids: Patch position ids tensor with shape
                ``(1, num_patches, 2)``.  Required by the pooler's
                ``as_export_module()`` to precompute pooling weights.
            **kwargs: Additional arguments (unused).

        Returns:
            Gemma4VisionModelPrefillExportAdapter wrapping this module.
        """
        # Assert standardize is True for export
        assert (
            self.config.standardize
        ), "Gemma4VisionModel export requires config.standardize=True"

        # Assert QUANT mode
        assert self._mode is Mode.QUANT, "Must be in QUANT mode for export"

        # Make sure that all observers are calibrated
        for obs in self._all_observers():
            assert obs.has_qparams, f"Observer {obs.name} has not been calibrated"

        # Store output_length for use in forward_export
        pooling_kernel_size = self.config.pooling_kernel_size
        max_patches = pixel_position_ids.shape[-2]
        self.output_length = max_patches // (pooling_kernel_size * pooling_kernel_size)
        assert (
            self.output_length * (pooling_kernel_size * pooling_kernel_size)
            == max_patches
        ), "max_patches must be divisible by pooling_kernel_size^2"

        # Recursively convert submodules to their export adapters.
        # Store as separate attributes to avoid mutating the original wrappers.
        # forward_export() will use these via getattr(..., self.<original>).
        self.patch_embedder_export = self.patch_embedder.as_export_module(mode=mode)

        # Encoder: no as_export_module yet — will use original wrapper
        # in forward_export via getattr fallback.

        # Pooler: requires pixel_position_ids to precompute pooling weights
        assert pixel_position_ids is not None, (
            "pixel_position_ids is required by the pooler's as_export_module() "
            "to precompute pooling weights for static export."
        )
        self.pooler_export = self.pooler.as_export_module(
            mode=mode,
            output_length=self.output_length,
            pixel_position_ids=pixel_position_ids,
        )

        # Precompute padding mask from pixel_position_ids
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        self.register_buffer("padding_positions", padding_positions)

        register_fake_quant_meta_kernels_for_dynamic_export()

        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionModelPrefillExportAdapter,
        )

        return Gemma4VisionModelPrefillExportAdapter(wrapped_model=self)

    def _all_observers(self) -> Iterable:
        """Return all observers owned by this wrapper."""
        yield self.obs_minus_bias
        yield self.obs_last_hidden_state
        yield self.obs_strip_padding
        if self.obs_std_bias is not None:
            yield self.obs_std_bias
        if self.obs_std_scale is not None:
            yield self.obs_std_scale

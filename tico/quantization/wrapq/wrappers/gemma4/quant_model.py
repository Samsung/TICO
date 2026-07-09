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
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    assert_gemma4_e2b_no_moe,
    dynamic_placeholder_fuse,
    fixed_slot_fuse,
    validate_static_visual_layout,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


def _get_placeholder_mask(
    input_ids: torch.Tensor,
    config: object,
) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
    """Return boolean masks for image, video, and audio placeholder tokens.

    Mirrors ``Gemma4Model.get_placeholder_mask`` but works on raw tensors
    without requiring the full model reference.
    """
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)
    audio_token_id = getattr(config, "audio_token_id", None)

    image_mask = (
        input_ids == image_token_id
        if image_token_id is not None
        else torch.zeros_like(input_ids, dtype=torch.bool)
    )
    video_mask = (
        input_ids == video_token_id
        if video_token_id is not None
        else torch.zeros_like(input_ids, dtype=torch.bool)
    )
    audio_mask = (
        input_ids == audio_token_id
        if audio_token_id is not None
        else torch.zeros_like(input_ids, dtype=torch.bool)
    )
    return image_mask, video_mask, audio_mask


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4Model")
class QuantGemma4Model(QuantModuleBase):
    """PTQ wrapper skeleton for image-text Gemma4 E2B.

    The default eager/PTQ path uses dynamic placeholder-mask fusion so natural
    PyTorch benchmarks and general calibration follow HF-style placeholder
    positions. The static export/runtime path uses strict fixed-slot fusion.

    ``validate_static_layout`` is opt-in because benchmark prompts may differ
    from deployment/NPU prompts. Enable it only for deployment-style calibration
    or NPU-proxy benchmarks whose inputs follow the static runtime layout.
    """

    # Test-only override that exercises the static fusion branch without running
    # torch.export. Unit tests should reset it after use.
    force_export: bool = False

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        assert_gemma4_e2b_no_moe(fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.config = fp_model.config
        self.vision_tower = (
            PTQWrapper(
                fp_model.vision_tower,
                qcfg=qcfg.child("vision_tower") if qcfg else None,
                fp_name=join_name(fp_name, "vision_tower"),
            )
            if fp_model.vision_tower is not None
            else None
        )
        self.language_model = PTQWrapper(
            fp_model.language_model,
            qcfg=qcfg.child("language_model") if qcfg else None,
            fp_name=join_name(fp_name, "language_model"),
        )
        self.embed_vision = PTQWrapper(
            fp_model.embed_vision,
            qcfg=qcfg.child("embed_vision") if qcfg else None,
            fp_name=join_name(fp_name, "embed_vision"),
        )
        # Leave audio tower in FP (not quantized) - audio is not supported yet.
        self.audio_tower = (
            fp_model.audio_tower
            if getattr(fp_model, "audio_tower", None) is not None
            else None
        )

        vision_args = self.qcfg.model_args.get("vision", {})
        self.visual_start_idx = int(vision_args.get("visual_start_idx", 0))
        self.num_visual_tokens = int(vision_args.get("num_visual_tokens", 0))
        self.validate_static_layout = bool(
            vision_args.get("validate_static_layout", False)
        )
        self.obs_mm_fusion = self._make_obs("mm_fusion")
        self.obs_per_layer_inputs = self._make_obs("per_layer_inputs")

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ):
        """Return projected image soft tokens."""
        assert self.vision_tower is not None, "vision_tower is not available"
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            return_dict=True,
        )
        return self.embed_vision(vision_outputs.last_hidden_state)

    def _uses_static_fusion(self) -> bool:
        """Return whether the current call should use static fixed-slot fusion."""
        return bool(torch.compiler.is_compiling() or self.force_export)

    def _validate_runtime_multimodal_masks(
        self,
        *,
        image_mask: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
    ) -> None:
        """Reject unsupported or incomplete multimodal eager inputs."""
        if video_mask.any() or audio_mask.any():
            raise NotImplementedError(
                "Gemma4 PTQ wrapper supports image-text inputs only. "
                "Video and audio placeholder tokens are not supported."
            )
        if pixel_values is None and image_mask.any():
            raise ValueError(
                "pixel_values must be provided when input_ids contain image placeholders."
            )

    def _validate_static_image_layout(
        self,
        image_mask: torch.Tensor,
        seq_len: int,
    ) -> None:
        """Validate image placeholders against the static layout when enabled.

        This is opt-in for eager/PTQ because benchmark prompts may not match
        deployment/NPU prompt layouts.
        """
        if not self.validate_static_layout:
            return
        validate_static_visual_layout(
            image_mask,
            visual_start_idx=self.visual_start_idx,
            num_visual_tokens=self.num_visual_tokens,
            seq_len=seq_len,
        )

    def _validate_static_visual_token_count(self, image_embeds: torch.Tensor) -> None:
        """Validate vision output length for static export/runtime layouts."""
        expected = int(self.num_visual_tokens)
        actual = int(image_embeds.shape[1])
        if expected <= 0:
            raise ValueError(
                "Static Gemma4 visual-token validation requires "
                "model_args.vision.num_visual_tokens to be positive."
            )
        if actual != expected:
            raise ValueError(
                "Vision tower produced "
                f"{actual} visual tokens, but static Gemma4 config expects "
                f"{expected}. Check model_args.vision.image_height, "
                "model_args.vision.image_width, and "
                "model_args.vision.num_visual_tokens."
            )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask=None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run Gemma4 image-text forward with eager and static fusion paths.

        Eager/PTQ uses dynamic placeholder-mask fusion by default, matching
        HF-style image placeholder positions. Static export/runtime uses strict
        fixed-slot fusion. Static layout validation is opt-in in eager mode and
        should be enabled only for deployment-style calibration or NPU-proxy
        benchmarks whose prompts follow the final static runtime layout.

        Image-size alignment and static-layout validation solve different
        problems: resizing stabilizes visual-token count, while layout
        validation checks visual-token positions in the sequence.
        """
        # --- Input validation (matches original Gemma4Model.forward) ------
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds."
            )
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError(
                "You cannot specify per_layer_inputs if input_ids is provided."
            )

        # --- Token embedding with placeholder replacement -----------------
        # QuantGemma4TextModel is the concrete type behind the PTQWrapper.
        text_model = self.language_model.wrapped  # QuantGemma4TextModel
        llm_input_ids: Optional[torch.Tensor] = None
        image_mask: Optional[torch.BoolTensor] = None

        if inputs_embeds is None:
            assert input_ids is not None  # guaranteed by validation above
            # Replace multimodal placeholder token IDs with pad_token_id so
            # the embedding lookup returns a neutral pad embedding at those
            # positions.  Image features are fused later.
            llm_input_ids = input_ids.clone()
            image_mask, video_mask, audio_mask = _get_placeholder_mask(
                input_ids, self.config
            )
            if not torch.compiler.is_compiling():
                self._validate_runtime_multimodal_masks(
                    image_mask=image_mask,
                    video_mask=video_mask,
                    audio_mask=audio_mask,
                    pixel_values=pixel_values,
                )
            multimodal_mask = image_mask | video_mask | audio_mask
            if multimodal_mask.any():
                pad_token_id = self.config.text_config.pad_token_id
                llm_input_ids = torch.where(
                    multimodal_mask, pad_token_id, llm_input_ids
                )
            inputs_embeds = text_model.embed_tokens(llm_input_ids)

        # --- Per-Layer Embeddings (PLE) -----------------------------------
        # E2B / E4B models use PLE.  When input_ids was provided we compute
        # PLE here; when inputs_embeds was provided the caller must also
        # supply per_layer_inputs (validated by QuantGemma4TextModel).
        text_config = self.config.get_text_config()
        hidden_size_per_layer_input = int(
            getattr(text_config, "hidden_size_per_layer_input", 0) or 0
        )
        if per_layer_inputs is None and hidden_size_per_layer_input:
            # PLE needs llm_input_ids and a pad-replaced version of
            # inputs_embeds.  When input_ids was provided we already have
            # llm_input_ids.  When inputs_embeds was provided directly we
            # cannot reliably recover input_ids, so the caller must supply
            # per_layer_inputs explicitly.
            if llm_input_ids is None:
                raise ValueError(
                    "per_layer_inputs must be provided when inputs_embeds is "
                    "given and PLE is enabled (hidden_size_per_layer_input > 0)."
                )
            # Replace embeddings at multimodal positions with the pad embedding
            # before computing PLE, matching the original Gemma4Model.forward.
            embed_tokens_fp = text_model.embed_tokens.wrapped.module
            pad_embedding = embed_tokens_fp.weight[text_config.pad_token_id, :]
            image_mask_emb, video_mask_emb, audio_mask_emb = _get_placeholder_mask(
                llm_input_ids, self.config
            )
            multimodal_mask_emb = image_mask_emb | video_mask_emb | audio_mask_emb
            multimodal_mask_emb = multimodal_mask_emb.to(inputs_embeds.device)
            llm_inputs_embeds = torch.where(
                multimodal_mask_emb[..., None],
                pad_embedding.view(1, 1, -1).to(inputs_embeds),
                inputs_embeds,
            )
            per_layer_inputs = text_model.get_per_layer_inputs(
                llm_input_ids, llm_inputs_embeds
            )

        # Collect PLE statistics for the export path.  The calibration path
        # also exercises PLE inside each decoder layer's
        # ``_apply_per_layer_input()``, but the export path receives PLE as
        # an external input that must be fake-quantized at this level.
        if per_layer_inputs is not None:
            self.obs_per_layer_inputs.collect(per_layer_inputs)

        # --- Multimodal fusion (image only for v0) -----------------------
        if pixel_values is not None:
            image_embeds = self.get_image_features(
                pixel_values, image_position_ids=image_position_ids
            )
            assert inputs_embeds is not None  # guaranteed after embedding step
            # Align dtype/device to match text embeddings before fusion.
            image_embeds = image_embeds.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

            if self._uses_static_fusion():
                self._validate_static_visual_token_count(image_embeds)
                if image_mask is not None and not torch.compiler.is_compiling():
                    validate_static_visual_layout(
                        image_mask,
                        visual_start_idx=self.visual_start_idx,
                        num_visual_tokens=self.num_visual_tokens,
                        seq_len=int(inputs_embeds.shape[1]),
                    )
                inputs_embeds = fixed_slot_fuse(
                    inputs_embeds,
                    image_embeds,
                    visual_start_idx=self.visual_start_idx,
                    num_visual_tokens=self.num_visual_tokens,
                )
            else:
                if image_mask is None:
                    raise ValueError(
                        "input_ids must be provided with pixel_values when running "
                        "Gemma4 eager/PTQ dynamic placeholder fusion."
                    )
                if self.validate_static_layout:
                    self._validate_static_visual_token_count(image_embeds)
                    self._validate_static_image_layout(
                        image_mask,
                        seq_len=int(inputs_embeds.shape[1]),
                    )
                inputs_embeds = dynamic_placeholder_fuse(
                    inputs_embeds,
                    image_embeds,
                    image_mask,
                )
            inputs_embeds = self._fq(inputs_embeds, self.obs_mm_fusion)

        # --- Language model -----------------------------------------------
        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            per_layer_inputs=per_layer_inputs,
            **kwargs,
        )

    def forward_export(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_masks: Optional[dict] = None,
        position_embeddings: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run Gemma4 text decoder with static shapes for torch.export.

        This method assumes the CPU runtime has already performed:
        - Token embedding (with placeholder replacement)
        - Vision tower + projection
        - Multimodal fusion (fixed-slot)
        - PLE computation (if enabled)
        - Mask and RoPE generation per layer type

        This method IS exportable via torch.export — no dynamic control flow
        on tensor values, no references to dynamic tensor shapes.

        Args:
            inputs_embeds: Pre-fused text+image embeddings, shape ``(1, S, H)``.
            per_layer_inputs: PLE tensor, shape ``(1, S, L, P)`` or None.
            attention_masks: Dict mapping layer type to additive mask tensors.
            position_embeddings: Dict mapping layer type to ``(cos, sin)`` tuples.

        Returns:
            Final hidden states after the text decoder and final norm,
            shape ``(1, S, H)``.
        """
        if not hasattr(self, "prefill_layers"):
            raise RuntimeError(
                "forward_export() requires as_export_module() to be called first."
            )

        text_model = self.language_model.wrapped  # QuantGemma4TextModel
        text_config = self.config.get_text_config()

        # mypy: attention_masks and position_embeddings are required when
        # running decoder layers; narrow away None before the loop.
        assert attention_masks is not None
        assert position_embeddings is not None

        hidden_states = inputs_embeds

        # Fake-quantize PLE inputs so the exported graph carries qparam
        # metadata on the placeholder and its derived slice nodes.
        if per_layer_inputs is not None:
            per_layer_inputs = self._fq(per_layer_inputs, self.obs_per_layer_inputs)

        # Run text decoder layers with precomputed masks and RoPE.
        for i, decoder_layer in enumerate(self.prefill_layers):
            layer_type = text_config.layer_types[i]
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            output = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
            )
            # Layer output is (hidden_states,) or (hidden_states, kv).
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

        # Final norm.
        hidden_states = text_model.norm(hidden_states)
        return hidden_states

    def as_export_module(
        self,
        mode: str = "prefill",
        *,
        pixel_position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """Prepare the model for torch.export by precomputing static tensors.

        This method:
        1. Asserts that the model is in QUANT mode
        2. Verifies all observers are calibrated
        3. Creates text decoder layer export adapters via ``as_export_module()``
        4. Returns a ``Gemma4ModelPrefillExportAdapter`` wrapping this module

        Args:
            mode: Export mode (only "prefill" is supported).
            pixel_position_ids: Accepted for API compatibility but not yet
                used.  Vision tower precomputation is handled separately by
                ``Gemma4VisionPrefillExportAdapter``.
            **kwargs: Additional arguments (unused).

        Returns:
            ``Gemma4ModelPrefillExportAdapter`` wrapping this module.
        """
        assert self._mode is Mode.QUANT, "Must be in QUANT mode for export"

        # Make sure that all observers are calibrated.
        for obs in self._all_observers():
            assert obs.has_qparams, f"Observer {obs.name} has not been calibrated"

        text_model = self.language_model.wrapped  # QuantGemma4TextModel
        text_config = self.config.get_text_config()

        # Create text decoder layer export adapters.
        if mode == "prefill":
            self.prefill_layers = nn.ModuleList(
                [
                    layer.wrapped.as_export_module(mode="prefill", return_kv=True)
                    for layer in text_model.layers
                ]
            )
        elif mode == "decode":
            self.prefill_layers = nn.ModuleList(
                [
                    layer.wrapped.as_export_module(mode="decode", return_kv=True)
                    for layer in text_model.layers
                ]
            )
        else:
            raise ValueError(f"Unsupported export mode: {mode!r}")

        # Register fake-quant meta kernels for dynamic export.
        from tico.quantization.wrapq.wrappers.llama.export_adapters import (
            register_fake_quant_meta_kernels_for_dynamic_export,
        )

        register_fake_quant_meta_kernels_for_dynamic_export()

        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4ModelPrefillExportAdapter,
        )

        return Gemma4ModelPrefillExportAdapter(wrapped_model=self)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_mm_fusion, self.obs_per_layer_inputs)

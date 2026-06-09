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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from tico.quantization.algorithm.spinquant.qwen3_vl_model_utils import (
    assert_tied_word_embedding,
    resolve_qwen3_vl_spinquant_components,
    validate_qwen3_vl_for_spinquant,
)
from tico.quantization.algorithm.spinquant.qwen3_vl_rotation_utils import (
    apply_qwen3_vl_embedding_side_rotation,
    apply_qwen3_vl_lm_head_side_rotation,
    build_qwen3_vl_r1,
    build_qwen3_vl_r2,
    build_qwen3_vl_vision_r1,
    build_qwen3_vl_vision_r2,
    extract_and_reset_qwen3_vl_final_norm_scale,
    fuse_qwen3_vl_text_layer_norms,
    fuse_qwen3_vl_vision_layer_norms,
    get_qwen3_vl_head_dim,
    get_qwen3_vl_text_hidden_size,
    get_qwen3_vl_vision_head_dim,
    get_qwen3_vl_vision_hidden_size,
    rotate_qwen3_vl_deepstack_outputs,
    rotate_qwen3_vl_ov_r2,
    rotate_qwen3_vl_text_layer_r1,
    rotate_qwen3_vl_vision_layer_r1,
    rotate_qwen3_vl_vision_merger_inputs,
    rotate_qwen3_vl_vision_ov_r2,
    rotate_qwen3_vl_vision_patch_and_position_embeddings,
)
from tico.quantization.algorithm.spinquant.rotation_utils import (
    infer_device,
    infer_dtype,
)
from tico.quantization.algorithm.spinquant.spin_qwen3_vl import (
    SpinQwen3VLForConditionalGeneration,
)
from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


@register_quantizer(Qwen3VLSpinQuantConfig)
class Qwen3VLSpinQuantQuantizer(BaseQuantizer):
    """
    Quantizer that applies tied-embedding-safe SpinQuant to Qwen3-VL.

    This implementation assumes Qwen3-VL word embeddings are tied. It preserves
    the tied storage by using runtime boundary Linear modules:
        - model.language_model.rotate_embedding
        - rotate_lm_head

    Decoder-layer text R1/R2 rotations, optional vision-tower rotations, and
    DeepStack output rotations are fused into weights during conversion.
    """

    def __init__(self, config: Qwen3VLSpinQuantConfig):
        """
        Initialize the Qwen3-VL SpinQuant quantizer.

        Parameters:
            config: Qwen3-VL SpinQuant configuration.
        """
        super().__init__(config)
        self.config = config

    @torch.no_grad()
    def prepare(
        self,
        model: nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Convert a Qwen3-VL model into a SpinQwen3VL model.

        Parameters:
            model: Target Qwen3-VL model.
            args: Unused. Kept for API compatibility.
            kwargs: Unused. Kept for API compatibility.

        Returns:
            A SpinQwen3VL model initialized from the source model.

        Raises:
            TypeError: If the input is not an nn.Module.
            ValueError: If the model is not a tied Qwen3-VL model.
        """
        assert isinstance(self.config, Qwen3VLSpinQuantConfig)
        validate_qwen3_vl_for_spinquant(
            model,
            self.config,
            require_spin_runtime=False,
        )
        return self._convert_to_spin_qwen3_vl(model)

    @torch.no_grad()
    def convert(
        self,
        model: nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Apply Qwen3-VL SpinQuant rotation fusion.

        Parameters:
            model: Prepared SpinQwen3VL model.
            args: Unused. Kept for API compatibility.
            kwargs: Unused. Kept for API compatibility.

        Returns:
            The rotated and fused model.

        Raises:
            ValueError: If the prepared model violates the tied embedding assumption.
        """
        assert isinstance(self.config, Qwen3VLSpinQuantConfig)
        validate_qwen3_vl_for_spinquant(
            model,
            self.config,
            require_spin_runtime=True,
        )

        fuse_qwen3_vl_text_layer_norms(model, self.config)
        if self.config.fuse_vision_layer_norms:
            fuse_qwen3_vl_vision_layer_norms(model, self.config)

        hidden_size = get_qwen3_vl_text_hidden_size(model)
        device = infer_device(model)

        if self.config.enable_r1:
            r1 = build_qwen3_vl_r1(model, self.config)
        else:
            r1 = torch.eye(hidden_size, device=device, dtype=torch.float64)

        final_norm_scale = extract_and_reset_qwen3_vl_final_norm_scale(
            model,
            self.config,
        )

        apply_qwen3_vl_embedding_side_rotation(model, self.config, r1)
        apply_qwen3_vl_lm_head_side_rotation(
            model,
            self.config,
            r1,
            norm_scale=final_norm_scale,
        )

        components = resolve_qwen3_vl_spinquant_components(model, self.config)
        layers = components.text_layers
        head_dim = get_qwen3_vl_head_dim(model)

        iterator = enumerate(layers)
        if self.config.show_progress:
            iterator = tqdm(
                iterator,
                total=len(layers),
                desc="Applying Qwen3-VL text SpinQuant rotations",
            )

        for layer_idx, layer in iterator:
            if self.config.enable_r1:
                rotate_qwen3_vl_text_layer_r1(layer, r1)

            if self.config.enable_r2:
                r2 = build_qwen3_vl_r2(
                    init_method=self.config.init_method,
                    r2_map=self.config.r2_map,
                    layer_idx=layer_idx,
                    head_dim=head_dim,
                    device=device,
                )
                rotate_qwen3_vl_ov_r2(layer, r2, head_dim)

        if self.config.enable_vision_r1 or self.config.enable_vision_r2:
            self._convert_vision_tower(model)

        if self.config.enable_r1:
            rotate_qwen3_vl_deepstack_outputs(model, self.config, r1)

        assert_tied_word_embedding(model, self.config)
        return model

    @torch.no_grad()
    def _convert_vision_tower(self, model: nn.Module) -> None:
        """
        Apply optional vision-side SpinQuant rotations to the Qwen3-VL vision tower.

        Parameters:
            model: Prepared SpinQwen3VL model.
        """
        assert isinstance(self.config, Qwen3VLSpinQuantConfig)
        components = resolve_qwen3_vl_spinquant_components(model, self.config)
        vision_layers = components.vision_blocks
        vision_head_dim = get_qwen3_vl_vision_head_dim(model)
        device = infer_device(model)

        vision_method = self.config.vision_init_method or self.config.init_method

        if self.config.enable_vision_r1:
            vision_hidden_size = get_qwen3_vl_vision_hidden_size(model)
            vision_r1 = build_qwen3_vl_vision_r1(model, self.config)
        else:
            vision_hidden_size = get_qwen3_vl_vision_hidden_size(model)
            vision_r1 = torch.eye(
                vision_hidden_size,
                device=device,
                dtype=torch.float64,
            )

        if self.config.enable_vision_r1:
            rotate_qwen3_vl_vision_patch_and_position_embeddings(
                model,
                self.config,
                vision_r1,
            )

        iterator = enumerate(vision_layers)
        if self.config.show_progress:
            iterator = tqdm(
                iterator,
                total=len(vision_layers),
                desc="Applying Qwen3-VL vision SpinQuant rotations",
            )

        for layer_idx, layer in iterator:
            if self.config.enable_vision_r1:
                rotate_qwen3_vl_vision_layer_r1(layer, vision_r1)

            if self.config.enable_vision_r2:
                vision_r2 = build_qwen3_vl_vision_r2(
                    init_method=vision_method,
                    r2_map=self.config.vision_r2_map,
                    layer_idx=layer_idx,
                    head_dim=vision_head_dim,
                    device=device,
                )
                rotate_qwen3_vl_vision_ov_r2(layer, vision_r2, vision_head_dim)

        if self.config.enable_vision_r1:
            rotate_qwen3_vl_vision_merger_inputs(model, self.config, vision_r1)

    @torch.no_grad()
    def _convert_to_spin_qwen3_vl(
        self,
        model: nn.Module,
    ) -> SpinQwen3VLForConditionalGeneration:
        """
        Create a SpinQwen3VLForConditionalGeneration instance from Qwen3-VL.

        Parameters:
            model: Source Qwen3-VL model.

        Returns:
            A SpinQwen3VL model with copied source weights and identity runtime
            rotation layers.
        """
        target_device = infer_device(model)
        target_dtype = infer_dtype(model)

        spin_model = SpinQwen3VLForConditionalGeneration(model.config)
        spin_model.to(device=target_device, dtype=target_dtype)

        missing_keys, unexpected_keys = spin_model.load_state_dict(
            model.state_dict(),
            strict=False,
        )

        self._validate_state_dict_keys(
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
        )

        self._initialize_spin_weights(spin_model)
        self._force_tied_word_embedding(spin_model)
        assert isinstance(self.config, Qwen3VLSpinQuantConfig)
        assert_tied_word_embedding(spin_model, self.config)
        self._copy_runtime_attributes(src=model, dst=spin_model)
        self._force_tied_word_embedding(spin_model)
        assert_tied_word_embedding(spin_model, self.config)

        if model.training:
            spin_model.train()
        else:
            spin_model.eval()

        return spin_model

    @torch.no_grad()
    def _initialize_spin_weights(
        self,
        model: SpinQwen3VLForConditionalGeneration,
    ) -> None:
        """
        Initialize Qwen3-VL SpinQuant runtime layers as identity transforms.

        Parameters:
            model: Prepared SpinQwen3VL model.
        """
        assert isinstance(self.config, Qwen3VLSpinQuantConfig)
        components = resolve_qwen3_vl_spinquant_components(model, self.config)
        self._init_linear_as_identity(components.language_model.rotate_embedding)
        self._init_linear_as_identity(model.rotate_lm_head)

    @torch.no_grad()
    def _force_tied_word_embedding(
        self,
        model: SpinQwen3VLForConditionalGeneration,
    ) -> None:
        """
        Force the SpinQwen3VL input embedding and LM head to share storage.

        `load_state_dict` copies tensor values into existing Parameter objects
        and does not preserve aliasing between tied parameters. Some transformers
        versions also do not materialize Qwen3-VL ties through `tie_weights()`
        for small synthetic configs. SpinQuant relies on the tied-weight
        invariant, so the alias is restored explicitly here.

        Parameters:
            model: Prepared SpinQwen3VL model to update in-place.
        """
        assert isinstance(self.config, Qwen3VLSpinQuantConfig)
        components = resolve_qwen3_vl_spinquant_components(model, self.config)
        embed_tokens = getattr(components.language_model, "embed_tokens")
        if not isinstance(embed_tokens, nn.Embedding):
            raise TypeError(
                "Expected language_model.embed_tokens to be nn.Embedding, "
                f"got {type(embed_tokens).__name__}."
            )

        components.lm_head.weight = embed_tokens.weight

        if hasattr(model, "config"):
            model.config.tie_word_embeddings = True
            if hasattr(model.config, "text_config"):
                model.config.text_config.tie_word_embeddings = True

    @torch.no_grad()
    def _init_linear_as_identity(self, layer: nn.Linear) -> None:
        """
        Initialize a square Linear layer as identity.

        Parameters:
            layer: Target Linear layer.

        Raises:
            ValueError: If the layer is not square.
        """
        if layer.in_features != layer.out_features:
            raise ValueError(
                "SpinQuant runtime layers must be square, but got "
                f"in_features={layer.in_features}, out_features={layer.out_features}."
            )

        eye = torch.eye(
            layer.in_features,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        layer.weight.copy_(eye)

        if layer.bias is not None:
            layer.bias.zero_()

    def _validate_state_dict_keys(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
    ) -> None:
        """
        Validate state_dict loading results for SpinQwen3VL conversion.

        Parameters:
            missing_keys: Keys missing when loading the source state_dict.
            unexpected_keys: Extra keys from the source state_dict.

        Raises:
            ValueError: If unexpected missing or unexpected keys are found.
        """
        expected_missing = {
            "model.language_model.rotate_embedding.weight",
            "rotate_lm_head.weight",
        }

        missing_set = set(missing_keys)
        unexpected_missing = missing_set - expected_missing

        if unexpected_missing:
            raise ValueError(
                "Unexpected missing keys were found while converting to "
                f"SpinQwen3VL: {sorted(unexpected_missing)}"
            )

        if unexpected_keys:
            raise ValueError(
                "Unexpected keys were found while converting to "
                f"SpinQwen3VL: {sorted(unexpected_keys)}"
            )

    def _copy_runtime_attributes(
        self,
        src: nn.Module,
        dst: SpinQwen3VLForConditionalGeneration,
    ) -> None:
        """
        Copy runtime attributes from the source model to the SpinQwen3VL model.

        Parameters:
            src: Source Qwen3-VL model.
            dst: Destination SpinQwen3VL model.
        """
        if hasattr(src, "generation_config"):
            dst.generation_config = src.generation_config

        if hasattr(src, "name_or_path"):
            dst.name_or_path = src.name_or_path

        if hasattr(src, "_keep_in_fp32_modules"):
            dst._keep_in_fp32_modules = src._keep_in_fp32_modules

        if hasattr(src, "hf_device_map"):
            dst.hf_device_map = src.hf_device_map

        if hasattr(src, "config"):
            dst.config = src.config

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
from transformers import PretrainedConfig

from tico.quantization.algorithm.spinquant.fuse_norm_utils import (
    fuse_spinquant_layer_norms,
)
from tico.quantization.algorithm.spinquant.rotation_utils import (
    apply_embedding_side_rotation,
    apply_lm_head_side_rotation,
    build_r1,
    build_r2,
    extract_and_reset_final_norm_scale,
    get_decoder_layers,
    infer_device,
    infer_dtype,
    rotate_attention_inputs,
    rotate_attention_output,
    rotate_mlp_input,
    rotate_mlp_output,
    rotate_ov_proj,
)
from tico.quantization.algorithm.spinquant.spin_llama import SpinLlamaForCausalLM
from tico.quantization.config.spinquant import SpinQuantConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


@register_quantizer(SpinQuantConfig)
class SpinQuantQuantizer(BaseQuantizer):
    """
    Quantizer that applies SpinQuant-style offline rotation fusion.

    This implementation intentionally supports only the offline-fusable
    rotations required by the current PTQ framework:
        - R1
        - R2

    To preserve tied embeddings, embedding-side and LM-head-side rotations are
    not fused into the original embedding table or LM head directly. Instead,
    they are stored in the auxiliary rotation modules provided by the custom
    SpinLlama model:
        - model.model.rotate_embedding
        - model.rotate_lm_head
    """

    def __init__(self, config: SpinQuantConfig):
        """
        Initialize the SpinQuant quantizer.

        Parameters:
            config: SpinQuant configuration.
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
        Convert the input model into a SpinLlamaForCausalLM model.

        Parameters:
            model: The target PyTorch model.
            args: Unused. Kept for API compatibility.
            kwargs: Unused. Kept for API compatibility.

        Returns:
            A SpinLlamaForCausalLM model initialized from the original model.

        Raises:
            TypeError: If the input is not an nn.Module.
            ValueError: If the input model is not a supported LLaMA causal LM.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected `model` to be an nn.Module, but got: {type(model)}"
            )

        self._validate_model(model)
        return self._convert_to_spin_llama(model)

    @torch.no_grad()
    def convert(
        self,
        model: nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Apply SpinQuant offline fusion to the model.

        Parameters:
            model: Target model.
            args: Unused. Kept for API compatibility.
            kwargs: Unused. Kept for API compatibility.

        Returns:
            The rotated and fused model.

        Raises:
            AttributeError: If the model does not expose the expected SpinLlama structure.
            ValueError: If any required rotation matrix is invalid.
        """
        fuse_spinquant_layer_norms(
            model,
            center_input_embeddings=False,
            fuse_lm_head=False,
        )

        layers = get_decoder_layers(model)
        hidden_size = int(model.config.hidden_size)
        num_heads = int(model.config.num_attention_heads)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_heads})."
            )

        head_dim = hidden_size // num_heads
        device = infer_device(model)

        assert isinstance(self.config, SpinQuantConfig)
        r1 = build_r1(
            model=model,
            init_method=self.config.init_method,
            r1=self.config.r1,
        )

        final_norm_scale = extract_and_reset_final_norm_scale(model)

        apply_embedding_side_rotation(model, r1)
        apply_lm_head_side_rotation(model, r1, norm_scale=final_norm_scale)

        iterator = enumerate(layers)
        if self.config.show_progress:
            iterator = tqdm(
                iterator, total=len(layers), desc="Applying SpinQuant rotations"
            )
        for layer_idx, layer in iterator:
            rotate_attention_inputs(layer, r1)
            rotate_attention_output(layer, r1)
            rotate_mlp_input(layer, r1)
            rotate_mlp_output(layer, r1)

            r2 = build_r2(
                init_method=self.config.init_method,
                r2_map=self.config.r2_map,
                layer_idx=layer_idx,
                head_dim=head_dim,
                device=device,
            )
            rotate_ov_proj(layer, r2, head_dim)

        return model

    def _validate_model(self, model: nn.Module) -> None:
        """
        Validate that the input model looks like a supported Hugging Face LLaMA
        causal language model.

        Parameters:
            model: The model to validate.

        Raises:
            ValueError: If the model is not compatible.
        """
        config = getattr(model, "config", None)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "The input model does not have a valid Hugging Face config."
            )

        model_type = getattr(config, "model_type", None)
        if model_type != "llama":
            raise ValueError(
                "SpinQuantQuantizer currently supports only LLaMA models, "
                f"but got model_type={model_type!r}."
            )

        if not hasattr(model, "model"):
            raise ValueError("The input model is missing the `model` submodule.")

        if not hasattr(model, "lm_head"):
            raise ValueError("The input model is missing the `lm_head` submodule.")

    @torch.no_grad()
    def _convert_to_spin_llama(self, model: nn.Module) -> SpinLlamaForCausalLM:
        """
        Create a SpinLlamaForCausalLM instance from an existing Hugging Face
        LLaMA causal language model.

        The conversion is performed by:
            - constructing a new SpinLlama model from the same config,
            - copying compatible weights from the source model,
            - initializing the newly added spin-related weights,
            - preserving selected runtime attributes.

        This approach is intentionally chosen instead of mutating the original
        model instance in-place, which makes the conversion logic more robust
        when custom parameters or forward-path changes are introduced.

        Parameters:
            model: The original Hugging Face model.

        Returns:
            A newly constructed SpinLlamaForCausalLM instance with copied weights.
        """
        target_device = infer_device(model)
        target_dtype = infer_dtype(model)

        spin_model = SpinLlamaForCausalLM(model.config)
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
        self._copy_runtime_attributes(src=model, dst=spin_model)

        if model.training:
            spin_model.train()
        else:
            spin_model.eval()

        return spin_model

    @torch.no_grad()
    def _initialize_spin_weights(self, model: SpinLlamaForCausalLM) -> None:
        """
        Initialize the spin-related weights.

        The newly introduced rotation layers are initialized as identity
        transforms so that the converted model preserves the original model
        behavior before SpinQuant rotations are fused.

        Parameters:
            model: The converted SpinLlamaForCausalLM model.
        """
        self._init_linear_as_identity(model.model.rotate_embedding)
        self._init_linear_as_identity(model.rotate_lm_head)

    @torch.no_grad()
    def _init_linear_as_identity(self, layer: nn.Linear) -> None:
        """
        Initialize a square linear layer as an identity transform.

        Parameters:
            layer: The target linear layer.

        Raises:
            ValueError: If the layer is not square.
        """
        if layer.in_features != layer.out_features:
            raise ValueError(
                "Spin rotation layers must be square to be initialized as identity, "
                f"but got in_features={layer.in_features}, "
                f"out_features={layer.out_features}."
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
        Validate state_dict loading results.

        Expected missing keys are the newly introduced spin-related weights.
        Unexpected keys are treated as errors because they usually indicate
        a mismatch between the source model and the SpinLlama model definition.

        Parameters:
            missing_keys: Keys missing when loading the source state_dict.
            unexpected_keys: Extra keys from the source state_dict that do not
                exist in the destination model.

        Raises:
            ValueError: If unexpected missing or unexpected keys are found.
        """
        expected_missing = {
            "model.rotate_embedding.weight",
            "rotate_lm_head.weight",
        }

        missing_set = set(missing_keys)
        unexpected_missing = missing_set - expected_missing

        if unexpected_missing:
            raise ValueError(
                "Unexpected missing keys were found while converting to SpinLlama: "
                f"{sorted(unexpected_missing)}"
            )

        if unexpected_keys:
            raise ValueError(
                "Unexpected keys were found while converting to SpinLlama: "
                f"{sorted(unexpected_keys)}"
            )

    def _copy_runtime_attributes(
        self,
        src: nn.Module,
        dst: SpinLlamaForCausalLM,
    ) -> None:
        """
        Copy important runtime attributes from the source model to the converted model.

        Parameters:
            src: The source Hugging Face model.
            dst: The converted SpinLlama model.
        """
        if hasattr(src, "generation_config"):
            dst.generation_config = src.generation_config

        if hasattr(src, "name_or_path"):
            dst.name_or_path = src.name_or_path

        if hasattr(src, "_keep_in_fp32_modules"):
            dst._keep_in_fp32_modules = src._keep_in_fp32_modules

        if hasattr(src, "config"):
            dst.config = src.config

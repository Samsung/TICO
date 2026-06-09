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

from typing import Dict, Literal, Optional

import torch

from tico.quantization.config.base import BaseConfig


class Qwen3VLSpinQuantConfig(BaseConfig):
    """
    Configuration for tied-embedding-safe SpinQuant on Qwen3-VL.

    This configuration covers the text-side SpinQuant path and optional
    vision-tower rotations:
        - text decoder R1 fusion
        - optional text attention OV-side R2 fusion
        - tied-safe text input and output boundary rotations
        - DeepStack visual output rotation fusion into the text residual basis
        - optional vision LayerNorm affine folding
        - optional vision residual-basis R1 rotation
        - optional vision attention OV-side R2 rotation

    Qwen3-VL word embeddings are assumed to be tied. Therefore, text boundary
    rotations are stored in dedicated runtime Linear modules instead of directly
    modifying the shared embedding table.

    The vision tower uses LayerNorm rather than RMSNorm. For exact R1 fusion,
    vision R1 must preserve the all-ones direction so that identity-affine
    LayerNorm remains rotation equivariant. The built-in random and hadamard
    vision initializers therefore construct a dense orthogonal rotation that
    fixes the all-ones vector. External vision R1 matrices are validated by
    default.
    """

    def __init__(
        self,
        init_method: Literal["random", "hadamard", "external"] = "random",
        r1: Optional[torch.Tensor] = None,
        r2_map: Optional[Dict[str, torch.Tensor]] = None,
        enable_r1: bool = True,
        enable_r2: bool = True,
        fuse_deepstack_visual_outputs: bool = True,
        show_progress: bool = True,
        language_model_attr: str = "model.language_model",
        text_layers_attr: str = "model.language_model.layers",
        visual_model_attr: str = "model.visual",
        vision_blocks_attr: str = "model.visual.blocks",
        visual_deepstack_mergers_attr: str = "model.visual.deepstack_merger_list",
        lm_head_attr: str = "lm_head",
        fuse_vision_layer_norms: bool = False,
        enable_vision_r1: bool = False,
        enable_vision_r2: bool = False,
        vision_init_method: Optional[Literal["random", "hadamard", "external"]] = None,
        vision_r1: Optional[torch.Tensor] = None,
        vision_r2_map: Optional[Dict[str, torch.Tensor]] = None,
        require_vision_r1_layernorm_compatible: bool = True,
        vision_rotation_tolerance: float = 1e-4,
    ):
        """
        Initialize Qwen3-VL SpinQuant configuration.

        Parameters:
            init_method:
                Strategy for resolving text rotation matrices.

                - "random": use random orthogonal matrices.
                - "hadamard": use randomized Hadamard orthogonal matrices.
                - "external": use user-provided matrices.

            r1:
                Optional text global hidden-dimension rotation matrix. This is
                required when ``init_method == "external"`` and ``enable_r1`` is
                enabled.

            r2_map:
                Optional mapping from text module keys to per-layer head-dimension
                rotation matrices.

                Supported example keys:
                    - "model.language_model.layers.0.self_attn.R2"
                    - "model.layers.0.self_attn.R2"

            enable_r1:
                Whether to apply the text global hidden-dimension R1 rotation.

            enable_r2:
                Whether to apply the text OV-side head-dimension R2 rotation.

            fuse_deepstack_visual_outputs:
                Whether to fuse text R1 into DeepStack visual output projections.
                Main visual merger outputs are not fused because they pass
                through the text input-side runtime rotation.

            show_progress:
                If True, display tqdm progress bars during conversion.

            language_model_attr:
                Dotted path to the Qwen3-VL text model.

            text_layers_attr:
                Dotted path to the Qwen3-VL text decoder layers.

            visual_model_attr:
                Dotted path to the Qwen3-VL vision model.

            vision_blocks_attr:
                Dotted path to the Qwen3-VL vision transformer blocks.

            visual_deepstack_mergers_attr:
                Dotted path to the DeepStack merger module list.

            lm_head_attr:
                Dotted path to the final language modeling head.

            fuse_vision_layer_norms:
                Whether to fold vision LayerNorm affine parameters into adjacent
                Linear layers. This must be enabled when ``enable_vision_r1`` is
                enabled.

            enable_vision_r1:
                Whether to rotate the vision tower residual stream with a
                LayerNorm-compatible R1 matrix.

            enable_vision_r2:
                Whether to rotate the vision attention V/proj path with per-head
                R2 matrices.

            vision_init_method:
                Optional rotation initialization mode for vision R1/R2. If None,
                it falls back to ``init_method``.

            vision_r1:
                Optional vision hidden-dimension R1 matrix. This is required when
                ``vision_init_method == "external"`` and ``enable_vision_r1`` is
                enabled.

            vision_r2_map:
                Optional mapping from vision block keys to per-layer head-dimension
                R2 matrices.

                Supported example keys:
                    - "model.visual.blocks.0.attn.R2"
                    - "model.visual.blocks.0.attn.vision_R2"

            require_vision_r1_layernorm_compatible:
                If True, external vision R1 matrices must be orthogonal and must
                preserve the all-ones direction.

            vision_rotation_tolerance:
                Absolute tolerance used when validating external vision rotations.
        """
        self.init_method = init_method
        self.r1 = r1
        self.r2_map = r2_map
        self.enable_r1 = enable_r1
        self.enable_r2 = enable_r2
        self.fuse_deepstack_visual_outputs = fuse_deepstack_visual_outputs
        self.show_progress = show_progress
        self.language_model_attr = language_model_attr
        self.text_layers_attr = text_layers_attr
        self.visual_model_attr = visual_model_attr
        self.vision_blocks_attr = vision_blocks_attr
        self.visual_deepstack_mergers_attr = visual_deepstack_mergers_attr
        self.lm_head_attr = lm_head_attr
        self.fuse_vision_layer_norms = fuse_vision_layer_norms
        self.enable_vision_r1 = enable_vision_r1
        self.enable_vision_r2 = enable_vision_r2
        self.vision_init_method = vision_init_method
        self.vision_r1 = vision_r1
        self.vision_r2_map = vision_r2_map
        self.require_vision_r1_layernorm_compatible = (
            require_vision_r1_layernorm_compatible
        )
        self.vision_rotation_tolerance = vision_rotation_tolerance

        self._validate()

    def _validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If a configuration value is invalid.
            TypeError: If a field has an unexpected type.
        """
        valid_methods = {"random", "hadamard", "external"}
        if self.init_method not in valid_methods:
            raise ValueError(f"Unsupported init_method: {self.init_method!r}")

        if (
            self.vision_init_method is not None
            and self.vision_init_method not in valid_methods
        ):
            raise ValueError(
                f"Unsupported vision_init_method: {self.vision_init_method!r}"
            )

        if self.enable_r1 and self.init_method == "external" and self.r1 is None:
            raise ValueError(
                "`r1` must be provided when init_method='external' and enable_r1=True."
            )

        vision_method = self.vision_init_method or self.init_method
        if (
            self.enable_vision_r1
            and vision_method == "external"
            and self.vision_r1 is None
        ):
            raise ValueError(
                "`vision_r1` must be provided when vision_init_method='external' "
                "and enable_vision_r1=True."
            )

        if self.enable_vision_r1 and not self.fuse_vision_layer_norms:
            raise ValueError(
                "`fuse_vision_layer_norms` must be True when enable_vision_r1=True."
            )

        if self.r1 is not None and not isinstance(self.r1, torch.Tensor):
            raise TypeError(f"`r1` must be a torch.Tensor, got {type(self.r1)}.")

        if self.vision_r1 is not None and not isinstance(self.vision_r1, torch.Tensor):
            raise TypeError(
                f"`vision_r1` must be a torch.Tensor, got {type(self.vision_r1)}."
            )

        if self.r2_map is not None:
            self._validate_r2_map("r2_map", self.r2_map)

        if self.vision_r2_map is not None:
            self._validate_r2_map("vision_r2_map", self.vision_r2_map)

        if not isinstance(self.vision_rotation_tolerance, (float, int)):
            raise TypeError(
                "`vision_rotation_tolerance` must be a float, "
                f"got {type(self.vision_rotation_tolerance)}."
            )
        if self.vision_rotation_tolerance <= 0:
            raise ValueError(
                "`vision_rotation_tolerance` must be positive, "
                f"got {self.vision_rotation_tolerance}."
            )

        for field_name in (
            "language_model_attr",
            "text_layers_attr",
            "visual_model_attr",
            "vision_blocks_attr",
            "visual_deepstack_mergers_attr",
            "lm_head_attr",
        ):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(
                    f"{field_name} must be a non-empty string, got {field_value!r}."
                )

    def _validate_r2_map(
        self,
        field_name: str,
        value: Dict[str, torch.Tensor],
    ) -> None:
        """
        Validate an R2 map field.

        Parameters:
            field_name: Configuration field name used in error messages.
            value: Candidate mapping from module keys to tensors.

        Raises:
            TypeError: If the map or one of its values has an invalid type.
            ValueError: If a key is invalid.
        """
        if not isinstance(value, dict):
            raise TypeError(
                f"`{field_name}` must be a dict[str, torch.Tensor], got {type(value)}."
            )

        for key, tensor in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"Invalid {field_name} key: {key!r}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{field_name}[{key!r}] must be a torch.Tensor, got {type(tensor)}."
                )

    @property
    def name(self) -> str:
        return "spinquant"

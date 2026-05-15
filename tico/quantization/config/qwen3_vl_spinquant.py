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

    This configuration is scoped to Phase 1:
        - text decoder R1 fusion
        - optional attention OV-side R2 fusion
        - tied-safe input and output boundary rotations
        - DeepStack visual output rotation fusion

    The word embedding and LM head are assumed to be tied. Therefore, the
    boundary rotations are stored in dedicated runtime Linear modules instead
    of directly modifying the shared embedding table.
    """

    def __init__(
        self,
        init_method: Literal["random", "hadamard", "external"] = "random",
        r1: Optional[torch.Tensor] = None,
        r2_map: Optional[Dict[str, torch.Tensor]] = None,
        apply_r1: bool = True,
        apply_r2: bool = True,
        fuse_deepstack_visual_outputs: bool = True,
        show_progress: bool = True,
        language_model_attr: str = "model.language_model",
        text_layers_attr: str = "model.language_model.layers",
        visual_deepstack_mergers_attr: str = "model.visual.deepstack_merger_list",
        lm_head_attr: str = "lm_head",
    ):
        """
        Initialize Qwen3-VL SpinQuant configuration.

        Parameters:
            init_method:
                Strategy for resolving rotation matrices.

                - "random": use random orthogonal matrices.
                - "hadamard": use randomized Hadamard orthogonal matrices.
                - "external": use user-provided matrices.

            r1:
                Optional global hidden-dimension rotation matrix. This is
                required when ``init_method == "external"`` and ``apply_r1`` is
                enabled.

            r2_map:
                Optional mapping from module keys to per-layer head-dimension
                rotation matrices.

                Supported example keys:
                    - "model.language_model.layers.0.self_attn.R2"
                    - "model.layers.0.self_attn.R2"

            apply_r1:
                Whether to apply the global hidden-dimension R1 rotation.

            apply_r2:
                Whether to apply the OV-side head-dimension R2 rotation.

            fuse_deepstack_visual_outputs:
                Whether to fuse R1 into DeepStack visual output projections.
                Main visual merger outputs are not fused because they pass
                through the input-side runtime rotation.

            show_progress:
                If True, display a tqdm progress bar during conversion.

            language_model_attr:
                Dotted path to the Qwen3-VL text model.

            text_layers_attr:
                Dotted path to the Qwen3-VL text decoder layers.

            visual_deepstack_mergers_attr:
                Dotted path to the DeepStack merger module list.

            lm_head_attr:
                Dotted path to the final language modeling head.
        """
        self.init_method = init_method
        self.r1 = r1
        self.r2_map = r2_map
        self.apply_r1 = apply_r1
        self.apply_r2 = apply_r2
        self.fuse_deepstack_visual_outputs = fuse_deepstack_visual_outputs
        self.show_progress = show_progress
        self.language_model_attr = language_model_attr
        self.text_layers_attr = text_layers_attr
        self.visual_deepstack_mergers_attr = visual_deepstack_mergers_attr
        self.lm_head_attr = lm_head_attr

        self._validate()

    def _validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If a configuration value is invalid.
            TypeError: If a field has an unexpected type.
        """
        if self.init_method not in {"random", "hadamard", "external"}:
            raise ValueError(f"Unsupported init_method: {self.init_method!r}")

        if self.apply_r1 and self.init_method == "external" and self.r1 is None:
            raise ValueError(
                "`r1` must be provided when init_method='external' and apply_r1=True."
            )

        if self.r1 is not None and not isinstance(self.r1, torch.Tensor):
            raise TypeError(f"`r1` must be a torch.Tensor, got {type(self.r1)}.")

        if self.r2_map is not None:
            if not isinstance(self.r2_map, dict):
                raise TypeError(
                    f"`r2_map` must be a dict[str, torch.Tensor], got {type(self.r2_map)}."
                )

            for key, value in self.r2_map.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Invalid r2_map key: {key!r}")
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"r2_map[{key!r}] must be a torch.Tensor, got {type(value)}."
                    )

        for field_name in (
            "language_model_attr",
            "text_layers_attr",
            "visual_deepstack_mergers_attr",
            "lm_head_attr",
        ):
            field_value = getattr(self, field_name)
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(
                    f"{field_name} must be a non-empty string, got {field_value!r}."
                )

    @property
    def name(self) -> str:
        return "spinquant"

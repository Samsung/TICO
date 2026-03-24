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


class SpinQuantConfig(BaseConfig):
    """
    Configuration for SpinQuant offline fusion.

    This configuration is intentionally scoped to the subset used by the PTQ
    framework:
        - global hidden-dimension rotation (R1)
        - head-dimension rotations (R2)

    Embedding tying is preserved by applying the corresponding transforms to
    dedicated rotation modules in the custom SpinLlama model instead of
    directly modifying the embedding table or LM head.
    """

    def __init__(
        self,
        init_method: Literal["random", "hadamard", "external"] = "random",
        r1: Optional[torch.Tensor] = None,
        r2_map: Optional[Dict[str, torch.Tensor]] = None,
        show_progress=True,
    ):
        """
        Initialize the SpinQuant configuration.

        Parameters:
            init_method:
                Strategy for resolving rotation matrices.

                - "random": use random orthogonal matrices.
                - "hadamard": use randomized Hadamard orthogonal matrices.
                - "external": use user-provided matrices.

            r1:
                Global hidden-dimension rotation matrix.

                This is required when init_method == "external".

            r2_map:
                Optional mapping from module keys to per-layer head-dimension
                rotation matrices.

                Example key:
                    "model.layers.0.self_attn.R2"

            show_progress:
                If True, display a tqdm progress bar during SpinQuant rotation
                application in `convert()`.
        """
        self.init_method = init_method
        self.r1 = r1
        self.r2_map = r2_map
        self.show_progress = show_progress

        self._validate()

    def _validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if self.init_method not in {"random", "hadamard", "external"}:
            raise ValueError(f"Unsupported init_method: {self.init_method}")

        if self.init_method == "external" and self.r1 is None:
            raise ValueError("`r1` must be provided when init_method='external'.")

        if self.r1 is not None and not isinstance(self.r1, torch.Tensor):
            raise ValueError(f"`r1` must be a torch.Tensor, got {type(self.r1)}.")

        if self.r2_map is not None:
            for key, value in self.r2_map.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"r2_map[{key}] must be a torch.Tensor, got {type(value)}."
                    )

    @property
    def name(self) -> str:
        return "spinquant"

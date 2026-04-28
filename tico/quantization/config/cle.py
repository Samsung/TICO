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

from typing import Literal, Optional, Sequence, Tuple

from tico.quantization.config.base import BaseConfig


class CLEConfig(BaseConfig):
    """
    Configuration for Cross-Layer Equalization.

    Cross-Layer Equalization is a data-free PTQ preprocessing method that
    rescales adjacent layers to balance their per-channel weight ranges.

    `pairs` accepts exact module-name pairs or glob-style pattern pairs.

    Example:
        pairs=[
            ("model.layers.*.mlp.up_proj", "model.layers.*.mlp.down_proj"),
        ]
    """

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        method: Literal["absmax", "range"] = "absmax",
        eps: float = 1e-12,
        scale_range: Tuple[float, float] = (1e-8, 1e8),
        max_iter: int = 1,
        equalize_bias: bool = True,
        show_progress: bool = True,
    ):
        self.pairs = pairs
        self.method = method
        self.eps = eps
        self.scale_range = scale_range
        self.max_iter = max_iter
        self.equalize_bias = equalize_bias
        self.show_progress = show_progress

    @property
    def name(self) -> str:
        return "cle"

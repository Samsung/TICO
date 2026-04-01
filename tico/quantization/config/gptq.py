# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from dataclasses import dataclass, field

import torch

from tico.quantization.config.base import BaseConfig


@dataclass
class GPTQConfig(BaseConfig):
    """
    Configuration for GPTQ weight quantization.

    Attributes
    ----------
    weight_bits : int
        Default bit-width applied to quantized weights.
    weight_bits_overrides : dict[str, int]
        Optional per-module bit-width overrides.

        Supported keys are matched in the following order:
          1) Full module name, for example `model.layers.0.self_attn.o_proj`
          2) Layer-local module name, for example `self_attn.o_proj`
          3) Full-name suffix, for example `self_attn.o_proj` or `down_proj`

        This makes it possible to keep a default bit-width for most modules
        while selectively increasing precision for specific projections.
    """

    # general
    verbose: bool = False
    show_progress: bool = True

    # quantizer.configure params (weight quantization spec)
    weight_bits: int = 8
    weight_bits_overrides: dict[str, int] = field(default_factory=dict)
    perchannel: bool = True
    symmetric: bool = False
    mse: str | None = None
    sensitivity: dict[str, torch.Tensor] | None = None

    # GPTQ.fasterquant params (algorithm hyperparams)
    percdamp: float = 0.01
    groupsize: int = -1
    actorder: bool = True
    static_groups: bool = False

    @property
    def name(self) -> str:
        return "gptq"

    def validate(self) -> None:
        if self.weight_bits <= 0:
            raise ValueError(f"weight_bits must be positive. got {self.weight_bits}")
        for module_name, bits in self.weight_bits_overrides.items():
            if bits <= 0:
                raise ValueError(
                    f"weight_bits_overrides[{module_name!r}] must be positive. got {bits}"
                )
        if self.groupsize != -1 and self.groupsize <= 0:
            raise ValueError(f"groupsize must be -1 or positive. got {self.groupsize}")
        if not (0.0 < self.percdamp <= 1.0):
            raise ValueError(f"percdamp must be in (0, 1]. got {self.percdamp}")

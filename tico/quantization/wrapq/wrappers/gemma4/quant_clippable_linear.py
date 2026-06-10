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
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4ClippableLinear")
class QuantGemma4ClippableLinear(QuantModuleBase):
    """PTQ wrapper for Gemma4 clippable linear layers.

    The wrapped module owns an inner ``nn.Linear`` and optional input/output
    clamp buffers. This wrapper keeps the clamp semantics while delegating
    weight and activation quantization to the inner ``QuantLinear`` wrapper.
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
        self.use_clipped_linears = bool(getattr(fp, "use_clipped_linears", False))
        self.linear = PTQWrapper(
            fp.linear, qcfg=qcfg.child("linear") if qcfg else None, fp_name=fp_name
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the clippable linear layer with quantized inner linear weights."""
        if self.use_clipped_linears:
            hidden_states = torch.clamp(
                hidden_states, self.module.input_min, self.module.input_max
            )
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(
                hidden_states, self.module.output_min, self.module.output_max
            )
        return hidden_states

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return ()

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

"""WrapQ support for ``tico.ops.Concat`` modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch

from tico.ops import Concat
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(Concat)
class QuantConcat(QuantModuleBase):
    """Force all concatenation inputs and the output into one quantization domain."""

    def __init__(
        self,
        fp: Concat,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ) -> None:
        """Create one shared observer for every connected concatenation tensor."""
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.obs_act_out = self._make_obs("act_out")

    def forward(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Concatenate inputs after applying one shared fake-quantization domain."""
        values = tuple(tensors)
        if not values:
            raise ValueError("QuantConcat requires at least one input tensor.")
        inputs_q = tuple(self._fq(value, self.obs_act_out) for value in values)
        output = self.module(inputs_q)
        return self._fq(output, self.obs_act_out)

    def _all_observers(self):
        """Return the single shared concatenation observer."""
        return (self.obs_act_out,)

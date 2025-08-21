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

from typing import Callable, Optional

import torch
import torch.nn as nn

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import register


class QuantElementwise(QuantModuleBase):
    """
    Generic wrapper for any 1-to-1 element-wise op `y = f(x)`.

    Sub-classes only need to implement:
        • `FUNC`: a Callable that maps tensor→tensor
    """

    # subclass must set this
    FUNC: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is QuantElementwise:
            return
        if cls.FUNC is None:
            raise NotImplementedError(
                f"{cls.__name__} must define a staticmethod `FUNC(tensor) -> tensor`"
            )

    def __init__(
        self,
        fp_module: nn.Module,
        *,
        qcfg: Optional[QuantConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_module
        self.act_in_obs = self._make_obs("act_in")
        self.act_out_obs = self._make_obs("act_out")

    # ------------------------------------------------------------
    def forward(self, x):
        x_q = self._fq(x, self.act_in_obs)
        assert self.FUNC is not None
        y = self.FUNC(x_q)  # element-wise op
        y_q = self._fq(y, self.act_out_obs)
        return y_q

    # ------------------------------------------------------------
    def _all_observers(self):
        return (self.act_in_obs, self.act_out_obs)


"""
Why `FUNC` is a `staticmethod`

- Prevents automatic binding: calling `self.FUNC(x)` will not inject `self`,
  so the callable keeps the expected signature `Tensor -> Tensor`
  (e.g., `torch.sigmoid(x)`), avoiding TypeErrors.

- Expresses purity and statelessness: `FUNC` is a pure, element-wise transform
  that must not read or mutate module state (params, buffers, config).

- Tracing/export friendly (FX / TorchScript): the call is captured as
  `call_function(torch.*)` instead of a bound `call_method`, which makes graph
  rewrites/pattern-matching and backends' substitutions more reliable.

- Avoids submodule pollution: we keep a functional op (`torch.relu`) rather
  than an `nn.Module` instance that would appear in the module tree.

- Small perf/alloc win: no bound-method objects are created on each call.
"""

# Sigmoid
@register(nn.Sigmoid)
class QuantSigmoid(QuantElementwise):
    FUNC = staticmethod(torch.sigmoid)


# Tanh
@register(nn.Tanh)
class QuantTanh(QuantElementwise):
    FUNC = staticmethod(torch.tanh)


# ReLU
@register(nn.ReLU)
class QuantReLU(QuantElementwise):
    FUNC = staticmethod(torch.relu)


# GELU (approximate)
@register(nn.GELU)
class QuantGELU(QuantElementwise):
    FUNC = staticmethod(torch.nn.functional.gelu)

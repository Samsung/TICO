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

from typing import Any, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


class QuantElementwise(QuantModuleBase):
    """
    Generic wrapper for any 1-to-1 element-wise op `y = f(x)`.

    Sub-classes only need to implement:
        • `FUNC`: a Callable that maps tensor→tensor
    """

    # subclass must set this
    FUNC: Any = None

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
        qcfg: Optional[PTQConfig] = None,
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
Q1) Why `FUNC` is a `staticmethod`

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

Q2) Why we define small Python wrappers (_relu, _tanh, etc.)

- torch.relu / torch.tanh / torch.sigmoid are CPython built-ins.
  Their type is `builtin_function_or_method`, not a Python `FunctionType`.
  This causes `torch.export` (and FX tracing) to fail with:
    "expected FunctionType, found builtin_function_or_method".

- By defining a thin Python wrapper (e.g., `def _tanh(x): return torch.tanh(x)`),
  we convert it into a normal Python function object (`FunctionType`),
  which satisfies export/tracing requirements.

- Functionally, this adds zero overhead and preserves semantics,
  but makes the callable introspectable (has __code__, __name__, etc.)
  and compatible with TorchDynamo / FX graph capture.

- It also keeps FUNC pure and stateless, ensuring the elementwise op
  is represented as `call_function(_tanh)` in the traced graph
  rather than a bound `call_method` or module attribute access.
"""


def _relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def _tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)


@register(nn.Sigmoid)
class QuantSigmoid(QuantElementwise):
    @staticmethod
    def FUNC(x: torch.Tensor) -> torch.Tensor:
        return _sigmoid(x)


@register(nn.Tanh)
class QuantTanh(QuantElementwise):
    @staticmethod
    def FUNC(x: torch.Tensor) -> torch.Tensor:
        return _tanh(x)


@register(nn.ReLU)
class QuantReLU(QuantElementwise):
    @staticmethod
    def FUNC(x: torch.Tensor) -> torch.Tensor:
        return _relu(x)


@register(nn.GELU)
class QuantGELU(QuantElementwise):
    @staticmethod
    def FUNC(x: torch.Tensor) -> torch.Tensor:
        return _gelu(x)

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
from typing import Any, Dict, Mapping, Type

from tico.experimental.quantization.algorithm.ptq.dtypes import DType
from tico.experimental.quantization.algorithm.ptq.observers.base import ObserverBase
from tico.experimental.quantization.algorithm.ptq.observers.minmax import MinMaxObserver


@dataclass
class QuantConfig:
    """
    *One* object describes the quantization preferences for a single wrapper
    **and its descendants**.

    Parameters
    ----------
    default_dtype : DType
        Fallback dtype for every observer that **does not** receive an explicit
        override.
    default_observer : Type[ObserverBase], optional
        Observer class to instantiate when the caller (or an override) does
         not provide a `observer` key.
    overrides : Mapping[str, Mapping[str, Any]]
        Two-level mapping of *scopes* → *observer-kwargs*.

        • **Scope** can be either
            - the *attribute name* of a child wrapper
              (e.g. ``"gate_proj"`` or ``"up_proj"``), or
            - an *observer logical name* inside *this* wrapper
              (e.g. ``"mul"``, ``"act_in"``).

        • **Observer-kwargs** is forwarded verbatim to the observer constructor
          (`dtype`, `num_bits`, `observer`, …).

    Example
    -------
    ```python
    from ptq.observers import PercentileObserver

    cfg = QuantConfig(
        default_dtype   = DType.uint(8),
        default_observer = PercentileObserver,   # <- global algorithm
        overrides={
            # local override: input observer now MinMax & 4-bit
            "act_in": {"observer": MinMaxObserver,
                       "dtype":   DType.uint(4)},
        },
    )
    ```
    """

    default_dtype: DType = DType.uint(8)
    default_observer: Type[ObserverBase] = MinMaxObserver
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def get_kwargs(self, obs_name: str) -> Dict[str, Any]:
        """
        Return user-specified kwargs for *obs_name* inside **this** wrapper.

        NOTE:
        Do NOT inject a dtype here. `_make_obs()` resolves precedence:
            1) user override (kw_cfg["dtype"])
            2) wrapper's default passed to `_make_obs(..., dtype=...)`
            3) self.default_dtype
        """
        return dict(self.overrides.get(obs_name, {}))

    def child(self, scope: str) -> "QuantConfig":
        """
        Produce a *view* for a child wrapper.

        The child inherits:
          • same `default_dtype`
          • overrides under `self.overrides.get(scope, {})`

        Other scopes remain invisible to the child.
        """
        sub_overrides = self.overrides.get(scope, {})
        return QuantConfig(self.default_dtype, self.default_observer, sub_overrides)

    def __repr__(self):
        return f"QuantConfig(default_dtype={self.default_dtype}, default_observer={self.default_observer}, overrides={dict(self.overrides)})"

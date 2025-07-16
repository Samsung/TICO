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
from typing import Any, Dict, Mapping

from tico.experimental.quantization.custom.dtypes import DType


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
    overrides : Mapping[str, Mapping[str, Any]]
        Two-level mapping of *scopes* → *observer-kwargs*.

        • **Scope** can be either
            - the *attribute name* of a child wrapper
              (e.g. ``"gate_proj"`` or ``"up_proj"``), or
            - an *observer logical name* inside *this* wrapper
              (e.g. ``"mul"``, ``"act_in"``).

        • **Observer-kwargs** is forwarded verbatim to the observer constructor
          (`dtype`, `num_bits`, `factory`, …).

    Example
    -------
    ```python
    cfg = QuantConfig(
        default_dtype=DType.uint(8),
        overrides={
            # apply only to this wrapper
            "mul": {"dtype": DType.uint(4)},

            # apply to a child wrapper (gate_proj)
            "gate_proj": {
                "act_in": {"dtype": DType.uint(4)}
            },
        },
    )
    ```
    """

    default_dtype: DType = DType.uint(8)
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def get_kwargs(self, obs_name: str) -> Dict[str, Any]:
        """
        Return kwargs to construct *obs_name* inside **this** wrapper.

        • Always inject `"dtype"` if the caller didn’t specify it.
        """
        kw: Dict[str, Any] = dict(self.overrides.get(obs_name, {}))
        kw.setdefault("dtype", self.default_dtype)
        return kw

    def child(self, scope: str) -> "QuantConfig":
        """
        Produce a *view* for a child wrapper.

        The child inherits:
          • same `default_dtype`
          • overrides under `self.overrides.get(scope, {})`

        Other scopes remain invisible to the child.
        """
        sub_overrides = self.overrides.get(scope, {})
        return QuantConfig(self.default_dtype, sub_overrides)

    def __repr__(self):
        return f"QuantConfig(default_dtype={self.default_dtype}, overrides={dict(self.overrides)})"

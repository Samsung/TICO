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
from typing import Any, Dict, Literal, Mapping, Type

from tico.quantization.config.base import BaseConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme

WrapperVariant = Literal["common", "prefill", "decode"]


@dataclass
class PTQConfig(BaseConfig):
    """
    One object describes the quantization preferences for a single wrapper
    and its descendants.

    Parameters
    ----------
    default_dtype : DType
        Fallback dtype for every observer that DOES NOT receive an explicit
        override.
    default_observer : Type[ObserverBase], optional
        Observer class to instantiate when the caller (or an override) does
         not provide a `observer` key.
    default_qscheme : QScheme
        Fallback quantization scheme (per-tensor / per-channel,
        asymmetric / symmetric) for observers that DO NOT receive an explicit
        override.
    wrapper_variant : str
        Execution specialization used when resolving quantization wrappers.

        Typical values:
            - "prefill" : full-sequence execution (prompt processing).
            - "decode"  : single-token autoregressive decoding.
            - "common"  : variant-independent implementation shared by
                          multiple execution modes

        The "common" variant is used for modules whose computation does not
        depend on the execution mode (e.g., Linear, LayerNorm, or MLP blocks).
        When a wrapper for the requested variant is not available, the registry
        will prefer a "common" implementation before falling back to other
        variants.

        The selected variant propagates automatically to child configurations,
        allowing entire model subgraphs to switch execution mode consistently.
    overrides : Mapping[str, Mapping[str, Any]]
        Two-level mapping of scopes → observer-kwargs.

        • SCOPE can be either
            - the attribute name of a child wrapper
              (e.g. "gate_proj" or "up_proj"), or
            - an observer logical name inside this wrapper
              (e.g. "mul", "act_in").

        • "Observer-kwargs" is forwarded verbatim to the observer constructor
          (`dtype`, `qscheme`, `channel_axis`, `observer`, …).

    Example
    -------
    ```python
    from wrapq.observers import PercentileObserver

    cfg = PTQConfig(
        default_dtype   = DType.uint(8),
        default_qscheme  = QScheme.PER_TENSOR_SYMM,        # <- global scheme
        default_observer = PercentileObserver,             # <- global algorithm
        wrapper_variant = "prefill",
        overrides={
            # local override: input observer now MinMax & 4-bit, per-channel asymmetric
            "act_in": {"observer": MinMaxObserver,
                       "dtype":    DType.uint(4),
                       "qscheme":  QScheme.PER_CHANNEL_ASYMM},
        },
    )
    ```
    """

    default_dtype: DType = DType.uint(8)
    default_observer: Type[ObserverBase] = MinMaxObserver  # type: ignore[type-abstract]
    default_qscheme: QScheme = QScheme.PER_TENSOR_ASYMM
    wrapper_variant: WrapperVariant = "common"
    overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # If True, any module that cannot be wrapped will raise.
    strict_wrap: bool = True

    @property
    def name(self) -> str:
        return "ptq"

    def get_kwargs(self, obs_name: str) -> Dict[str, Any]:
        """
        Return user-specified kwargs for *obs_name* inside **this** wrapper.

        NOTE:
        Do NOT inject a dtype/qscheme here. `_make_obs()` resolves precedence:
            1) user override (kw_cfg["dtype" | "qscheme"])
            2) wrapper's default passed to `_make_obs(..., dtype=..., qscheme=...)`
            3) self.default_dtype / `self.default_qscheme`
        """
        return dict(self.overrides.get(obs_name, {}))

    def child(self, scope: str) -> "PTQConfig":
        """
        Produce a *view* for a child wrapper.

        The child inherits:
          • same `default_dtype`
          • same `default_observer`
          • same `default_qscheme`
          • overrides under `self.overrides.get(scope, {})`

        Other scopes remain invisible to the child.
        """
        sub_overrides = self.overrides.get(scope, {})
        return PTQConfig(
            self.default_dtype,
            self.default_observer,
            default_qscheme=self.default_qscheme,
            wrapper_variant=self.wrapper_variant,
            overrides=sub_overrides,
            strict_wrap=self.strict_wrap,
        )

    def __repr__(self):
        return f"PTQConfig(default_dtype={self.default_dtype}, default_observer={self.default_observer}, default_qscheme={self.default_qscheme}, overrides={dict(self.overrides)}, strict_wrap={self.strict_wrap})"

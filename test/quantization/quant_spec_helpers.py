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

"""Test helpers for explicit QuantSpec-based PTQ configuration.

The helpers in this module are not compatibility shims for the production API.
They keep test setup concise while all tests still build PTQConfig through the
new ``activation=affine(...)`` and ``weight=affine(...)`` fields.
"""

from typing import Any, Mapping, Optional, Type

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme


def make_affine_ptq_config(
    *,
    dtype: DType = DType.uint(8),
    qscheme: Optional[QScheme] = None,
    observer: Type[ObserverBase] = MinMaxObserver,
    overrides: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> PTQConfig:
    """Build a PTQConfig with the same affine spec for activations and params."""
    spec = affine(dtype, qscheme=qscheme, observer=observer)
    return PTQConfig(
        activation=spec,
        weight=spec,
        overrides={} if overrides is None else overrides,
        **kwargs,
    )

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

"""NumPy/PyTorch reference runtime for serialized Circle models.

See ``tico/circle/README.md`` ("Reference runtime") for the supported operator
set, dynamic-shape rules, quantized execution modes, and debugging options.
"""

from tico.circle.runtime.errors import (
    CircleRuntimeError,
    CircleRuntimeValidationError,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.executor import (
    CircleReferenceRuntime,
    ExecutionResult,
    run_circle,
)
from tico.circle.runtime.kernels import (
    default_kernel_registry,
    ExecutionMode,
    KernelContext,
    KernelRegistry,
)
from tico.circle.runtime.program import CircleProgram, RuntimeOperator, RuntimeTensor

__all__ = [
    "CircleProgram",
    "CircleReferenceRuntime",
    "CircleRuntimeError",
    "CircleRuntimeValidationError",
    "ExecutionMode",
    "ExecutionResult",
    "KernelContext",
    "KernelRegistry",
    "RuntimeOperator",
    "RuntimeTensor",
    "UnsupportedCircleOperatorError",
    "default_kernel_registry",
    "run_circle",
]

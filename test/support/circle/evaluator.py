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

"""Test-facing wrapper around the Circle reference runtime.

Value tests historically used a small NumPy evaluator that lived here. The
numerical kernels now live in ``tico.circle.runtime``; this module keeps the
value-test API (``evaluate`` returning outputs plus every intermediate tensor)
and the dtype helpers used by :class:`CircleModelBuilder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.value import default_tensor_type_registry


@dataclass(frozen=True)
class CircleEvaluationResult:
    """Store graph outputs and all tensor values produced during evaluation."""

    outputs: tuple[np.ndarray, ...]
    tensor_values: dict[int, np.ndarray]


def numpy_dtype_from_circle_tensor_type(tensor_type: int) -> np.dtype[Any]:
    """Return the logical NumPy dtype of a Circle tensor type."""

    return np.dtype(
        default_tensor_type_registry().by_value(int(tensor_type)).logical_dtype
    )


def circle_tensor_type_from_numpy_dtype(dtype: np.dtype[Any] | type[Any]) -> int:
    """Return the dense Circle tensor type corresponding to a NumPy dtype."""

    normalized = np.dtype(dtype).newbyteorder("=")
    for spec in default_tensor_type_registry().specs:
        if spec.packed or spec.name == "BFLOAT16":
            continue
        if spec.logical_dtype.newbyteorder("=") == normalized:
            return spec.tensor_type
    raise NotImplementedError(
        f"Circle value-test fixtures do not support NumPy dtype {normalized}."
    )


class CircleReferenceEvaluator:
    """Evaluate a Circle document with the reference runtime and keep all values.

    Every intermediate tensor is retained (``trace=True``) so that extraction
    value tests can compare region boundaries against the source graph.
    """

    def __init__(self, *, mode: ExecutionMode = ExecutionMode.NATIVE) -> None:
        """Create an evaluator for the given execution mode."""

        self.mode = mode

    def evaluate(
        self,
        document: CircleDocument,
        inputs: tuple[np.ndarray, ...],
        *,
        subgraph_index: int = 0,
    ) -> CircleEvaluationResult:
        """Evaluate one subgraph and return outputs plus intermediate values."""

        runtime = CircleReferenceRuntime(document, subgraph_index=subgraph_index)
        result = runtime.run(
            tuple(np.asarray(value) for value in inputs),
            mode=self.mode,
            trace=True,
        )
        assert result.tensor_values is not None
        return CircleEvaluationResult(
            outputs=result.outputs,
            tensor_values=result.tensor_values,
        )

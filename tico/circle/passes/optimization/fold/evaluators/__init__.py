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

from __future__ import annotations

from typing import Any

from tico.circle._schema import circle_schema
from tico.circle.passes.optimization.fold.evaluators.base import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    ConstantEvaluatorRegistry,
)
from tico.circle.passes.optimization.fold.evaluators.elementwise import (
    BinaryElementwiseEvaluator,
    CastEvaluator,
)
from tico.circle.passes.optimization.fold.evaluators.heavy import (
    DensifyEvaluator,
    DepthwiseConv2DEvaluator,
    DequantizeEvaluator,
    FullyConnectedEvaluator,
    HeavyConstantEvaluatorPolicy,
    register_heavy_constant_evaluators,
    SparseToDenseEvaluator,
)
from tico.circle.passes.optimization.fold.evaluators.indexing import GatherEvaluator
from tico.circle.passes.optimization.fold.evaluators.shape import (
    ReshapeEvaluator,
    ShapeEvaluator,
    SqueezeEvaluator,
)


def default_constant_evaluator_registry() -> ConstantEvaluatorRegistry:
    """Create the first-stage registry for common Circle constant operations."""

    return ConstantEvaluatorRegistry(
        (
            (_builtin_operator_value("ADD"), BinaryElementwiseEvaluator("ADD", _add)),
            (_builtin_operator_value("MUL"), BinaryElementwiseEvaluator("MUL", _mul)),
            (_builtin_operator_value("CAST"), CastEvaluator()),
            (_builtin_operator_value("RESHAPE"), ReshapeEvaluator()),
            (_builtin_operator_value("SHAPE"), ShapeEvaluator()),
            (_builtin_operator_value("SQUEEZE"), SqueezeEvaluator()),
            (_builtin_operator_value("GATHER"), GatherEvaluator()),
        )
    )


def _builtin_operator_value(name: str) -> int:
    """Return one generated BuiltinOperator enum value by symbolic name."""

    schema = circle_schema()
    enum_module = getattr(schema, "BuiltinOperator", None)
    enum_type = (
        getattr(enum_module, "BuiltinOperator", None)
        if enum_module is not None
        else None
    )
    if enum_type is None:
        enum_type = getattr(schema, "BuiltinOperator", None)
    if enum_type is None or not hasattr(enum_type, name):
        raise RuntimeError(f"Circle schema does not provide BuiltinOperator.{name}.")
    return int(getattr(enum_type, name))


def _add(lhs: Any, rhs: Any) -> Any:
    """Apply NumPy addition without importing a mutable ufunc into the registry API."""

    return lhs + rhs


def _mul(lhs: Any, rhs: Any) -> Any:
    """Apply multiplication without exposing a mutable ufunc in the registry API."""

    return lhs * rhs


__all__ = [
    "BinaryElementwiseEvaluator",
    "CastEvaluator",
    "ConstantEvaluation",
    "ConstantEvaluationContext",
    "ConstantEvaluator",
    "ConstantEvaluatorRegistry",
    "DensifyEvaluator",
    "DepthwiseConv2DEvaluator",
    "DequantizeEvaluator",
    "FullyConnectedEvaluator",
    "HeavyConstantEvaluatorPolicy",
    "SparseToDenseEvaluator",
    "GatherEvaluator",
    "ReshapeEvaluator",
    "ShapeEvaluator",
    "SqueezeEvaluator",
    "default_constant_evaluator_registry",
    "register_heavy_constant_evaluators",
]

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

from collections.abc import Mapping

from tico.circle._object import ObjectFactory
from tico.circle.passes.optimization.fold.constant_subgraph import (
    ConstantFoldPolicy,
    FoldConstantSubgraphPass,
)
from tico.circle.passes.optimization.fold.evaluators import (
    ConstantEvaluatorRegistry,
    default_constant_evaluator_registry,
)
from tico.circle.passes.optimization.fold.evaluators.heavy import (
    HeavyConstantEvaluatorPolicy,
    register_heavy_constant_evaluators,
)
from tico.circle.value import TensorValueCodec


def heavy_constant_evaluator_registry(
    *,
    policy: HeavyConstantEvaluatorPolicy | None = None,
    builtin_codes: Mapping[str, int] | None = None,
    padding_values: Mapping[str, int] | None = None,
) -> ConstantEvaluatorRegistry:
    """Create the default registry extended with opt-in expensive evaluators."""

    registry = default_constant_evaluator_registry()
    return register_heavy_constant_evaluators(
        registry,
        policy=policy,
        builtin_codes=builtin_codes,
        padding_values=padding_values,
    )


class FoldHeavyConstantSubgraphPass(FoldConstantSubgraphPass):
    """Fold common and heavy constant operators under the shared fold budget."""

    def __init__(
        self,
        *,
        fold_policy: ConstantFoldPolicy | None = None,
        evaluator_policy: HeavyConstantEvaluatorPolicy | None = None,
        evaluator_registry: ConstantEvaluatorRegistry | None = None,
        builtin_codes: Mapping[str, int] | None = None,
        padding_values: Mapping[str, int] | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create a heavy fold pass with injectable schema-independent services."""

        registry = evaluator_registry or heavy_constant_evaluator_registry(
            policy=evaluator_policy,
            builtin_codes=builtin_codes,
            padding_values=padding_values,
        )
        super().__init__(
            policy=fold_policy,
            evaluator_registry=registry,
            codec=codec,
            object_factory=object_factory,
        )


__all__ = [
    "FoldHeavyConstantSubgraphPass",
    "heavy_constant_evaluator_registry",
]

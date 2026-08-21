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
from enum import Enum

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


class ConstantFoldingProfile(str, Enum):
    """Select the evaluator family used by constant folding."""

    BASIC = "basic"
    HEAVY = "heavy"


def heavy_constant_evaluator_registry(
    *,
    policy: HeavyConstantEvaluatorPolicy | None = None,
    builtin_codes: Mapping[str, int] | None = None,
    padding_values: Mapping[str, int] | None = None,
) -> ConstantEvaluatorRegistry:
    """Create the basic registry extended with expensive evaluators."""

    registry = default_constant_evaluator_registry()
    return register_heavy_constant_evaluators(
        registry,
        policy=policy,
        builtin_codes=builtin_codes,
        padding_values=padding_values,
    )


class FoldConstantsPass(FoldConstantSubgraphPass):
    """Fold constants using one explicitly selected evaluator profile."""

    def __init__(
        self,
        *,
        profile: ConstantFoldingProfile | str = ConstantFoldingProfile.BASIC,
        policy: ConstantFoldPolicy | None = None,
        evaluator_policy: HeavyConstantEvaluatorPolicy | None = None,
        evaluator_registry: ConstantEvaluatorRegistry | None = None,
        builtin_codes: Mapping[str, int] | None = None,
        padding_values: Mapping[str, int] | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Build the selected registry and delegate to the fold engine."""

        selected = ConstantFoldingProfile(profile)
        if evaluator_registry is None and selected is ConstantFoldingProfile.HEAVY:
            evaluator_registry = heavy_constant_evaluator_registry(
                policy=evaluator_policy,
                builtin_codes=builtin_codes,
                padding_values=padding_values,
            )
        # BASIC intentionally leaves the registry unset so the existing
        # fold engine preserves its lazy schema initialization.
        self.profile = selected
        super().__init__(
            policy=policy,
            evaluator_registry=evaluator_registry,
            codec=codec,
            object_factory=object_factory,
        )


class FoldHeavyConstantSubgraphPass(FoldConstantsPass):
    """Backward-compatible spelling for the heavy folding profile."""

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
        """Preserve the former constructor while selecting HEAVY."""

        super().__init__(
            profile=ConstantFoldingProfile.HEAVY,
            policy=fold_policy,
            evaluator_policy=evaluator_policy,
            evaluator_registry=evaluator_registry,
            builtin_codes=builtin_codes,
            padding_values=padding_values,
            codec=codec,
            object_factory=object_factory,
        )


__all__ = [
    "ConstantFoldingProfile",
    "FoldConstantsPass",
    "FoldHeavyConstantSubgraphPass",
    "heavy_constant_evaluator_registry",
]

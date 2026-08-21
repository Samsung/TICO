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


from tico.circle.passes.optimization.fold.constant_subgraph import (
    ConstantFoldPlan,
    ConstantFoldPolicy,
    FoldConstantSubgraphPass,
)
from tico.circle.passes.optimization.fold.evaluators import (
    ConstantEvaluation,
    ConstantEvaluationContext,
    ConstantEvaluator,
    ConstantEvaluatorRegistry,
    default_constant_evaluator_registry,
    HeavyConstantEvaluatorPolicy,
    register_heavy_constant_evaluators,
)
from tico.circle.passes.optimization.fold.profiles import (
    ConstantFoldingProfile,
    FoldConstantsPass,
    FoldHeavyConstantSubgraphPass,
    heavy_constant_evaluator_registry,
)

__all__ = [
    "ConstantEvaluation",
    "ConstantEvaluationContext",
    "ConstantEvaluator",
    "ConstantEvaluatorRegistry",
    "ConstantFoldPlan",
    "ConstantFoldPolicy",
    "ConstantFoldingProfile",
    "FoldConstantsPass",
    "FoldConstantSubgraphPass",
    "FoldHeavyConstantSubgraphPass",
    "HeavyConstantEvaluatorPolicy",
    "default_constant_evaluator_registry",
    "heavy_constant_evaluator_registry",
    "register_heavy_constant_evaluators",
]

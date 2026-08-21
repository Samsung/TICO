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


from tico.circle.passes.optimization.simplify.arithmetic import (
    ArithmeticCanonicalizationPolicy,
    CanonicalizeArithmeticPass,
    SimplifyArithmeticPass,
)
from tico.circle.passes.optimization.simplify.identity_ops import (
    EliminateIdentityOpsPass,
    RemoveNoOpOperatorsPass,
)
from tico.circle.passes.optimization.simplify.reductions import (
    ReductionSimplificationPolicy,
    SimplifyReductionOpsPass,
)
from tico.circle.passes.optimization.simplify.transpose_region import (
    EliminateTransposeBoundedLayoutRegionPass,
)
from tico.circle.passes.optimization.simplify.views import SimplifyViewOpsPass

__all__ = [
    "ArithmeticCanonicalizationPolicy",
    "CanonicalizeArithmeticPass",
    "EliminateIdentityOpsPass",
    "EliminateTransposeBoundedLayoutRegionPass",
    "ReductionSimplificationPolicy",
    "RemoveNoOpOperatorsPass",
    "SimplifyArithmeticPass",
    "SimplifyReductionOpsPass",
    "SimplifyViewOpsPass",
]

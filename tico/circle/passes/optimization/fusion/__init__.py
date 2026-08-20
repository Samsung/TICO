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

from tico.circle.passes.optimization.fusion.composite_ops import (
    CompositeFusionPolicy,
    FuseCompositeOpsPass,
)
from tico.circle.passes.optimization.fusion.legacy_fc_gelu_fc import (
    FuseLegacyFCGeluFCPass,
    LegacyFCGeluFCFusionPolicy,
)
from tico.circle.passes.optimization.fusion.linear_ops import (
    FuseLinearOpsPass,
    LinearFusionPolicy,
)
from tico.circle.passes.optimization.fusion.reduction_ops import (
    ReductionSimplificationPolicy,
    SimplifyReductionOpsPass,
)
from tico.circle.passes.optimization.fusion.transpose_conv_slice import (
    FuseTransposeConvSlicePass,
    TransposeConvSliceFusionPolicy,
)

__all__ = [
    "CompositeFusionPolicy",
    "FuseCompositeOpsPass",
    "FuseLegacyFCGeluFCPass",
    "FuseLinearOpsPass",
    "FuseTransposeConvSlicePass",
    "LegacyFCGeluFCFusionPolicy",
    "LinearFusionPolicy",
    "ReductionSimplificationPolicy",
    "SimplifyReductionOpsPass",
    "TransposeConvSliceFusionPolicy",
]

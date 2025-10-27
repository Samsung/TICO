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

from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch

from tico.quantization.algorithm.pt2e.annotation.config import QuantizationConfig

AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        torch.fx.Node,
        Optional[QuantizationConfig],
        Optional[Callable[[torch.fx.Node], bool]],
    ],
    None,
]
OP_TO_ANNOTATOR: Dict[torch._ops.OpOverload, AnnotatorType] = {}
OP_TO_SHARE_QUANT_SPEC: List[Callable] = [
    torch.ops.aten.view_copy.default,
    torch.ops.aten.view.default,
]


def register_annotator(target: List[torch._ops.OpOverload]):
    def decorator(annotator: AnnotatorType):
        for t in target:
            OP_TO_ANNOTATOR[t] = annotator
        return annotator

    return decorator

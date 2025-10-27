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

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.ao.quantization.quantizer import SharedQuantizationSpec

import tico.quantization.algorithm.pt2e.annotation.spec as annot_spec
import tico.quantization.algorithm.pt2e.annotation.utils as annot_utils
import tico.quantization.algorithm.pt2e.utils as quant_utils
from tico.quantization.algorithm.pt2e.annotation.config import QuantizationConfig
from tico.utils.validate_args_kwargs import AdaptiveAvgPool2dArgs


@annot_spec.register_annotator([torch.ops.aten.adaptive_avg_pool2d.default])
def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
):
    if (
        node.op != "call_function"
        or node.target != torch.ops.aten.adaptive_avg_pool2d.default
    ):
        return
    if filter_fn and not filter_fn(node):
        return
    if quant_utils.is_annotated(node):
        return

    args = AdaptiveAvgPool2dArgs(*node.args)  # type: ignore[arg-type]
    input = args.input

    assert isinstance(input, torch.fx.Node)
    if (
        "quantization_annotation" not in input.meta
        or not input.meta["quantization_annotation"]._annotated
        or input.meta["quantization_annotation"].output_qspec is None
    ):
        input_act_qspec = quant_utils.get_input_act_qspec(quantization_config)
    else:
        input_act_qspec = SharedQuantizationSpec(input)
    annot_utils.annotate_input_qspec_map(node, input, input_act_qspec)

    output_act_qspec = SharedQuantizationSpec((input, node))
    annot_utils.annotate_output_qspec(node, output_act_qspec)

    annot_utils.mark_nodes_as_annotated(node)

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
from torch.ao.quantization.quantizer import DerivedQuantizationSpec

import tico.quantization.algorithm.pt2e.annotation.spec as annot_spec
import tico.quantization.algorithm.pt2e.annotation.utils as annot_utils
import tico.quantization.algorithm.pt2e.utils as quant_utils
from tico.quantization.algorithm.pt2e.annotation.config import QuantizationConfig
from tico.utils.validate_args_kwargs import Conv2DArgs


@annot_spec.register_annotator(
    [torch.ops.aten.conv2d.default, torch.ops.aten.conv2d.padding]
)
def _annotate_conv2d(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
):
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
        ]:
            continue
        if filter_fn and not filter_fn(node):
            continue
        if quant_utils.is_annotated(node):
            continue

        args = Conv2DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input_ = args.input
        weight = args.weight
        bias = args.bias

        input_act_qspec = quant_utils.get_input_act_qspec(quantization_config)
        weight_qspec = quant_utils.get_weight_qspec(quantization_config)
        annot_utils.annotate_input_qspec_map(node, input_, input_act_qspec)
        annot_utils.annotate_input_qspec_map(node, weight, weight_qspec)
        nodes_to_mark_annotated = [input_, weight, node]
        if bias:

            def _derive_bias_qparams_from_act_and_weight_qparams(obs_or_fqs):
                act_scale, _ = obs_or_fqs[0].calculate_qparams()
                weight_scale, _ = obs_or_fqs[1].calculate_qparams()
                bias_scale = act_scale * weight_scale
                bias_zero_point = torch.zeros_like(bias_scale, dtype=torch.int32)
                return bias_scale, bias_zero_point

            bias_qspec = DerivedQuantizationSpec(
                derived_from=[
                    (input_, node),
                    (weight, node),
                ],
                derive_qparams_fn=_derive_bias_qparams_from_act_and_weight_qparams,
                dtype=torch.int32,
                quant_min=-(2**31),
                quant_max=2**31 - 1,
                qscheme=weight_qspec.qscheme,
                ch_axis=0 if weight_qspec.qscheme == torch.per_channel_affine else None,
            )
            annot_utils.annotate_input_qspec_map(
                node,
                bias,
                bias_qspec,
            )
            nodes_to_mark_annotated.append(bias)

        output_act_qspec = quant_utils.get_output_act_qspec(quantization_config)
        annot_utils.annotate_output_qspec(node, output_act_qspec)

        annot_utils.mark_nodes_as_annotated(nodes_to_mark_annotated)

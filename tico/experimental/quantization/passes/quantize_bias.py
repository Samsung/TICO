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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import copy

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam, to_qparam_dtype
from tico.utils import logging
from tico.utils.graph import add_placeholder, get_torch_param_value, is_torch_param
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import LinearArgs


@trace_graph_diff_on_pass
class QuantizeBias(PassBase):
    """
    Quantize bias.

    This pass identifies fp32 biases, quantizes them using scales of input and weights.

    This pass assumes that if bias is fp32, input and weights must have been quantized.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.linear.default:
                lin_args = LinearArgs(*node.args, **node.kwargs)
                inp = lin_args.input
                weights = lin_args.weight
                bias = lin_args.bias

                if bias is None:
                    continue

                # Only support bias is Parameter
                # TODO Is it possible that bias is not Parameter?
                if not is_torch_param(bias, exported_program):
                    continue

                bias_val: torch.Tensor = get_torch_param_value(bias, exported_program)
                if bias_val.dtype != torch.float32:
                    continue

                if QPARAM_KEY not in inp.meta:
                    continue

                if QPARAM_KEY not in weights.meta:
                    continue

                quant_dtype = None
                if inp.meta[QPARAM_KEY].dtype == "int16":
                    quant_dtype = torch.int64
                elif inp.meta[QPARAM_KEY].dtype == "uint8":
                    quant_dtype = torch.int32
                else:
                    continue

                type_info = torch.iinfo(quant_dtype)

                assert quant_dtype is not None

                i_scale = inp.meta[QPARAM_KEY].scale
                w_scale = weights.meta[QPARAM_KEY].scale

                assert i_scale is not None
                assert w_scale is not None
                assert len(i_scale) == 1
                assert len(w_scale) == bias_val.shape[0]

                bias_scale = torch.tensor(i_scale) * torch.tensor(w_scale)
                q_bias = torch.round(bias_val / bias_scale)
                q_bias = torch.clamp(q_bias, min=type_info.min, max=type_info.max)
                q_bias = q_bias.to(quant_dtype)

                q_bias_node = add_placeholder(exported_program, q_bias, bias.name)

                qparam = QuantParam()
                qparam.scale = bias_scale.tolist()
                assert qparam.scale is not None
                qparam.zero_point = [0] * len(qparam.scale)
                qparam.dtype = to_qparam_dtype(quant_dtype)
                qparam.quantized_dimension = 0
                q_bias_node.meta[QPARAM_KEY] = qparam

                node.update_arg(2, q_bias_node)

                logger.debug(f"Bias ({bias.name}) is quantized to {q_bias_node.name}.")

            # TODO Support more ops.

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)

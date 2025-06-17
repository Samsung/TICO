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

from typing import List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch.fx
import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    is_buffer,
    is_lifted_tensor_constant,
)
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam, to_qparam_dtype
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import (
    DequantizePerChannelArgs,
    DequantizePerTensorArgs,
)


def get_constant(exported_program: ExportedProgram, node: torch.fx.Node):
    assert isinstance(node, torch.fx.Node)
    if node.name in exported_program.constants:
        return exported_program.constants[node.name]
    elif is_buffer(exported_program, node):
        return get_buffer(exported_program, node)
    elif is_lifted_tensor_constant(exported_program, node):
        return get_lifted_tensor_constant(exported_program, node)
    else:
        raise RuntimeError("NYI constant")


class ValRange:
    def __init__(self, val: Union[torch.Tensor, List[int]]):
        if isinstance(val, torch.Tensor):
            self.max = torch.max(val).item()
            self.min = torch.min(val).item()
        elif type(val) == list:
            self.max = max(val)
            self.min = min(val)
        else:
            raise RuntimeError("Wrong dtype (val)")

    def within(self, min_val, max_val):
        return self.min >= min_val and self.max <= max_val


# Infer dtype using weight, zero point, and dtype
def infer_dtype(weight: torch.Tensor, zerop: List[int], dtype: torch.dtype) -> str:
    weight_val = ValRange(weight)
    zp_val = ValRange(zerop)

    if weight_val.within(0, 15) and zp_val.within(0, 15) and dtype == torch.uint8:
        return "uint4"
    else:
        return to_qparam_dtype(dtype)


@trace_graph_diff_on_pass
class RemoveWeightDequantOp(PassBase):
    """
    This pass identifies and removes any remaining Dequantize ops associated with
     quantized weights.

    Since weights already quantized earlier (and possibly kept in float by
     attaching a DQ), the final stage of the quantization pipeline typically
    does not require those DQ ops anymore.

    NOTE Removing 'DQ' causes a sementic change: f32 -> quantized

    [BEFORE]
      W (quantized) - Dequantize (float)

    [AFTER]
      W (quantized)
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for dq in graph.nodes:
            if not dq.op == "call_function":
                continue

            if dq.target not in [
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]:
                continue
            dq_args: Optional[DequantizePerChannelArgs | DequantizePerTensorArgs] = None

            if (
                dq.target
                == torch.ops.quantized_decomposed.dequantize_per_channel.default
            ):
                dq_args = DequantizePerChannelArgs(*dq.args, **dq.kwargs)
            elif (
                dq.target
                == torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                dq_args = DequantizePerTensorArgs(*dq.args, **dq.kwargs)
            else:
                raise RuntimeError(f"Invalid DQ target: {dq.target}")

            q_weight = dq_args.input
            # All weights are placehoders.
            if q_weight.op != "placeholder":
                continue
            # Check if DQ already has quant param because DQ can be shared.
            if QPARAM_KEY in q_weight.meta:
                continue

            q_weight_meta = q_weight.meta["val"]
            assert isinstance(q_weight_meta, FakeTensor)
            # Weight should have quantized values.
            assert q_weight_meta.dtype != torch.float

            q_weight_val = get_constant(exported_program, q_weight)
            assert isinstance(q_weight_val, torch.Tensor)

            quant_param = QuantParam()
            if isinstance(dq_args, DequantizePerChannelArgs):
                scales = get_constant(exported_program, dq_args.scales)
                zero_ps = get_constant(exported_program, dq_args.zero_points)

                # Sometimes users can give fp32 zero point. Let's update dtype here.
                zero_ps = zero_ps.to(torch.int64)
                quant_param.scale = scales.tolist()
                quant_param.zero_point = zero_ps.tolist()
                assert quant_param.zero_point is not None  # To avoid mypy error
                quant_param.quantized_dimension = dq_args.axis
                quant_param.dtype = infer_dtype(
                    q_weight_val, quant_param.zero_point, q_weight_meta.dtype
                )
            elif isinstance(dq_args, DequantizePerTensorArgs):
                quant_param.scale = [dq_args.scale]
                quant_param.zero_point = [dq_args.zero_point]
                assert quant_param.zero_point is not None  # To avoid mypy error
                quant_param.dtype = infer_dtype(
                    q_weight_val, quant_param.zero_point, q_weight_meta.dtype
                )
            else:
                raise RuntimeError(f"Invalid DQ target: {dq.target}")

            q_weight.meta[QPARAM_KEY] = quant_param
            dq.replace_all_uses_with(q_weight, propagate_meta=False)
            logger.debug(f"{dq.name} is removed.")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)

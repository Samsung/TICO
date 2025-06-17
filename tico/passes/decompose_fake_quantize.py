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
import torch

# To import torch.ops.quantized_decomposed related operator
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import FakeQuantizePerChannelArgs


def get_quant_type(min: int, max: int) -> torch.dtype:
    if min == 0 and max == 15:
        # torch can't represent "uint4".
        # Let's set torch.uint8 and infer dtype with quant_min/quant_max instead.
        return torch.uint8
    if min == 0 and max == 255:
        return torch.uint8
    if min == -32768 and max == 32767:
        return torch.int16
    if min == -32767 and max == 32767:
        return torch.int16

    raise RuntimeError(f"Not supported min/max values: {min}/{max}")


@trace_graph_diff_on_pass
class DecomposeFakeQuantize(PassBase):
    """
    Decompose fake quantize operator to quant/dequant operators.
    Otherwise, it can't be converted to the edge IR because fake quantize operator is not Aten Canonical.

    [Before]
    def forward(self, x):
        fake_quantize_per_tensor_affine = torch.ops.aten.fake_quantize_per_tensor_affine.default(tensor, scale, zero_p, quant_min, quant_max);  x = None
        return (fake_quantize_per_tensor_affine,)

    [After]
    def forward(self, x):
        quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(tensor, scale, zero_p, quant_min, quant_max, dtype = ${torch.dtype});  x = None
        dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, scale, zero_p, quant_min, quant_max, dtype = ${torch.dtype});  quantize_per_tensor_default = None
        return (dequantize_per_tensor_default,)
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)
        modified = False

        gm = exported_program.graph_module
        g = gm.graph
        qd = torch.ops.quantized_decomposed  # type: ignore[return]
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in [torch.ops.aten.fake_quantize_per_tensor_affine.default]:
                # tensor, scale, zero_p, quant_min, quant_max
                assert len(node.args) == 5
                _, _, _, quant_min, quant_max = node.args

                quant_kwargs = {
                    **node.kwargs,
                    **{"dtype": get_quant_type(quant_min, quant_max)},
                }
                with gm.graph.inserting_before(node):
                    quant = create_node(
                        g,
                        qd.quantize_per_tensor.default,
                        args=node.args,
                        kwargs=quant_kwargs,
                        origin=node,
                    )
                    dequnt = create_node(
                        g,
                        qd.dequantize_per_tensor.default,
                        args=(quant, *quant.args[1:]),
                        kwargs=quant.kwargs,
                    )
                    node.replace_all_uses_with(dequnt, propagate_meta=True)
                modified = True

            if node.target in [torch.ops.aten.fake_quantize_per_channel_affine.default]:
                fq_args = FakeQuantizePerChannelArgs(*node.args, **node.kwargs)
                quant_min = fq_args.quant_min
                quant_max = fq_args.quant_max

                quant_kwargs = {
                    **node.kwargs,
                    **{"dtype": get_quant_type(quant_min, quant_max)},
                }
                with gm.graph.inserting_before(node):
                    quant = create_node(
                        g,
                        qd.quantize_per_channel.default,
                        args=node.args,
                        kwargs=quant_kwargs,
                        origin=node,
                    )
                    dequnt = create_node(
                        g,
                        qd.dequantize_per_channel.default,
                        args=(quant, *quant.args[1:]),
                        kwargs=quant.kwargs,
                    )
                    node.replace_all_uses_with(dequnt, propagate_meta=True)
                modified = True

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)

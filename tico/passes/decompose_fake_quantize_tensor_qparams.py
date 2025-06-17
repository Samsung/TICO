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

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch._ops
    import torch.fx
import torch
from torch._export.utils import (
    get_buffer,
    get_lifted_tensor_constant,
    is_buffer,
    is_lifted_tensor_constant,
)

# To import torch.ops.quantized_decomposed related operator
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.validate_args_kwargs import FakeQuantizePerTensorTQParamArgs


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

    raise RuntimeError("Not supported min/max values")


def get_constant_from_tensor(
    node: Union[torch.fx.Node, float], ep: ExportedProgram
) -> Union[torch.fx.Node, float]:
    """
    There are some nodes that can do constant folding.
      Case 1. With constant tensors
      Case 2. With `torch.ones.` or `torch.zeros`

    Please refer to the below `DecomposeFakeQuantizeTensorQParams` docs for the detailed explanations.
    """
    if isinstance(node, float):
        return node
    if is_buffer(ep, node):
        buf = get_buffer(ep, node)
        assert isinstance(buf, torch.Tensor)
        return buf.item()
    elif is_lifted_tensor_constant(ep, node):
        lifted = get_lifted_tensor_constant(ep, node)
        assert isinstance(lifted, torch.Tensor)
        return lifted.item()
    assert isinstance(node.target, torch._ops.OpOverload)
    if node.target.__name__ == "mul.Tensor":
        assert len(node.args) == 2
        x = get_constant_from_tensor(node.args[0], ep)  # type: ignore[arg-type]
        y = get_constant_from_tensor(node.args[1], ep)  # type: ignore[arg-type]
        return x * y  # type: ignore[operator]
    if node.target.__name__ == "zeros.default":
        assert len(node.args) == 1
        assert node.args[0] == [1]
        return 0
    if node.target.__name__ == "ones.default":
        assert len(node.args) == 1
        assert node.args[0] == [1]
        return 1
    if node.target.__name__ == "view.default":
        assert len(node.args) == 2
        tensor, shape = node.args
        assert shape == [-1]
        return get_constant_from_tensor(tensor, ep)  # type: ignore[arg-type]
    if node.target.__name__ == "_to_copy.default":
        assert len(node.args) == 1
        return get_constant_from_tensor(node.args[0], ep)  # type: ignore[arg-type]
    if node.target.__name__ == "lift_fresh_copy.default":
        assert len(node.args) == 1
        assert isinstance(node.args[0], torch.fx.Node)
        lifted_tensor: torch.fx.Node = node.args[0]
        lifted_tensor_constants = ep.graph_signature.inputs_to_lifted_tensor_constants
        assert lifted_tensor.name in lifted_tensor_constants
        tensor_name = lifted_tensor_constants[lifted_tensor.name]
        value = ep.constants[tensor_name].item()
        return value
    if node.target.__name__ in ["detach.default", "detach_.default"]:
        assert len(node.args) == 1
        return get_constant_from_tensor(node.args[0], ep)  # type: ignore[arg-type]

    raise RuntimeError(f"Not supported node {node.target.__name__}")


@trace_const_diff_on_pass
@trace_graph_diff_on_pass
class DecomposeFakeQuantizeTensorQParams(PassBase):
    """
    Decompose fake quantize with tensor QParams operator to quant/dequant operators.
    Otherwise, it can't be converted to the edge IR because fake quantize operator is not Aten Canonical.

    As of now, we don't support the (de)quantize op that has scale/zp whose dtypes are tensors. They should be scalars.
    But, fake quantize with tensor QParams can be decomposed only when those tensors can be removed by constant foldings.

    We consider below cases for now.

    [CASE 1] With constant tensors

    s = torch.tensor(0.1)
    zp = torch.tensor(0)
    fq_enabled = torch.tensor(True)
    x = torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x, s, zp, fq_enabled, 0, 255
    )

    [Before pass]

    def forward(self, c_lifted_tensor_0, c_lifted_tensor_1, c_lifted_tensor_2, x):
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_1);  c_lifted_tensor_1 = None
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_2);  c_lifted_tensor_2 = None
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparams = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(x, lift_fresh_copy, lift_fresh_copy_1, lift_fresh_copy_2, quant_min, quant_max);  x = lift_fresh_copy = lift_fresh_copy_1 = lift_fresh_copy_2 = None
        getitem = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams[0]
        getitem_1 = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams[1];  _fake_quantize_per_tensor_affine_cachemask_tensor_qparams = None
        return (getitem, getitem_1)

    [After pass]

    def forward(self, c_lifted_tensor_0, c_lifted_tensor_1, c_lifted_tensor_2, x):
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(c_lifted_tensor_1);  c_lifted_tensor_1 = None
        quantize_per_tensor_tensor = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(x, lift_fresh_copy, lift_fresh_copy_1, quant_min, quant_max, dtype = ${torch.dtype});  x = None
        dequantize_per_tensor_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor(quantize_per_tensor_tensor, lift_fresh_copy, lift_fresh_copy_1, quant_min, quant_max, dtype = ${torch.dtype});  quantize_per_tensor_tensor = lift_fresh_copy = lift_fresh_copy_1 = None
        return (dequantize_per_tensor_tensor,)

    `s` and `zp` are tensors but they can be removed after constant foldings. When they are transformed to fx graph, they are
    lifted as a placeholder and become an argument of the `aten.lift_fresh_copy`.


    [CASE 2] With `torch.ones` or `torch.zeros`

    n_bits=16
    scale=torch.ones([1])
    Qp = 2**(n_bits-1)-1
    scale=scale*(1/Qp)
    z = torch.fake_quantize_per_tensor_affine(x, scale, torch.zeros([1]).int().view(-1), -Qp, Qp)

    `torch.ones([1])` or `torch.zeros([1])` is just number 1 or 0 but it is transformed to aten IR node, which prevents it from
    being pre-calculated to the number.

    For example, `n_bits * 1` would be just number 16 when the transformation, but `n_bits * torch.ones([1])`
    would be `aten.Mul(16, aten.full)`, which is the reason why `torch.fake_quantize_per_tensor_affine` is trasnformed to
    `aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams` whose scale/zp argument types are tensors rather than scalars.

    So, if we manually compute such things like `n_bits * torch.ones([1])`, we can decompose fake quantize with qparam tensors.

    [Before pass]

    def forward(self, x):
        ones = torch.ops.aten.ones.default([1], device = device(type='cpu'), pin_memory = False)
        mul = torch.ops.aten.mul.Tensor(ones, 3.051850947599719e-05);  ones = None
        zeros = torch.ops.aten.zeros.default([1], device = device(type='cpu'), pin_memory = False)
        _to_copy = torch.ops.aten._to_copy.default(zeros, dtype = torch.int32);  zeros = None
        view = torch.ops.aten.view.default(_to_copy, [-1]);  _to_copy = None
        ones_1 = torch.ops.aten.ones.default([1], dtype = torch.int64, layout = torch.strided, device = device(type='cpu'))
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_default = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(x, mul, view, ones_1, -32767, 32767);  x = mul = view = ones_1 = None
        getitem = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_default[0];  _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_default = None
        return (getitem,)

    [After pass]
    def forward(self, x: "f32[4]"):
            quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 3.051850947599719e-05, 0, -32767, 32767, dtype = torch.int16);  x = None
            dequantize_per_tensor_default: "f32[4]" = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 3.051850947599719e-05, 0, -32767, 32767, dtype = torch.int16);  quantize_per_tensor_default = None
            return (dequantize_per_tensor_default,)
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        modified = False

        gm = exported_program.graph_module
        g = gm.graph
        qd = torch.ops.quantized_decomposed  # type: ignore[return]
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                == torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
            ):
                # tensor, scale, zero_p, fake_quant_enabled, quant_min, quant_max
                # TODO Support `fake_quant_enabled`
                assert len(node.args) == 6
                tensor, s, zp, _, quant_min, quant_max = node.args
                # Get constant tensors
                ep = exported_program
                s_value = get_constant_from_tensor(s, ep)
                zp_value = get_constant_from_tensor(zp, ep)
                # This op has one user: `getitem` for the output.
                # TODO Investigate why the op is generated like this.
                # node.users = {getitem: None}
                get_item, *mask = node.users.keys()
                # assert len(mask) == 0, "Not supported yet."
                quant_kwargs = {
                    **node.kwargs,
                    **{"dtype": get_quant_type(quant_min, quant_max)},
                }
                with gm.graph.inserting_before(node):
                    quant = create_node(
                        g,
                        qd.quantize_per_tensor.default,
                        args=(tensor, s_value, zp_value, quant_min, quant_max),
                        kwargs=quant_kwargs,
                        origin=node,
                    )
                    dequant = create_node(
                        g,
                        qd.dequantize_per_tensor.default,
                        args=(quant, *quant.args[1:]),
                        kwargs=quant.kwargs,
                    )
                    get_item.replace_all_uses_with(dequant, propagate_meta=True)
                    # If `mask` can be graph output, which prevents `eliminate_dead_code()` from eliminating `mask`.
                    # So, let's remove `mask` from the output.args first.
                    # mask_user(output).args == (dequantize_per_tensor.tensor, mask)
                    if mask:
                        len(mask) == 1
                        mask_user = list(mask[0].users.keys())[0]
                        assert len(mask_user.args) == 1
                        mask_user.args = ((mask_user.args[0][0],),)
                modified = True
            if (
                node.target
                == torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams
            ):
                fq_args = FakeQuantizePerTensorTQParamArgs(*node.args, **node.kwargs)
                tensor = fq_args.input
                s = fq_args.scale
                zp = fq_args.zero_point
                quant_min = fq_args.quant_min
                quant_max = fq_args.quant_max

                # Get constant tensors
                ep = exported_program
                s_value = get_constant_from_tensor(s, ep)
                zp_value = get_constant_from_tensor(zp, ep)
                quant_kwargs = {
                    **node.kwargs,
                    **{"dtype": get_quant_type(quant_min, quant_max)},
                }
                with gm.graph.inserting_before(node):
                    quant = create_node(
                        g,
                        qd.quantize_per_tensor.default,
                        args=(tensor, s_value, zp_value, quant_min, quant_max),
                        kwargs=quant_kwargs,
                        origin=node,
                    )
                    dequant = create_node(
                        g,
                        qd.dequantize_per_tensor.default,
                        args=(quant, *quant.args[1:]),
                        kwargs=quant.kwargs,
                    )
                    node.replace_all_uses_with(dequant, propagate_meta=True)
                modified = True

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)

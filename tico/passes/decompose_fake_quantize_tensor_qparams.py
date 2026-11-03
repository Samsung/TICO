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

import operator
from typing import TYPE_CHECKING, Dict, Union

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
from torch.export import ExportedProgram

from tico.passes.decompose_fake_quantize import get_quant_type
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.validate_args_kwargs import FakeQuantizePerTensorTQParamArgs


def get_constant_from_tensor(
    node: Union[torch.fx.Node, bool, int, float],
    ep: ExportedProgram,
    *,
    _cache: Union[Dict[Union[str, bool, int, float], Union[bool, int, float]], None] = None,
) -> Union[bool, int, float]:
    """Memoizing wrapper around `_get_constant_from_tensor_impl`.

    One cache dict per pass run collapses the repeated scale/zero-point
    constant-chain walks shared by thousands of fake-quantize nodes.

    `_cache` is internal and keyword-only: pass one dict per pass run to
    share memoized values across calls, or omit it for a one-off lookup.
    """
    if _cache is None:
        return _get_constant_from_tensor_impl(node, ep)
    key = node if isinstance(node, (bool, int, float)) else node.name
    if key not in _cache:
        _cache[key] = _get_constant_from_tensor_impl(node, ep)
    return _cache[key]


def _get_constant_from_tensor_impl(
    node: Union[torch.fx.Node, bool, int, float], ep: ExportedProgram
) -> Union[bool, int, float]:
    """Evaluate supported singleton constant expressions, then extract a scalar.

    Keep tensor dtype and rank through multiplication, views and casts: converting
    operands to Python scalars first loses integer overflow and scalar promotion.
    Only the existing small expression vocabulary is supported. Evaluation uses
    CPU singleton tensors rather than allocating on an exported device target.
    """

    def resolve(value):
        if isinstance(value, (bool, int, float)):
            return value
        assert isinstance(value, torch.fx.Node)
        if is_buffer(ep, value):
            tensor = get_buffer(ep, value)
        elif is_lifted_tensor_constant(ep, value):
            tensor = get_lifted_tensor_constant(ep, value)
        else:
            tensor = None
        if tensor is not None:
            assert isinstance(tensor, torch.Tensor)
            if tensor.numel() != 1:
                raise RuntimeError("Expected a singleton constant tensor.")
            return tensor.detach().cpu()

        assert isinstance(value.target, torch._ops.OpOverload)
        target_name = value.target.__name__
        if target_name == "mul.Tensor":
            assert len(value.args) == 2
            return value.target(resolve(value.args[0]), resolve(value.args[1]))
        if target_name in ("zeros.default", "ones.default"):
            assert len(value.args) == 1
            shape = value.args[0]
            assert shape in ([1], [])
            return value.target(shape, dtype=value.meta["val"].dtype, device="cpu")
        if target_name == "view.default":
            assert len(value.args) == 2
            source, shape = value.args
            assert shape == [-1]
            return resolve(source).view(shape)
        if target_name in (
            "_to_copy.default",
            "to.dtype",
            "to.dtype_layout",
            "to.device",
        ):
            return resolve(value.args[0]).to(dtype=value.meta["val"].dtype)
        if target_name == "lift_fresh_copy.default":
            assert len(value.args) == 1
            return resolve(value.args[0]).clone()
        if target_name in ("detach.default", "detach_.default"):
            assert len(value.args) == 1
            return resolve(value.args[0]).detach()

        raise RuntimeError(f"Not supported node {target_name}")

    result = resolve(node)
    if isinstance(result, torch.Tensor):
        result = result.item()
    if not isinstance(result, (bool, int, float)):
        raise RuntimeError("Expected a real scalar constant.")
    return result


@trace_const_diff_on_pass
@trace_graph_diff_on_pass
class DecomposeFakeQuantizeTensorQParams(PassBase):
    """
    Decompose fake quantize with tensor QParams operator to quant/dequant operators.
    Otherwise, it can't be converted to the edge IR because fake quantize operator is not Aten Canonical.

    Enabled fake quantization requires scale/zp tensors that can be constant-folded
    to the scalar qparams accepted by the Q/DQ operators. Disabled cachemask sites
    bypass Q/DQ without inspecting scale/zp.

    Cachemask fake quantization also requires a constant enable flag. When disabled,
    it clones the input and produces an all-true mask without applying Q/DQ.
    Enabled mask outputs are unsupported and rejected rather than discarded.

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
        getitem = _fake_quantize_per_tensor_affine_cachemask_tensor_qparams[0];  _fake_quantize_per_tensor_affine_cachemask_tensor_qparams = None
        return (getitem,)

    [After pass]

    def forward(self, c_lifted_tensor_0, c_lifted_tensor_1, c_lifted_tensor_2, x):
        quantize_per_tensor_default = torch.ops.quantized_decomposed.quantize_per_tensor.default(x, 0.1, 0, 0, 255, dtype=torch.uint8)
        dequantize_per_tensor_default = torch.ops.quantized_decomposed.dequantize_per_tensor.default(quantize_per_tensor_default, 0.1, 0, 0, 255, dtype=torch.uint8)
        return (dequantize_per_tensor_default,)

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
        _const_cache: Dict[Union[str, bool, int, float], Union[bool, int, float]] = {}

        # Validate live tuple uses before rewriting any cachemask operation.
        # Dead masks and their dead consumers do not restrict inference export.
        g.lint()
        live_nodes: set[torch.fx.Node] = set()
        for node in reversed(g.nodes):
            if node.is_impure() or any(user in live_nodes for user in node.users):
                live_nodes.add(node)

        cachemask_outputs: dict[
            torch.fx.Node,
            tuple[bool, list[torch.fx.Node], list[torch.fx.Node]],
        ] = {}
        for node in g.nodes:
            if (
                node not in live_nodes
                or node.op != "call_function"
                or node.target
                != torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
            ):
                continue
            assert len(node.args) == 6
            try:
                enabled = get_constant_from_tensor(node.args[3], exported_program, _cache=_const_cache)
            except (AssertionError, AttributeError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"{node.target}: fake_quant_enabled must be a compile-time "
                    "constant; runtime enable flags are not supported."
                ) from exc
            if not isinstance(enabled, (bool, int, float)):
                raise RuntimeError(
                    f"{node.target}: fake_quant_enabled must be a scalar "
                    "compile-time constant."
                )
            try:
                enabled = bool(int(enabled))
            except (OverflowError, ValueError) as exc:
                raise RuntimeError(
                    f"{node.target}: fake_quant_enabled must be a finite "
                    "scalar compile-time constant."
                ) from exc

            values: list[torch.fx.Node] = []
            masks: list[torch.fx.Node] = []
            for user in node.users:
                if user not in live_nodes:
                    continue
                if not (
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.args) == 2
                    and user.args[0] is node
                    and isinstance(user.args[1], int)
                    and user.args[1] in (0, 1, -2, -1)
                    and not user.kwargs
                ):
                    raise RuntimeError(
                        f"{node.target}: only tensor and mask getitem outputs "
                        "are supported; direct tuple uses are not supported."
                    )
                if user.args[1] in (0, -2):
                    values.append(user)
                else:
                    masks.append(user)
            if enabled and masks:
                raise RuntimeError(
                    f"{node.target}: enabled fake-quantization mask output is "
                    "not supported; export only the tensor output (index 0 or -2)."
                )
            cachemask_outputs[node] = (bool(enabled), values, masks)

        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                == torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
            ):
                if node not in cachemask_outputs:
                    continue
                enabled, values, masks = cachemask_outputs[node]
                tensor, s, zp, _, quant_min, quant_max = node.args
                if not enabled:
                    if values:
                        with g.inserting_before(node):
                            cloned = create_node(
                                g,
                                torch.ops.aten.clone.default,
                                args=(tensor,),
                                origin=node,
                            )
                        cloned.meta.update(values[0].meta)
                        for value in values:
                            value.replace_all_uses_with(cloned)
                    if masks:
                        with g.inserting_before(node):
                            mask = create_node(
                                g,
                                torch.ops.aten.full_like.default,
                                args=(tensor, True),
                                kwargs={"dtype": torch.bool},
                                origin=node,
                            )
                        mask.meta.update(masks[0].meta)
                        for mask_output in masks:
                            mask_output.replace_all_uses_with(mask)
                    modified = True
                    continue
                # Get constant tensors
                ep = exported_program
                s_value = get_constant_from_tensor(s, ep, _cache=_const_cache)
                zp_value = int(get_constant_from_tensor(zp, ep, _cache=_const_cache))
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
                    dequant.meta.update(values[0].meta)
                    for value in values:
                        value.replace_all_uses_with(dequant)
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
                s_value = get_constant_from_tensor(s, ep, _cache=_const_cache)
                zp_value = int(get_constant_from_tensor(zp, ep, _cache=_const_cache))
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

        modified |= gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()

        return PassResult(modified)

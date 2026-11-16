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

"""Exact CPU semantics and conservative matching for affine boundary movement."""

import copy
import unittest

import torch

from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.fill_meta_val import FillMetaVal
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.move_affine_quantize_before_layout import (
    MoveAffineQuantizeBeforeLayout,
)
from tico.quantization.passes.preserve_embedding_quantization import EXPLICIT_EMBEDDING
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.quantization.passes.propagate_qparam_forward import PropagateQParamForward
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.mx.mx_ops import quantize_mx
from tico.utils.passes import PassManager

_Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
_DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
_LAYOUTS = (torch.ops.aten.reshape.default, torch.ops.aten.permute.default)


def _affine(value):
    return torch.fake_quantize_per_tensor_affine(value, 0.003, 7, -32768, 32767)


class _Layout(torch.nn.Module):
    def __init__(self, layout="chain", source="mx", boundary=True, fanout=None):
        super().__init__()
        self.layout = layout
        self.source = source
        self.boundary = boundary
        self.fanout = fanout

    def forward(self, x):
        if self.source == "mx":
            source = quantize_mx(x, "int8", -1)
        elif self.source == "fp4":
            source = quantize_mx(x, "fp4", -1)
        elif self.source == "affine":
            source = _affine(x)
        else:
            source = x
        if self.layout == "reshape":
            value = source.reshape(4, 32)
        elif self.layout == "permute":
            value = source.permute(1, 0)
        elif self.layout == "transpose":
            value = source.transpose(0, 1)
        elif self.layout == "slice":
            value = source[:, :32]
        elif self.layout == "nonlinear":
            value = source.reshape(2, 4, 16).relu().permute(1, 0, 2)
        elif self.layout == "quantizer":
            value = _affine(source.reshape(2, 4, 16)).permute(1, 0, 2)
        else:
            value = source.reshape(2, 4, 16).permute(1, 0, 2).reshape(4, 32)
        if self.boundary == "qdq":
            quantized = _Q(value, 0.003, 7, -32768, 32767, torch.int16)
            result = _DQ(quantized, 0.003, 7, -32768, 32767, torch.int16)
        else:
            result = _affine(value) if self.boundary else value
        if self.fanout == "producer":
            return result, source + source, source
        if self.fanout == "intermediate":
            return result, value + value
        if self.fanout == "output":
            return result, value
        if self.fanout == "final":
            return result, result + result, result
        return result


class _LogicalMXInterpreter(torch.fx.Interpreter):
    """Execute logical MX Q/DQ using their existing eager fake-quant oracle.

    Logical MX Q/DQ deliberately have no eager kernels. Quantize computes the
    unchanged MX fake-quant operator, and DQ exposes those rounded real values.
    Every affine Q/DQ and layout still executes its actual CPU implementation.
    """

    def __init__(self, module):
        super().__init__(module)
        self.values = {}

    def call_function(self, target, args, kwargs):
        if target == torch.ops.circle_custom.quantize_mx.default:
            return torch.ops.circle_custom.mx_fake_quantize.default(*args, **kwargs)
        if target == torch.ops.circle_custom.dequantize_mx.default:
            return args[0]
        return super().call_function(target, args, kwargs)

    def run_node(self, node):
        value = super().run_node(node)
        self.values[node] = value
        return value


def _input():
    return torch.linspace(-120.125, 120.125, 128).reshape(2, 64)


def _export(module=None, *, dynamic=False):
    module = _Layout() if module is None else module
    dynamic_shapes = None
    if dynamic:
        module.boundary = "qdq"
        dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=4)}}
    ep = torch.export.export(module.eval(), (_input(),), dynamic_shapes=dynamic_shapes)
    PassManager(
        [DecomposeFakeQuantize(), ConvertLayoutOpToReshape(), FillMetaVal()]
    ).run(ep)
    if module.layout == "transpose":
        # Match the public export decomposition's transpose -> permute spelling.
        ep = ep.run_decompositions()
        PassManager([ConvertLayoutOpToReshape(), FillMetaVal()]).run(ep)
    return ep


def _only(ep, target):
    nodes = [node for node in ep.graph.nodes if node.target == target]
    if len(nodes) != 1:
        raise AssertionError(f"Expected one {target}, found {len(nodes)}")
    return nodes[0]


def _execute(ep, value=None):
    interpreter = _LogicalMXInterpreter(ep.graph_module)
    result = interpreter.run(_input() if value is None else value)
    return result, interpreter.values


class MoveAffineQuantizeBeforeLayoutTest(unittest.TestCase):
    def assert_outputs_equal(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual):
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(before.shape, after.shape)
            self.assertEqual(before.dtype, after.dtype)

    def assert_unchanged(self, ep):
        graph = str(ep.graph)
        self.assertFalse(MoveAffineQuantizeBeforeLayout().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        ep.graph.lint()

    def test_moves_reshape_permute_and_chain_with_exact_values(self):
        for layout in ("reshape", "permute", "transpose", "chain"):
            with self.subTest(layout=layout):
                ep = _export(_Layout(layout))
                before, values = _execute(ep)
                q, dq = _only(ep, _Q), _only(ep, _DQ)
                q_args, q_kwargs = q.args[1:], q.kwargs.copy()
                dq_args, dq_kwargs = dq.args[1:], dq.kwargs.copy()
                mxq = _only(ep, torch.ops.circle_custom.quantize_mx.default)
                mxdq = _only(ep, torch.ops.circle_custom.dequantize_mx.default)
                mx_args, mx_kwargs = mxq.args, mxq.kwargs.copy()
                mx_dq_args, mx_dq_kwargs = mxdq.args, mxdq.kwargs.copy()
                # Exercise actual saturation and rounding, not an identity grid.
                codes = values[q]
                self.assertTrue(torch.any(codes == -32768))
                self.assertTrue(torch.any(codes == 32767))
                source_values = values[mxdq]
                self.assertTrue(
                    torch.any(
                        source_values / 0.003 != torch.round(source_values / 0.003)
                    )
                )
                self.assertTrue(MoveAffineQuantizeBeforeLayout().call(ep).modified)
                moved_q, moved_dq = _only(ep, _Q), _only(ep, _DQ)
                self.assertIs(moved_q.args[0], mxdq)
                self.assertIs(moved_dq.args[0], moved_q)
                self.assertEqual(moved_q.args[1:], q_args)
                self.assertEqual(moved_q.kwargs, q_kwargs)
                self.assertEqual(moved_dq.args[1:], dq_args)
                self.assertEqual(moved_dq.kwargs, dq_kwargs)
                self.assertEqual(mxq.args, mx_args)
                self.assertEqual(mxq.kwargs, mx_kwargs)
                self.assertEqual(mxdq.args, mx_dq_args)
                self.assertEqual(mxdq.kwargs, mx_dq_kwargs)
                self.assert_outputs_equal(before, _execute(ep)[0])
                # Removing the MX rounding changes this deterministic result.
                without_mx = _Layout(layout, source="none")(_input())
                self.assertFalse(torch.equal(before[0], without_mx))
                self.assert_unchanged(ep)

    def test_rank_change_refreshes_value_and_tensor_metadata(self):
        ep = _export()
        before, _ = _execute(ep)
        # ShapeProp metadata may coexist with FakeTensor val metadata.
        q, dq = _only(ep, _Q), _only(ep, _DQ)
        from torch.fx.passes.shape_prop import _extract_tensor_metadata

        q.meta["tensor_meta"] = _extract_tensor_metadata(
            torch.empty(4, 32, dtype=torch.int16)
        )
        dq.meta["tensor_meta"] = _extract_tensor_metadata(torch.empty(4, 32))
        self.assertTrue(MoveAffineQuantizeBeforeLayout().call(ep).modified)
        _, actual_values = _execute(ep)
        for node in ep.graph.nodes:
            if node.target not in (*_LAYOUTS, _Q, _DQ):
                continue
            actual = actual_values[node]
            self.assertEqual(node.meta["val"].shape, actual.shape)
            self.assertEqual(node.meta["val"].dtype, actual.dtype)
            if "tensor_meta" in node.meta:
                self.assertEqual(node.meta["tensor_meta"].shape, actual.shape)
                self.assertEqual(node.meta["tensor_meta"].dtype, actual.dtype)
        self.assertEqual(_only(ep, _Q).meta["val"].shape, (2, 64))
        self.assert_outputs_equal(before, _execute(ep)[0])

    def test_shared_mx_producer_and_graph_output_are_preserved(self):
        ep = _export(_Layout(fanout="producer"))
        before, _ = _execute(ep)
        mxdq = _only(ep, torch.ops.circle_custom.dequantize_mx.default)
        other = _only(ep, torch.ops.aten.add.Tensor)
        self.assertTrue(MoveAffineQuantizeBeforeLayout().call(ep).modified)
        self.assertEqual(other.args, (mxdq, mxdq))
        self.assertIs(list(ep.graph.nodes)[-1].args[0][2], mxdq)
        self.assert_outputs_equal(before, _execute(ep)[0])
        PassManager(
            [FoldQuantOps(), PropagateQParamForward(), PropagateQParamBackward()]
        ).run(ep)
        source = other.args[0]
        self.assertEqual(source.meta[QPARAM_KEY].dtype, "mxint8")
        self.assertEqual(source.meta[QPARAM_KEY].quantized_dimension, 1)
        self.assertIs(list(ep.graph.nodes)[-1].args[0][2], source)
        q = _only(ep, _Q)
        self.assertIs(q.args[0], source)
        for node in ep.graph.nodes:
            if node.target in (*_LAYOUTS, _Q):
                qp = node.meta[QPARAM_KEY]
                self.assertEqual(
                    (qp.dtype, qp.scale, qp.zero_point), ("int16", [0.003], [7])
                )
                self.assertIsNone(qp.quantized_dimension)

    def test_final_fake_quant_result_may_have_multiple_users(self):
        ep = _export(_Layout(fanout="final"))
        before, _ = _execute(ep)
        self.assertTrue(MoveAffineQuantizeBeforeLayout().call(ep).modified)
        self.assert_outputs_equal(before, _execute(ep)[0])

    def test_intermediate_fanout_and_graph_output_are_not_moved(self):
        for fanout in ("intermediate", "output"):
            with self.subTest(fanout=fanout):
                self.assert_unchanged(_export(_Layout(fanout=fanout)))

    def test_earlier_layout_fanout_blocks_the_complete_rewrite(self):
        for graph_output in (False, True):
            with self.subTest(graph_output=graph_output):
                ep = _export()
                middle = _only(ep, torch.ops.aten.permute.default)
                output = list(ep.graph.nodes)[-1]
                if graph_output:
                    extra = middle
                else:
                    with ep.graph.inserting_before(output):
                        extra = ep.graph.call_function(
                            torch.ops.aten.add.Tensor, (middle, middle)
                        )
                output.args = ((*output.args[0], extra),)
                ep.graph_module.recompile()
                self.assert_unchanged(ep)

    def test_aliasing_mutation_between_layout_and_quantize_is_not_moved(self):
        ep = _export(_Layout("permute"))
        source = _only(ep, torch.ops.circle_custom.dequantize_mx.default)
        q = _only(ep, _Q)
        with ep.graph.inserting_before(q):
            mutation = ep.graph.call_function(torch.ops.aten.add_.Tensor, (source, 1.0))
        self.assertTrue(mutation.is_impure())
        ep.graph_module.recompile()
        before, _ = _execute(ep)
        self.assert_unchanged(ep)
        self.assert_outputs_equal(before, _execute(ep)[0])

    def test_non_target_paths_are_not_moved(self):
        cases = (
            _Layout(boundary=False),
            _Layout(source="affine"),
            _Layout(source="fp4"),
            _Layout(source="none"),
            _Layout(layout="nonlinear"),
            _Layout(layout="slice"),
        )
        for model in cases:
            with self.subTest(
                source=model.source, layout=model.layout, boundary=model.boundary
            ):
                self.assert_unchanged(_export(model))

    def test_intervening_quantizer_stops_the_complete_chain(self):
        # The affine boundary immediately after the first reshape is movable;
        # the later pair must stay after its permute (source is then affine).
        ep = _export(_Layout(layout="quantizer"))
        pairs = [node for node in ep.graph.nodes if node.target == _Q]
        trailing = pairs[-1]
        trailing_source = trailing.args[0]
        before, _ = _execute(ep)
        self.assertTrue(MoveAffineQuantizeBeforeLayout().call(ep).modified)
        self.assertIs(trailing.args[0], trailing_source)
        self.assertEqual(trailing_source.target, torch.ops.aten.permute.default)
        self.assert_outputs_equal(before, _execute(ep)[0])

    def test_mismatched_affine_qdq_settings_are_not_moved(self):
        for field, value in ((1, 0.004), (2, 8), (3, -32767), (4, 32766)):
            with self.subTest(field=field):
                ep = _export()
                dq = _only(ep, _DQ)
                args = list(dq.args)
                args[field] = value
                dq.args = tuple(args)
                self.assert_unchanged(ep)
        ep = _export()
        dq = _only(ep, _DQ)
        dq.kwargs = {"dtype": torch.int32}
        self.assert_unchanged(ep)

    def test_explicit_output_dtype_is_conservatively_skipped(self):
        # The later folding dataclass cannot represent explicit out_dtype yet.
        for dtype in (None, torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                ep = _export()
                dq = _only(ep, _DQ)
                dq.kwargs = {**dq.kwargs, "out_dtype": dtype}
                self.assert_unchanged(ep)

    def test_keyword_arguments_keep_their_exact_representation(self):
        ep = _export(_Layout("reshape"))
        for node in ep.graph.nodes:
            if node.target in (*_LAYOUTS, _Q, _DQ):
                names = [arg.name for arg in node.target._schema.arguments]
                node.kwargs = {**dict(zip(names, node.args)), **node.kwargs}
                node.args = ()
        ep.graph_module.recompile()
        before, _ = _execute(ep)
        q, dq = _only(ep, _Q), _only(ep, _DQ)
        expected = {key: value for key, value in q.kwargs.items() if key != "input"}
        self.assertTrue(MoveAffineQuantizeBeforeLayout().call(ep).modified)
        self.assertEqual(q.args, ())
        self.assertEqual(dq.args, ())
        self.assertEqual(
            {key: value for key, value in q.kwargs.items() if key != "input"}, expected
        )
        self.assert_outputs_equal(before, _execute(ep)[0])

    def test_mismatched_mx_pair_is_not_moved(self):
        ep = _export()
        dq = _only(ep, torch.ops.circle_custom.dequantize_mx.default)
        dq.args = (dq.args[0], "int8", 0)
        self.assert_unchanged(ep)

    def test_shared_integer_quantize_output_is_not_moved(self):
        ep = _export()
        q = _only(ep, _Q)
        with ep.graph.inserting_before(list(ep.graph.nodes)[-1]):
            use = ep.graph.call_function(torch.ops.aten.clone.default, (q,))
        output = list(ep.graph.nodes)[-1]
        output.args = ((*output.args[0], use),)
        ep.graph_module.recompile()
        self.assert_unchanged(ep)

    def test_protected_embedding_boundaries_are_not_moved(self):
        for target in (_Q, _DQ, torch.ops.aten.permute.default):
            with self.subTest(target=target):
                ep = _export()
                _only(ep, target).meta[EXPLICIT_EMBEDDING] = True
                self.assert_unchanged(ep)

    def test_dynamic_shape_and_missing_metadata_are_not_moved(self):
        self.assert_unchanged(_export(_Layout("permute"), dynamic=True))
        ep = _export()
        _only(ep, torch.ops.aten.permute.default).meta.pop("val")
        self.assert_unchanged(ep)

    def test_tensor_qparams_and_per_channel_overloads_are_not_moved(self):
        for target in (
            torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
        ):
            with self.subTest(target=target):
                ep = _export()
                q, dq = _only(ep, _Q), _only(ep, _DQ)
                if target == torch.ops.quantized_decomposed.quantize_per_tensor.tensor:
                    # Runtime placeholders deliberately cannot be scalar-folded.
                    with ep.graph.inserting_before(q):
                        scale = ep.graph.placeholder("runtime_scale")
                        zero = ep.graph.placeholder("runtime_zero")
                    q.target = target
                    q.args = (q.args[0], scale, zero, -32768, 32767)
                    dq.target = (
                        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor
                    )
                    dq.args = (q, scale, zero, -32768, 32767)
                else:
                    q.target = target
                    q.args = (
                        q.args[0],
                        torch.ones(4),
                        torch.zeros(4, dtype=torch.int64),
                        0,
                        -32768,
                        32767,
                        torch.int16,
                    )
                    q.kwargs = {}
                    dq.target = (
                        torch.ops.quantized_decomposed.dequantize_per_channel.default
                    )
                    dq.args = (q, *q.args[1:])
                    dq.kwargs = {}
                self.assert_unchanged(ep)

    def test_pass_manager_converges_and_preserves_signature(self):
        ep = _export()
        original_inputs = copy.deepcopy(ep.graph_signature.input_specs)
        manager = PassManager([MoveAffineQuantizeBeforeLayout()])
        manager.run(ep)
        graph = str(ep.graph)
        manager.run(ep)
        self.assertEqual(str(ep.graph), graph)
        self.assertEqual(ep.graph_signature.input_specs, original_inputs)
        output = list(ep.graph.nodes)[-1].args[0][0]
        self.assertEqual(ep.graph_signature.output_specs[0].arg.name, output.name)
        self.assert_outputs_equal((_Layout()(_input()),), _execute(ep)[0])
        ep.validate()


if __name__ == "__main__":
    unittest.main()

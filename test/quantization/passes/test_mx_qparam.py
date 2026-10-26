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

"""MX block membership and axis propagation across tensor layouts."""

import copy
import operator
import unittest

import torch

from tico.ops import Concat
from tico.quantization.passes.mx_qparam import remap_mx_qparam
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.quantization.passes.propagate_qparam_forward import PropagateQParamForward
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx


def _mx(axis: int = -1, dtype: str = "mxint8") -> QuantParam:
    return QuantParam(dtype=dtype, quantized_dimension=axis)


def _layout_node(shape, target, args):
    graph = torch.fx.Graph()
    source = graph.placeholder("x")
    source.meta["val"] = torch.empty(shape)
    node = graph.call_function(target, (source, *args))
    node.meta["val"] = target(source.meta["val"], *args)
    graph.output(node)
    return source, node


class MXLayoutMetadataTest(unittest.TestCase):
    def test_permute_normalizes_axes_and_round_trips(self):
        _, node = _layout_node(
            (2, 64, 96), torch.ops.aten.permute.default, ([-1, 0, -2],)
        )
        source_qparam = _mx(-2)
        mapped = remap_mx_qparam(source_qparam, node)
        assert mapped is not None
        self.assertEqual(mapped, _mx(2))
        self.assertEqual(remap_mx_qparam(mapped, node, backward=True), _mx(1))
        self.assertEqual(source_qparam, _mx(-2))
        self.assertIsNot(mapped, source_qparam)

    def test_permute_preserves_numerical_block_membership(self):
        torch.manual_seed(42)
        value = torch.randn(2, 64, 96)
        _, node = _layout_node(
            value.shape, torch.ops.aten.permute.default, ([2, 0, 1],)
        )
        mapped = remap_mx_qparam(_mx(1), node)
        assert mapped is not None and mapped.quantized_dimension is not None
        actual = quantize_mx(value, elem_format="int8", axis=1).permute(2, 0, 1)
        expected = quantize_mx(
            value.permute(2, 0, 1),
            elem_format="int8",
            axis=mapped.quantized_dimension,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_safe_reshape_axis_mappings(self):
        cases = (
            ((1, 5, 128), (1, 5, 2, 64), -1, 3),
            ((1, 5, 2, 64), (1, 5, 128), -1, 2),
            ((2, 64, 3), (128, 3), 1, 0),
            ((2, 64), (2, 1, 64, 1), 1, 2),
            ((2, 15, 64), (30, 64), 2, 1),
        )
        for input_shape, output_shape, input_axis, output_axis in cases:
            with self.subTest(input_shape=input_shape, output_shape=output_shape):
                _, node = _layout_node(
                    input_shape, torch.ops.aten.reshape.default, (list(output_shape),)
                )
                mapped = remap_mx_qparam(_mx(input_axis), node)
                assert mapped is not None
                self.assertEqual(mapped, _mx(output_axis))
                restored = remap_mx_qparam(mapped, node, backward=True)
                self.assertEqual(restored, _mx(input_axis % len(input_shape)))

    def test_head_split_preserves_numerical_blocks(self):
        torch.manual_seed(42)
        value = torch.randn(1, 5, 128)
        actual = quantize_mx(value, elem_format="int8", axis=-1).reshape(1, 5, 2, 64)
        expected = quantize_mx(value.reshape(1, 5, 2, 64), elem_format="int8", axis=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reshape_rejects_reblocking_and_changed_inner_stride(self):
        cases = (
            ((2, 16, 3), (32, 3), 1),
            ((2, 64, 3), (2, 3, 64), 1),
            ((4, 8), (2, 16), -1),
        )
        for input_shape, output_shape, axis in cases:
            with self.subTest(input_shape=input_shape, output_shape=output_shape):
                _, node = _layout_node(
                    input_shape, torch.ops.aten.reshape.default, (list(output_shape),)
                )
                with self.assertRaisesRegex(NotYetSupportedError, "changes the groups"):
                    remap_mx_qparam(_mx(axis), node)

    def test_slice_preserves_complete_blocks_and_nonblock_dimensions(self):
        cases = (
            ((2, 96), (1, 32, 64), -1),
            ((2, 95), (1, 32, 95), -1),
            ((2, 96), (0, 1, 2), -1),
        )
        for shape, args, axis in cases:
            with self.subTest(shape=shape, args=args):
                _, node = _layout_node(shape, torch.ops.aten.slice.Tensor, args)
                self.assertEqual(remap_mx_qparam(_mx(axis), node), _mx(1))

    def test_slice_rejects_partial_or_strided_blocks(self):
        for args in ((1, 1, 65), (1, 0, 48), (1, 0, 96, 2)):
            with self.subTest(args=args):
                _, node = _layout_node((2, 96), torch.ops.aten.slice.Tensor, args)
                with self.assertRaisesRegex(NotYetSupportedError, "cuts or regroups"):
                    remap_mx_qparam(_mx(), node)

    def test_expand_prepends_dimensions(self):
        _, node = _layout_node((1, 64), torch.ops.aten.expand.default, ([3, 2, 64],))
        self.assertEqual(remap_mx_qparam(_mx(-1), node), _mx(2))

    def test_padding_rejects_shifted_blocks(self):
        _, node = _layout_node(
            (2, 64), torch.ops.aten.constant_pad_nd.default, ([1, 0],)
        )
        with self.assertRaisesRegex(NotYetSupportedError, "shifts the start"):
            remap_mx_qparam(_mx(), node)

    def test_zero_padding_allows_complete_blocks_and_trailing_zeros(self):
        _, node = _layout_node(
            (2, 45), torch.ops.aten.constant_pad_nd.default, ([32, 7],)
        )
        self.assertEqual(remap_mx_qparam(_mx(), node), _mx(1))

    def test_invalid_axis_is_rejected(self):
        _, node = _layout_node((2, 64), torch.ops.aten.permute.default, ([1, 0],))
        for axis in (-3, 2):
            with (
                self.subTest(axis=axis),
                self.assertRaisesRegex(ValueError, "out of range"),
            ):
                remap_mx_qparam(_mx(axis), node)

    def test_affine_qparam_is_not_reinterpreted_as_mx(self):
        _, node = _layout_node((2, 3), torch.ops.aten.reshape.default, ([3, 2],))
        self.assertIsNone(
            remap_mx_qparam(QuantParam(dtype="uint8", quantized_dimension=1), node)
        )

    def test_split_blocks_and_tail(self):
        graph = torch.fx.Graph()
        source = graph.placeholder("x")
        source.meta["val"] = torch.empty(2, 95)
        split = graph.call_function(
            torch.ops.aten.split_with_sizes.default, (source, [32, 63], 1)
        )
        for index, length in enumerate((32, 63)):
            node = graph.call_function(operator.getitem, (split, index))
            node.meta["val"] = torch.empty(2, length)
            self.assertEqual(remap_mx_qparam(_mx(), node), _mx(1))
        split.args = (source, [31, 64], 1)
        with self.assertRaisesRegex(NotYetSupportedError, "cuts or regroups"):
            remap_mx_qparam(_mx(), node)

    def test_concat_preserves_blocks_and_rejects_incomplete_join(self):
        for sizes, valid in (((32, 45), True), ((31, 45), False)):
            with self.subTest(sizes=sizes):
                graph = torch.fx.Graph()
                sources = [graph.placeholder("x"), graph.placeholder("y")]
                for source, size in zip(sources, sizes):
                    source.meta["val"] = torch.empty(2, size)
                    source.meta[QPARAM_KEY] = _mx()
                node = graph.call_function(torch.ops.aten.cat.default, (sources, 1))
                node.meta["val"] = torch.empty(2, sum(sizes))
                if valid:
                    self.assertEqual(remap_mx_qparam(_mx(), node), _mx(1))
                    sources[1].meta[QPARAM_KEY] = _mx(0)
                    with self.assertRaisesRegex(
                        NotYetSupportedError, "different MX domains"
                    ):
                        remap_mx_qparam(_mx(), node)
                else:
                    with self.assertRaisesRegex(
                        NotYetSupportedError, "incomplete MX blocks"
                    ):
                        remap_mx_qparam(_mx(), node)


class _Permute(torch.nn.Module):
    def forward(self, value):
        return torch.ops.aten.permute.default(value, [0, 2, 1])


class _WidthConcat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.concat = Concat(dim=1, allow_distinct_input_qparams=True)

    def forward(self, x, y):
        return self.concat((x, y))


class MXPropagationPassTest(unittest.TestCase):
    def _export(self):
        ep = torch.export.export(_Permute().eval(), (torch.randn(1, 3, 64),))
        source = next(node for node in ep.graph.nodes if node.op == "placeholder")
        target = next(node for node in ep.graph.nodes if node.op == "call_function")
        return ep, source, target

    def test_forward_remaps_and_is_idempotent(self):
        ep, source, target = self._export()
        source.meta[QPARAM_KEY] = _mx()
        PropagateQParamForward().call(ep)
        self.assertEqual(target.meta[QPARAM_KEY], _mx(1))
        before = copy.deepcopy(target.meta[QPARAM_KEY])
        PropagateQParamForward().call(ep)
        self.assertEqual(target.meta[QPARAM_KEY], before)
        self.assertEqual(source.meta[QPARAM_KEY], _mx())

    def test_forward_rejects_same_dtype_conflicting_destination(self):
        ep, source, target = self._export()
        source.meta[QPARAM_KEY] = _mx()
        target.meta[QPARAM_KEY] = _mx(2)
        with self.assertRaisesRegex(
            NotYetSupportedError, "Conflicting MX output domain"
        ):
            PropagateQParamForward().call(ep)
        self.assertEqual(target.meta[QPARAM_KEY], _mx(2))

    def test_backward_uses_inverse_permutation(self):
        ep, source, target = self._export()
        target.meta[QPARAM_KEY] = _mx(1)
        PropagateQParamBackward().call(ep)
        self.assertEqual(source.meta[QPARAM_KEY], _mx(2))
        self.assertEqual(target.meta[QPARAM_KEY], _mx(1))

    def test_backward_preserves_known_producer_domain(self):
        ep, source, target = self._export()
        source.meta[QPARAM_KEY] = _mx(0)
        target.meta[QPARAM_KEY] = _mx(1)
        PropagateQParamBackward().call(ep)
        self.assertEqual(source.meta[QPARAM_KEY], _mx(0))

    def test_backward_does_not_quantize_shared_float_producer(self):
        ep, source, target = self._export()
        output = next(node for node in ep.graph.nodes if node.op == "output")
        output.args = ((target, source),)
        target.meta[QPARAM_KEY] = _mx(1)
        PropagateQParamBackward().call(ep)
        self.assertNotIn(QPARAM_KEY, source.meta)

    def test_width_concat_affine_exception_does_not_skip_mx_domains(self):
        for backward in (False, True):
            with self.subTest(backward=backward):
                ep = torch.export.export(
                    _WidthConcat().eval(),
                    (torch.randn(1, 2, 64), torch.randn(1, 3, 64)),
                )
                sources = [node for node in ep.graph.nodes if node.op == "placeholder"]
                target = next(
                    node
                    for node in ep.graph.nodes
                    if node.target == torch.ops.aten.cat.default
                )
                if backward:
                    target.meta[QPARAM_KEY] = _mx(-1)
                    PropagateQParamBackward().call(ep)
                    for source in sources:
                        self.assertEqual(source.meta[QPARAM_KEY], _mx(2))
                else:
                    for source in sources:
                        source.meta[QPARAM_KEY] = _mx(-1)
                    PropagateQParamForward().call(ep)
                    self.assertEqual(target.meta[QPARAM_KEY], _mx(2))

    def test_concat_rejects_mixed_mx_and_float_inputs(self):
        ep = torch.export.export(
            _WidthConcat().eval(),
            (torch.randn(1, 2, 64), torch.randn(1, 3, 64)),
        )
        sources = [node for node in ep.graph.nodes if node.op == "placeholder"]
        sources[1].meta[QPARAM_KEY] = _mx(-1)
        with self.assertRaisesRegex(NotYetSupportedError, "different MX domains"):
            PropagateQParamForward().call(ep)

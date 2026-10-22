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

import unittest

import torch
from tico.quantization.evaluation.metric import compute_peir
from tico.utils.mx.mx_ops import quantize_mx

from torch.export import export

from test.support.helper import num_of_ops


# TODO Move this to test/modules/op
class SimpleMXINT8(torch.nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return quantize_mx(x, "int8", axis=self.axis)

    def get_example_inputs(self):
        return (torch.ones(1, 2, 3),), {}


class QuantizeMXTest(unittest.TestCase):
    # Test with a tensor filled with the same value
    # Fake-quant values must be the same with the original values
    def test_ones(self):
        input_ = torch.ones(1, 2, 3, dtype=torch.float32)

        output = quantize_mx(input_, "int8", 0)
        self.assertTrue(torch.allclose(input_, output))

        output = quantize_mx(input_, "int8", 1)
        self.assertTrue(torch.allclose(input_, output))

        output = quantize_mx(input_, "int8", 2)
        self.assertTrue(torch.allclose(input_, output))

    # Test with randomly generated values
    # Fake-quant values must be
    # (1) different from the original values
    # (2) but the difference should be small (PEIR < 0.01)
    def test_random_values(self):
        torch.manual_seed(0)
        input_ = torch.randn(1, 32, 32)

        output = quantize_mx(input_, "int8", 0)
        self.assertFalse(torch.allclose(input_, output))
        self.assertLess(compute_peir(input_, output), 0.01)

        output = quantize_mx(input_, "int8", 1)
        self.assertFalse(torch.allclose(input_, output))
        self.assertLess(compute_peir(input_, output), 0.01)

        output = quantize_mx(input_, "int8", 2)
        self.assertFalse(torch.allclose(input_, output))
        self.assertLess(compute_peir(input_, output), 0.01)

    def test_export(self):
        m = SimpleMXINT8(axis=2)
        args, kwargs = m.get_example_inputs()

        with torch.no_grad():
            ep = export(m.eval(), args, kwargs)

        self.assertEqual(
            1, num_of_ops(ep, [torch.ops.circle_custom.mx_fake_quantize.default])
        )


class ResidualAroundMXFakeQuantize(torch.nn.Module):
    """A residual branch next to an MX fake-quant of the same tensor."""

    def forward(self, x):
        return x + quantize_mx(x, "int8", axis=-1)

    def get_example_inputs(self):
        torch.manual_seed(0)
        return (torch.randn(2, 32),), {}


class MXFakeQuantizeExportAliasTest(unittest.TestCase):
    def test_eager_output_does_not_alias_input(self):
        for elem_format in ("int8", "fp4"):
            with self.subTest(elem_format=elem_format):
                x = torch.arange(64, dtype=torch.float32).reshape(2, 32) / 7
                original = x.clone()
                value = quantize_mx(x, elem_format, -1)
                value.zero_()
                torch.testing.assert_close(x, original, atol=0, rtol=0)

    def test_export_does_not_alias_output_to_input(self):
        """The fake kernel must not alias its input, or export rewires the residual.

        With an aliasing fake kernel the exported ``add`` consumed the
        ``mx_fake_quantize`` output twice; the untouched residual operand must
        stay the graph input.
        """
        m = ResidualAroundMXFakeQuantize().eval()
        args, kwargs = m.get_example_inputs()
        with torch.no_grad():
            ep = export(m, args, kwargs)
        add = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.add.Tensor)
        lhs, rhs = add.args
        self.assertEqual(lhs.op, "placeholder")
        self.assertEqual(rhs.target, torch.ops.circle_custom.mx_fake_quantize.default)
        self.assertIsNot(lhs, rhs)
        torch.testing.assert_close(ep.module()(*args), m(*args), atol=0, rtol=0)


class LogicalMXExportAliasTest(unittest.TestCase):
    def test_logical_operators_have_fresh_fake_outputs_and_remain_export_only(self):
        class LogicalMXResidual(torch.nn.Module):
            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, x):
                return x + self.op(x, "int8", -1)

        x = torch.arange(64, dtype=torch.float32).reshape(2, 32)
        for op in (
            torch.ops.circle_custom.quantize_mx.default,
            torch.ops.circle_custom.dequantize_mx.default,
        ):
            with self.subTest(op=op):
                with self.assertRaisesRegex(RuntimeError, "internal logical"):
                    op(x, "int8", -1)
                ep = export(LogicalMXResidual(op).eval(), (x,))
                add = next(
                    n for n in ep.graph.nodes if n.target == torch.ops.aten.add.Tensor
                )
                self.assertEqual(add.args[0].op, "placeholder")
                self.assertEqual(add.args[1].target, op)
                self.assertIsNot(add.args[0], add.args[1])


class MXFakeQuantizeOutputLayout(torch.nn.Module):
    """Exercise view operations and stride-dependent branches after MX quantization."""

    def __init__(self, elem_format: str, axis: int):
        super().__init__()
        self.elem_format = elem_format
        self.axis = axis

    def forward(self, x):
        value = quantize_mx(x, self.elem_format, self.axis)
        offset = 1 if value.is_contiguous() else -1
        # Keep stride differences observable even when the MX tensor is empty.
        stride = value.new_full((1,), value.stride(0))
        return value.contiguous().view(-1), value + offset, stride


class MXFakeQuantizeExportLayoutTest(unittest.TestCase):
    def _assert_export_layout(self, x, elem_format, axis):
        module = MXFakeQuantizeOutputLayout(elem_format, axis).eval()
        eager_value = quantize_mx(x, elem_format, axis)
        with torch.no_grad():
            ep = export(module, (x,))
        node = next(
            n
            for n in ep.graph.nodes
            if n.target == torch.ops.circle_custom.mx_fake_quantize.default
        )
        fake_value = node.meta["val"]
        self.assertEqual(eager_value.shape, x.shape)
        self.assertEqual(eager_value.dtype, x.dtype)
        self.assertEqual(eager_value.device, x.device)
        self.assertTrue(eager_value.is_contiguous())
        self.assertEqual(fake_value.shape, eager_value.shape)
        self.assertEqual(fake_value.dtype, eager_value.dtype)
        self.assertEqual(fake_value.device, eager_value.device)
        self.assertEqual(fake_value.stride(), eager_value.stride())
        torch.testing.assert_close(ep.module()(x), module(x), atol=0, rtol=0)

    def test_partial_blocks_with_sliced_and_contiguous_inputs(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = torch.arange(64, dtype=dtype).reshape(2, 32) / 7
            sliced = values[:, :3]
            for layout, x in (
                ("sliced", sliced),
                ("contiguous", sliced.contiguous()),
                ("singleton", values[:1, :3]),
            ):
                for elem_format in ("int8", "fp4"):
                    with self.subTest(
                        dtype=dtype, layout=layout, elem_format=elem_format
                    ):
                        self._assert_export_layout(x, elem_format, -1)

    def test_transposed_and_channels_last_inputs(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            values = torch.arange(64, dtype=dtype).reshape(2, 32) / 7
            channels_last = (
                torch.arange(120, dtype=dtype)
                .reshape(2, 3, 4, 5)
                .contiguous(memory_format=torch.channels_last)
            )
            for layout, x, axis in (
                ("transposed_aligned", values.t(), 0),
                ("transposed_partial", values.t()[:3], 0),
                ("channels_last", channels_last, 1),
            ):
                for elem_format in ("int8", "fp4"):
                    with self.subTest(
                        dtype=dtype, layout=layout, elem_format=elem_format
                    ):
                        self._assert_export_layout(x, elem_format, axis)

    def test_empty_non_reduction_dimensions(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shape in ((0, 3), (1, 0, 3)):
                for elem_format in ("int8", "fp4"):
                    with self.subTest(
                        dtype=dtype, shape=shape, elem_format=elem_format
                    ):
                        self._assert_export_layout(
                            torch.empty(shape, dtype=dtype), elem_format, -1
                        )

    def test_partial_block_then_view(self):
        class MXThenView(torch.nn.Module):
            def forward(self, x):
                y = torch.ops.circle_custom.mx_fake_quantize(x, "int8", -1)
                return y.contiguous().view(-1)

        x = torch.arange(64, dtype=torch.float32).reshape(2, 32)[:, :3]
        module = MXThenView().eval()
        ep = export(module, (x,))
        torch.testing.assert_close(ep.module()(x), module(x), atol=0, rtol=0)

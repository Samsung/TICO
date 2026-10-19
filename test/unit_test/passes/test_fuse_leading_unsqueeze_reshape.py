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

import math
import unittest

import tico
import torch
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.fuse_leading_unsqueeze_reshape import FuseLeadingUnsqueezeReshape
from tico.passes.ops import aten

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class LeadingUnsqueezeReshapeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.reshape(x, shape=(20, 8, 64))
        z = torch.permute(y, dims=(1, 0, 2))
        k = torch.reshape(z, shape=(1, 8, 20, 64))
        return k

    def get_example_inputs(self):
        return (torch.randn(20, 1, 512),), {}


class LeadingUnsqueezeReshapeNetTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(LeadingUnsqueezeReshapeNet())
        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), aten.reshape), 2)
        self.assertEqual(num_of_ops(self.exported_program(), aten.permute), 1)

        self.run_value_test(FuseLeadingUnsqueezeReshape())
        self.assertEqual(num_of_ops(self.exported_program(), aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), aten.permute), 1)


class LeadingUnsqueezeReshapeNet2k(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.reshape(x, shape=(20, 8, 64))
        z = torch.permute(y, dims=(1, 0, 2))
        k = torch.reshape(z, shape=(1, 1, 8, 20, 64))
        return k

    def get_example_inputs(self):
        return (torch.randn(20, 1, 512),), {}


class LeadingUnsqueezeReshapeNet2kTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(LeadingUnsqueezeReshapeNet2k())
        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), aten.reshape), 2)
        self.assertEqual(num_of_ops(self.exported_program(), aten.permute), 1)

        self.run_value_test(FuseLeadingUnsqueezeReshape())
        self.assertEqual(num_of_ops(self.exported_program(), aten.reshape), 1)
        self.assertEqual(num_of_ops(self.exported_program(), aten.permute), 1)


class _LeadingUnsqueeze(torch.nn.Module):
    def __init__(self, shape, dims, output_shape, reuse=False):
        super().__init__()
        self.shape = shape
        self.dims = dims
        self.output_shape = output_shape
        self.reuse = reuse

    def forward(self, x):
        front = x.reshape(self.shape)
        permuted = front.permute(self.dims)
        result = permuted.reshape(self.output_shape)
        if self.reuse:
            return result, permuted, front
        return result


class LeadingUnsqueezeAxesTest(unittest.TestCase):
    def _assert_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def _apply(self, ep):
        input_specs = tuple(ep.graph_signature.input_specs)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(FuseLeadingUnsqueezeReshape().call(ep).modified)
        self.assertEqual(tuple(ep.graph_signature.input_specs), input_specs)
        ep.graph.lint()
        graph = str(ep.graph)
        self.assertFalse(FuseLeadingUnsqueezeReshape().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)

    def test_positive_negative_and_mixed_axes(self):
        for shape, axes, normalized in (
            ((2, 3), ((1, 0), (-1, -2), (1, -2)), (1, 0)),
            ((2, 3, 4), ((2, 0, 1), (-1, -3, -2), (2, -3, 1)), (2, 0, 1)),
        ):
            for dims in axes:
                for k in (1, 2):
                    for dtype in (torch.float32, torch.int64):
                        with self.subTest(shape=shape, dims=dims, k=k, dtype=dtype):
                            output_shape = (1,) * k + tuple(
                                shape[d] for d in normalized
                            )
                            model = _LeadingUnsqueeze(shape, dims, output_shape).eval()
                            x = torch.arange(math.prod(shape), dtype=dtype)
                            ep = torch.export.export(model, (x,))
                            expected = ep.module()(x)
                            self._apply(ep)
                            self._assert_equal(ep.module()(x), expected)
                            self.assertEqual(num_of_ops(ep, aten.reshape), 1)
                            self.assertEqual(num_of_ops(ep, aten.permute), 1)
                            permute = next(
                                n for n in ep.graph.nodes if n.target in aten.permute
                            )
                            reshape = permute.args[0]
                            self.assertIsInstance(reshape, torch.fx.Node)
                            assert isinstance(reshape, torch.fx.Node)
                            self.assertEqual(reshape.args[1], [1] * k + list(shape))
                            self.assertEqual(
                                permute.args[1],
                                list(range(k)) + [d + k for d in normalized],
                            )
                            self.assertEqual(
                                tuple(permute.meta["val"].shape), output_shape
                            )
                            self.assertEqual(permute.meta["val"].dtype, dtype)

    def test_multiple_users_keep_original_views(self):
        for dims in ((1, 0), (-1, -2), (1, -2)):
            for k in (1, 2):
                with self.subTest(dims=dims, k=k):
                    model = _LeadingUnsqueeze(
                        (2, 3), dims, (1,) * k + (3, 2), reuse=True
                    ).eval()
                    x = torch.arange(6, dtype=torch.float32)
                    ep = torch.export.export(model, (x,))
                    expected = ep.module()(x)
                    original_permute = next(
                        n for n in ep.graph.nodes if n.target in aten.permute
                    )
                    original_reshape = original_permute.args[0]
                    self._apply(ep)
                    actual = ep.module()(x)
                    self.assertEqual(len(actual), len(expected))
                    for result, reference in zip(actual, expected):
                        self._assert_equal(result, reference)
                    output = next(n for n in ep.graph.nodes if n.op == "output")
                    self.assertIs(output.args[0][1], original_permute)
                    self.assertIs(output.args[0][2], original_reshape)
                    self.assertEqual(original_permute.args[1], list(dims))
                    self.assertEqual(num_of_ops(ep, aten.reshape), 2)
                    self.assertEqual(num_of_ops(ep, aten.permute), 2)

    def test_non_leading_reshape_is_unchanged(self):
        for dims in ((1, 0), (-1, -2), (1, -2)):
            for output_shape in ((3, 2, 1), (1, 6), (3, 2)):
                with self.subTest(dims=dims, output_shape=output_shape):
                    model = _LeadingUnsqueeze((2, 3), dims, output_shape).eval()
                    x = torch.arange(6, dtype=torch.int64)
                    ep = torch.export.export(model, (x,))
                    graph = str(ep.graph)
                    self.assertFalse(FuseLeadingUnsqueezeReshape().call(ep).modified)
                    self.assertEqual(str(ep.graph), graph)
                    self._assert_equal(ep.module()(x), model(x))

    def test_public_conversion_with_default_circle_optimization(self):
        for dims in ((1, 0), (-1, -2), (1, -2)):
            for k in (1, 2):
                with self.subTest(dims=dims, k=k):
                    model = _LeadingUnsqueeze((2, 3), dims, (1,) * k + (3, 2)).eval()
                    x = torch.arange(6, dtype=torch.float32)
                    circle = tico.convert(model, (x,))
                    self._assert_equal(torch.from_numpy(circle(x)), model(x))

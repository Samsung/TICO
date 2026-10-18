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

import unittest

import tico
import torch
from tico.passes.restore_linear import RestoreLinear


class _Matmul(torch.nn.Module):
    def __init__(self, dims=None, bias=False, alpha=1, beta=1, reuse=False):
        super().__init__()
        self.dims = dims
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.reuse = reuse

    def forward(self, x, weight, bias):
        rhs = weight.t() if self.dims is None else weight.permute(self.dims)
        if self.bias:
            result = torch.addmm(bias, x, rhs, alpha=self.alpha, beta=self.beta)
        else:
            result = torch.mm(x, rhs)
        if self.reuse:
            return result, rhs, weight + 1
        return result


def _inputs(transpose=True, rows=2):
    x = torch.arange(1, rows * 3 + 1, dtype=torch.float32).reshape(rows, 3)
    weight = torch.arange(1, 13, dtype=torch.float32).reshape(
        (4, 3) if transpose else (3, 4)
    )
    return x, weight, torch.arange(11, 15, dtype=torch.float32)


class RestoreLinearTest(unittest.TestCase):
    def _run(self, ep):
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            result = RestoreLinear().call(ep)
        ep.graph.lint()
        return result.modified

    def _assert_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_nondefault_addmm_coefficients_are_not_restored(self):
        inputs = _inputs()
        for alpha, beta in ((2, 1), (1, 3), (2, 3), (0, 1), (1, 0)):
            for dims in (None, (1, 0)):
                with self.subTest(alpha=alpha, beta=beta, dims=dims):
                    model = _Matmul(dims, bias=True, alpha=alpha, beta=beta).eval()
                    ep = torch.export.export(model, inputs)
                    graph = str(ep.graph)
                    self.assertFalse(self._run(ep))
                    self.assertEqual(str(ep.graph), graph)
                    self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_identity_permute_is_not_restored(self):
        square_inputs = (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
            torch.tensor([11.0, 13.0]),
        )
        for inputs in (_inputs(transpose=False), square_inputs):
            for bias in (False, True):
                with self.subTest(bias=bias, weight_shape=inputs[1].shape):
                    model = _Matmul((0, 1), bias=bias).eval()
                    ep = torch.export.export(model, inputs)
                    graph = str(ep.graph)
                    self.assertFalse(self._run(ep))
                    self.assertEqual(str(ep.graph), graph)
                    self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_transposes_restore_linear_and_preserve_metadata(self):
        inputs = _inputs()
        for dims in (None, (1, 0), (-1, -2), (1, -2), (-1, 0)):
            for bias in (False, True):
                with self.subTest(dims=dims, bias=bias):
                    model = _Matmul(dims, bias=bias).eval()
                    ep = torch.export.export(model, inputs)
                    original = next(
                        n
                        for n in ep.graph.nodes
                        if n.target
                        in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)
                    )
                    original.meta["stack_trace"] = "matmul provenance"
                    metadata = dict(original.meta)
                    signature = tuple(ep.graph_signature.input_specs)
                    placeholders = tuple(
                        n for n in ep.graph.nodes if n.op == "placeholder"
                    )
                    self.assertTrue(self._run(ep))
                    linear = next(
                        n
                        for n in ep.graph.nodes
                        if n.target == torch.ops.aten.linear.default
                    )
                    self.assertEqual(
                        linear.args, placeholders if bias else placeholders[:2]
                    )
                    for key, value in metadata.items():
                        self.assertIs(linear.meta[key], value)
                    self.assertEqual(tuple(ep.graph_signature.input_specs), signature)
                    self.assertFalse(
                        any(n.target == original.target for n in ep.graph.nodes)
                    )
                    self._assert_equal(ep.module()(*inputs), model(*inputs))
                    graph = str(ep.graph)
                    self.assertFalse(self._run(ep))
                    self.assertEqual(str(ep.graph), graph)

    def test_other_transpose_users_and_input_ownership_are_preserved(self):
        inputs = _inputs()
        originals = tuple(value.clone() for value in inputs)
        for bias in (False, True):
            with self.subTest(bias=bias):
                model = _Matmul((1, 0), bias=bias, reuse=True).eval()
                ep = torch.export.export(model, inputs)
                self.assertTrue(self._run(ep))
                self.assertEqual(
                    sum(
                        n.target == torch.ops.aten.permute.default
                        for n in ep.graph.nodes
                    ),
                    1,
                )
                for actual, expected in zip(ep.module()(*inputs), model(*inputs)):
                    self._assert_equal(actual, expected)
                for actual, original in zip(inputs, originals):
                    self._assert_equal(actual, original)

    def test_dynamic_batch_remains_symbolic(self):
        inputs = _inputs(rows=3)
        batch = torch.export.Dim("batch", min=2, max=6)
        model = _Matmul(bias=True).eval()
        ep = torch.export.export(model, inputs, dynamic_shapes=({0: batch}, {}, {}))
        constraints = dict(ep.range_constraints)
        self.assertTrue(self._run(ep))
        linear = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten.linear.default
        )
        self.assertIsInstance(linear.meta["val"].shape[0], torch.SymInt)
        self.assertEqual(ep.range_constraints, constraints)
        for rows in (2, 5):
            inputs = _inputs(rows=rows)
            self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_permute_copy_is_not_added_to_supported_patterns(self):
        inputs = _inputs()
        model = _Matmul((1, 0)).eval()
        ep = torch.export.export(model, inputs)
        permute = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten.permute.default
        )
        permute.target = torch.ops.aten.permute_copy.default
        ep.graph_module.recompile()
        graph = str(ep.graph)
        self.assertFalse(self._run(ep))
        self.assertEqual(str(ep.graph), graph)
        self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_default_circle_conversion_preserves_values(self):
        cases = (
            (_Matmul(), True),
            (_Matmul((1, 0), bias=True), True),
            (_Matmul((0, 1)), False),
            (_Matmul((0, 1), bias=True), False),
            (_Matmul(bias=True, alpha=2), True),
            (_Matmul(bias=True, beta=3), True),
            (_Matmul(bias=True, alpha=2, beta=3), True),
        )
        for model, transpose in cases:
            with self.subTest(
                dims=model.dims,
                bias=model.bias,
                alpha=model.alpha,
                beta=model.beta,
            ):
                inputs = _inputs(transpose=transpose)
                if not transpose:
                    inputs = (inputs[0], inputs[1][:, :3], inputs[2][:3])
                model.eval()
                expected = model(*inputs)
                actual = torch.from_numpy(tico.convert(model, inputs)(*inputs))
                self._assert_equal(actual, expected)

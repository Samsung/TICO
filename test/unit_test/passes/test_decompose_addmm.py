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

from tico.passes.decompose_addmm import DecomposeAddmm
from tico.passes.fill_meta_val import FillMetaVal
from torch.export.graph_signature import InputKind, TensorArgument


class _Addmm(torch.nn.Module):
    def __init__(self, alpha=1, beta=1, reuse=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reuse = reuse

    def forward(self, bias, lhs, rhs):
        result = torch.addmm(bias, lhs, rhs, alpha=self.alpha, beta=self.beta)
        if self.reuse:
            return result, bias + 1, lhs + 1, rhs + 1
        return result


def _inputs(dtype, bias_rank=1):
    bias = torch.tensor([16777217, 2], dtype=dtype)
    if bias_rank == 2:
        bias = bias.repeat(2, 1)
    return (
        bias,
        torch.tensor([[1, 2], [3, 4]], dtype=dtype),
        torch.tensor([[2, 3], [5, 7]], dtype=dtype),
    )


class DecomposeAddmmTest(unittest.TestCase):
    def _assert_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected))

    def _lower(self, model, inputs):
        ep = torch.export.export(model.eval(), inputs)
        addmm = next(
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.addmm.default
        )
        metadata = addmm.meta["val"]
        user_specs = tuple(ep.graph_signature.input_specs)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(DecomposeAddmm().call(ep).modified)
        self.assertFalse(
            any(node.target == torch.ops.aten.addmm.default for node in ep.graph.nodes)
        )
        output = next(node for node in ep.graph.nodes if node.op == "output")
        results = output.args[0]
        assert isinstance(results, tuple)
        result = results[0]
        assert isinstance(result, torch.fx.Node)
        self.assertIs(result.meta["val"], metadata)
        self.assertEqual(
            tuple(
                spec
                for spec in ep.graph_signature.input_specs
                if spec.kind == InputKind.USER_INPUT
            ),
            user_specs,
        )
        placeholders = tuple(
            node for node in ep.graph.nodes if node.op == "placeholder"
        )
        input_names = []
        for spec in ep.graph_signature.input_specs:
            assert isinstance(spec.arg, TensorArgument)
            input_names.append(spec.arg.name)
        self.assertEqual(
            tuple(node.name for node in placeholders),
            tuple(input_names),
        )
        zero_shapes = []
        if model.beta == 0:
            zero_shapes.append(inputs[0].shape)
        if model.alpha == 0:
            zero_shapes.append(torch.Size((inputs[1].shape[0], inputs[2].shape[1])))
        self.assertEqual(len(ep.constants), len(zero_shapes))
        for constant, shape in zip(ep.constants.values(), zero_shapes):
            self.assertEqual(constant.dtype, inputs[0].dtype)
            self.assertEqual(constant.device, torch.device("cpu"))
            self.assertEqual(constant.shape, shape)
            self.assertEqual(torch.count_nonzero(constant).item(), 0)
            for value in inputs:
                self.assertNotEqual(constant.data_ptr(), value.data_ptr())
        FillMetaVal().call(ep)
        self.assertEqual(result.meta["val"].dtype, inputs[0].dtype)
        ep.graph.lint()
        graph = str(ep.graph)
        constants = tuple(ep.constants.values())
        self.assertFalse(DecomposeAddmm().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        for actual, expected in zip(ep.constants.values(), constants):
            self.assertIs(actual, expected)
        return ep

    def test_zero_coefficients_preserve_dtype_values_and_constants(self):
        for dtype in (torch.int32, torch.int64, torch.float32):
            for alpha, beta in ((0, 1), (1, 0), (0, 0)):
                for bias_rank in (1, 2):
                    with self.subTest(
                        dtype=dtype, alpha=alpha, beta=beta, rank=bias_rank
                    ):
                        inputs = _inputs(dtype, bias_rank)
                        model = _Addmm(alpha, beta)
                        ep = self._lower(model, inputs)
                        self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_nonzero_coefficients_do_not_create_zero_constants(self):
        for dtype in (torch.int32, torch.int64, torch.float32):
            for alpha, beta in ((1, 1), (2, 3)):
                with self.subTest(dtype=dtype, alpha=alpha, beta=beta):
                    inputs = _inputs(dtype)
                    model = _Addmm(alpha, beta)
                    ep = self._lower(model, inputs)
                    self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_default_circle_conversion_preserves_integer_dtype_and_precision(self):
        for dtype in (torch.int32, torch.int64):
            for alpha, beta in ((0, 1), (1, 0), (0, 0), (1, 1), (2, 3)):
                for bias_rank in (1, 2):
                    with self.subTest(
                        dtype=dtype, alpha=alpha, beta=beta, rank=bias_rank
                    ):
                        inputs = _inputs(dtype, bias_rank)
                        model = _Addmm(alpha, beta).eval()
                        circle = tico.convert(model, inputs)
                        self.assertEqual(circle.runtime, "reference")
                        self._assert_equal(
                            torch.from_numpy(circle(*inputs)), model(*inputs)
                        )

    def test_float32_zeros_ignore_global_default_dtype(self):
        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            for alpha, beta in ((0, 1), (1, 0), (0, 0), (1, 1)):
                with self.subTest(alpha=alpha, beta=beta):
                    inputs = _inputs(torch.float32)
                    model = _Addmm(alpha, beta).eval()
                    ep = self._lower(model, inputs)
                    self._assert_equal(ep.module()(*inputs), model(*inputs))
                    self._assert_equal(
                        torch.from_numpy(tico.convert(model, inputs)(*inputs)),
                        model(*inputs),
                    )
        finally:
            torch.set_default_dtype(default_dtype)

    def test_zero_beta_still_ignores_nan_bias(self):
        for bias_rank in (1, 2):
            with self.subTest(rank=bias_rank):
                inputs = _inputs(torch.float32, bias_rank)
                inputs[0].fill_(float("nan"))
                model = _Addmm(beta=0).eval()
                expected = torch.mm(inputs[1], inputs[2])
                ep = self._lower(model, inputs)
                self._assert_equal(ep.module()(*inputs), expected)
                self._assert_equal(
                    torch.from_numpy(tico.convert(model, inputs)(*inputs)), expected
                )

    def test_other_users_and_input_ownership_are_preserved(self):
        inputs = _inputs(torch.int64)
        originals = tuple(value.clone() for value in inputs)
        model = _Addmm(alpha=0, reuse=True)
        ep = self._lower(model, inputs)
        outputs = ep.module()(*inputs)
        for actual, expected in zip(outputs, model(*inputs)):
            self._assert_equal(actual, expected)
        outputs[0].fill_(-1)
        for actual, expected in zip(inputs, originals):
            self._assert_equal(actual, expected)

    def test_graph_without_addmm_is_unchanged(self):
        model = torch.nn.Linear(2, 2).eval()
        inputs = (torch.tensor([[1.0, 2.0]]),)
        ep = torch.export.export(model, inputs)
        graph = str(ep.graph)
        signature = tuple(ep.graph_signature.input_specs)
        self.assertFalse(DecomposeAddmm().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        self.assertEqual(tuple(ep.graph_signature.input_specs), signature)
        self._assert_equal(ep.module()(*inputs), model(*inputs))

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
from tico.passes.lower_copy import LowerCopy
from tico.serialize.quant_param import QPARAM_KEY, QuantParam


class _Copy(torch.nn.Module):
    def forward(self, dst, src):
        return dst.clone().copy_(src)


class _CopyAndReuse(torch.nn.Module):
    def forward(self, dst, src):
        return dst.clone().copy_(src), src * 2


class LowerCopyTest(unittest.TestCase):
    def _lower(self, model, inputs, copy_metadata=None):
        ep = torch.export.export(model.eval(), inputs).run_decompositions({})
        self.assertEqual(
            sum(n.target == torch.ops.aten.copy.default for n in ep.graph.nodes), 1
        )
        if copy_metadata is not None:
            copy = next(
                n for n in ep.graph.nodes if n.target == torch.ops.aten.copy.default
            )
            copy.meta.update(copy_metadata)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(LowerCopy().call(ep).modified)
        self.assertFalse(
            any(n.target == torch.ops.aten.copy.default for n in ep.graph.nodes)
        )
        graph = str(ep.graph)
        self.assertFalse(LowerCopy().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        return ep

    def _assert_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def test_cast_precedes_same_shape_and_broadcast_lowering(self):
        for dst_shape, src_shape in (((2,), (2,)), ((2, 2), (2,)), ((2, 2), ())):
            with self.subTest(dst_shape=dst_shape, src_shape=src_shape):
                dst = torch.zeros(dst_shape, dtype=torch.int32)
                src = torch.tensor([1.5, -2.5]) if src_shape else torch.tensor(1.5)
                ep = self._lower(_Copy(), (dst, src))
                casts = [
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten._to_copy.default
                ]
                self.assertEqual(len(casts), 1)
                cast = casts[0]
                source = next(n for n in ep.graph.nodes if n.name == "src")
                self.assertEqual(cast.args, (source,))
                self.assertEqual(cast.kwargs, {"dtype": dst.dtype})
                self.assertIn("nn_module_stack", cast.meta)
                self.assertEqual(cast.meta["val"].dtype, dst.dtype)
                self.assertEqual(cast.meta["val"].shape, src.shape)
                expands = [
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten.expand.default
                ]
                self.assertEqual(len(expands), int(dst_shape != src_shape))
                if expands:
                    expanded = expands[0]
                    self.assertEqual(expanded.args, (cast, list(dst_shape)))
                    self.assertIn("nn_module_stack", expanded.meta)
                    expected = src.to(dst.dtype).expand(dst_shape)
                    self.assertEqual(expanded.meta["val"].dtype, expected.dtype)
                    self.assertEqual(expanded.meta["val"].shape, expected.shape)
                    self.assertEqual(expanded.meta["val"].stride(), expected.stride())
                    if "tensor_meta" in expanded.meta:
                        self.assertEqual(
                            expanded.meta["tensor_meta"].stride, expected.stride()
                        )
                actual = ep.module()(dst, src)
                self._assert_equal(actual, _Copy()(dst, src))
                self.assertNotEqual(actual.data_ptr(), src.data_ptr())

    def test_same_dtype_keeps_existing_shape_paths_without_cast(self):
        src = torch.tensor([1.5, -2.5])
        for dst_shape in ((2,), (2, 2)):
            with self.subTest(dst_shape=dst_shape):
                dst = torch.zeros(dst_shape)
                ep = self._lower(_Copy(), (dst, src))
                self.assertFalse(
                    any(
                        n.target == torch.ops.aten._to_copy.default
                        for n in ep.graph.nodes
                    )
                )
                actual = ep.module()(dst, src)
                self._assert_equal(actual, _Copy()(dst, src))
                self.assertEqual(actual.data_ptr(), src.data_ptr())

    def test_broadcast_preserves_quantization_and_provenance_metadata(self):
        qparam = QuantParam(scale=[0.5], zero_point=[0], dtype="int32")
        for src_dtype in (torch.int32, torch.float32):
            with self.subTest(src_dtype=src_dtype):
                inputs = (
                    torch.zeros(2, 2, dtype=torch.int32),
                    torch.tensor([-2, 3], dtype=src_dtype),
                )
                ep = self._lower(
                    _Copy(),
                    inputs,
                    {QPARAM_KEY: qparam, "stack_trace": "copy provenance"},
                )
                expanded = next(
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten.expand.default
                )
                self.assertIs(expanded.meta[QPARAM_KEY], qparam)
                self.assertEqual(expanded.meta["stack_trace"], "copy provenance")
                self.assertNotIn("tensor_meta", expanded.meta)
                self._assert_equal(ep.module()(*inputs), _Copy()(*inputs))

    def test_cast_leaves_other_source_users_and_inputs_unchanged(self):
        inputs = (torch.zeros(2, 2, dtype=torch.int32), torch.tensor([1.5, -2.5]))
        originals = tuple(value.clone() for value in inputs)
        model = _CopyAndReuse().eval()
        ep = self._lower(model, inputs)
        source = next(n for n in ep.graph.nodes if n.name == "src")
        multiply = next(
            n for n in ep.graph.nodes if n.target == torch.ops.aten.mul.Tensor
        )
        self.assertIs(multiply.args[0], source)
        self.assertEqual(source.meta["val"].dtype, torch.float32)
        for actual, expected in zip(ep.module()(*inputs), model(*inputs)):
            self._assert_equal(actual, expected)
        for actual, expected in zip(inputs, originals):
            self._assert_equal(actual, expected)

    def test_inplace_copy_is_not_matched(self):
        inputs = (torch.zeros(2, dtype=torch.int32), torch.tensor([1.5, -2.5]))
        ep = torch.export.export(_Copy().eval(), inputs)
        graph = str(ep.graph)
        self.assertTrue(
            any(n.target == torch.ops.aten.copy_.default for n in ep.graph.nodes)
        )
        self.assertFalse(LowerCopy().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)

    def test_default_conversion_preserves_copy_dtype_and_values(self):
        cases = (
            (torch.zeros(2, dtype=torch.int32), torch.tensor([1.5, -2.5])),
            (torch.zeros(2, 2, dtype=torch.int32), torch.tensor([1.5, -2.5])),
            (torch.zeros(2, 2, dtype=torch.int32), torch.tensor(-2.5)),
            (torch.zeros(()), torch.tensor(3, dtype=torch.int32)),
            (torch.zeros(2), torch.tensor([1, -2], dtype=torch.int32)),
            (torch.zeros(2, 2), torch.tensor([True, False])),
            (torch.zeros(2, dtype=torch.bool), torch.tensor([0.0, -2.5])),
        )
        model = _Copy().eval()
        for dst, src in cases:
            with self.subTest(
                dst_shape=dst.shape, src_shape=src.shape, dtype=dst.dtype
            ):
                originals = (dst.clone(), src.clone())
                expected = model(dst, src)
                circle = tico.convert(model, (dst, src))
                self.assertEqual(circle.runtime, "reference")
                for _ in range(2):
                    self._assert_equal(torch.from_numpy(circle(dst, src)), expected)
                for actual, original in zip((dst, src), originals):
                    self._assert_equal(actual, original)

    def test_default_conversion_preserves_other_source_users(self):
        model = _CopyAndReuse().eval()
        inputs = (torch.zeros(2, 2, dtype=torch.int32), torch.tensor([1.5, -2.5]))
        originals = tuple(value.clone() for value in inputs)
        expected = model(*inputs)
        circle = tico.convert(model, inputs)
        self.assertEqual(circle.runtime, "reference")
        actual = circle(*inputs)
        self.assertEqual(len(actual), len(expected))
        for result, reference in zip(actual, expected):
            self._assert_equal(torch.from_numpy(result), reference)
        for result, original in zip(inputs, originals):
            self._assert_equal(result, original)

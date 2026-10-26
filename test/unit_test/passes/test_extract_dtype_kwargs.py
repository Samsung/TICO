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
from typing import Any

import tico
import torch
from tico.passes.extract_dtype_kwargs import _extract_to_output, ExtractDtypeKwargsPass
from tico.serialize.circle_graph import CircleModel, CircleSubgraph
from tico.serialize.operators.op_full_like import FullLikeVisitor
from tico.utils.validate_args_kwargs import FullLikeArgs

_CASES: tuple[
    tuple[torch.dtype, int | float, dict[str, torch.dtype | None], torch.dtype], ...
] = (
    (torch.int32, 0.5, {"dtype": torch.float32}, torch.float32),
    (torch.bool, 2, {"dtype": torch.int32}, torch.int32),
    (torch.float32, 2**24 + 1, {"dtype": torch.int64}, torch.int64),
    (torch.int32, -7, {"dtype": None}, torch.int32),
    (torch.int32, -7, {}, torch.int32),
    (torch.float32, 0.5, {"dtype": torch.float32}, torch.float32),
)


class _FullLike(torch.nn.Module):
    def __init__(self, fill_value, kwargs):
        super().__init__()
        self.fill_value = fill_value
        self.kwargs = kwargs

    def forward(self, value, addend):
        return torch.full_like(value, self.fill_value, **self.kwargs) + addend


class ExtractDtypeKwargsTest(unittest.TestCase):
    def test_full_like_keeps_creation_dtype_and_graph_unchanged(self):
        for input_dtype, fill_value, kwargs, output_dtype in _CASES:
            with self.subTest(input_dtype=input_dtype, kwargs=kwargs):
                model = _FullLike(fill_value, kwargs).eval()
                inputs = (
                    torch.tensor([0, 1], dtype=input_dtype),
                    torch.zeros(2, dtype=output_dtype),
                )
                ep = torch.export.export(model, inputs)
                full_like = next(
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten.full_like.default
                )
                original_meta = full_like.meta["val"]
                original_graph = str(ep.graph)
                expected = model(*inputs)

                for _ in range(2):
                    self.assertFalse(ExtractDtypeKwargsPass().call(ep).modified)
                    self.assertEqual(str(ep.graph), original_graph)
                    self.assertIs(full_like.meta["val"], original_meta)
                    self.assertEqual(full_like.meta["val"].dtype, output_dtype)
                    self.assertEqual(full_like.meta["val"].shape, inputs[0].shape)
                    self.assertTrue(torch.equal(ep.module()(*inputs), expected))

    def test_full_like_visitor_builds_constants_in_requested_dtype(self):
        for input_dtype, fill_value, kwargs, output_dtype in _CASES:
            with self.subTest(input_dtype=input_dtype, kwargs=kwargs):
                inputs = (
                    torch.tensor([0, 1], dtype=input_dtype),
                    torch.zeros(2, dtype=output_dtype),
                )
                ep = torch.export.export(_FullLike(fill_value, kwargs).eval(), inputs)
                full_like = next(
                    n
                    for n in ep.graph.nodes
                    if n.target == torch.ops.aten.full_like.default
                )
                model = CircleModel()
                graph = CircleSubgraph(model)
                graph.add_tensor_from_node(full_like)

                self.assertIsNone(FullLikeVisitor({}, graph).define_node(full_like))
                tensor = graph.get_tensor(full_like)
                expected = torch.tensor([fill_value, fill_value], dtype=output_dtype)
                self.assertEqual(
                    bytes(model.buffers[tensor.buffer].data), expected.numpy().tobytes()
                )

    def test_full_like_default_conversion_preserves_values_and_inputs(self):
        for input_dtype, fill_value, kwargs, output_dtype in _CASES:
            with self.subTest(input_dtype=input_dtype, kwargs=kwargs):
                model = _FullLike(fill_value, kwargs).eval()
                inputs = (
                    torch.tensor([0, 1], dtype=input_dtype),
                    torch.tensor([1, 0], dtype=output_dtype),
                )
                originals = tuple(value.clone() for value in inputs)
                expected = model(*inputs)
                circle = tico.convert(model, inputs)
                self.assertEqual(circle.runtime, "reference")

                actual = torch.from_numpy(circle(*inputs))
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.equal(actual, expected))
                for value, original in zip(inputs, originals):
                    self.assertTrue(torch.equal(value, original))

    def test_full_like_keeps_memory_format_and_pin_memory_options(self):
        value = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32).t()
        original = value.clone()
        for memory_format in (torch.contiguous_format, torch.preserve_format):
            with self.subTest(memory_format=memory_format):
                model = _FullLike(
                    0.5,
                    {
                        "dtype": torch.float32,
                        "memory_format": memory_format,
                        "pin_memory": False,
                    },
                ).eval()
                inputs = (value, torch.zeros(2, 2))
                ep = torch.export.export(model, inputs)
                original_graph = str(ep.graph)
                self.assertFalse(ExtractDtypeKwargsPass().call(ep).modified)
                self.assertEqual(str(ep.graph), original_graph)
                actual = torch.from_numpy(tico.convert(model, inputs)(*inputs))
                self.assertTrue(torch.equal(actual, model(*inputs)))
                self.assertTrue(torch.equal(value, original))

    def test_full_like_args_keep_existing_positional_options(self):
        node = torch.fx.Graph().placeholder("value")
        args = FullLikeArgs(node, 1, False, torch.preserve_format)
        self.assertIs(args.input, node)
        self.assertIs(args.pin_memory, False)
        self.assertEqual(args.memory_format, torch.preserve_format)
        self.assertIsNone(args.dtype)

    def test_full_like_explicit_handler_registration_still_runs(self):
        # Zero is exactly representable in both dtypes, so this caller's opt-in
        # extraction is safe and exercises the existing handler interface.
        model = _FullLike(0, {"dtype": torch.float32}).eval()
        inputs = (torch.tensor([0, 1], dtype=torch.int32), torch.zeros(2))
        ep = torch.export.export(model, inputs)
        rewrite = ExtractDtypeKwargsPass()
        rewrite.target_ops[torch.ops.aten.full_like.default] = _extract_to_output

        self.assertTrue(rewrite.call(ep).modified)
        self.assertTrue(torch.equal(ep.module()(*inputs), model(*inputs)))
        self.assertEqual(ep.module()(*inputs).dtype, torch.float32)
        self.assertFalse(rewrite.call(ep).modified)

    def test_full_like_args_keep_device_and_layout_rejection(self):
        node = torch.fx.Graph().placeholder("value")
        for name, value in (("device", torch.device("cpu")), ("layout", torch.strided)):
            with self.subTest(option=name):
                kwargs: dict[str, Any] = {name: value}
                with self.assertRaisesRegex(
                    AssertionError, f"{name} needs a type hint"
                ):
                    FullLikeArgs(node, 1, **kwargs)

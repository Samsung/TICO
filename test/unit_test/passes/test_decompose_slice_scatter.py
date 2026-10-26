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
from tico.passes.decompose_slice_scatter import DecomposeSliceScatter
from tico.passes.fill_meta_val import FillMetaVal

# start, end, source width, and the expected first output row.
_CASES = (
    (None, None, 5, [10, 11, 12, 13, 14]),
    (-2, None, 2, [0, 1, 2, 10, 11]),
    (-4, -2, 2, [0, 10, 11, 3, 4]),
    (-20, 2, 2, [10, 11, 2, 3, 4]),
    (2, 20, 3, [0, 1, 10, 11, 12]),
    (-20, 20, 5, [10, 11, 12, 13, 14]),
    (20, None, 0, [0, 1, 2, 3, 4]),
    (None, -20, 0, [0, 1, 2, 3, 4]),
    (4, 2, 0, [0, 1, 2, 3, 4]),
    (2, 2, 0, [0, 1, 2, 3, 4]),
    (1, 3, 2, [0, 10, 11, 3, 4]),
    (None, 3, 3, [10, 11, 12, 3, 4]),
)


class _SliceScatter(torch.nn.Module):
    def __init__(self, start, end, dim=1, step=1, reuse=False):
        super().__init__()
        self.start = start
        self.end = end
        self.dim = dim
        self.step = step
        self.reuse = reuse

    def forward(self, x, src):
        result = torch.slice_scatter(
            x, src, dim=self.dim, start=self.start, end=self.end, step=self.step
        )
        if self.reuse:
            return result, x + 1, src + 1
        return result


def _inputs(width, dtype=torch.float32, rows=2):
    offsets = torch.arange(rows, dtype=dtype).unsqueeze(1) * 100
    return offsets + torch.arange(5, dtype=dtype), offsets + torch.arange(
        10, 10 + width, dtype=dtype
    )


class DecomposeSliceScatterTest(unittest.TestCase):
    def _assert_equal(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(torch.equal(actual, expected))

    def _lower(self, model, inputs, dynamic_shapes=None):
        ep = torch.export.export(model.eval(), inputs, dynamic_shapes=dynamic_shapes)
        scatter = next(
            n
            for n in ep.graph.nodes
            if n.target == torch.ops.aten.slice_scatter.default
        )
        metadata = scatter.meta["val"]
        signature = tuple(ep.graph_signature.input_specs)
        constraints = dict(ep.range_constraints)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(DecomposeSliceScatter().call(ep).modified)
        self.assertFalse(
            any(
                n.target == torch.ops.aten.slice_scatter.default for n in ep.graph.nodes
            )
        )
        cat = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.cat.default)
        self.assertIs(cat.meta["val"], metadata)
        self.assertEqual(tuple(ep.graph_signature.input_specs), signature)
        self.assertEqual(ep.range_constraints, constraints)
        ep.graph.lint()
        graph = str(ep.graph)
        self.assertFalse(DecomposeSliceScatter().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        return ep

    def test_bounds_preserve_values_shape_dtype_and_metadata(self):
        for dtype in (torch.float32, torch.int32):
            for start, end, width, expected_row in _CASES:
                with self.subTest(dtype=dtype, start=start, end=end):
                    inputs = _inputs(width, dtype)
                    ep = self._lower(_SliceScatter(start, end), inputs)
                    FillMetaVal().call(ep)
                    expected = torch.tensor(expected_row, dtype=dtype).repeat(2, 1)
                    expected[1] += 100
                    actual = ep.module()(*inputs)
                    self._assert_equal(actual, expected)
                    for node in ep.graph.nodes:
                        if node.target == torch.ops.aten.slice_copy.Tensor:
                            self.assertGreater(node.meta["val"].shape[1], 0)
                            self.assertEqual(node.meta["val"].dtype, dtype)

    def test_negative_dimension_and_keyword_bounds(self):
        inputs = _inputs(2)
        model = _SliceScatter(-4, -2, dim=-1)
        ep = torch.export.export(model, inputs)
        scatter = next(
            n
            for n in ep.graph.nodes
            if n.target == torch.ops.aten.slice_scatter.default
        )
        scatter.args = scatter.args[:2]
        scatter.kwargs = {"dim": -1, "start": -4, "end": -2, "step": 1}
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(DecomposeSliceScatter().call(ep).modified)
        ep.graph.lint()
        self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_other_users_and_input_ownership_are_preserved(self):
        inputs = _inputs(2)
        originals = tuple(value.clone() for value in inputs)
        model = _SliceScatter(-4, -2, reuse=True)
        ep = self._lower(model, inputs)
        outputs = ep.module()(*inputs)
        for actual, expected in zip(outputs, model(*inputs)):
            self._assert_equal(actual, expected)
        outputs[0].fill_(-1)
        for value, original in zip(inputs, originals):
            self._assert_equal(value, original)

    def test_dynamic_non_scattered_dimension_remains_symbolic(self):
        rows = torch.export.Dim("rows", min=2, max=6)
        model = _SliceScatter(-4, -2)
        ep = self._lower(model, _inputs(2, rows=3), ({0: rows}, {0: rows}))
        cat = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.cat.default)
        self.assertIsInstance(cat.meta["val"].shape[0], torch.SymInt)
        for count in (2, 5):
            inputs = _inputs(2, rows=count)
            self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_full_replacement_of_dynamic_axis_remains_symbolic(self):
        size = torch.export.Dim("size", min=2, max=8)
        model = _SliceScatter(None, None, dim=0)
        inputs = (torch.arange(4.0), torch.arange(10.0, 14.0))
        ep = self._lower(model, inputs, ({0: size}, {0: size}))
        cat = next(n for n in ep.graph.nodes if n.target == torch.ops.aten.cat.default)
        self.assertIsInstance(cat.meta["val"].shape[0], torch.SymInt)
        self.assertFalse(
            any(n.target == torch.ops.aten.slice_copy.Tensor for n in ep.graph.nodes)
        )
        for count in (3, 6):
            inputs = (torch.arange(float(count)), torch.arange(10.0, 10.0 + count))
            self._assert_equal(ep.module()(*inputs), model(*inputs))

    def test_default_conversion_reference_parity_and_input_nonmutation(self):
        for start, end, width, _ in _CASES:
            with self.subTest(start=start, end=end):
                model = _SliceScatter(start, end, dim=-1).eval()
                inputs = _inputs(width)
                originals = tuple(value.clone() for value in inputs)
                circle = tico.convert(model, inputs)
                self.assertEqual(circle.runtime, "reference")
                for _ in range(2):
                    actual = torch.from_numpy(circle(*inputs))
                    self._assert_equal(actual, model(*inputs))
                    for value, original in zip(inputs, originals):
                        self._assert_equal(value, original)

    def test_step_greater_than_one_still_fails_without_rewriting(self):
        ep = torch.export.export(_SliceScatter(None, None, step=2), _inputs(3))
        graph = str(ep.graph)
        with self.assertRaisesRegex(RuntimeError, "step > 1 is not yet supported"):
            DecomposeSliceScatter().call(ep)
        self.assertEqual(str(ep.graph), graph)

    def test_unrelated_slice_is_unchanged(self):
        class Slice(torch.nn.Module):
            def forward(self, x):
                return x[:, -4:-2]

        ep = torch.export.export(Slice(), (_inputs(2)[0],))
        graph = str(ep.graph)
        self.assertFalse(DecomposeSliceScatter().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)

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

import operator
import unittest

import tico
import torch

from tico.passes.decompose_group_norm import DecomposeGroupNorm
from tico.passes.fill_meta_val import FillMetaVal
from tico.utils.convert import traced_run_decompositions
from tico.utils.errors import NotYetSupportedError

_NORM_TARGETS = (
    torch.ops.aten.native_layer_norm.default,
    torch.ops.aten.native_group_norm.default,
)


class _NativeNorm(torch.nn.Module):
    def __init__(self, kind, indices=(0,), groups=1, reuse=False):
        super().__init__()
        self.kind = kind
        self.indices = indices
        self.groups = groups
        self.reuse = reuse

    def forward(self, value):
        if self.kind == "layer":
            result = torch.ops.aten.native_layer_norm.default(
                value, [value.shape[-1]], None, None, 1e-5
            )
        else:
            result = torch.ops.aten.native_group_norm.default(
                value, None, None, 1, 2, value.shape[-1], self.groups, 1e-5
            )
        if self.reuse:
            return result[0] + 1, result[0] * 2
        if len(self.indices) == 1:
            return result[self.indices[0]]
        return tuple(result[index] for index in self.indices)


class _TwoNorms(torch.nn.Module):
    def forward(self, value):
        normalized = torch.ops.aten.native_layer_norm.default(
            value, [3], None, None, 1e-5
        )[0]
        mean = torch.ops.aten.native_group_norm.default(
            value, None, None, 1, 2, 3, 1, 1e-5
        )[1]
        return normalized, mean


class _UnusedNorm(torch.nn.Module):
    def forward(self, value):
        torch.ops.aten.native_layer_norm.default(value, [3], None, None, 1e-5)
        return value + 1


class DecomposeGroupNormTest(unittest.TestCase):
    def _input(self, last_dim=3):
        if last_dim == 1:
            return torch.tensor([[[1.0], [4.0]]])
        return torch.tensor([[[1.0, 2.0, 4.0], [-1.0, 3.0, 5.0]]])

    def _assert_close(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        # The existing decomposition changes floating-point evaluation order.
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def _lower(self, ep):
        signature = tuple(ep.graph_signature.input_specs)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(DecomposeGroupNorm().call(ep).modified)
        self.assertFalse(any(node.target in _NORM_TARGETS for node in ep.graph.nodes))
        self.assertFalse(
            any(node.target == operator.getitem for node in ep.graph.nodes)
        )
        self.assertEqual(tuple(ep.graph_signature.input_specs), signature)
        FillMetaVal().call(ep)
        ep.graph.lint()
        graph = str(ep.graph)
        self.assertFalse(DecomposeGroupNorm().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)

    def _assert_rejected_without_mutation(self, ep):
        nodes = tuple(ep.graph.nodes)
        metadata = {node: dict(node.meta) for node in nodes}
        graph = str(ep.graph)
        code = ep.graph_module.code
        signature = repr(ep.graph_signature)
        constants = dict(ep.constants)
        with self.assertRaisesRegex(NotYetSupportedError, "only normalized output"):
            DecomposeGroupNorm().call(ep)
        self.assertEqual(tuple(ep.graph.nodes), nodes)
        self.assertEqual(str(ep.graph), graph)
        self.assertEqual(ep.graph_module.code, code)
        self.assertEqual(repr(ep.graph_signature), signature)
        self.assertEqual(ep.constants.keys(), constants.keys())
        for key, value in constants.items():
            self.assertIs(ep.constants[key], value)
        for node in nodes:
            self.assertEqual(node.meta.keys(), metadata[node].keys())
            for key, value in metadata[node].items():
                self.assertIs(node.meta[key], value)

    def test_normalized_output_preserves_values_metadata_and_signature(self):
        value = self._input()
        for kind, groups in (("layer", 1), ("group", 1), ("group", 2)):
            for index in (0, -3):
                with self.subTest(kind=kind, groups=groups, index=index):
                    model = _NativeNorm(kind, (index,), groups).eval()
                    ep = torch.export.export(model, (value,))
                    output = next(
                        node for node in ep.graph.nodes if node.op == "output"
                    )
                    original = output.args[0][0]
                    original_value = original.meta["val"]
                    self._lower(ep)
                    self.assertIs(output.args[0][0].meta["val"], original_value)
                    self._assert_close(ep.module()(value), model(value))

    def test_dead_statistics_and_duplicate_normalized_users_do_not_select_statistics(
        self,
    ):
        value = self._input()
        for kind in ("layer", "group"):
            with self.subTest(kind=kind):
                model = _NativeNorm(kind, reuse=True).eval()
                ep = torch.export.export(model, (value,))
                norm = next(
                    node for node in ep.graph.nodes if node.target in _NORM_TARGETS
                )
                normalized = next(user for user in norm.users if user.args[1] == 0)
                statistics = next(user for user in norm.users if user.args[1] == 1)
                # Put a dead statistic first in the norm's users without changing semantics.
                normalized.args = ((), 0)
                normalized.args = (norm, 0)
                with ep.graph.inserting_after(statistics):
                    dead = ep.graph.call_function(
                        torch.ops.aten.neg.default, (statistics,)
                    )
                with ep.graph.inserting_after(norm):
                    duplicate = ep.graph.call_function(operator.getitem, (norm, -3))
                duplicate.meta = dict(normalized.meta)
                mul = next(
                    node
                    for node in ep.graph.nodes
                    if node.target == torch.ops.aten.mul.Tensor
                )
                mul.update_arg(0, duplicate)
                ep.graph.lint()
                ep.graph_module.recompile()
                self.assertIn(dead, statistics.users)
                self._lower(ep)
                for actual, expected in zip(ep.module()(value), model(value)):
                    self._assert_close(actual, expected)

    def test_statistics_outputs_are_rejected_before_mutation(self):
        value = self._input()
        for kind in ("layer", "group"):
            for indices in ((1,), (2,), (-2,), (-1,), (0, 1), (2, 0, 1)):
                with self.subTest(kind=kind, indices=indices):
                    model = _NativeNorm(kind, indices).eval()
                    ep = traced_run_decompositions(torch.export.export(model, (value,)))
                    self._assert_rejected_without_mutation(ep)

    def test_later_unsupported_norm_does_not_mutate_earlier_supported_norm(self):
        ep = torch.export.export(_TwoNorms(), (self._input(),))
        self._assert_rejected_without_mutation(ep)

    def test_statistics_used_only_by_impure_node_are_rejected(self):
        value = self._input()
        ep = torch.export.export(_NativeNorm("layer"), (value,))
        norm = next(node for node in ep.graph.nodes if node.target in _NORM_TARGETS)
        statistics = next(user for user in norm.users if user.args[1] == 1)
        output = next(node for node in ep.graph.nodes if node.op == "output")
        with ep.graph.inserting_before(output):
            total = ep.graph.call_function(torch.ops.aten.sum.default, (statistics,))
            positive = ep.graph.call_function(torch.ops.aten.gt.Scalar, (total, 0))
            check = ep.graph.call_function(
                torch.ops.aten._assert_async.msg, (positive, "mean must be positive")
            )
        ep.graph_module.recompile()
        self.assertTrue(check.is_impure())
        self.assertFalse(check.users)
        self._assert_close(ep.module()(value), _NativeNorm("layer")(value))
        self._assert_rejected_without_mutation(ep)

    def test_unused_normalization_is_removed_without_building_a_decomposition(self):
        value = self._input()
        model = _UnusedNorm()
        ep = torch.export.export(model, (value,))
        self.assertTrue(any(node.target in _NORM_TARGETS for node in ep.graph.nodes))
        self._lower(ep)
        self.assertFalse(
            any(node.target == torch.ops.aten.mean.dim for node in ep.graph.nodes)
        )
        self._assert_close(ep.module()(value), model(value))

    def test_direct_tuple_use_is_rejected_before_mutation(self):
        value = self._input()
        for kind in ("layer", "group"):
            with self.subTest(kind=kind):
                ep = torch.export.export(_NativeNorm(kind), (value,))
                norm = next(
                    node for node in ep.graph.nodes if node.target in _NORM_TARGETS
                )
                output = next(node for node in ep.graph.nodes if node.op == "output")
                output.args = ((norm,),)
                ep.graph_module.recompile()
                self._assert_rejected_without_mutation(ep)

    def test_graph_without_normalization_is_unchanged(self):
        ep = torch.export.export(torch.nn.ReLU(), (self._input(),))
        graph = str(ep.graph)
        signature = repr(ep.graph_signature)
        self.assertFalse(DecomposeGroupNorm().call(ep).modified)
        self.assertEqual(str(ep.graph), graph)
        self.assertEqual(repr(ep.graph_signature), signature)

    def test_public_default_conversion_preserves_normalized_output(self):
        for kind in ("layer", "group"):
            value = self._input()
            model = _NativeNorm(kind, reuse=True).eval()
            with self.subTest(kind=kind):
                circle = tico.convert(model, (value,))
                self.assertEqual(circle.runtime, "reference")
                for actual, expected in zip(circle(value), model(value)):
                    self._assert_close(torch.from_numpy(actual), expected)

    def test_public_conversion_rejects_statistics_even_when_output_shapes_match(self):
        for kind in ("layer", "group"):
            for last_dim in (1, 3):
                value = self._input(last_dim)
                for indices in ((1,), (2,), (0, 1)):
                    with self.subTest(kind=kind, last_dim=last_dim, indices=indices):
                        model = _NativeNorm(kind, indices).eval()
                        with self.assertRaisesRegex(
                            NotYetSupportedError, "only normalized output"
                        ):
                            tico.convert(model, (value,))

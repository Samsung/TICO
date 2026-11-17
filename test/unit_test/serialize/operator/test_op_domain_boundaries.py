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

import torch
from circle_schema import circle

from tico.circle.io import model_from_bytes
from tico.serialize.circle_serializer import build_circle_bytes
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError
from tico.utils.model import CircleModel


class _Binary(torch.nn.Module):
    def __init__(self, target, alpha):
        super().__init__()
        self.target = target
        self.alpha = alpha

    def forward(self, x, y):
        return self.target(x, y, alpha=self.alpha)


class _Split(torch.nn.Module):
    def forward(self, x):
        parts = torch.split(x, [32, 32, 31], dim=-1)
        return parts[0], parts[1], parts[2]


class DomainBoundarySerializerTest(unittest.TestCase):
    def test_explicit_unit_alpha_has_native_add_sub_contract(self):
        for target in (torch.add, torch.sub):
            ep = torch.export.export(_Binary(target, 1), (torch.ones(2), torch.ones(2)))
            model = model_from_bytes(build_circle_bytes(ep))
            self.assertEqual(len(model.subgraphs[0].operators), 1)

    def test_nonunit_alpha_is_never_silently_dropped(self):
        for target in (torch.add, torch.sub):
            for alpha in (-2, 0.5):
                ep = torch.export.export(
                    _Binary(target, alpha), (torch.ones(2), torch.ones(2))
                )
                with self.assertRaisesRegex(
                    NotYetSupportedError, "operand other has alpha="
                ):
                    build_circle_bytes(ep)

    def _split(self, dtype):
        ep = torch.export.export(_Split(), (torch.arange(95.0).reshape(1, 95),))
        projections = [n for n in ep.graph.nodes if n.target == operator.getitem]
        # One slot extracted twice, including a negative index. Export usually
        # canonicalizes these; the serializer must handle equivalent FX graphs.
        projections[1].update_arg(1, -3)
        for node in ep.graph.nodes:
            if node.op == "placeholder" or node.target == operator.getitem:
                node.meta[QPARAM_KEY] = (
                    QuantParam(dtype="mxint8", quantized_dimension=1)
                    if dtype == "mxint8"
                    else QuantParam(dtype="int16", scale=[0.125], zero_point=[0])
                )
        return ep, projections

    def test_split_repeated_negative_getitems_alias_one_slot(self):
        for dtype in ("mxint8", "int16"):
            ep, _ = self._split(dtype)
            model = model_from_bytes(build_circle_bytes(ep))
            graph = model.subgraphs[0]
            split = graph.operators[0]
            self.assertEqual(len(split.outputs), 3)
            self.assertEqual(graph.outputs[0], graph.outputs[1])
            self.assertEqual(graph.outputs[0], split.outputs[0])
            self.assertEqual(graph.outputs[2], split.outputs[2])
            for tid in split.outputs:
                tensor = graph.tensors[tid]
                expected = (
                    circle.TensorType.TensorType.MXINT8
                    if dtype == "mxint8"
                    else circle.TensorType.TensorType.INT16
                )
                self.assertEqual(tensor.type, expected)
                if dtype == "mxint8":
                    self.assertEqual(tensor.quantization.quantizedDimension, 1)
                else:
                    self.assertEqual(list(tensor.quantization.scale), [0.125])

    def test_per_channel_split_does_not_inherit_an_unsliced_input_grid(self):
        # Per-channel activation slicing is outside this policy. Preserve the
        # prior serializer treatment instead of copying all 95 scales onto
        # a 32-element unused slot or requiring used slots to keep that vector.
        ep, projections = self._split("int16")
        source = next(node for node in ep.graph.nodes if node.op == "placeholder")
        source.meta[QPARAM_KEY] = QuantParam(
            dtype="int16",
            scale=[0.125] * 95,
            zero_point=[0] * 95,
            quantized_dimension=1,
        )
        for node, count in zip(projections, (32, 32, 31)):
            node.meta[QPARAM_KEY] = QuantParam(
                dtype="int16",
                scale=[0.125] * count,
                zero_point=[0] * count,
                quantized_dimension=1,
            )
        model = model_from_bytes(build_circle_bytes(ep))
        graph = model.subgraphs[0]
        split = graph.operators[0]
        self.assertEqual(len(graph.tensors[split.inputs[0]].quantization.scale), 95)
        self.assertEqual(len(graph.tensors[split.outputs[0]].quantization.scale), 32)
        self.assertIsNone(graph.tensors[split.outputs[1]].quantization)
        self.assertEqual(len(graph.tensors[split.outputs[2]].quantization.scale), 31)

    def test_repeated_negative_split_float_values(self):
        value = torch.arange(95.0).reshape(1, 95)
        ep = torch.export.export(_Split(), (value,))
        projections = [n for n in ep.graph.nodes if n.target == operator.getitem]
        projections[1].update_arg(1, -3)
        actual = CircleModel(build_circle_bytes(ep))(value)
        for result, expected in zip(
            actual, (value[:, :32], value[:, :32], value[:, 64:])
        ):
            torch.testing.assert_close(
                torch.from_numpy(result), expected, rtol=0, atol=0
            )

    def test_repeated_split_output_needs_explicit_conversion(self):
        ep, projections = self._split("int16")
        projections[1].meta[QPARAM_KEY] = QuantParam(
            dtype="int16", scale=[0.5], zero_point=[0]
        )
        with self.assertRaisesRegex(
            NotYetSupportedError, "repeated split output.*different quantization"
        ):
            build_circle_bytes(ep)


if __name__ == "__main__":
    unittest.main()

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

"""Narrow INT16 clipping survives serialization and Circle O1."""

import unittest

import numpy as np
import torch
import torch.ao.quantization.fx._decomposed  # register quantization operators
from circle_schema import circle

from tico.circle import CircleDocument
from tico.circle.export import optimize_for_export
from tico.circle.runtime import ExecutionMode, run_circle
from tico.serialize.circle_serializer import build_circle_bytes
from tico.serialize.quant_param import QPARAM_KEY, QuantParam


class _Quantize(torch.nn.Module):
    def __init__(self, qmin):
        super().__init__()
        self.qmin = qmin

    def forward(self, x):
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, 0.125, 7, self.qmin, 32767, torch.int16
        )


class NarrowQuantizeSerializerTest(unittest.TestCase):
    def _binary(self, qmin, *, preserve_clipping=True, embedding_clamp=False):
        ep = torch.export.export(_Quantize(qmin), (torch.zeros(1, 8),))
        quantize = next(n for n in ep.graph.nodes if n.op == "call_function")
        quantize.meta["_preserve_affine_clipping"] = preserve_clipping
        quantize.args[0].meta["_embedding_range_clamp_"] = embedding_clamp
        quantize.meta[QPARAM_KEY] = QuantParam(
            dtype="int16", scale=[0.125], zero_point=[7]
        )
        return build_circle_bytes(ep)

    def test_narrow_saturation_and_same_grid_maximum_survive_o1(self):
        # No halfway cases: Circle affine rounding is independently specified
        # by its runtime. This test isolates the missing storage-endpoint clamp.
        value = np.array(
            [[-5000, -4097, -4096.9, -4096.7, -1.1, 0, 4090.1, 5000]],
            dtype=np.float32,
        )
        codes = np.clip(np.round(value / 0.125) + 7, -32767, 32767)
        expected = (codes - 7) * np.float32(0.125)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                binary = self._binary(-32767)
                if optimize:
                    binary = optimize_for_export(binary)
                document = CircleDocument.from_bytes(binary)
                document.verify()
                graph = document.model.subgraphs[0]
                names = [
                    document.model.operatorCodes[op.opcodeIndex].builtinCode
                    for op in graph.operators
                ]
                self.assertEqual(
                    names,
                    [
                        circle.BuiltinOperator.BuiltinOperator.QUANTIZE,
                        circle.BuiltinOperator.BuiltinOperator.MAXIMUM,
                    ],
                )
                maximum = graph.operators[1]
                for tid in (*maximum.inputs, *maximum.outputs):
                    tensor = graph.tensors[tid]
                    self.assertEqual(tensor.type, circle.TensorType.TensorType.INT16)
                    self.assertEqual(list(tensor.quantization.scale), [0.125])
                    self.assertEqual(list(tensor.quantization.zeroPoint), [7])
                result = run_circle(binary, [value], mode=ExecutionMode.FAKE_QUANTIZE)
                np.testing.assert_array_equal(result.outputs[0], expected)

    def test_existing_narrow_export_keeps_its_independent_clipping_policy(self):
        document = CircleDocument.from_bytes(
            self._binary(-32767, preserve_clipping=False)
        )
        self.assertEqual(len(document.model.subgraphs[0].operators), 1)

    def test_embedding_preclamp_is_not_duplicated(self):
        document = CircleDocument.from_bytes(self._binary(-32767, embedding_clamp=True))
        self.assertEqual(len(document.model.subgraphs[0].operators), 1)

    def test_full_int16_keeps_one_quantize(self):
        document = CircleDocument.from_bytes(self._binary(-32768))
        self.assertEqual(len(document.model.subgraphs[0].operators), 1)


if __name__ == "__main__":
    unittest.main()

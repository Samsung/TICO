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

"""Reduced INT16 range keeps a float DQ/clamp edge throughout real export."""

import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import tico
import torch
import torch.ao.quantization.fx._decomposed
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.passes import preserve_embedding_quantization as preservation
from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
    PreserveEmbeddingQuantization,
)
from tico.serialize.quant_param import QPARAM_KEY


class _ReducedRangeEmbedding(torch.nn.Module):
    scale: torch.Tensor

    def __init__(self, *, with_mul=False):
        super().__init__()
        self.with_mul = with_mul
        self.register_buffer(
            "weight",
            torch.tensor(
                [[0, 100, 200, 255], [1, 4, 8, 0], [0, 2, 3, 1]],
                dtype=torch.uint8,
            ),
        )
        self.register_buffer("scales", torch.tensor([10.0, 0.25, 1.0]))
        self.register_buffer(
            "zero_points", torch.tensor([100, 4, 2], dtype=torch.int64)
        )
        self.register_buffer("scale", torch.tensor(1.5))

    def forward(self, ids):
        table = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.weight, self.scales, self.zero_points, 0, 0, 255, torch.uint8
        )
        value = torch.nn.functional.embedding(ids, table)
        value = torch.fake_quantize_per_tensor_affine(value, 1.0, 0, -32768, 32767)
        if self.with_mul:
            scale = torch.fake_quantize_per_tensor_affine(self.scale, 0.25, 0, 0, 255)
            value = value * scale
        return torch.fake_quantize_per_tensor_affine(value, 0.01, 0, -32767, 32767)


class _QuantizersFirstSet(set):
    """Force a legal adverse order in the production pass's membership set.

    The legacy pass iterated this set while inserting clamps, allowing Q_B to
    insert its clamp before DQ_A was classified. Only iteration is controlled:
    the real converter, pass implementation, serializer and kernels all execute.
    """

    def __iter__(self):
        values = tuple(super().__iter__())
        quantizers = tuple(
            node
            for node in values
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        others = tuple(node for node in values if node not in quantizers)
        return iter(quantizers + others)


class ExplicitEmbeddingReducedRangeTest(unittest.TestCase):
    def _roundtrip(self, *, with_mul=False, ids=None):
        tokens = torch.tensor([[0, 1, 2]] if ids is None else ids, dtype=torch.int64)
        program = torch.export.export(
            _ReducedRangeEmbedding(with_mul=with_mul).eval(), (tokens,)
        )
        # Capture the real pipeline's final graph: decomposition may return a
        # replacement ExportedProgram rather than mutating the supplied one.
        conversion = importlib.import_module("tico.utils.convert")
        prepare = conversion._prepare_exported_program
        prepared = []

        def capture(*args, **kwargs):
            result = prepare(*args, **kwargs)
            prepared.append(result)
            return result

        # This is the public production path, including quant legalization and O1.
        with patch.object(conversion, "_prepare_exported_program", side_effect=capture):
            converted = tico.convert_from_exported_program(program)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.circle"
            converted.save(path)
            document = CircleDocument.load(path)
            self.assertTrue(document.verify().ok)
            runtime = CircleReferenceRuntime(document)
        return prepared[-1], runtime, tokens.numpy()

    def _assert_float_contract(self, program, runtime, *, with_mul):
        clamps = [
            node
            for node in program.graph.nodes
            if node.meta.get("_embedding_range_clamp_")
        ]
        self.assertEqual(len(clamps), 1)
        clamp = clamps[0]
        self.assertEqual(clamp.target, torch.ops.aten.clamp.default)
        for node in (clamp, clamp.args[0]):
            # _prepare_exported_program has already run LegalizeQuantizedClamp.
            self.assertTrue(node.meta[FLOAT_EMBEDDING])
            self.assertNotIn(QPARAM_KEY, node.meta)
        dq = clamp.args[0].args[0] if with_mul else clamp.args[0]
        self.assertEqual(
            dq.target, torch.ops.quantized_decomposed.dequantize_per_tensor.default
        )
        self.assertTrue(dq.meta[EXPLICIT_EMBEDDING])
        self.assertTrue(dq.meta[FLOAT_EMBEDDING])
        self.assertNotIn(QPARAM_KEY, dq.meta)
        self.assertEqual(dq.args[0].meta[QPARAM_KEY].scale, [1.0])
        maximum = next(op for op in runtime.program.operators if op.name == "MAXIMUM")
        for index in (maximum.inputs[0], maximum.outputs[0]):
            tensor = runtime.program.tensor(index)
            self.assertEqual(tensor.dtype, np.dtype(np.float32))
            self.assertFalse(tensor.is_quantized)
        output = runtime.output_tensors[0]
        self.assertEqual(output.dtype, np.dtype(np.int16))
        self.assertEqual(output.quantization.scale, (float(np.float32(0.01)),))
        self.assertEqual(output.quantization.zero_point, (0,))
        self.assertTrue(runtime.rowwise_gather_plans)

    def _assert_native(self, runtime, tokens, expected):
        result = runtime.run([tokens], mode=ExecutionMode.NATIVE).outputs[0]
        self.assertEqual(result.dtype, np.int16)
        np.testing.assert_array_equal(result, np.asarray(expected, dtype=np.int16))
        full_dq = CircleReferenceRuntime(runtime.program, enable_rowwise_gather=False)
        np.testing.assert_array_equal(
            result, full_dq.run([tokens], mode=ExecutionMode.NATIVE).outputs[0]
        )

    def test_no_mul_reduced_range_survives_full_export_and_roundtrip(self):
        program, runtime, tokens = self._roundtrip()
        self._assert_float_contract(program, runtime, with_mul=False)
        # row 0 starts at -1000: Q_A(-1000, scale=1) = -1000, then
        # Q_B(-1000, scale=.01, min=-32767) must saturate to -32767.
        self._assert_native(
            runtime,
            tokens,
            [[[-32767, 0, 32767, 32767], [-100, 0, 100, -100], [-200, 0, 100, -100]]],
        )

    def test_existing_mul_continuation_and_clipping_free_values(self):
        program, runtime, tokens = self._roundtrip(with_mul=True)
        self._assert_float_contract(program, runtime, with_mul=True)
        self._assert_native(
            runtime,
            tokens,
            [[[-32767, 0, 32767, 32767], [-150, 0, 150, -150], [-300, 0, 150, -150]]],
        )

    def test_positive_clipping_free_and_repeated_decode(self):
        program, runtime, tokens = self._roundtrip(ids=[[2]])
        self._assert_float_contract(program, runtime, with_mul=False)
        for _ in range(2):
            self._assert_native(runtime, tokens, [[[-200, 0, 100, -100]]])

    def test_adverse_membership_iteration_uses_real_production_pipeline(self):
        with patch.object(preservation, "set", _QuantizersFirstSet, create=True):
            program, runtime, tokens = self._roundtrip(ids=[[0]])
        self._assert_native(runtime, tokens, [[[-32767, 0, 32767, 32767]]])
        self._assert_float_contract(program, runtime, with_mul=False)

    def test_repeated_preservation_keeps_generated_clamp_and_dq_float(self):
        preserve = PreserveEmbeddingQuantization.call

        def repeat(actual_pass, program):
            # Stress repeated invocations at their real production-pipeline
            # position, without replacing the implementation under test.
            for _ in range(3):
                result = preserve(actual_pass, program)
                clamps = [
                    node
                    for node in program.graph.nodes
                    if node.meta.get("_embedding_range_clamp_")
                ]
                self.assertEqual(len(clamps), 1)
                clamp = clamps[0]
                dq = clamp.args[0]
                self.assertEqual(
                    dq.target,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                )
                self.assertTrue(dq.meta[FLOAT_EMBEDDING])
                self.assertNotIn(QPARAM_KEY, clamp.meta)
                self.assertNotIn(QPARAM_KEY, dq.meta)
            return result

        with patch.object(
            PreserveEmbeddingQuantization, "call", autospec=True, side_effect=repeat
        ):
            program, runtime, tokens = self._roundtrip(ids=[[0]])
        self._assert_float_contract(program, runtime, with_mul=False)
        self._assert_native(runtime, tokens, [[[-32767, 0, 32767, 32767]]])


if __name__ == "__main__":
    unittest.main()

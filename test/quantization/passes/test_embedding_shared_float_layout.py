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

"""A shared float side input keeps independent quantizers behind float layouts."""

import copy
import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
    PreserveEmbeddingQuantization,
)
from tico.serialize.quant_param import QPARAM_KEY, QuantParam

from test.quantization.passes.test_embedding_float_continuation import (
    _lookup_float,
    _q,
    _Table,
)


class _SharedFloatLayout(_Table):
    def __init__(self, layout, quantized_first):
        super().__init__()
        self.layout = layout
        self.quantized_first = quantized_first

    def independent_input(self, side):
        if self.layout == "direct":
            return side
        if self.layout in ("reshape", "shared"):
            return side.reshape(1, 4)
        if self.layout == "permute":
            return side.permute(0, 2, 1)
        return side.reshape(2, 2).permute(1, 0).reshape(1, 2, 2)

    def quantize_independent(self, value):
        value = torch.fake_quantize_per_tensor_affine(value, 0.25, 0, -32768, 32767)
        # Restore source element order after the independent quantizer. Its
        # scalar grid commutes with these integer selection operations.
        if self.layout in ("permute", "chain"):
            value = value.permute(0, 2, 1)
        return value.reshape(1, 4)

    def combine(self, value, side):
        if self.layout == "shared":
            side = side.reshape(1, 2, 2)
        return torch.fake_quantize_per_tensor_affine(
            torch.cat((value, side.permute(0, 2, 1)), dim=2),
            0.125,
            0,
            -32768,
            32767,
        )

    def forward(self, ids, side):
        value = self.lookup(ids)
        value = torch.ops.aten.slice_copy.Tensor(value, 2, 0, 2)
        if self.layout == "shared":
            side = self.independent_input(side)
        if self.quantized_first:
            independent = (
                side if self.layout == "shared" else self.independent_input(side)
            )
            separate = self.quantize_independent(independent)
            combined = self.combine(value, side)
        else:
            combined = self.combine(value, side)
            independent = (
                side if self.layout == "shared" else self.independent_input(side)
            )
            separate = self.quantize_independent(independent)
        return combined, separate


class EmbeddingSharedFloatLayoutTest(unittest.TestCase):
    def _roundtrip(self, layout, quantized_first):
        module = _SharedFloatLayout(layout, quantized_first).eval()
        ids = torch.tensor([[3, 0]], dtype=torch.int64)
        side = torch.tensor([[[0.0625, -0.1875], [0.3125, -0.4375]]])
        inputs = (ids, side)
        program = torch.export.export(module, inputs)
        conversion = importlib.import_module("tico.utils.convert")
        prepare = conversion._prepare_exported_program
        prepared = []

        def capture(*args, **kwargs):
            result = prepare(*args, **kwargs)
            prepared.append(result)
            return result

        # Observe, rather than substitute, the real pipeline. The returned
        # ExportedProgram can differ from the program supplied by the caller.
        with patch.object(conversion, "_prepare_exported_program", side_effect=capture):
            converted = tico.convert_from_exported_program(
                program,
                config=tico.CompileConfigV1(preserve_embedding_quantization=True),
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared-layout.circle"
            converted.save(path)
            document = CircleDocument.load(path)
            self.assertTrue(document.verify().ok)
            runtime = CircleReferenceRuntime(document)
        return module, prepared[-1], runtime, ids.numpy(), side.numpy()

    def _assert_float_and_quantizer_contracts(self, program, runtime):
        side = next(
            node
            for node in program.graph.nodes
            if node.op == "placeholder"
            and node.name in program.graph_signature.user_inputs
            and node.meta["val"].dtype == torch.float32
        )
        self.assertTrue(side.meta.get(FLOAT_EMBEDDING))
        self.assertNotIn(QPARAM_KEY, side.meta)
        self.assertEqual(runtime.input_tensors[1].dtype, np.dtype(np.float32))
        self.assertFalse(runtime.input_tensors[1].is_quantized)

        quantizers = [
            node
            for node in program.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        ]
        separate_quantizers = [
            node
            for node in quantizers
            if node.args[1] == 0.25
            and node.args[0].target != torch.ops.aten.embedding.default
        ]
        self.assertEqual(
            len(separate_quantizers),
            1,
            "The float side branch must retain its independent .25 Quantize",
        )
        separate = separate_quantizers[0]
        self.assertTrue(separate.meta.get(EXPLICIT_EMBEDDING))
        self.assertEqual(separate.args[1:], (0.25, 0, -32768, 32767, torch.int16))
        self.assertTrue(separate.args[0].meta.get(FLOAT_EMBEDDING))
        self.assertNotIn(QPARAM_KEY, separate.args[0].meta)
        qparam = separate.meta[QPARAM_KEY]
        self.assertEqual(
            qparam, QuantParam(scale=[0.25], zero_point=[0], dtype="int16")
        )
        # Every layout before this Q still represents the genuine float
        # source, including layouts used only by the independent branch.
        pending = [separate.args[0]]
        while pending:
            node = pending.pop()
            self.assertTrue(node.meta.get(FLOAT_EMBEDDING), node.name)
            self.assertEqual(node.meta["val"].dtype, torch.float32)
            self.assertNotIn(QPARAM_KEY, node.meta)
            pending.extend(node.all_input_nodes)
        final = next(node for node in quantizers if node.args[1] == 0.125)
        self.assertEqual(final.args[1:], (0.125, 0, -32768, 32767, torch.int16))
        self.assertEqual(
            final.meta[QPARAM_KEY],
            QuantParam(scale=[0.125], zero_point=[0], dtype="int16"),
        )
        for node in program.graph.nodes:
            if node.meta.get(FLOAT_EMBEDDING):
                self.assertEqual(node.meta["val"].dtype, torch.float32)
                self.assertNotIn(QPARAM_KEY, node.meta)

        # A layout changes shape/order, never the represented numeric domain.
        for operator in runtime.program.operators:
            if operator.name not in ("RESHAPE", "TRANSPOSE", "SLICE", "STRIDED_SLICE"):
                continue
            source = runtime.program.tensor(operator.inputs[0])
            target = runtime.program.tensor(operator.outputs[0])
            self.assertEqual(source.dtype, target.dtype, operator.name)
            self.assertEqual(source.quantization, target.quantization, operator.name)
        concat = next(
            op for op in runtime.program.operators if op.name == "CONCATENATION"
        )
        for index in (*concat.inputs, *concat.outputs):
            tensor = runtime.program.tensor(index)
            self.assertEqual(tensor.dtype, np.dtype(np.float32))
            self.assertFalse(tensor.is_quantized)
        for tensor, scale in zip(runtime.output_tensors, (0.125, 0.25)):
            self.assertEqual(tensor.dtype, np.dtype(np.int16))
            self.assertEqual(tensor.quantization.scale, (scale,))
            self.assertEqual(tensor.quantization.zero_point, (0,))

    def _assert_native_oracle(self, module, runtime, ids, side):
        # This oracle reads the original UINT8 codes and per-row qparams,
        # independently applies Q_A rounding, and keeps the side branch FP32.
        combined = np.concatenate(
            (_lookup_float(module, ids)[..., :2], side.transpose(0, 2, 1)), axis=2
        )
        expected = (_q(combined, 0.125), np.array([[0, -1, 1, -2]], np.int16))
        self.assertTrue(runtime.rowwise_gather_plans)
        actual = runtime.run((ids, side), mode=ExecutionMode.NATIVE).outputs
        for result, oracle in zip(actual, expected):
            np.testing.assert_array_equal(result, oracle)
        full_dq = CircleReferenceRuntime(runtime.program, enable_rowwise_gather=False)
        for result, oracle in zip(
            full_dq.run((ids, side), mode=ExecutionMode.NATIVE).outputs, expected
        ):
            np.testing.assert_array_equal(result, oracle)

    def test_shared_float_layouts_preserve_independent_quantizer_in_both_orders(self):
        for layout in ("direct", "reshape", "permute", "chain", "shared"):
            for quantized_first in (False, True):
                with self.subTest(layout=layout, quantized_first=quantized_first):
                    module, program, runtime, ids, side = self._roundtrip(
                        layout, quantized_first
                    )
                    self._assert_float_and_quantizer_contracts(program, runtime)
                    self._assert_native_oracle(module, runtime, ids, side)

    def test_repeated_policy_at_production_position_is_idempotent(self):
        preserve = PreserveEmbeddingQuantization.call

        def repeat(actual_pass, program):
            result = preserve(actual_pass, program)
            graph = str(program.graph)
            metadata = {
                node.name: copy.deepcopy(
                    (
                        node.meta.get(EXPLICIT_EMBEDDING),
                        node.meta.get(FLOAT_EMBEDDING),
                        node.meta.get(QPARAM_KEY),
                    )
                )
                for node in program.graph.nodes
            }
            for _ in range(2):
                result = preserve(actual_pass, program)
                self.assertEqual(str(program.graph), graph)
                self.assertEqual(
                    {
                        node.name: (
                            node.meta.get(EXPLICIT_EMBEDDING),
                            node.meta.get(FLOAT_EMBEDDING),
                            node.meta.get(QPARAM_KEY),
                        )
                        for node in program.graph.nodes
                    },
                    metadata,
                )
            return result

        with patch.object(
            PreserveEmbeddingQuantization, "call", autospec=True, side_effect=repeat
        ):
            module, program, runtime, ids, side = self._roundtrip("chain", True)
        self._assert_float_and_quantizer_contracts(program, runtime)
        self._assert_native_oracle(module, runtime, ids, side)

    def test_policy_after_real_folding_does_not_accumulate_layouts_or_quantizers(self):
        for layout in ("direct", "reshape", "permute", "chain", "shared"):
            for quantized_first in (False, True):
                with self.subTest(layout=layout, quantized_first=quantized_first):
                    module = _SharedFloatLayout(layout, quantized_first).eval()
                    inputs = (
                        torch.tensor([[3, 0]], dtype=torch.int64),
                        torch.tensor([[[0.0625, -0.1875], [0.3125, -0.4375]]]),
                    )
                    program = torch.export.export(module, inputs)
                    DecomposeFakeQuantize().call(program)
                    policy = PreserveEmbeddingQuantization()
                    policy.call(program)
                    FoldQuantOps().call(program)
                    after_fold = str(program.graph)
                    quantizer_count = sum(
                        node.target
                        == torch.ops.quantized_decomposed.quantize_per_tensor.default
                        for node in program.graph.nodes
                    )
                    self.assertEqual(quantizer_count, 3)
                    layout_count = sum(
                        node.target
                        in (
                            torch.ops.aten.reshape.default,
                            torch.ops.aten.permute.default,
                            torch.ops.aten.slice_copy.Tensor,
                        )
                        for node in program.graph.nodes
                    )
                    for _ in range(2):
                        policy.call(program)
                        self.assertEqual(str(program.graph), after_fold)
                        self.assertEqual(
                            sum(
                                node.target
                                == torch.ops.quantized_decomposed.quantize_per_tensor.default
                                for node in program.graph.nodes
                            ),
                            quantizer_count,
                        )
                        self.assertEqual(
                            sum(
                                node.target
                                in (
                                    torch.ops.aten.reshape.default,
                                    torch.ops.aten.permute.default,
                                    torch.ops.aten.slice_copy.Tensor,
                                )
                                for node in program.graph.nodes
                            ),
                            layout_count,
                        )
                        for node in program.graph.nodes:
                            if node.meta.get(FLOAT_EMBEDDING):
                                self.assertNotIn(QPARAM_KEY, node.meta, node.name)

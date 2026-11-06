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

"""Public compile configuration selects embedding export contracts explicitly."""

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import tico
import torch
import torch.ao.quantization.fx._decomposed
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.config import CompileConfigBase
from tico.pt2_to_circle import main
from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
)
from tico.utils.convert import _prepare_exported_program


class _Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "table", torch.tensor([[0, 3], [8, 15]], dtype=torch.uint8)
        )
        self.register_buffer("scales", torch.tensor([0.125, 0.25]))
        self.register_buffer("zero", torch.tensor([1, 4], dtype=torch.int32))

    def forward(self, left, right):
        weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.table, self.scales, self.zero, 0, 0, 255, torch.uint8
        )
        first = torch.nn.functional.embedding(left, weight)
        second = torch.nn.functional.embedding(right, weight)
        return (
            torch.fake_quantize_per_tensor_affine(first, 0.25, 120, 0, 255),
            torch.fake_quantize_per_tensor_affine(second, 0.5, 100, 0, 255),
        )


class _Relu(torch.nn.Module):
    def forward(self, value):
        value = torch.fake_quantize_per_tensor_affine(value, 0.25, 0, -32768, 32767)
        return torch.fake_quantize_per_tensor_affine(
            torch.relu(value), 0.5, 0, -32768, 32767
        )


def _contract(program):
    """Compare graph semantics and tensor contracts, independently of byte packing."""
    return (
        program.inputs,
        program.outputs,
        tuple((op.name, op.inputs, op.outputs) for op in program.operators),
        tuple(
            (
                tensor.dtype.str,
                tensor.shape,
                tensor.shape_signature,
                tensor.quantization,
                tensor.constant.tobytes() if tensor.is_constant else None,
            )
            for tensor in program.tensors
        ),
    )


class EmbeddingCompileConfigTest(unittest.TestCase):
    def setUp(self):
        self.inputs = (torch.tensor([[0, 1]]), torch.tensor([[1, 0]]))

    def _convert(self, api, options):
        # Every comparison starts with a new module and ExportedProgram. In
        # particular, the legacy case never reuses an opt-in graph or its metadata.
        module = _Embedding().eval()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "embedding.circle"
            args: tuple[Any, ...]
            if api in ("convert", "convert_to_file"):
                args = (module, self.inputs)
            else:
                exported = torch.export.export(module, self.inputs)
                if "pt2" in api:
                    source = Path(directory) / "embedding.pt2"
                    torch.export.save(exported, source)
                    args = (source,)
                else:
                    args = (exported,)
            if api.endswith("to_file"):
                getattr(tico, api)(*args, output, **options)
            else:
                getattr(tico, api)(*args, **options).save(output)
            document = CircleDocument.load(output)
        self.assertTrue(document.verify().ok)
        return CircleReferenceRuntime(document)

    def _assert_embedding_contract(self, runtime, enabled):
        program = runtime.program
        self.assertEqual(len(program.inputs), 2)
        self.assertEqual(len(program.outputs), 2)
        self.assertEqual(
            Counter(op.name for op in program.operators),
            (
                Counter({"DEQUANTIZE": 1, "GATHER": 2, "QUANTIZE": 2})
                if enabled
                else Counter({"GATHER": 2})
            ),
        )
        gathers = [op for op in program.operators if op.name == "GATHER"]
        self.assertEqual(gathers[0].inputs[0], gathers[1].inputs[0])
        self.assertEqual(tuple(op.inputs[1] for op in gathers), program.inputs)
        table_index = gathers[0].inputs[0]
        if enabled:
            dq = next(op for op in program.operators if op.name == "DEQUANTIZE")
            self.assertEqual(dq.outputs, (table_index,))
            table_index = dq.inputs[0]
            for gather in gathers:
                value = program.tensor(gather.outputs[0])
                self.assertEqual(value.dtype, np.dtype("float32"))
                self.assertIsNone(value.quantization)
        table = program.tensor(table_index)
        self.assertEqual(table.dtype, np.dtype("uint8"))
        self.assertEqual(table.shape, (2, 2))
        np.testing.assert_array_equal(table.constant, [[0, 3], [8, 15]])
        self.assertEqual(table.quantization.scale, (0.125, 0.25))
        self.assertEqual(table.quantization.zero_point, (1, 4))
        self.assertEqual(table.quantization.quantized_dimension, 0)
        for index in program.inputs:
            value = program.tensor(index)
            self.assertEqual(value.dtype, np.dtype("int64"))
            self.assertEqual(value.shape, (1, 2))
            self.assertIsNone(value.quantization)
        for index, scale, zero in zip(program.outputs, (0.25, 0.5), (120, 100)):
            value = program.tensor(index)
            self.assertEqual(value.dtype, np.dtype("uint8"))
            self.assertEqual(value.shape, (1, 2, 2))
            self.assertEqual(value.quantization.scale, (scale,))
            self.assertEqual(value.quantization.zero_point, (zero,))
            self.assertEqual(value.quantization.quantized_dimension, 0)
        if enabled:
            outputs = runtime.run(
                tuple(value.numpy() for value in self.inputs),
                mode=ExecutionMode.NATIVE,
            ).outputs
            # Independent fixed-grid oracle: row DQ then original activation Q,
            # with the Circle runtime's halfway-away-from-zero rounding.
            np.testing.assert_array_equal(outputs[0], [[[119, 121], [124, 131]]])
            np.testing.assert_array_equal(outputs[1], [[[102, 106], [100, 101]]])

    def test_default_dictionary_and_set_roundtrip(self):
        default = tico.get_default_config()
        self.assertIs(default.get("preserve_embedding_quantization"), False)
        self.assertIs(default.to_dict()["preserve_embedding_quantization"], False)
        for value in (False, True):
            config = tico.CompileConfigV1.from_dict(
                {"version": "1.0", "preserve_embedding_quantization": value}
            )
            self.assertIs(config.get("preserve_embedding_quantization"), value)
            self.assertIs(config.to_dict()["preserve_embedding_quantization"], value)
            config.set("preserve_embedding_quantization", not value)
            self.assertIs(config.get("preserve_embedding_quantization"), not value)
        self.assertIs(
            tico.CompileConfigV1.from_dict({"version": "1.0"}).get(
                "preserve_embedding_quantization"
            ),
            False,
        )

    def test_omitted_false_and_missing_key_keep_legacy_contract(self):
        # Fixed pre-feature contract from
        # f8e9b036cbd91cc030ae5e8e712d6a9702cf2e87: shared quantized table,
        # two quantized GATHER outputs, no serialized Q/DQ boundaries.
        expected = None
        for options in (
            {},
            {"config": tico.CompileConfigV1(preserve_embedding_quantization=False)},
            {"config": CompileConfigBase()},
            {"config": tico.CompileConfigV1.from_dict({"version": "1.0"})},
        ):
            with self.subTest(options=options):
                runtime = self._convert("convert_from_exported_program", options)
                self._assert_embedding_contract(runtime, False)
                contract = _contract(runtime.program)
                if expected is None:
                    expected = contract
                self.assertEqual(contract, expected)
        prepared = _prepare_exported_program(
            torch.export.export(_Embedding().eval(), self.inputs), CompileConfigBase()
        )
        for node in prepared.graph.nodes:
            self.assertNotIn(EXPLICIT_EMBEDDING, node.meta)
            self.assertNotIn(FLOAT_EMBEDDING, node.meta)

    def test_public_api_config_reaches_real_circle_exports(self):
        for enabled in (False, True):
            expected = None
            for api in (
                "convert",
                "convert_from_exported_program",
                "convert_from_pt2",
                "convert_to_file",
                "convert_from_exported_program_to_file",
                "convert_from_pt2_to_file",
            ):
                with self.subTest(api=api, enabled=enabled):
                    runtime = self._convert(
                        api,
                        {
                            "config": tico.CompileConfigV1.from_dict(
                                {"preserve_embedding_quantization": enabled}
                            )
                        },
                    )
                    self._assert_embedding_contract(runtime, enabled)
                    contract = _contract(runtime.program)
                    if expected is None:
                        expected = contract
                    self.assertEqual(contract, expected)

    def test_cli_yaml_selects_policy_and_omitted_key_keeps_default(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "embedding.pt2"
            destination = Path(directory) / "embedding.circle"
            config = Path(directory) / "compile.yaml"
            torch.export.save(
                torch.export.export(_Embedding().eval(), self.inputs), source
            )
            for value in (None, False, True):
                with self.subTest(value=value):
                    config.write_text(
                        'version: "1.0"\n'
                        + (
                            ""
                            if value is None
                            else f"preserve_embedding_quantization: {str(value).lower()}\n"
                        )
                    )
                    with patch.object(
                        sys,
                        "argv",
                        [
                            "pt2-to-circle",
                            "-i",
                            str(source),
                            "-o",
                            str(destination),
                            "--config",
                            str(config),
                        ],
                    ):
                        main()
                    document = CircleDocument.load(destination)
                    self.assertTrue(document.verify().ok)
                    self._assert_embedding_contract(
                        CircleReferenceRuntime(document), value is True
                    )

    def test_no_embedding_has_same_graph_with_option_on_and_off(self):
        contracts = []
        for enabled in (False, True):
            converted = tico.convert(
                _Relu().eval(),
                (torch.tensor([[-1.0, 2.0]]),),
                config=tico.CompileConfigV1(preserve_embedding_quantization=enabled),
            )
            runtime = CircleReferenceRuntime(
                CircleDocument.from_bytes(converted.circle_binary)
            )
            contracts.append(_contract(runtime.program))
        self.assertEqual(contracts[0], contracts[1])


if __name__ == "__main__":
    unittest.main()

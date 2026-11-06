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

"""Keep an embedding's calibrated rounding grid before downstream concat."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
import torch.ao.quantization.fx._decomposed
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.passes.preserve_embedding_quantization import EXPLICIT_EMBEDDING
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.convert import _prepare_exported_program
from tico.utils.validate_args_kwargs import parse_node_args, QuantizePerTensorArgs


class _EmbeddingConcat(torch.nn.Module):
    weight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    other: torch.Tensor

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.register_buffer(
            "weight", torch.tensor([[1, 3], [2, 4]], dtype=torch.uint8)
        )
        self.register_buffer("scales", torch.tensor([0.14, 0.31]))
        self.register_buffer("zero_points", torch.tensor([0, 0], dtype=torch.int32))
        shape = (1, 4, 1) if path == "reshape" else (1, 2, 2)
        self.register_buffer("other", torch.zeros(shape))

    def forward(self, ids):
        weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.weight, self.scales, self.zero_points, 0, 0, 255, torch.uint8
        )
        value = torch.nn.functional.embedding(ids, weight)
        value = torch.fake_quantize_per_tensor_affine(value, 0.25, 0, -32768, 32767)
        if self.path == "reshape":
            value = value.reshape(1, 4, 1)
        elif self.path == "permute":
            value = value.permute(0, 2, 1)
        other = torch.fake_quantize_per_tensor_affine(self.other, 0.3, 0, -32768, 32767)
        return torch.fake_quantize_per_tensor_affine(
            torch.cat((value, other), dim=-1), 0.3, 0, -32768, 32767
        )


class _PrivateQuantizeConcat(torch.nn.Module):
    def forward(self, value):
        quantized = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            value, 0.25, 0, -32768, 32767, torch.int16
        )
        return torch.cat((quantized, quantized), dim=-1)


def _round(value):
    # Independent oracle values are small and avoid floating-point near-tie cases.
    return np.copysign(np.floor(np.abs(value) + np.float32(0.5)), value)


def _oracle(module, ids):
    selected = module.weight.numpy()[ids].astype(np.float32)
    selected -= module.zero_points.numpy()[ids, None].astype(np.float32)
    selected *= module.scales.numpy()[ids, None]
    first_codes = np.clip(_round(selected / np.float32(0.25)), -32768, 32767)
    first_float = first_codes * np.float32(0.25)
    if module.path == "reshape":
        first_float = first_float.reshape(1, 4, 1)
    elif module.path == "permute":
        first_float = first_float.transpose(0, 2, 1)
    combined = np.concatenate((first_float, np.zeros_like(first_float)), axis=-1)
    return np.clip(_round(combined / np.float32(0.3)), -32768, 32767).astype(np.int16)


class ExplicitEmbeddingRetargetTest(unittest.TestCase):
    def test_production_graph_preserves_quantize_arguments_and_metadata(self):
        for path in ("direct", "reshape", "permute"):
            with self.subTest(path=path):
                exported = torch.export.export(
                    _EmbeddingConcat(path).eval(), (torch.tensor([[0, 1]]),)
                )
                fake_quantize = next(
                    node
                    for node in exported.graph.nodes
                    if node.target
                    == torch.ops.aten.fake_quantize_per_tensor_affine.default
                    and node.args[1] == 0.25
                )
                original = fake_quantize.args[1:]
                prepared = _prepare_exported_program(
                    exported,
                    config=tico.CompileConfigV1(preserve_embedding_quantization=True),
                )
                quantize = next(
                    node
                    for node in prepared.graph.nodes
                    if node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                    and node.meta.get(EXPLICIT_EMBEDDING)
                )
                args = parse_node_args(QuantizePerTensorArgs, quantize)
                self.assertEqual(
                    (args.scale, args.zero_p, args.quant_min, args.quant_max), original
                )
                self.assertEqual(args.dtype, torch.int16)
                self.assertEqual(
                    quantize.meta[QPARAM_KEY],
                    QuantParam(scale=[0.25], zero_point=[0], dtype="int16"),
                )
                requantizes = [
                    node
                    for node in prepared.graph.nodes
                    if node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                    and not node.meta.get(EXPLICIT_EMBEDDING)
                ]
                self.assertEqual(len(requantizes), 1)
                self.assertEqual(
                    parse_node_args(QuantizePerTensorArgs, requantizes[0]).scale, 0.3
                )

    def test_public_circle_file_roundtrip_native_integer_oracle(self):
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        for path in ("direct", "reshape", "permute"):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                module = _EmbeddingConcat(path).eval()
                # This public path performs torch.export and default Circle O1.
                converted = tico.convert(
                    module,
                    (ids,),
                    config=tico.CompileConfigV1(preserve_embedding_quantization=True),
                )
                destination = Path(directory) / "embedding.circle"
                converted.save(destination)
                document = CircleDocument.load(destination)
                self.assertTrue(document.verify().ok)
                runtime = CircleReferenceRuntime(document)
                actual = runtime.run((ids.numpy(),), mode=ExecutionMode.NATIVE).outputs[
                    0
                ]
                np.testing.assert_array_equal(actual, _oracle(module, ids.numpy()))
                self.assertEqual(actual.dtype, np.int16)
                self.assertEqual(
                    int(actual.flat[0]), 1
                )  # .14 -> .25 -> .3, not .14 -> .3
                program = runtime.program
                quantizes = [op for op in program.operators if op.name == "QUANTIZE"]
                self.assertEqual(len(quantizes), 2)
                first, second = quantizes
                first_grid = program.tensor(first.outputs[0]).quantization
                second_source = program.tensor(second.inputs[0]).quantization
                second_grid = program.tensor(second.outputs[0]).quantization
                assert first_grid is not None
                assert second_grid is not None
                self.assertEqual(first_grid.scale, (0.25,))
                if path == "direct":
                    assert second_source is not None
                    self.assertEqual(second_source.scale, (0.25,))
                else:
                    # Layouts in the CPU continuation remain real float.
                    # Q_B consumes DQ_A through that layout, retaining Q_A's
                    # rounding just as integer-domain requantization does.
                    self.assertIsNone(second_source)
                    producers = {
                        index: op for op in program.operators for index in op.outputs
                    }
                    layout = producers[second.inputs[0]]
                    self.assertEqual(
                        layout.name, "RESHAPE" if path == "reshape" else "TRANSPOSE"
                    )
                    dequantize = producers[layout.inputs[0]]
                    self.assertEqual(dequantize.name, "DEQUANTIZE")
                    self.assertEqual(dequantize.inputs, first.outputs)
                self.assertEqual(second_grid.scale, (float(np.float32(0.3)),))
                self.assertTrue(runtime.rowwise_gather_plans)

    def test_protected_grid_keeps_keyword_range_zero_point_and_metadata(self):
        exported = torch.export.export(
            _PrivateQuantizeConcat().eval(), (torch.tensor([[0.14, 0.42]]),)
        )
        quantize = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        concat = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        source = quantize.args[0]
        quantize.args = (source,)
        quantize.kwargs = {
            "scale": 0.25,
            "zero_point": 7,
            "quant_min": -32767,
            "quant_max": 32767,
            "dtype": torch.int16,
        }
        original_kwargs = dict(quantize.kwargs)
        original_grid = QuantParam(scale=[0.25], zero_point=[7], dtype="int16")
        unrelated = {"nested": [4, 5]}
        quantize.meta[QPARAM_KEY] = original_grid
        quantize.meta[EXPLICIT_EMBEDDING] = True
        quantize.meta["retained_metadata"] = unrelated
        concat.meta[QPARAM_KEY] = QuantParam(
            scale=[0.3], zero_point=[11], dtype="int16"
        )
        self.assertNotIn(QPARAM_KEY, source.meta)

        PropagateQParamBackward().call(exported)

        self.assertEqual(quantize.args, (source,))
        self.assertEqual(quantize.kwargs, original_kwargs)
        self.assertIs(quantize.meta[QPARAM_KEY], original_grid)
        self.assertEqual(
            quantize.meta[QPARAM_KEY],
            QuantParam(scale=[0.25], zero_point=[7], dtype="int16"),
        )
        self.assertIs(quantize.meta["retained_metadata"], unrelated)
        self.assertEqual(unrelated, {"nested": [4, 5]})
        self.assertTrue(quantize.meta[EXPLICIT_EMBEDDING])
        self.assertNotIn(QPARAM_KEY, source.meta)

    def test_unprotected_private_nonembedding_quantize_still_retargets(self):
        exported = torch.export.export(
            _PrivateQuantizeConcat().eval(), (torch.tensor([[0.14, 0.42]]),)
        )
        quantize = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        concat = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        quantize.meta[QPARAM_KEY] = QuantParam(
            scale=[0.25], zero_point=[0], dtype="int16"
        )
        concat.meta[QPARAM_KEY] = QuantParam(scale=[0.3], zero_point=[0], dtype="int16")
        self.assertNotIn(EXPLICIT_EMBEDDING, quantize.meta)
        PropagateQParamBackward().call(exported)
        self.assertEqual(parse_node_args(QuantizePerTensorArgs, quantize).scale, 0.3)
        self.assertEqual(quantize.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])


if __name__ == "__main__":
    unittest.main()

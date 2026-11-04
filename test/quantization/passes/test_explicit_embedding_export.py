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

"""Serialized native CPU embedding: explicit domains, ABI and shared storage."""

import unittest

import numpy as np

import tico
import torch
import torch.ao.quantization.fx._decomposed
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import dequantize_array


class ExplicitEmbedding(torch.nn.Module):
    """Deliberately unequal row grids and a non-grid-aligned scale operand."""

    weight: torch.Tensor
    scales: torch.Tensor
    zp: torch.Tensor
    embed_scale: torch.Tensor

    def __init__(self, *, tied=False, quant_min=-32768):
        super().__init__()
        self.tied = tied
        self.quant_min = quant_min
        self.register_buffer(
            "weight",
            torch.tensor(
                [[0, 128, 255, 1], [5, 7, 2, 250], [11, 30, 90, 180], [251, 3, 100, 0]],
                dtype=torch.uint8,
            ),
        )
        self.register_buffer("scales", torch.tensor([0.125, 0.25, 2.0, 0.03125]))
        self.register_buffer("zp", torch.tensor([128, 4, 90, 200], dtype=torch.int64))
        self.register_buffer("embed_scale", torch.tensor(1.7))

    def forward(self, ids, hidden=None):
        weight = torch.ops.quantized_decomposed.dequantize_per_channel.default(
            self.weight, self.scales, self.zp, 0, 0, 255, torch.uint8
        )
        value = torch.nn.functional.embedding(ids, weight)
        value = torch.fake_quantize_per_tensor_affine(value, 0.25, 120, 0, 255)
        scale = torch.fake_quantize_per_tensor_affine(self.embed_scale, 0.25, 0, 0, 255)
        value = torch.fake_quantize_per_tensor_affine(
            value * scale, 0.001, 0, self.quant_min, 32767
        )
        if self.tied:
            projected = torch.nn.functional.linear(hidden, weight)
            projected = torch.fake_quantize_per_tensor_affine(
                projected, 0.125, 128, 0, 255
            )
            return value, projected
        return value


def _has_gemma4():
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextScaledWordEmbedding,
        )
    except ImportError:
        return False
    return Gemma4TextScaledWordEmbedding is not None


class ExplicitEmbeddingExportTest(unittest.TestCase):
    def _runtime(self, module, inputs):
        converted = tico.convert(module.eval(), inputs)
        # Conversion includes default Circle O1; force a generated-schema repack/reload.
        document = CircleDocument.from_bytes(converted.circle_binary)
        self.assertTrue(document.verify().ok)
        loaded = CircleDocument.from_bytes(document.to_bytes())
        return CircleReferenceRuntime(loaded)

    def _assert_structure(self, runtime):
        program = runtime.program
        gather = next(op for op in program.operators if op.name == "GATHER")
        dq = next(op for op in program.operators if op.outputs == (gather.inputs[0],))
        self.assertEqual(dq.name, "DEQUANTIZE")
        weight = program.tensor(dq.inputs[0])
        self.assertEqual(weight.dtype, np.dtype("uint8"))
        self.assertEqual(weight.shape, (4, 4))
        self.assertEqual(weight.quantization.scale, (0.125, 0.25, 2.0, 0.03125))
        self.assertEqual(weight.quantization.zero_point, (128, 4, 90, 200))
        self.assertEqual(weight.quantization.quantized_dimension, 0)
        for tensor_index in (dq.outputs[0], gather.outputs[0]):
            tensor = program.tensor(tensor_index)
            self.assertEqual(tensor.dtype, np.dtype("float32"))
            self.assertFalse(tensor.is_quantized)
        self.assertGreaterEqual(
            sum(op.name == "QUANTIZE" for op in program.operators), 2
        )
        output = runtime.output_tensors[0]
        self.assertEqual(output.dtype, np.dtype("int16"))
        self.assertEqual(output.quantization.scale, (float(np.float32(0.001)),))
        self.assertEqual(output.quantization.zero_point, (0,))
        tables = [t for t in program.tensors if t.is_constant and t.shape == (4, 4)]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].index, weight.index)
        return weight

    def test_roundtrip_decode_prefill_exact_native_oracle(self):
        for ids in ([[3]], [[0, 3, 2, 1, 3, 0]]):
            tokens = torch.tensor(ids, dtype=torch.int64)
            runtime = self._runtime(ExplicitEmbedding(), (tokens,))
            self._assert_structure(runtime)
            oracle = CircleReferenceRuntime(
                runtime.program, enable_rowwise_gather=False
            )
            values = tokens.numpy()
            expected = oracle.run((values,), mode=ExecutionMode.NATIVE).outputs[0]
            for _ in range(2):
                actual = runtime.run((values,), mode=ExecutionMode.NATIVE).outputs[0]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.int16)
                self.assertEqual(actual.shape, (*values.shape, 4))
            # Independent arithmetic, including Q_A saturation and quantized scale 1.75.
            module = ExplicitEmbedding()
            row = (
                module.weight.numpy()[values].astype(np.float32)
                - module.zp.numpy()[values, None].astype(np.float32)
            ) * module.scales.numpy()[values, None]

            def rounded(x):
                return np.copysign(np.floor(np.abs(x) + 0.5), x)

            middle = (
                np.clip(rounded(row / np.float32(0.25)) + 120, 0, 255) - 120
            ) * np.float32(0.25)
            independent = np.clip(
                rounded(middle * np.float32(1.75) / np.float32(0.001)), -32768, 32767
            ).astype(np.int16)
            np.testing.assert_array_equal(actual, independent)
            if tokens.numel() > 1:
                self.assertIn(-32768, actual)
                self.assertIn(32767, actual)

    def test_declared_symmetric_int16_range_is_preserved(self):
        tokens = torch.tensor([[0, 2]])
        runtime = self._runtime(ExplicitEmbedding(quant_min=-32767), (tokens,))
        oracle = CircleReferenceRuntime(runtime.program, enable_rowwise_gather=False)
        actual = runtime.run((tokens.numpy(),)).outputs[0]
        np.testing.assert_array_equal(actual, oracle.run((tokens.numpy(),)).outputs[0])
        self.assertEqual(actual.min(), -32767)
        self.assertEqual(actual.max(), 32767)

    def test_shared_weight_linear_domain_and_codes_unchanged(self):
        module = ExplicitEmbedding(tied=True).eval()
        tokens = torch.tensor([[0, 3, 1]])
        hidden = torch.tensor([[0.0, 0.25, -0.5, 0.75]])
        before = {name: tensor.clone() for name, tensor in module.named_buffers()}
        runtime = self._runtime(module, (tokens, hidden))
        weight = self._assert_structure(runtime)
        linear = next(
            op for op in runtime.program.operators if op.name == "FULLY_CONNECTED"
        )
        self.assertEqual(linear.inputs[1], weight.index)
        self.assertEqual(
            runtime.program.tensor(linear.outputs[0]).quantization.scale, (0.125,)
        )
        codes = weight.constant.copy()
        qparams = weight.quantization
        # Quantized FC is a fake-quant reference operation; native embedding is
        # verified separately, not replaced by this check.
        result = runtime.run(
            (tokens.numpy(), hidden.numpy()), mode=ExecutionMode.FAKE_QUANTIZE
        )
        float_weight = (
            codes.astype(np.float32)
            - np.asarray(qparams.zero_point, np.float32)[:, None]
        ) * np.asarray(qparams.scale, np.float32)[:, None]
        projected = hidden.numpy() @ float_weight.T
        expected = (
            np.clip(
                np.copysign(np.floor(np.abs(projected / 0.125) + 0.5), projected) + 128,
                0,
                255,
            )
            - 128
        ) * 0.125
        np.testing.assert_array_equal(result.outputs[1], expected)
        np.testing.assert_array_equal(weight.constant, codes)
        self.assertEqual(weight.quantization, qparams)
        for name, tensor in module.named_buffers():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    @unittest.skipUnless(
        _has_gemma4(), "optional transformers Gemma4 module unavailable"
    )
    def test_wrapq_gemma4_export_native_contract(self):
        from tico.quantization import convert, prepare
        from tico.quantization.config.ptq import PTQConfig
        from tico.quantization.config.specs import affine
        from tico.quantization.wrapq.dtypes import DType
        from tico.quantization.wrapq.qscheme import QScheme
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextScaledWordEmbedding,
        )

        torch.manual_seed(31)
        module = Gemma4TextScaledWordEmbedding(
            4, 4, padding_idx=0, embed_scale=1.7
        ).eval()
        config = PTQConfig(
            activation=affine(DType.int(16), qscheme=QScheme.PER_TENSOR_SYMM),
            weight=affine(DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM),
        )
        tokens = torch.tensor([[0, 3, 2, 1, 3]])
        prepared = prepare(module, config)
        with torch.no_grad():
            prepared(tokens)
        quantized = convert(prepared).eval()
        runtime = self._runtime(quantized, (tokens,))
        self.assertEqual(runtime.output_tensors[0].dtype, np.dtype("int16"))
        self.assertTrue(runtime.rowwise_gather_plans)
        oracle = CircleReferenceRuntime(runtime.program, enable_rowwise_gather=False)
        actual = runtime.run((tokens.numpy(),)).outputs[0]
        np.testing.assert_array_equal(actual, oracle.run((tokens.numpy(),)).outputs[0])
        # WrapQ uses ties-to-even, Circle uses half-away. This deterministic
        # sample avoids exact ties; both expose the same calibrated grid.
        expected = quantized(tokens).detach().numpy()
        np.testing.assert_array_equal(
            dequantize_array(actual, runtime.output_tensors[0]), expected
        )


if __name__ == "__main__":
    unittest.main()

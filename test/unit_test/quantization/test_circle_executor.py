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

"""End-to-end tests for the fake-quantized Circle executor used by evaluate()."""

import unittest

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle.runtime import (
    CircleReferenceRuntime,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.evaluation.backend import BACKEND
from tico.quantization.evaluation.evaluate import evaluate
from tico.quantization.evaluation.executor.circle_executor import CircleExecutor
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.utils.model import CircleModel
from torch import nn


class TwoLinear(nn.Module):
    """Two chained linear layers; the first output is a quantized activation."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.second = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


def _quantize_and_export(seed: int) -> tuple[nn.Module, CircleModel, torch.Tensor]:
    """Calibrate a UINT8 PTQ model, freeze it, and export it to Circle."""

    torch.manual_seed(seed)
    model = TwoLinear().eval()
    config = PTQConfig(
        activation=affine(DType.uint(8), qscheme=QScheme.PER_TENSOR_ASYMM),
        weight=affine(DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM),
        strict_wrap=False,
    )
    prepared = prepare(model, config, inplace=True)
    with torch.inference_mode():
        for _ in range(8):
            prepared(torch.randn(2, 4))
    quantized = convert(prepared, inplace=True).eval()
    sample = torch.randn(2, 4)
    return quantized, tico.convert(quantized, (sample,)), sample


class CircleExecutorTest(unittest.TestCase):
    def test_run_inference_before_compile_raises_error(self):
        executor = CircleExecutor()
        with self.assertRaisesRegex(RuntimeError, "compile the model"):
            executor.run_inference([])

    def test_quantized_graph_is_exported_with_integer_tensors(self):
        """The executor must face a really quantized graph, not a float one."""

        _, circle_model, _ = _quantize_and_export(seed=1)
        runtime = CircleReferenceRuntime(circle_model.circle_binary)
        uint8 = circle.TensorType.TensorType.UINT8
        self.assertTrue(
            all(tensor.tensor_type == uint8 for tensor in runtime.input_tensors)
        )
        self.assertTrue(
            all(tensor.tensor_type == uint8 for tensor in runtime.output_tensors)
        )
        fully_connected = circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED
        self.assertEqual(
            sum(
                1
                for op in runtime.program.operators
                if op.builtin_code == fully_connected
            ),
            2,
        )

    def test_native_mode_rejects_integer_fully_connected(self):
        """Integer FC arithmetic is backend-defined and must not be approximated."""

        _, circle_model, sample = _quantize_and_export(seed=2)
        runtime = CircleReferenceRuntime(circle_model.circle_binary)
        input_tensor = runtime.input_tensors[0]
        assert input_tensor.quantization is not None
        scale = input_tensor.quantization.scale[0]
        zero_point = input_tensor.quantization.zero_point[0]
        quantized_input = np.clip(
            np.round(sample.numpy() / scale) + zero_point, 0, 255
        ).astype(np.uint8)
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "FULLY_CONNECTED[\\s\\S]*FAKE_QUANTIZE"
        ):
            runtime.run([quantized_input], mode=ExecutionMode.NATIVE)

    def test_fake_quantized_inference_matches_torch_fake_quant_model(self):
        """Outputs are FLOAT32 and follow the frozen fake-quant PyTorch module."""

        quantized, circle_model, sample = _quantize_and_export(seed=3)
        executor = CircleExecutor()
        executor.compile(circle_model)
        outputs = executor.run_inference([sample])

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].dtype, np.float32)
        self.assertEqual(outputs[0].shape, (2, 3))

        with torch.no_grad():
            expected = quantized(sample).numpy()
        runtime = CircleReferenceRuntime(circle_model.circle_binary)
        output_tensor = runtime.output_tensors[0]
        assert output_tensor.quantization is not None
        output_scale = output_tensor.quantization.scale[0]
        # Both sides quantize the output to the same UINT8 grid; a one-step
        # difference can only come from a rounding tie or accumulated float
        # error near a grid boundary.
        np.testing.assert_allclose(outputs[0], expected, atol=output_scale, rtol=0.0)
        self.assertTrue(np.all(np.isfinite(outputs[0])))

    def test_fake_quantized_output_lies_on_the_serialized_grid(self):
        """Every output value must equal (q - zero_point) * scale for some q."""

        _, circle_model, sample = _quantize_and_export(seed=4)
        executor = CircleExecutor()
        executor.compile(circle_model)
        output = executor.run_inference([sample])[0]
        runtime = CircleReferenceRuntime(circle_model.circle_binary)
        quantization = runtime.output_tensors[0].quantization
        assert quantization is not None
        scale = np.float32(quantization.scale[0])
        zero_point = np.float32(quantization.zero_point[0])
        levels = output / scale + zero_point
        np.testing.assert_allclose(levels, np.round(levels), atol=1e-3, rtol=0.0)
        self.assertTrue(np.all(levels >= -1e-3))
        self.assertTrue(np.all(levels <= 255 + 1e-3))

    def test_evaluate_with_circle_backend_reports_peir(self):
        """evaluate() runs the reference executor without any external toolchain."""

        quantized, circle_model, sample = _quantize_and_export(seed=5)
        results = evaluate(
            quantized, circle_model, BACKEND.CIRCLE, [sample], mode="return"
        )
        assert results is not None
        self.assertIn("peir", results)
        self.assertEqual(len(results["peir"]), 1)
        self.assertLess(results["peir"][0], 5.0)


if __name__ == "__main__":
    unittest.main()

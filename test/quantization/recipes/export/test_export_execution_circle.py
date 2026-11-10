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

"""Execution preflight preserves supported Circle contracts and bypass policy."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.export.circle import convert_and_save_circle
from tico.quantization.wrapq.dtypes import UINT8
from tico.quantization.wrapq.lifecycle import freeze_qparams, validate_frozen_model


def _linear(*, disabled_bfloat16=False):
    module = torch.nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[0.25, 0.5], [-0.25, 1.0]]))
    model = prepare(module, PTQConfig(activation=affine(UINT8), weight=affine(UINT8)))
    observer = model.wrapped.obs_weight
    if disabled_bfloat16:
        observer.disable_fake_quant()
    observer.load_qparams(
        torch.tensor(
            [0.125, 0.25],
            dtype=torch.bfloat16 if disabled_bfloat16 else torch.float32,
        ),
        torch.tensor([1, 3], dtype=torch.int32),
        lock=True,
    )
    return model


class TestExportExecutionCircle(unittest.TestCase):
    def _assert_artifact(self, runtime, *, disabled_bfloat16):
        self.assertEqual(
            [operator.name for operator in runtime.program.operators],
            ["FULLY_CONNECTED"],
        )
        for tensor, scale, zero_point in zip(
            (*runtime.input_tensors, *runtime.output_tensors),
            (0.007843137718737125, 0.007352941203862429),
            (127, 85),
            strict=True,
        ):
            self.assertEqual(tensor.shape, (1, 2))
            self.assertEqual(tensor.dtype, np.dtype(np.uint8))
            self.assertIsNotNone(tensor.quantization)
            self.assertEqual(tensor.quantization.scale, (scale,))
            self.assertEqual(tensor.quantization.zero_point, (zero_point,))
        constants = [tensor for tensor in runtime.program.tensors if tensor.is_constant]
        self.assertEqual(len(constants), 1)
        weight = constants[0]
        self.assertEqual(weight.shape, (2, 2))
        if disabled_bfloat16:
            self.assertEqual(weight.dtype, np.dtype(np.float32))
            self.assertIsNone(weight.quantization)
            np.testing.assert_array_equal(weight.constant, [[0.25, 0.5], [-0.25, 1.0]])
        else:
            self.assertEqual(weight.dtype, np.dtype(np.uint8))
            self.assertIsNotNone(weight.quantization)
            self.assertEqual(weight.quantization.scale, (0.125, 0.25))
            self.assertEqual(weight.quantization.zero_point, (1, 3))
            self.assertEqual(weight.quantization.quantized_dimension, 0)
            np.testing.assert_array_equal(weight.constant, [[3, 5], [2, 7]])

    def _exercise_export(self, *, disabled_bfloat16):
        source = _linear(disabled_bfloat16=disabled_bfloat16)
        source(torch.tensor([[-1.0, 1.0], [0.5, -0.5]]))
        sample = torch.tensor([[0.125, 0.375]])
        source(sample)
        source = convert(source).eval()
        before = _state_contract(source)
        expected = source(sample).detach().numpy()
        original_state = source.wrapped.obs_weight.require_frozen()
        self.assertEqual(original_state.fake_quant_enabled, not disabled_bfloat16)
        self.assertEqual(original_state.provenance, "imported")
        self.assertTrue(original_state.locked)
        self.assertEqual(
            original_state.scale.dtype,
            torch.bfloat16 if disabled_bfloat16 else torch.float32,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "linear.circle"
            convert_and_save_circle(source, (sample,), path)
            document = CircleDocument.load(path)
            self.assertFalse(document.verify().issues)
            runtime = CircleReferenceRuntime(document)
            self._assert_artifact(runtime, disabled_bfloat16=disabled_bfloat16)
            actual = runtime.run(
                [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
            ).outputs[0]
            np.testing.assert_array_equal(actual, expected)

            # Real conversion must leave the source usable for lifecycle and
            # exact, strict checkpoint restoration into an equivalent topology.
            validate_frozen_model(source)
            freeze_qparams(source)
            self.assertEqual(_state_contract(source), before)
            restored = _linear(disabled_bfloat16=disabled_bfloat16)
            restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
            validate_frozen_model(restored)
            self.assertEqual(_state_contract(restored), before)
            np.testing.assert_array_equal(restored(sample).detach().numpy(), expected)
            restored_path = Path(directory) / "restored.circle"
            convert_and_save_circle(restored, (sample,), restored_path)
            restored_document = CircleDocument.load(restored_path)
            self.assertFalse(restored_document.verify().issues)
            restored_runtime = CircleReferenceRuntime(restored_document)
            self._assert_artifact(restored_runtime, disabled_bfloat16=disabled_bfloat16)
            np.testing.assert_array_equal(
                restored_runtime.run(
                    [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs[0],
                actual,
            )
            self.assertEqual(_state_contract(restored), before)

    def test_supported_fp32_scale_export_keeps_integer_tensor_contracts(self):
        self._exercise_export(disabled_bfloat16=False)

    def test_disabled_bfloat16_grid_exports_float_weight_and_keeps_exact_state(self):
        self._exercise_export(disabled_bfloat16=True)

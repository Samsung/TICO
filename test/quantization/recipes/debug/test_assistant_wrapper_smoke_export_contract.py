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

"""The smoke assistant profile must accommodate all supplied shared KV."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
    _has_gemma4_assistant,
    Gemma4AssistantCoreCase,
)


@unittest.skipUnless(_has_gemma4_assistant().available, "Gemma4 assistant unavailable")
class TestAssistantWrapperSmokeExportContract(unittest.TestCase):
    """Exercise real calibration, export, saved-file replay and frozen state."""

    def test_default_profile_exports_complete_kv_and_preserves_frozen_state(self):
        case = Gemma4AssistantCoreCase()
        model, _ = case.build({})
        prepared = case.prepare_model(model, {})
        case.after_prepare(prepared, {})
        with torch.no_grad():
            for sample in case.calibration_inputs(prepared, {}):
                case.forward(prepared, sample)
            sample = case.eval_input(prepared, {})
            quantized = case.convert_model(prepared, {})
            before = _state_contract(quantized)
            invocation = case.export_invocation(quantized, sample, {})
            inputs = invocation.inputs.args
            # No cropping: every supplied KV element survives before padding.
            shared = sample.kwargs["shared_kv_states"]
            for offset, layer_type in ((1, "full_attention"), (3, "sliding_attention")):
                for value, source in zip(
                    inputs[offset : offset + 2], shared[layer_type]
                ):
                    self.assertGreaterEqual(value.shape[2], source.shape[2])
                    torch.testing.assert_close(
                        value[:, :, : source.shape[2]], source, atol=0, rtol=0
                    )
                    self.assertTrue(torch.all(value[:, :, source.shape[2] :] == 0))
            expected = invocation.module(*inputs)
            circle = tico.convert(invocation.module.eval(), inputs)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "assistant.circle"
                circle.save(path)
                self.assertGreater(path.stat().st_size, 0)
                document = CircleDocument.load(path)
                document.verify(raise_on_error=True)
                runtime = CircleReferenceRuntime(document)
                actual = runtime.run(
                    [value.numpy() for value in inputs],
                    mode=ExecutionMode.FAKE_QUANTIZE,
                ).outputs
            self.assertEqual(len(actual), 3)
            for output, reference, index in zip(
                actual, expected, runtime.program.outputs
            ):
                self.assertEqual(output.shape, tuple(reference.shape))
                self.assertEqual(output.dtype, np.float32)
                quantization = runtime.program.tensor(index).quantization
                assert quantization is not None
                # Existing assistant export contract: at most four output-grid
                # steps for two-layer CPU accumulation and fake quantization.
                np.testing.assert_allclose(
                    output, reference.numpy(), atol=4 * quantization.scale[0], rtol=0
                )
            self.assertEqual(before, _state_contract(quantized))

    def test_undersized_sliding_profile_still_rejects_complete_kv(self):
        case = Gemma4AssistantCoreCase()
        model, _ = case.build({})
        prepared = case.prepare_model(model, {})
        sample = case.eval_input(prepared, {})
        case.sliding_kv_length = 8
        with self.assertRaisesRegex(ValueError, "Sliding shared KV exceeds"):
            case.export_input(sample, {})


if __name__ == "__main__":
    unittest.main()

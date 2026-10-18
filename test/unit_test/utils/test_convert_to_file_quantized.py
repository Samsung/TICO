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

"""Packed UINT4 constants retain qparams on the explicit file conversion path."""

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tico
import torch
from circle_schema import circle
from tico.circle.document import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode

conversion = importlib.import_module("tico.utils.convert")


class _ConstantFakeQuantModule(torch.nn.Module):
    """Use the ordinary fake-quant constant flow from RemoveWeightDequantOp tests."""

    constant: torch.Tensor

    def __init__(self):
        super().__init__()
        # Scale 1/8 and zero point 7 produce the odd-length code vector
        # [0, 1, 15, 7, 8], including both four-bit range boundaries.
        self.register_buffer("constant", torch.tensor([-0.875, -0.75, 1.0, 0.0, 0.125]))

    def forward(self, input_):
        constant = torch.fake_quantize_per_tensor_affine(
            self.constant,
            scale=0.125,
            zero_point=7,
            quant_min=0,
            quant_max=15,
        )
        return input_ + constant


class QuantizedConvertToFileTest(unittest.TestCase):
    def test_uint4_payload_qparams_and_fake_quantize_parity(self):
        for optimize in (False, True):
            for appended in (False, True):
                with self.subTest(optimize=optimize, appended=appended):
                    module = _ConstantFakeQuantModule().eval()
                    original = module.constant.clone()
                    args = (torch.tensor([0.5, -1.0, 0.0, 2.0, -0.125]),)
                    config = tico.CompileConfigV1(circle_optimize=optimize)
                    expected = tico.convert(module, args, config=config).circle_binary
                    with tempfile.TemporaryDirectory() as temporary:
                        path = Path(temporary) / "uint4.circle"
                        budget = (
                            0 if appended else conversion._FILE_OUTPUT_INLINE_BUDGET
                        )
                        with patch.object(
                            conversion, "_FILE_OUTPUT_INLINE_BUDGET", budget
                        ):
                            tico.convert_to_file(module, args, path, config=config)
                        actual = path.read_bytes()
                        document = CircleDocument.from_bytes(actual)
                        document.verify()
                        if appended:
                            self.assertEqual(document.to_bytes(), expected)
                        else:
                            self.assertEqual(actual, expected)

                        weights = [
                            tensor
                            for tensor in document.model.subgraphs[0].tensors
                            if tensor.type == circle.TensorType.TensorType.UINT4
                        ]
                        self.assertEqual(len(weights), 1)
                        weight = weights[0]
                        np.testing.assert_array_equal(weight.shape, [5])
                        np.testing.assert_array_equal(
                            weight.quantization.scale, [0.125]
                        )
                        np.testing.assert_array_equal(
                            weight.quantization.zeroPoint, [7]
                        )
                        np.testing.assert_array_equal(
                            document.model.buffers[weight.buffer].data,
                            [0x10, 0x7F, 0x08],
                        )
                        root = circle.Model.Model.GetRootAsModel(actual, 0)
                        self.assertEqual(
                            root.Buffers(weight.buffer).Offset() > 0, appended
                        )

                        # These dyadic values require no rounding. This validates
                        # dequantized reference semantics, not integer backend parity.
                        with torch.no_grad():
                            reference = module(*args).numpy()
                        result = CircleReferenceRuntime(document).run(
                            args, mode=ExecutionMode.FAKE_QUANTIZE
                        )
                        self.assertEqual(len(result.outputs), 1)
                        np.testing.assert_array_equal(result.outputs[0], reference)
                    torch.testing.assert_close(
                        module.constant, original, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()

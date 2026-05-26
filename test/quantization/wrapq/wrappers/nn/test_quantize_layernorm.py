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

"""Smoke tests migrated from the LayerNorm quantization example."""

import copy
import os
import unittest

import tico.quantization

import torch
import torch.nn as nn
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
class TestNNLayerNormExample(unittest.TestCase):
    """Exercise the old LayerNorm PTQ example."""

    def test_prepare_convert_layernorm_flow_matches_example(self):
        """Quantize LayerNorm and validate output shape and finiteness."""
        torch.manual_seed(123)
        model = nn.LayerNorm((32,), eps=1e-5, elementwise_affine=True).eval()
        fp_ref = copy.deepcopy(model).eval()
        prepared = tico.quantization.prepare(model, PTQConfig(), inplace=True)
        self.assertIsInstance(prepared, PTQWrapper)

        calibration_data = [torch.randn(4, 32) for _ in range(3)]
        with torch.no_grad():
            for sample in calibration_data:
                prepared(sample)

        quantized = tico.quantization.convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            quant_out = quantized(calibration_data[0])
            fp_out = fp_ref(calibration_data[0])

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

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

"""Smoke tests migrated from the simple Linear quantization example."""

import copy
import unittest

import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


class TinyLinearModel(nn.Module):
    """A minimal model with a single Linear layer."""

    def __init__(self):
        """Create the toy model used by the migrated example test."""
        super().__init__()
        self.fc = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        """Run the single Linear layer."""
        return self.fc(x)


class TestNNLinearExample(unittest.TestCase):
    """Exercise the old tiny Linear PTQ example."""

    def test_prepare_convert_linear_flow_matches_example(self):
        """Quantize a Linear layer and validate output shape and finiteness."""
        torch.manual_seed(123)
        model = TinyLinearModel().eval()
        fp_ref = copy.deepcopy(model.fc).eval()
        model.fc = prepare(model.fc, PTQConfig())  # type: ignore[assignment]
        self.assertIsInstance(model.fc, PTQWrapper)
        self.assertIsInstance(model.fc.wrapped, QuantLinear)

        with torch.no_grad():
            for _ in range(4):
                model(torch.randn(4, 16))

        convert(model.fc)
        self.assertIs(model.fc._mode, Mode.QUANT)

        x = torch.randn(2, 16)
        with torch.no_grad():
            quant_out = model(x)
            fp_out = fp_ref(x)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

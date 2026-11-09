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

"""Smoke tests migrated from the Qwen3-VL text MLP quantization example."""

import math
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

from test.quantization.wrapq.wrappers.qwen_vl.text_mlp_fixture import (
    make_text_mlp_fixture,
)

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL text MLP tests"


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenTextMLPExample(unittest.TestCase):
    """Exercise the prepare-calibrate-convert flow for Qwen text MLP."""

    def setUp(self):
        """Create a tiny deterministic text MLP module."""
        self.fp_mlp, self.calibration, self.hidden = make_text_mlp_fixture()

    def test_prepare_convert_text_mlp_flow_matches_example(self):
        """Quantize a Qwen text MLP and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_mlp import (
            QuantQwen3VLTextMLP,
        )

        prepared = prepare(self.fp_mlp, PTQConfig()).eval()
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantQwen3VLTextMLP)
        self.assertIs(prepared._mode, Mode.CALIB)
        self.assertTrue(hasattr(prepared, "tico_quantizer"))

        with torch.no_grad():
            prepared(self.calibration)

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)
        self.assertIs(quantized.wrapped._mode, Mode.QUANT)
        self.assertFalse(hasattr(quantized, "tico_quantizer"))

        hidden = self.hidden
        # Independent FP formula for the duplicated identity projections, not
        # the wrapper's forward or the FP model that prepare() may mutate.
        golden = torch.tensor(
            [
                value * value / (1 + math.exp(-value))
                for value in hidden.flatten().tolist()
            ]
        ).reshape(hidden.shape)
        with torch.no_grad():
            quant_out = quantized(hidden)

        self.assertEqual(quant_out.shape, golden.shape)
        self.assertTrue(torch.isfinite(quant_out).all())
        # uint8 min/max gives at most h=1/(2*255) round error at every site:
        # all activation ranges are <=1, with calibration covering endpoints.
        # Identity/averaging projections have row norm 1 and exact grid weights.
        # Input/gate/SiLU-input accumulate <=4h; sigmoid (Lipschitz 1/4) <=2h;
        # SiLU product+round <=7h; up branch <=3h; gated product+round <=11h;
        # down input/output and final output add <=3h. Products have factors
        # in [0,1], so these absolute errors add without amplification. Clipping
        # cannot increase error against an in-range FP value. Reserve 1e-6 for
        # float32 arithmetic separately from this 14h quantization bound.
        torch.testing.assert_close(
            quant_out, golden, rtol=0, atol=14 / (2 * 255) + 1e-6
        )


if __name__ == "__main__":
    unittest.main()

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

"""Smoke tests migrated from the Llama MLP quantization example."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import INT16
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

from test.quantization.quant_spec_helpers import make_affine_ptq_config

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Llama MLP example tests"


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestLlamaMLPExample(unittest.TestCase):
    """Exercise the old Llama MLP prepare-calibrate-convert flow."""

    def setUp(self):
        """Create a tiny deterministic LlamaMLP module."""
        torch.manual_seed(123)
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaMLP

        self.cfg = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        self.fp_mlp = LlamaMLP(self.cfg).eval()
        self.fp_ref = copy.deepcopy(self.fp_mlp).eval()

    def test_prepare_convert_mlp_flow_matches_example(self):
        """Quantize a Llama MLP with the INT16 policy used by the example."""
        from tico.quantization.wrapq.wrappers.llama.quant_mlp import QuantLlamaMLP

        qcfg = make_affine_ptq_config(dtype=INT16, qscheme=QScheme.PER_TENSOR_SYMM)
        prepared = prepare(self.fp_mlp, qcfg)
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantLlamaMLP)

        with torch.no_grad():
            for _ in range(4):
                prepared(torch.randn(2, 5, self.cfg.hidden_size))

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden = torch.randn(2, 5, self.cfg.hidden_size)
        with torch.no_grad():
            quant_out = quantized(hidden)
            fp_out = self.fp_ref(hidden)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())
        self.assertLess((quant_out - fp_out).abs().mean().item(), 1.0)


if __name__ == "__main__":
    unittest.main()

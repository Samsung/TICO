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

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL text MLP tests"


def _make_text_config():
    """Create a tiny Qwen3-VL text config for synthetic smoke tests."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig

    return Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        use_cache=False,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenTextMLPExample(unittest.TestCase):
    """Exercise the prepare-calibrate-convert flow for Qwen text MLP."""

    def setUp(self):
        """Create a tiny deterministic text MLP module."""
        torch.manual_seed(123)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

        self.cfg = _make_text_config()
        self.fp_mlp = Qwen3VLTextMLP(self.cfg).eval()
        self.fp_ref = copy.deepcopy(self.fp_mlp).eval()

    def test_prepare_convert_text_mlp_flow_matches_example(self):
        """Quantize a Qwen text MLP and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_mlp import (
            QuantQwen3VLTextMLP,
        )

        prepared = prepare(self.fp_mlp, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantQwen3VLTextMLP)

        with torch.no_grad():
            for _ in range(3):
                prepared(torch.randn(2, 8, self.cfg.hidden_size))

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden = torch.randn(2, 8, self.cfg.hidden_size)
        with torch.no_grad():
            quant_out = quantized(hidden)
            fp_out = self.fp_ref(hidden)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

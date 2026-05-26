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

"""Smoke tests migrated from the Qwen3-VL text model quantization example."""

import copy
import os
import unittest

import tico.quantization

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL text model tests"


def _make_text_config():
    """Create a tiny Qwen3-VL text config for full text-model tests."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig

    cfg = Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=128,
        attention_dropout=0.0,
        use_cache=False,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenTextModelExample(unittest.TestCase):
    """Exercise the old text-model PTQ flow with synthetic token IDs."""

    def test_prepare_convert_text_model_flow_matches_example(self):
        """Quantize Qwen3VLTextModel and validate the hidden-state output."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(123)
        cfg = _make_text_config()
        model = Qwen3VLTextModel(cfg).eval()
        fp_ref = copy.deepcopy(model).eval()

        prepared = tico.quantization.prepare(model, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)

        with torch.no_grad():
            for _ in range(3):
                ids = torch.randint(0, cfg.vocab_size, (1, 8))
                prepared(input_ids=ids, use_cache=False)

        quantized = tico.quantization.convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)

        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            quant_out = quantized(
                input_ids=input_ids, use_cache=False
            ).last_hidden_state
            fp_out = fp_ref(input_ids=input_ids, use_cache=False).last_hidden_state

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

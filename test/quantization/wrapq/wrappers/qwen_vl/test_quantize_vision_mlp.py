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

"""Smoke tests migrated from the Qwen3-VL vision MLP quantization example."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_mlp import (
    QuantQwen3VLVisionMLP,
)

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = (
    "required transformers not installed — skipping Qwen3-VL vision MLP example tests"
)


def _make_tiny_qwen3vl_model():
    """Build a tiny Qwen3-VL model from config without downloading weights."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    cfg = Qwen3VLConfig(
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1],
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "num_hidden_layers": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 1024,
            "vocab_size": 1000,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=998,
        video_token_id=999,
    )
    return Qwen3VLModel(cfg).eval()


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenVisionMLPExample(unittest.TestCase):
    """Exercise the example flow for one Qwen3-VL vision MLP module."""

    def setUp(self):
        """Create one tiny Qwen3-VL vision MLP and a floating-point reference."""
        torch.manual_seed(123)
        model = _make_tiny_qwen3vl_model()
        self.fp_mlp = model.visual.blocks[0].mlp.eval()
        self.fp_ref = copy.deepcopy(self.fp_mlp).eval()
        self.hidden_size = model.config.vision_config.hidden_size

    def _calibrate(self, prepared: PTQWrapper) -> None:
        """Run the synthetic calibration sweep used by the original example."""
        with torch.no_grad():
            for _ in range(3):
                hidden = torch.randn(16, self.hidden_size)
                _ = prepared(hidden)

    def test_prepare_convert_vision_mlp_flow_matches_example(self):
        """Quantize one Qwen3-VL vision MLP and compare its quantized output."""
        prepared = prepare(self.fp_mlp, PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantQwen3VLVisionMLP)
        self.assertIs(prepared._mode, Mode.CALIB)

        self._calibrate(prepared)
        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden = torch.randn(16, self.hidden_size)
        with torch.no_grad():
            quant_out = quantized(hidden)
            fp_out = self.fp_ref(hidden)

        diff = (quant_out - fp_out).abs().mean().item()
        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.0)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

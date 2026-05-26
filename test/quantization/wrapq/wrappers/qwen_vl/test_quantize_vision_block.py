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

"""Smoke tests migrated from the Qwen3-VL vision block quantization example."""

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

skip_msg = "required transformers not installed — skipping Qwen3-VL vision block tests"


def _rope(seq_len: int, head_dim: int):
    """Create synthetic rotary position embeddings."""
    emb = torch.randn(seq_len, head_dim)
    return emb.cos(), emb.sin()


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenVisionBlockExample(unittest.TestCase):
    """Exercise the old vision block PTQ flow with synthetic hidden states."""

    def test_prepare_convert_vision_block_flow_matches_example(self):
        """Quantize Qwen3VLVisionBlock and validate output shape."""
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        torch.manual_seed(123)
        cfg = Qwen3VLVisionConfig(hidden_size=64, num_heads=4)
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"
        model = Qwen3VLVisionBlock(cfg).eval()
        fp_ref = copy.deepcopy(model).eval()
        prepared = tico.quantization.prepare(model, PTQConfig(), inplace=True)
        self.assertIsInstance(prepared, PTQWrapper)

        seq_len = 8
        cu_seqlens = torch.tensor([0, seq_len])
        pos = _rope(seq_len, cfg.hidden_size // cfg.num_heads)
        calibration_data = [torch.randn(seq_len, cfg.hidden_size) for _ in range(3)]
        with torch.no_grad():
            for hidden in calibration_data:
                prepared(hidden, cu_seqlens, position_embeddings=pos)

        quantized = tico.quantization.convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            quant_out = quantized(
                calibration_data[0], cu_seqlens, position_embeddings=pos
            )
            fp_out = fp_ref(calibration_data[0], cu_seqlens, position_embeddings=pos)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

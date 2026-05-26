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

"""Smoke tests migrated from the Qwen3-VL vision attention quantization example."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attention import (
    QuantQwen3VLVisionAttention,
)

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL vision attention example tests"


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


def _get_position_embeddings(visual_model, grid_thw: torch.Tensor):
    """Return Qwen3-VL vision RoPE embeddings for a synthetic image grid."""
    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)

    seq_len, _ = pos_embeds.size()
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return emb.cos(), emb.sin()


def _get_cu_seqlens(grid_thw: torch.Tensor):
    """Return cumulative sequence lengths for one synthetic Qwen3-VL image grid."""
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    return torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenVisionAttentionExample(unittest.TestCase):
    """Exercise the example flow for one Qwen3-VL vision attention module."""

    def setUp(self):
        """Create one tiny Qwen3-VL vision attention and its reference inputs."""
        torch.manual_seed(123)
        model = _make_tiny_qwen3vl_model()
        self.visual = model.visual
        self.fp_attn = model.visual.blocks[0].attn.eval()
        self.fp_ref = copy.deepcopy(self.fp_attn).eval()
        self.hidden_size = model.config.vision_config.hidden_size
        self.grid_thw = torch.tensor([[1, 8, 8]], dtype=torch.long)
        self.cu_seqlens = _get_cu_seqlens(self.grid_thw)
        self.position_embeddings = _get_position_embeddings(self.visual, self.grid_thw)
        self.seq_len = int(self.cu_seqlens[-1].item())

    def _make_hidden(self):
        """Create synthetic patch embeddings with the expected sequence length."""
        return torch.randn(self.seq_len, self.hidden_size)

    def _calibrate(self, prepared: PTQWrapper) -> None:
        """Run the synthetic calibration sweep used by the original example."""
        with torch.no_grad():
            for _ in range(3):
                hidden = self._make_hidden()
                _ = prepared(hidden, self.cu_seqlens, None, self.position_embeddings)

    def test_prepare_convert_vision_attention_flow_matches_example(self):
        """Quantize one Qwen3-VL vision attention block and compare its output."""
        prepared = prepare(self.fp_attn, PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantQwen3VLVisionAttention)
        self.assertIs(prepared._mode, Mode.CALIB)

        self._calibrate(prepared)
        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        hidden = self._make_hidden()
        with torch.no_grad():
            quant_out = quantized(
                hidden, self.cu_seqlens, None, self.position_embeddings
            )
            fp_out = self.fp_ref(
                hidden, self.cu_seqlens, None, self.position_embeddings
            )

        diff = (quant_out - fp_out).abs().mean().item()
        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 1.5)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

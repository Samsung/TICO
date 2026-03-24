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

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
    QuantQwen3VLTextDecoderLayer,
)

skip_msg = "required transformers not installed — skipping Qwen3VLTextDecoderLayer tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLTextDecoderLayer(unittest.TestCase):
    fp_layer: torch.nn.Module

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        cls.cfg = Qwen3VLTextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=256,
        )
        if not hasattr(cls.cfg, "_attn_implementation"):
            setattr(cls.cfg, "_attn_implementation", "eager")
        else:
            cls.cfg._attn_implementation = "eager"

        cls.fp_layer = Qwen3VLTextDecoderLayer(cls.cfg, layer_idx=0)

    def _rand_rope(self, B: int, S: int):
        h = self.cfg.head_dim
        emb = torch.randn(B, S, h)
        return emb.cos(), emb.sin()

    def test_mode_transitions(self):
        qlayer = QuantQwen3VLTextDecoderLayer(self.fp_layer)
        self.assertIs(qlayer._mode, Mode.NO_QUANT)

        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)

        SEQ_LEN = 16
        hidden = torch.randn(1, SEQ_LEN, self.cfg.hidden_size)
        _ = qlayer(hidden)

        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

    def test_forward_diff(self):
        qlayer = QuantQwen3VLTextDecoderLayer(self.fp_layer)
        qlayer.enable_calibration()

        SEQ_LEN = 16
        for _ in range(4):
            hidden = torch.randn(1, SEQ_LEN, self.cfg.hidden_size)
            _ = qlayer(hidden)
        qlayer.freeze_qparams()

        hidden = torch.randn(1, SEQ_LEN, self.cfg.hidden_size)
        pos = self._rand_rope(1, SEQ_LEN)

        mask = torch.full((1, 1, SEQ_LEN, SEQ_LEN), float("-120"))
        mask.triu_(1)

        with torch.no_grad():
            q_out = qlayer(hidden)
            q_out = q_out[0] if isinstance(q_out, tuple) else q_out

            fp_out = self.fp_layer(
                hidden, attention_mask=mask, position_embeddings=pos
            )
            fp_out = fp_out[0] if isinstance(fp_out, tuple) else fp_out

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_with_precomputed_embeddings(self):
        """position_embeddings injected from outside (model-level sharing pattern)."""
        qlayer = QuantQwen3VLTextDecoderLayer(self.fp_layer)
        qlayer.enable_calibration()

        SEQ_LEN = 16
        hidden = torch.randn(1, SEQ_LEN, self.cfg.hidden_size)
        pos = self._rand_rope(1, SEQ_LEN)

        mask = torch.full((1, 1, SEQ_LEN, SEQ_LEN), float("-120"))
        mask.triu_(1)

        _ = qlayer(hidden, attention_mask=mask, position_embeddings=pos)
        qlayer.freeze_qparams()
        self.assertIs(qlayer._mode, Mode.QUANT)

    def test_dtype_override(self):
        cfg = PTQConfig(
            default_dtype=DType.int(16),
            overrides={
                "mlp_residual_out": {"dtype": DType.uint(8)},
            },
        )
        qlayer = QuantQwen3VLTextDecoderLayer(self.fp_layer, qcfg=cfg)
        self.assertEqual(qlayer.obs_mlp_residual_out.dtype, DType.uint(8))

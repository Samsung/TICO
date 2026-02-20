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

"""
The tests run only if *transformers* is available (they depend on the genuine
`transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionAttention`).
"""

import importlib.util
import unittest

import torch
from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attn import (
    QuantQwen3VLVisionAttention,
)


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping LlamaAttention tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLAttention(unittest.TestCase):
    fp_attn: torch.nn.Module
    head_dim: int
    hidden_size: int

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionAttention,
        )

        cls.hidden_size = 16
        cfg = Qwen3VLVisionConfig(hidden_size=cls.hidden_size, num_heads=2)

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_attn = Qwen3VLVisionAttention(cfg)
        cls.head_dim = cls.fp_attn.head_dim

    # dummy RoPE tables with correct last dim
    def _rand_rope(self, S):
        h = self.head_dim
        emb = torch.randn(S, h)
        return emb.cos(), emb.sin()

    def test_mode_transitions(self):
        qattn = QuantQwen3VLVisionAttention(self.fp_attn)
        self.assertIs(qattn._mode, Mode.NO_QUANT)

        qattn.enable_calibration()
        self.assertIs(qattn._mode, Mode.CALIB)
        seq_len = 12
        x = torch.randn(seq_len, self.hidden_size)
        pos = self._rand_rope(seq_len)
        _ = qattn(x, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=pos)

        qattn.freeze_qparams()
        self.assertIs(qattn._mode, Mode.QUANT)

    def test_forward_diff(self):
        seq_len = 12
        cu_seqlens = torch.tensor([0, seq_len])
        qattn = QuantQwen3VLVisionAttention(self.fp_attn)
        qattn.enable_calibration()
        for _ in range(4):
            inp = torch.randn(seq_len, self.hidden_size)
            pos = self._rand_rope(seq_len)
            _ = qattn(inp, cu_seqlens=cu_seqlens, position_embeddings=pos)
        qattn.freeze_qparams()

        x = torch.randn(seq_len, self.hidden_size)
        pos = self._rand_rope(seq_len)
        with torch.no_grad():
            q_out = qattn(x, cu_seqlens=cu_seqlens, position_embeddings=pos)
            fp_out = self.fp_attn(inp, cu_seqlens=cu_seqlens, position_embeddings=pos)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_per_projection_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "qkv": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                },
                "proj": {
                    "act_in": {"dtype": DType.int(16)},
                    "act_out": {"dtype": DType.int(16)},
                },
            },
        )
        qattn = QuantQwen3VLVisionAttention(self.fp_attn, qcfg=cfg)

        q_lin = qattn.proj.wrapped
        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.obs_act_in.dtype, DType.int(16))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.int(16))

        q_lin = qattn.qkv.wrapped
        self.assertIsInstance(q_lin, QuantLinear)
        self.assertEqual(q_lin.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.uint(4))

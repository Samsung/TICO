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

import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention


import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.qparams import inject_gptq_qparams

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.linear_folding import (
    reused_weight_qparam_scale_multiplier,
)
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attention import (
    QuantQwen3VLVisionAttention,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config

skip_msg = "required transformers not installed — skipping LlamaAttention tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLAttention(unittest.TestCase):
    fp_attn: "Qwen3VLVisionAttention"
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
        cfg = Qwen3VLVisionConfig(  # pylint: disable=unexpected-keyword-arg
            hidden_size=cls.hidden_size, num_heads=2
        )

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
        cu_seqlens = torch.tensor([0, seq_len])
        x = torch.randn(seq_len, self.hidden_size)
        pos = self._rand_rope(seq_len)
        _ = qattn(
            x, cu_seqlens=cu_seqlens, rotary_pos_emb=None, position_embeddings=pos
        )

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
            fp_out = self.fp_attn(x, cu_seqlens=cu_seqlens, position_embeddings=pos)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.1)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_static_attention_split_sizes_match_dynamic_cu_seqlens(self):
        """Static export split sizes should preserve eager attention results."""
        split_sizes = (4, 8)
        seq_len = sum(split_sizes)
        cu_seqlens = torch.tensor([0, split_sizes[0], seq_len])
        qattn = QuantQwen3VLVisionAttention(self.fp_attn)

        x = torch.randn(seq_len, self.hidden_size)
        pos = self._rand_rope(seq_len)
        with torch.no_grad():
            dynamic_output = qattn(
                x,
                cu_seqlens=cu_seqlens,
                position_embeddings=pos,
            )
            static_output = qattn(
                x,
                cu_seqlens=cu_seqlens,
                position_embeddings=pos,
                attention_split_sizes=split_sizes,
            )

        torch.testing.assert_close(static_output, dynamic_output)

    def test_per_projection_override(self):
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
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

    def test_k_fold_records_weight_qparam_multiplier(self):
        qattn = QuantQwen3VLVisionAttention(self.fp_attn)
        qkv_lin = qattn.qkv.wrapped.module
        assert isinstance(qkv_lin, torch.nn.Linear)
        mult = reused_weight_qparam_scale_multiplier(qattn.qkv.wrapped)
        assert isinstance(mult, torch.Tensor)
        n = qkv_lin.weight.shape[0] // 3
        torch.testing.assert_close(mult[:n], torch.ones(n))
        torch.testing.assert_close(
            mult[n : 2 * n], torch.full((n,), self.head_dim**-0.5)
        )
        torch.testing.assert_close(mult[2 * n :], torch.ones(n))

    def test_inject_gptq_qparams_rescales_folded_k_rows(self):
        qattn = QuantQwen3VLVisionAttention(
            self.fp_attn, qcfg=make_affine_ptq_config(dtype=DType.uint(4))
        )
        qkv_lin = qattn.qkv.wrapped.module
        assert isinstance(qkv_lin, torch.nn.Linear)
        n_rows = qkv_lin.weight.shape[0]
        base_scale = torch.linspace(0.01, 0.02, n_rows).reshape(-1, 1)

        class _Quantizer:
            maxq = 15
            perchannel = True
            scale = base_scale
            zero = torch.zeros(n_rows, 1)

        stats = inject_gptq_qparams(qattn, {"qkv": _Quantizer()})
        self.assertEqual(stats["matched"], 1)

        obs = qattn.qkv.wrapped.get_observer("weight")
        assert obs is not None
        qparams = obs.compute_qparams()
        assert qparams is not None
        got_scale, _ = qparams
        n = n_rows // 3
        expected = base_scale.reshape(-1) * torch.cat(
            [torch.ones(n), torch.full((n,), self.head_dim**-0.5), torch.ones(n)]
        )
        torch.testing.assert_close(got_scale, expected)

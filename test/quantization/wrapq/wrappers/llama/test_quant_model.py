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
`transformers.models.llama.modeling_llama.LlamaModel`).
"""

import unittest

import torch
from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_model import QuantLlamaModel

skip_msg = "required transformers not installed — skipping LlamaModel tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaModel(unittest.TestCase):
    seq_len: int
    vocab_size: int
    fp_model: torch.nn.Module

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel

        cls.seq_len = 16
        cls.vocab_size = 10000
        cfg = LlamaConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            num_hidden_layers=2,
            max_position_embeddings=cls.seq_len,
            use_cache=False,
            return_dict=False,
        )
        cls.fp_model = LlamaModel(cfg)

    def test_mode_transitions(self):
        qmodel = QuantLlamaModel(self.fp_model)
        self.assertIs(qmodel._mode, Mode.NO_QUANT)

        qmodel.enable_calibration()
        self.assertIs(qmodel._mode, Mode.CALIB)

        x = torch.randint(
            0,
            self.vocab_size,
            (
                1,
                self.seq_len,
            ),
        )
        _ = qmodel(x)  # gather stats

        qmodel.freeze_qparams()
        self.assertIs(qmodel._mode, Mode.QUANT)

    def test_forward_diff(self):
        qmodel = QuantLlamaModel(self.fp_model)
        qmodel.enable_calibration()
        calib_set = []
        for _ in range(4):
            inp = torch.randint(
                0,
                self.vocab_size,
                (
                    1,
                    self.seq_len,
                ),
            )
            _ = qmodel(inp)
            calib_set.append(inp)
        qmodel.freeze_qparams()

        with torch.no_grad():
            q_out = qmodel(calib_set[0])[0]
            fp_out = self.fp_model(calib_set[0])[0]

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

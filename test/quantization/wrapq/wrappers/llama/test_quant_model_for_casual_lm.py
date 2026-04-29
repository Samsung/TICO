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
`transformers.models.llama.modeling_llama.LlamaForCausalLM`).
"""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    QuantLlamaForCausalLMDecodeExportAdapter,
    QuantLlamaForCausalLMPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm import (
    QuantLlamaForCausalLM,
)

skip_msg = "required transformers not installed — skipping LlamaForCausalLM tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaForCausalLM(unittest.TestCase):
    seq_len: int
    vocab_size: int
    fp_model: torch.nn.Module

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

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
            vocab_size=cls.vocab_size,
            return_dict=True,
        )

        cls.fp_model = LlamaForCausalLM(cfg)

    def test_mode_transitions(self):
        qmodel = QuantLlamaForCausalLM(self.fp_model, qcfg=PTQConfig())
        self.assertIs(qmodel._mode, Mode.NO_QUANT)

        qmodel.enable_calibration()
        self.assertIs(qmodel._mode, Mode.CALIB)

        x = torch.randint(0, self.vocab_size, (1, self.seq_len // 2))
        _ = qmodel(x, return_dict=True, use_cache=False)

        qmodel.freeze_qparams()
        self.assertIs(qmodel._mode, Mode.QUANT)

    def test_forward_diff(self):
        qmodel = QuantLlamaForCausalLM(self.fp_model, qcfg=PTQConfig())
        qmodel.enable_calibration()

        calib_set = []
        for index in range(4):
            inp = torch.randint(
                0,
                self.vocab_size,
                (1, self.seq_len // (index + 1)),
            )
            _ = qmodel(inp, return_dict=True, use_cache=False)
            calib_set.append(inp)

        qmodel.freeze_qparams()

        with torch.no_grad():
            q_out = qmodel(
                calib_set[0],
                return_dict=True,
                use_cache=False,
            ).logits

            fp_out = self.fp_model(
                calib_set[0],
                return_dict=True,
                use_cache=False,
            ).logits

        diff = (fp_out - q_out).abs().mean().item()

        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_export_adapter_without_cache_return_logits_only(self):
        qmodel = QuantLlamaForCausalLM(self.fp_model, qcfg=PTQConfig())
        prefill_adapter = QuantLlamaForCausalLMPrefillExportAdapter(qmodel)

        batch_size = 1
        inp = torch.randint(
            0,
            self.vocab_size,
            (1, self.seq_len),
        )

        with torch.no_grad():
            logits = prefill_adapter(
                input_ids=inp,
            )

        self.assertEqual(logits.shape, (batch_size, self.seq_len, self.vocab_size))

    def test_export_adapter_with_cache_return_logits_and_kv(self):
        qmodel = QuantLlamaForCausalLM(self.fp_model, qcfg=PTQConfig())
        prefill_adapter = QuantLlamaForCausalLMPrefillExportAdapter(
            qmodel, return_kv=True
        )

        batch_size = 1
        inp = torch.randint(
            0,
            self.vocab_size,
            (1, self.seq_len),
        )

        with torch.no_grad():
            out = prefill_adapter(
                input_ids=inp,
            )
        # out should be a tuple (logits, past_key_values)
        self.assertIsInstance(out, tuple)
        logits, past_key_values = out
        self.assertEqual(logits.shape, (batch_size, self.seq_len, self.vocab_size))
        # Expect past_key_values length equals number of layers (2)
        self.assertEqual(len(past_key_values), 2)
        for pkv in past_key_values:
            self.assertIsInstance(pkv, tuple)
            self.assertEqual(len(pkv), 2)
            k, v = pkv
            self.assertEqual(k.shape, (batch_size, 1, self.seq_len, 4))
            self.assertEqual(v.shape, (batch_size, 1, self.seq_len, 4))

    def test_decode_adapter_with_cache_return_logits_and_kv(self):
        qmodel = QuantLlamaForCausalLM(self.fp_model, qcfg=PTQConfig())
        decode_adapter = QuantLlamaForCausalLMDecodeExportAdapter(qmodel)

        batch_size = 1
        inp = torch.randint(
            0,
            self.vocab_size,
            (1, 1),
        )
        past_key_values = [
            (
                torch.randn(batch_size, 1, self.seq_len - 1, 4),
                torch.randn(batch_size, 1, self.seq_len - 1, 4),
            )
            for _ in range(2)
        ]

        with torch.no_grad():
            out = decode_adapter(
                input_ids=inp,
                past_key_values=past_key_values,
            )
        # out should be a tuple (logits, new_key_values)
        self.assertIsInstance(out, tuple)
        logits, new_key_values = out
        self.assertEqual(logits.shape, (batch_size, 1, self.vocab_size))
        self.assertEqual(len(new_key_values), 2)

    # for nkv in new_key_values:
    #     self.assertIsInstance(nkv, tuple)
    #     self.assertEqual(len(nkv), 2)
    #     k, v = nkv
    #     self.assertEqual(k.shape, (batch_size, 1, 1, 4))
    #     self.assertEqual(v.shape, (batch_size, 1, 1, 4))

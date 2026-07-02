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

"""Smoke tests for Gemma4ForCausalLM prepare-calibrate-convert flow."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4ForCausalLM,
        )
    except Exception:
        return False
    return True


def _make_text_config():
    """Create a tiny Gemma4 text config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0}
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        enable_moe_block=False,
        # Enable logit softcapping to exercise that code path.
        final_logit_softcapping=30.0,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4ForCausalLMSmoke(unittest.TestCase):
    """Exercise Gemma4ForCausalLM wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4ForCausalLM modules."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        self.text_cfg = _make_text_config()
        self.fp_model = Gemma4ForCausalLM(self.text_cfg).eval()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        self.batch_size = 1
        self.seq_len = 16

    def _ptq_config(self):
        """Build the PTQ config used by smoke tests."""
        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=0,
        )

    def _text_sample(self):
        """Create one synthetic text-only input."""
        return {
            "input_ids": torch.randint(
                0, self.text_cfg.vocab_size, (self.batch_size, self.seq_len)
            ),
        }

    def test_no_quant_model_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_for_causal_lm import (
            QuantGemma4ForCausalLM,
        )

        wrapped = QuantGemma4ForCausalLM(self.fp_model, qcfg=self._ptq_config()).eval()
        sample = self._text_sample()

        with torch.no_grad():
            quant_out = wrapped(**sample)
            fp_out = self.fp_ref(**sample).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_prepare_convert_flow(self):
        """Quantize Gemma4ForCausalLM and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_for_causal_lm import (
            QuantGemma4ForCausalLM,
        )

        prepared = prepare(self.fp_model, self._ptq_config())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4ForCausalLM)

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample)
            fp_out = self.fp_ref(**sample).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_logits_to_keep(self):
        """logits_to_keep=1 should produce logits only for the last position."""
        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)

        sample = self._text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample, logits_to_keep=1)
            fp_out = self.fp_ref(**sample, logits_to_keep=1).logits

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertEqual(quant_out.shape[1], 1)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_logit_softcapping_bounded(self):
        """When final_logit_softcapping is set, logits should be bounded."""
        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)

        sample = self._text_sample()
        with torch.no_grad():
            quant_out = quantized(**sample)

        softcap = self.text_cfg.final_logit_softcapping
        if softcap is not None:
            self.assertTrue(quant_out.abs().max().item() <= softcap + 1e-5)

    def test_as_export_module_flow(self):
        """Test the as_export_module flow for Circle export."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4ForCausalLMExportAdapter,
        )

        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)

        export_module = quantized.wrapped.as_export_module(mode="prefill")

        # as_export_module returns Gemma4ForCausalLMExportAdapter
        self.assertIsInstance(export_module, Gemma4ForCausalLMExportAdapter)
        self.assertEqual(export_module.logits_to_keep, 0)

        # Build precomputed export inputs (simulating CPU runtime output)
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(self.batch_size, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(
                self.batch_size, self.seq_len, num_layers, ple_dim
            )

        attention_masks = {}
        position_embeddings = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(
                self.batch_size, 1, self.seq_len, self.seq_len
            )
            cos = torch.ones(self.batch_size, self.seq_len, head_dim)
            sin = torch.zeros(self.batch_size, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        with torch.no_grad():
            output = export_module(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_masks=attention_masks,
                position_embeddings=position_embeddings,
            )

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(
            output.shape,
            (self.batch_size, self.seq_len, self.text_cfg.vocab_size),
        )

    def test_as_export_module_decode_flow(self):
        """Test the as_export_module flow in decode mode (logits_to_keep=1)."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4ForCausalLMExportAdapter,
        )

        prepared = prepare(self.fp_model, self._ptq_config())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._text_sample())

        quantized = convert(prepared)

        export_module = quantized.wrapped.as_export_module(mode="decode")

        self.assertIsInstance(export_module, Gemma4ForCausalLMExportAdapter)
        self.assertEqual(export_module.logits_to_keep, 1)

        # Build precomputed export inputs for decode (seq_len=1)
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        decode_seq_len = 1
        inputs_embeds = torch.randn(self.batch_size, decode_seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(
                self.batch_size, decode_seq_len, num_layers, ple_dim
            )

        attention_masks = {}
        position_embeddings = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(
                self.batch_size, 1, decode_seq_len, decode_seq_len
            )
            cos = torch.ones(self.batch_size, decode_seq_len, head_dim)
            sin = torch.zeros(self.batch_size, decode_seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        with torch.no_grad():
            output = export_module(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                attention_masks=attention_masks,
                position_embeddings=position_embeddings,
            )

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(
            output.shape,
            (self.batch_size, 1, self.text_cfg.vocab_size),
        )


if __name__ == "__main__":
    unittest.main()

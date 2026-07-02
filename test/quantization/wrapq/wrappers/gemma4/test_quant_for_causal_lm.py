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

"""Smoke tests for QuantGemma4ForCausalLM wrapper."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.quant_for_causal_lm import (
    QuantGemma4ForCausalLM,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
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


def _make_gemma4_for_causal_lm():
    """Create a tiny Gemma4ForCausalLM for testing."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = _make_text_config()
    return Gemma4ForCausalLM(config).eval()


# ---------------------------------------------------------------------------
# Smoke tests for QuantGemma4ForCausalLM forward
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4ForCausalLMSmoke(unittest.TestCase):
    """Exercise QuantGemma4ForCausalLM wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4ForCausalLM modules."""
        torch.manual_seed(2026)
        self.fp_model = _make_gemma4_for_causal_lm()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        self.config = self.fp_model.config
        self.seq_len = 8
        self.batch_size = 1
        self.qcfg = PTQConfig()

    def _text_only_sample(self):
        """Create a text-only sample."""
        return {
            "input_ids": torch.randint(
                0, self.config.vocab_size, (self.batch_size, self.seq_len)
            ),
        }

    def test_prepare_returns_correct_wrapper(self):
        """prepare() should return a PTQWrapper wrapping QuantGemma4ForCausalLM."""
        qmodel = prepare(self.fp_model, self.qcfg)
        self.assertIsInstance(qmodel, PTQWrapper)
        self.assertIsInstance(qmodel.wrapped, QuantGemma4ForCausalLM)

    def test_text_only_forward_is_finite(self):
        """Text-only forward through the wrapper should produce finite logits."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        # The wrapper returns logits directly (not a Gemma4CausalLMOutputWithPast).
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_output_shape(self):
        """Forward output should have shape (batch, seq_len, vocab_size)."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        self.assertEqual(
            out.shape,
            (self.batch_size, self.seq_len, self.config.vocab_size),
        )

    def test_logits_to_keep(self):
        """logits_to_keep=1 should produce logits only for the last position."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample, logits_to_keep=1)

        self.assertEqual(
            out.shape,
            (self.batch_size, 1, self.config.vocab_size),
        )

    def test_logit_softcapping_applied(self):
        """When final_logit_softcapping is set, logits should be bounded."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        with torch.no_grad():
            out = quantized(**sample)

        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            self.assertTrue(out.abs().max().item() <= softcap + 1e-5)

    def test_prepare_convert_flow(self):
        """Full prepare → calibrate → convert flow should succeed."""
        qmodel = prepare(self.fp_model, self.qcfg)
        self.assertIsInstance(qmodel, PTQWrapper)
        self.assertIsInstance(qmodel.wrapped, QuantGemma4ForCausalLM)

        sample = self._text_only_sample()
        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            out = quantized(**sample)

        self.assertEqual(
            out.shape,
            (self.batch_size, self.seq_len, self.config.vocab_size),
        )
        self.assertTrue(torch.isfinite(out).all())

    def test_observers_calibrated(self):
        """After convert, the softcapping and logits observers should have qparams."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        wrapped = getattr(quantized, "wrapped", quantized)
        for obs in wrapped._all_observers():
            self.assertTrue(
                obs.has_qparams,
                f"Observer {obs.name} was not calibrated",
            )

    def test_export_module_prefill(self):
        """as_export_module(mode='prefill') should return an adapter with logits_to_keep=0."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        wrapped = getattr(quantized, "wrapped", quantized)
        export_module = wrapped.as_export_module(mode="prefill").eval()

        self.assertEqual(export_module.logits_to_keep, 0)

    def test_export_module_decode(self):
        """as_export_module(mode='decode') should return an adapter with logits_to_keep=1."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        wrapped = getattr(quantized, "wrapped", quantized)
        export_module = wrapped.as_export_module(mode="decode").eval()

        self.assertEqual(export_module.logits_to_keep, 1)

    def test_export_forward_produces_logits(self):
        """The export adapter forward should produce finite logits."""
        qmodel = prepare(self.fp_model, self.qcfg).eval()
        sample = self._text_only_sample()

        with torch.no_grad():
            for _ in range(3):
                qmodel(**sample)

        quantized = convert(qmodel)
        wrapped = getattr(quantized, "wrapped", quantized)
        export_module = wrapped.as_export_module(mode="prefill").eval()

        # Build precomputed inputs (simulating CPU runtime).
        hidden_size = int(self.config.hidden_size)
        head_dim = int(self.config.head_dim)
        num_layers = int(self.config.num_hidden_layers)
        ple_dim = int(getattr(self.config, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.config.layer_types)

        ex_inputs_embeds = torch.randn(self.batch_size, self.seq_len, hidden_size)
        ex_per_layer_inputs = None
        if ple_dim > 0:
            ex_per_layer_inputs = torch.randn(
                self.batch_size, self.seq_len, num_layers, ple_dim
            )

        ex_attention_masks = {}
        ex_position_embeddings = {}
        for layer_type in layer_types:
            ex_attention_masks[layer_type] = torch.zeros(
                self.batch_size, 1, self.seq_len, self.seq_len
            )
            cos = torch.ones(self.batch_size, self.seq_len, head_dim)
            sin = torch.zeros(self.batch_size, self.seq_len, head_dim)
            ex_position_embeddings[layer_type] = (cos, sin)

        with torch.no_grad():
            export_out = export_module(
                inputs_embeds=ex_inputs_embeds,
                per_layer_inputs=ex_per_layer_inputs,
                attention_masks=ex_attention_masks,
                position_embeddings=ex_position_embeddings,
            )

        self.assertEqual(
            export_out.shape,
            (self.batch_size, self.seq_len, self.config.vocab_size),
        )
        self.assertTrue(torch.isfinite(export_out).all())


if __name__ == "__main__":
    unittest.main()

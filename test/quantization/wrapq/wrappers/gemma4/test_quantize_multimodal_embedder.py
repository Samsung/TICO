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

"""Smoke tests for Gemma4 multimodal embedder prepare-calibrate-convert flow."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4MultimodalEmbedder,
        )
    except Exception:
        return False
    return True


def _make_vision_config():
    """Create a tiny Gemma4 vision config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=4,
        position_embedding_size=8,
        pooling_kernel_size=2,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        standardize=True,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


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
class TestGemma4MultimodalEmbedderSmoke(unittest.TestCase):
    """Exercise Gemma4 multimodal embedder wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4 multimodal embedder modules."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

        self.vision_cfg = _make_vision_config()
        self.text_cfg = _make_text_config()
        self.fp_embedder = Gemma4MultimodalEmbedder(
            self.vision_cfg, self.text_cfg
        ).eval()
        self.fp_ref = copy.deepcopy(self.fp_embedder).eval()
        self.seq_len = 16
        self.multimodal_hidden_size = self.fp_embedder.multimodal_hidden_size
        self.text_hidden_size = self.fp_embedder.text_hidden_size

    def _sample(self):
        """Create one synthetic Gemma4 multimodal embedder sample."""
        batch_size = 1
        return {
            "inputs_embeds": torch.randn(
                batch_size, self.seq_len, self.multimodal_hidden_size
            ),
        }

    def test_no_quant_multimodal_embedder_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_multimodal_embedder import (
            QuantGemma4MultimodalEmbedder,
        )

        wrapped = QuantGemma4MultimodalEmbedder(
            self.fp_embedder, qcfg=PTQConfig()
        ).eval()
        sample = self._sample()

        with torch.no_grad():
            quant_out = wrapped(**sample)
            fp_out = self.fp_ref(**sample)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_prepare_convert_multimodal_embedder_flow(self):
        """Quantize Gemma4 multimodal embedder and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_multimodal_embedder import (
            QuantGemma4MultimodalEmbedder,
        )

        prepared = prepare(self.fp_embedder, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4MultimodalEmbedder)

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._sample()
        with torch.no_grad():
            quant_out = quantized(**sample)
            fp_out = self.fp_ref(**sample)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_as_export_module_flow(self):
        """Test the as_export_module flow for Circle export."""
        prepared = prepare(self.fp_embedder, PTQConfig())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample())

        quantized = convert(prepared)

        export_module = quantized.wrapped.as_export_module(mode="prefill")

        # Verify export module forward works
        sample = self._sample()
        with torch.no_grad():
            out = export_module(**sample)

        self.assertEqual(out.shape, (1, self.seq_len, self.text_hidden_size))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()

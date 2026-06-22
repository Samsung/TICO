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

"""Smoke tests for Gemma4 text scaled word embedding prepare-calibrate-convert flow."""

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
            Gemma4TextScaledWordEmbedding,
        )
    except Exception:
        return False
    return True


def _make_embedding(vocab_size=100, embedding_dim=32, padding_idx=0, embed_scale=0.125):
    """Create a tiny Gemma4TextScaledWordEmbedding for testing."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    return Gemma4TextScaledWordEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        embed_scale=embed_scale,
    ).eval()


def _sample_input_ids(batch_size=1, seq_len=16, vocab_size=100):
    """Create synthetic input IDs."""
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4TextScaledWordEmbeddingSmoke(unittest.TestCase):
    """Exercise Gemma4 text scaled word embedding wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4TextScaledWordEmbedding modules."""
        torch.manual_seed(2026)
        self.vocab_size = 100
        self.embedding_dim = 32
        self.padding_idx = 0
        self.embed_scale = 0.125
        self.seq_len = 16
        self.fp_embedding = _make_embedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
            embed_scale=self.embed_scale,
        )
        self.fp_ref = copy.deepcopy(self.fp_embedding).eval()

    def _sample(self):
        """Create one synthetic sample."""
        return _sample_input_ids(
            batch_size=1, seq_len=self.seq_len, vocab_size=self.vocab_size
        )

    def test_no_quant_embedding_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_scaled_word_embedding import (
            QuantGemma4TextScaledWordEmbedding,
        )

        wrapped = QuantGemma4TextScaledWordEmbedding(
            self.fp_embedding, qcfg=PTQConfig()
        ).eval()
        input_ids = self._sample()

        with torch.no_grad():
            quant_out = wrapped(input_ids)
            fp_out = self.fp_ref(input_ids)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_prepare_convert_embedding_flow(self):
        """Quantize Gemma4 text scaled word embedding and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_scaled_word_embedding import (
            QuantGemma4TextScaledWordEmbedding,
        )

        prepared = prepare(self.fp_embedding, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4TextScaledWordEmbedding)

        with torch.no_grad():
            for _ in range(3):
                prepared(self._sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        input_ids = self._sample()
        with torch.no_grad():
            quant_out = quantized(input_ids)
            fp_out = self.fp_ref(input_ids)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_as_export_module_flow(self):
        """Test the as_export_module flow for Circle export."""
        prepared = prepare(self.fp_embedding, PTQConfig())

        with torch.no_grad():
            for _ in range(3):
                prepared(self._sample())

        quantized = convert(prepared)

        export_module = quantized.wrapped.as_export_module(mode="prefill")

        # Verify export module forward works
        input_ids = self._sample()
        with torch.no_grad():
            out = export_module(input_ids)

        self.assertEqual(out.shape, (1, self.seq_len, self.embedding_dim))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()

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

"""Unit tests for the Gemma4 text scaled word embedding PTQ wrapper."""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.quant_text_scaled_word_embedding import (
    QuantGemma4TextScaledWordEmbedding,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


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


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4TextScaledWordEmbedding(unittest.TestCase):
    """Validate Gemma4 text scaled word embedding wrapper behavior."""

    def setUp(self):
        """Create deterministic inputs."""
        torch.manual_seed(2026)
        self.vocab_size = 100
        self.embedding_dim = 32
        self.padding_idx = 0
        self.embed_scale = 0.125
        self.seq_len = 16
        self.batch_size = 1

    def _sample_inputs(self, batch_size=None, seq_len=None):
        """Create synthetic input IDs."""
        batch_size = batch_size or self.batch_size
        seq_len = seq_len or self.seq_len
        return _sample_input_ids(batch_size, seq_len, self.vocab_size)

    # ------------------------------------------------------------------
    # NO_QUANT mode
    # ------------------------------------------------------------------

    def test_no_quant_forward_matches_fp(self):
        """In NO_QUANT mode the wrapper should match the floating-point module."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        self.assertIs(q_embedding._mode, Mode.NO_QUANT)

        input_ids = self._sample_inputs()
        with torch.no_grad():
            q_out = q_embedding(input_ids)
            fp_out = fp_embedding(input_ids)

        # Shapes must match
        self.assertEqual(q_out.shape, fp_out.shape)

        # Values must be close (the wrapper delegates to the original module)
        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_no_quant_output_shape(self):
        """Check that the output has the expected static shape."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        input_ids = self._sample_inputs()
        with torch.no_grad():
            output = q_embedding(input_ids)

        expected_shape = (self.batch_size, self.seq_len, self.embedding_dim)
        self.assertEqual(output.shape, expected_shape)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def test_mode_transitions(self):
        """Check the calibration lifecycle: NO_QUANT → CALIB → QUANT."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        self.assertIs(q_embedding._mode, Mode.NO_QUANT)

        q_embedding.enable_calibration()
        self.assertIs(q_embedding._mode, Mode.CALIB)

        input_ids = self._sample_inputs()
        with torch.no_grad():
            _ = q_embedding(input_ids)

        q_embedding.freeze_qparams()
        self.assertIs(q_embedding._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns all 4 observers."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        all_obs = list(q_embedding._all_observers())
        self.assertEqual(len(all_obs), 4)
        self.assertIs(all_obs[0], q_embedding.obs_weight)
        self.assertIs(all_obs[1], q_embedding.obs_embedding)
        self.assertIs(all_obs[2], q_embedding.obs_embed_scale)
        self.assertIs(all_obs[3], q_embedding.obs_act_out)

    # ------------------------------------------------------------------
    # Calibration and fake quantization
    # ------------------------------------------------------------------

    def test_weight_is_observed_in_calib_mode(self):
        """In CALIB mode the weight should be observed through obs_weight."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        with mock.patch.object(
            q_embedding.obs_weight,
            "collect",
            wraps=q_embedding.obs_weight.collect,
        ) as mock_collect:
            # Weight is collected in enable_calibration
            q_embedding.enable_calibration()
            mock_collect.assert_called_once()

    def test_embed_scale_is_observed_in_calib_mode(self):
        """In CALIB mode the embed_scale should be observed."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        with mock.patch.object(
            q_embedding.obs_embed_scale,
            "collect",
            wraps=q_embedding.obs_embed_scale.collect,
        ) as mock_collect:
            # embed_scale is collected in enable_calibration
            q_embedding.enable_calibration()
            mock_collect.assert_called_once()

    def test_output_is_fake_quantized_in_quant_mode(self):
        """In QUANT mode the output should be fake-quantized through obs_act_out."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()
        q_embedding.enable_calibration()

        input_ids = self._sample_inputs()
        with torch.no_grad():
            _ = q_embedding(input_ids)
        q_embedding.freeze_qparams()

        with mock.patch.object(
            q_embedding.obs_act_out,
            "fake_quant",
            wraps=q_embedding.obs_act_out.fake_quant,
        ) as mock_fq:
            with torch.no_grad():
                _ = q_embedding(input_ids)
            mock_fq.assert_called()

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()
        q_embedding.enable_calibration()

        input_ids = self._sample_inputs()
        with torch.no_grad():
            _ = q_embedding(input_ids)
        q_embedding.freeze_qparams()

        with torch.no_grad():
            output = q_embedding(input_ids)

        expected_shape = (self.batch_size, self.seq_len, self.embedding_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    # ------------------------------------------------------------------
    # dtype override
    # ------------------------------------------------------------------

    def test_dtype_override(self):
        """Check that PTQConfig overrides propagate to observers."""
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "weight": {
                    "dtype": DType.uint(4),
                    "qscheme": QScheme.PER_CHANNEL_ASYMM,
                },
                "act_out": {"dtype": DType.uint(4)},
            },
        )
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding, qcfg=cfg).eval()

        self.assertIsInstance(q_embedding.obs_weight, AffineObserverBase)
        self.assertIsInstance(q_embedding.obs_act_out, AffineObserverBase)
        self.assertEqual(q_embedding.obs_weight.dtype, DType.uint(4))
        self.assertEqual(q_embedding.obs_act_out.dtype, DType.uint(4))

    def test_weight_uses_per_channel_asymm_by_default(self):
        """Check that weight observer uses PER_CHANNEL_ASYMM by default."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        self.assertEqual(q_embedding.obs_weight.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(q_embedding.obs_weight.channel_axis, 0)

    # ------------------------------------------------------------------
    # as_export_module
    # ------------------------------------------------------------------

    def test_as_export_module_requires_quant_mode(self):
        """as_export_module should assert that mode is QUANT."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()

        # Should fail in NO_QUANT mode
        with self.assertRaises(AssertionError):
            q_embedding.as_export_module(mode="prefill")

    def test_as_export_module_returns_self(self):
        """as_export_module should return self."""
        fp_embedding = _make_embedding()
        q_embedding = QuantGemma4TextScaledWordEmbedding(fp_embedding).eval()
        q_embedding.enable_calibration()

        input_ids = self._sample_inputs()
        with torch.no_grad():
            _ = q_embedding(input_ids)
        q_embedding.freeze_qparams()

        export_module = q_embedding.as_export_module(mode="prefill")
        self.assertIs(export_module, q_embedding)


if __name__ == "__main__":
    unittest.main()

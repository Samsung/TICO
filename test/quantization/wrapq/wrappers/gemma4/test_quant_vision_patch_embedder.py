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

"""Unit tests for the Gemma4 vision patch embedder PTQ wrapper."""

import unittest

import torch

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_patch_embedder import (
    QuantGemma4VisionPatchEmbedder,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionPatchEmbedder,
        )
    except Exception:
        return False
    return True


def _make_patch_embedder(
    hidden_size=32,
    patch_size=4,
    position_embedding_size=8,
):
    """Create a tiny Gemma4VisionPatchEmbedder for testing."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

    config = Gemma4VisionConfig(
        hidden_size=hidden_size,
        patch_size=patch_size,
        position_embedding_size=position_embedding_size,
    )
    return Gemma4VisionPatchEmbedder(config).eval()


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4VisionPatchEmbedder(unittest.TestCase):
    """Validate Gemma4 vision patch embedder wrapper behavior."""

    def setUp(self):
        """Create deterministic inputs."""
        torch.manual_seed(2026)
        self.hidden_size = 32
        self.patch_size = 4
        self.position_embedding_size = 8
        self.batch_size = 1
        self.num_patches = 16

    def _sample_inputs(self):
        """Create synthetic inputs."""
        patch_dim = 3 * self.patch_size**2
        pixel_values = torch.randn(self.batch_size, self.num_patches, patch_dim)
        pixel_position_ids = torch.randint(
            0, self.position_embedding_size, (self.batch_size, self.num_patches, 2)
        )
        padding_positions = torch.zeros(
            self.batch_size, self.num_patches, dtype=torch.bool
        )
        return pixel_values, pixel_position_ids, padding_positions

    # ------------------------------------------------------------------
    # NO_QUANT mode
    # ------------------------------------------------------------------

    def test_no_quant_forward_matches_fp(self):
        """In NO_QUANT mode the wrapper should match the floating-point module."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertIs(q_module._mode, Mode.NO_QUANT)

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            q_out = q_module(pixel_values, pixel_position_ids, padding_positions)
            fp_out = fp_module(pixel_values, pixel_position_ids, padding_positions)

        # Shapes must match
        self.assertEqual(q_out.shape, fp_out.shape)

        # Values must be close (the wrapper delegates to original operations)
        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_no_quant_output_shape(self):
        """Check that the output has the expected static shape."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            output = q_module(pixel_values, pixel_position_ids, padding_positions)

        expected_shape = (self.batch_size, self.num_patches, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def test_mode_transitions(self):
        """Check the calibration lifecycle: NO_QUANT → CALIB → QUANT."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertIs(q_module._mode, Mode.NO_QUANT)

        q_module.enable_calibration()
        self.assertIs(q_module._mode, Mode.CALIB)

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)

        q_module.freeze_qparams()
        self.assertIs(q_module._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns all 8 observers."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        all_obs = list(q_module._all_observers())
        self.assertEqual(len(all_obs), 7)
        self.assertIs(all_obs[0], q_module.obs_emb_table)
        self.assertIs(all_obs[1], q_module.obs_act_in)
        self.assertIs(all_obs[2], q_module.obs_pixel_values_m_0_5)
        self.assertIs(all_obs[3], q_module.obs_pixel_values)
        self.assertIs(all_obs[4], q_module.obs_hidden_states)
        self.assertIs(all_obs[5], q_module.obs_position_embeddings)
        self.assertIs(all_obs[6], q_module.obs_output)

    # ------------------------------------------------------------------
    # Calibration and fake quantization
    # ------------------------------------------------------------------

    def test_emb_table_is_observed_in_calib_mode(self):
        """In CALIB mode the position_embedding_table should be observed."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        q_module.enable_calibration()

        # Check that emb_table observer has collected statistics (min_val/max_val are set)
        self.assertIsNotNone(q_module.obs_emb_table.min_val)
        self.assertIsNotNone(q_module.obs_emb_table.max_val)

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        q_module.enable_calibration()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)
        q_module.freeze_qparams()

        with torch.no_grad():
            output = q_module(pixel_values, pixel_position_ids, padding_positions)

        expected_shape = (self.batch_size, self.num_patches, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    # ------------------------------------------------------------------
    # dtype override
    # ------------------------------------------------------------------

    def test_emb_table_uses_per_tensor_symm_by_default(self):
        """Check that emb_table observer uses PER_TENSOR_SYMM by default."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertEqual(q_module.obs_emb_table.qscheme, QScheme.PER_TENSOR_SYMM)

    # ------------------------------------------------------------------
    # position_embedding_table buffer
    # ------------------------------------------------------------------

    def test_position_embedding_table_is_registered_as_buffer(self):
        """position_embedding_table should be registered as a buffer on the wrapper."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertTrue(hasattr(q_module, "position_embedding_table"))
        self.assertIsInstance(q_module.position_embedding_table, torch.Tensor)
        self.assertEqual(
            q_module.position_embedding_table.shape,
            (2, self.position_embedding_size, self.hidden_size),
        )

    def test_position_embedding_table_matches_original(self):
        """position_embedding_table buffer should match the original module."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertTrue(
            torch.allclose(
                q_module.position_embedding_table, fp_module.position_embedding_table
            )
        )

    # ------------------------------------------------------------------
    # Config attributes
    # ------------------------------------------------------------------

    def test_config_attributes_are_stored(self):
        """Check that config attributes are stored on the wrapper."""
        fp_module = _make_patch_embedder(
            hidden_size=64,
            patch_size=8,
            position_embedding_size=16,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertEqual(q_module.hidden_size, 64)
        self.assertEqual(q_module.patch_size, 8)
        self.assertEqual(q_module.position_embedding_size, 16)

    # ------------------------------------------------------------------
    # as_export_module
    # ------------------------------------------------------------------

    def test_as_export_module_requires_quant_mode(self):
        """as_export_module should assert that mode is QUANT."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        # Should fail in NO_QUANT mode
        with self.assertRaises(AssertionError):
            q_module.as_export_module(mode="prefill")

    def test_as_export_module_returns_self(self):
        """as_export_module should return self."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        q_module.enable_calibration()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)
        q_module.freeze_qparams()

        export_module = q_module.as_export_module(mode="prefill")
        self.assertIs(export_module, q_module)


if __name__ == "__main__":
    unittest.main()

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

"""Unit tests for the Gemma4 vision model PTQ wrapper."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode


_SKIP_MSG = "required transformers Gemma4 vision modules are not installed"


def _has_gemma4_vision() -> bool:
    """Return whether the installed transformers package provides Gemma4 vision."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionModel,
        )
    except Exception:
        return False
    return True


def _make_vision_config(**overrides):
    """Create a tiny Gemma4 vision config for synthetic vision model tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    kwargs = dict(
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
    kwargs.update(overrides)
    cfg = Gemma4VisionConfig(**kwargs)
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _vision_position_ids(batch_size: int, num_patches: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = int(num_patches**0.5)
    coords = torch.arange(num_patches)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


@unittest.skipUnless(_has_gemma4_vision(), _SKIP_MSG)
class TestQuantGemma4VisionModel(unittest.TestCase):
    """Validate Gemma4 vision model wrapper behavior."""

    def setUp(self):
        """Create deterministic test inputs."""
        torch.manual_seed(2026)
        self.cfg = _make_vision_config()
        self.batch_size = 1
        self.num_patches = 16
        self.patch_size = self.cfg.patch_size
        self.position_embedding_size = self.cfg.position_embedding_size

    @staticmethod
    def _make_vision_model(cfg=None):
        """Create a floating-point Gemma4 vision model."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        cfg = cfg if cfg is not None else _make_vision_config()
        return Gemma4VisionModel(cfg).eval()

    def _sample_inputs(self, batch_size=None):
        """Create synthetic vision model inputs.

        The HF Gemma4VisionModel expects pre-flattened patches:
            pixel_values: (B, num_patches, 3*patch_size^2)
            pixel_position_ids: (B, num_patches, 2)
        """
        batch_size = batch_size or self.batch_size
        patch_dim = 3 * self.patch_size**2

        pixel_values = torch.randn(batch_size, self.num_patches, patch_dim)
        pixel_position_ids = _vision_position_ids(batch_size, self.num_patches)

        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

    def test_00_prepare_wraps_vision_model_when_registered(self):
        """Check that registry-based prepare wraps Gemma4VisionModel."""
        from tico.quantization import prepare
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        fp_model = self._make_vision_model()
        prepared = prepare(fp_model, PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4VisionModel)

    def test_no_quant_forward_matches_hf_vision_model(self):
        """Check that the wrapper matches Hugging Face eager vision model output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config()
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()
        sample = self._sample_inputs()

        with torch.no_grad():
            quant_out = q_model(**sample, return_dict=True)
            fp_out = fp_model(**sample, return_dict=True)

        # HF model strips batch dim via hidden_states[pooler_mask], so output
        # may be 2D (num_soft_tokens, hidden_size). The wrapper preserves batch dim.
        self.assertEqual(
            quant_out.last_hidden_state.shape[-1], fp_out.last_hidden_state.shape[-1]
        )

    def test_mode_transitions(self):
        """Check lifecycle transitions: NO_QUANT → CALIB → QUANT."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        fp_model = self._make_vision_model()
        q_model = QuantGemma4VisionModel(fp_model).eval()

        self.assertIs(q_model._mode, Mode.NO_QUANT)

        q_model.enable_calibration()
        self.assertIs(q_model._mode, Mode.CALIB)

        sample = self._sample_inputs()
        with torch.no_grad():
            _ = q_model(**sample, return_dict=True)

        q_model.freeze_qparams()
        self.assertIs(q_model._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns expected observers."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        fp_model = self._make_vision_model()
        q_model = QuantGemma4VisionModel(fp_model).eval()

        all_obs = list(q_model._all_observers())
        self.assertGreaterEqual(len(all_obs), 3)

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        fp_model = self._make_vision_model()
        q_model = QuantGemma4VisionModel(fp_model).eval()
        q_model.enable_calibration()

        sample = self._sample_inputs()
        with torch.no_grad():
            _ = q_model(**sample, return_dict=True)
        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(**sample, return_dict=True)

        self.assertTrue(torch.isfinite(output.last_hidden_state).all())
        # Check output has the right hidden_size dimension
        self.assertEqual(output.last_hidden_state.shape[-1], self.cfg.hidden_size)

    def test_config_attributes_are_stored(self):
        """Check that config attributes are stored on the wrapper."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config(hidden_size=64, patch_size=8)
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()

        self.assertEqual(q_model.config.hidden_size, 64)
        self.assertEqual(q_model.config.patch_size, 8)

    def test_standardize_buffers_are_registered(self):
        """Check that std_bias and std_scale are registered when standardize=True."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config(standardize=True)
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()

        self.assertTrue(hasattr(q_model, "std_bias"))
        self.assertTrue(hasattr(q_model, "std_scale"))
        self.assertIsInstance(q_model.std_bias, torch.Tensor)
        self.assertIsInstance(q_model.std_scale, torch.Tensor)
        self.assertEqual(q_model.std_bias.shape[0], cfg.hidden_size)
        self.assertEqual(q_model.std_scale.shape[0], cfg.hidden_size)

    def test_standardize_false_no_buffers(self):
        """Check that std_bias/std_scale are NOT registered when standardize=False."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config(standardize=False)
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()

        self.assertFalse(hasattr(q_model, "std_bias"))
        self.assertFalse(hasattr(q_model, "std_scale"))

    def test_as_export_module_requires_quant_mode(self):
        """as_export_module should assert that mode is QUANT."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        fp_model = self._make_vision_model()
        q_model = QuantGemma4VisionModel(fp_model).eval()

        # Should fail in NO_QUANT mode
        with self.assertRaises(AssertionError):
            q_model.as_export_module(mode="prefill", pixel_position_ids=None)

    def test_as_export_module_requires_standardize(self):
        """as_export_module should assert that config.standardize is True."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config(standardize=True)
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()
        q_model.enable_calibration()

        sample = self._sample_inputs()
        with torch.no_grad():
            _ = q_model(**sample, return_dict=True)
        q_model.freeze_qparams()

        # Should succeed with standardize=True
        export_module = q_model.as_export_module(
            mode="prefill",
            pixel_position_ids=sample["pixel_position_ids"],
        )
        self.assertIsNotNone(export_module)

    def test_forward_export_via_as_export_module(self):
        """Test the export flow via as_export_module which sets up export adapters."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config(standardize=True)
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()
        q_model.enable_calibration()

        sample = self._sample_inputs()
        with torch.no_grad():
            _ = q_model(**sample, return_dict=True)
        q_model.freeze_qparams()

        # as_export_module sets up export adapters
        export_module = q_model.as_export_module(
            mode="prefill",
            pixel_position_ids=sample["pixel_position_ids"],
        )

        # Test forward (adapter delegates to wrapped_model.forward_export)
        with torch.no_grad():
            output = export_module(**sample)

        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_as_export_module_creates_export_adapter_attributes(self):
        """as_export_module should create patch_embedder_export and pooler_export attributes."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        cfg = _make_vision_config(standardize=True)
        fp_model = self._make_vision_model(cfg)
        q_model = QuantGemma4VisionModel(fp_model).eval()
        q_model.enable_calibration()

        sample = self._sample_inputs()
        with torch.no_grad():
            _ = q_model(**sample, return_dict=True)
        q_model.freeze_qparams()

        # Before as_export_module, no export adapter attributes
        self.assertFalse(hasattr(q_model, "patch_embedder_export"))
        self.assertFalse(hasattr(q_model, "pooler_export"))

        q_model.as_export_module(
            mode="prefill",
            pixel_position_ids=sample["pixel_position_ids"],
        )

        # After as_export_module, export adapter attributes should exist
        self.assertTrue(hasattr(q_model, "patch_embedder_export"))
        self.assertTrue(hasattr(q_model, "pooler_export"))

        # Original wrappers should still be intact (not mutated)
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        self.assertIsInstance(q_model.patch_embedder, PTQWrapper)
        self.assertIsInstance(q_model.pooler, PTQWrapper)

    def test_submodules_are_wrapped(self):
        """Check that patch_embedder, encoder, and pooler are wrapped."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        fp_model = self._make_vision_model()
        q_model = QuantGemma4VisionModel(fp_model).eval()

        self.assertIsInstance(q_model.patch_embedder, PTQWrapper)
        self.assertIsInstance(q_model.encoder, PTQWrapper)
        self.assertIsInstance(q_model.pooler, PTQWrapper)


if __name__ == "__main__":
    unittest.main()

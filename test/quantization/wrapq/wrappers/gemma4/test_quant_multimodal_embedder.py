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

"""Unit tests for the Gemma4 multimodal embedder PTQ wrapper."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.quant_multimodal_embedder import (
    QuantGemma4MultimodalEmbedder,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


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


def _make_vision_config(**overrides):
    """Create a tiny Gemma4 vision config for synthetic tests."""
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


def _make_text_config(**overrides):
    """Create a tiny Gemma4 text config for synthetic tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    kwargs = dict(
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
    kwargs.update(overrides)
    cfg = Gemma4TextConfig(**kwargs)
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _make_multimodal_embedder(vision_cfg=None, text_cfg=None):
    """Create a tiny Gemma4MultimodalEmbedder for testing."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

    vision_cfg = vision_cfg if vision_cfg is not None else _make_vision_config()
    text_cfg = text_cfg if text_cfg is not None else _make_text_config()
    return Gemma4MultimodalEmbedder(vision_cfg, text_cfg).eval()


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4MultimodalEmbedder(unittest.TestCase):
    """Validate Gemma4 multimodal embedder wrapper behavior."""

    def setUp(self):
        """Create deterministic inputs."""
        torch.manual_seed(2026)
        self.vision_cfg = _make_vision_config()
        self.text_cfg = _make_text_config()
        self.batch_size = 1
        self.seq_len = 16
        self.multimodal_hidden_size = self.vision_cfg.hidden_size
        self.text_hidden_size = self.text_cfg.hidden_size

    def _sample_inputs(self):
        """Create synthetic inputs."""
        inputs_embeds = torch.randn(
            self.batch_size, self.seq_len, self.multimodal_hidden_size
        )
        return (inputs_embeds,)

    # ------------------------------------------------------------------
    # NO_QUANT mode
    # ------------------------------------------------------------------

    def test_no_quant_forward_matches_fp(self):
        """In NO_QUANT mode the wrapper should match the floating-point module."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()

        self.assertIs(q_module._mode, Mode.NO_QUANT)

        inputs = self._sample_inputs()
        with torch.no_grad():
            q_out = q_module(*inputs)
            fp_out = fp_module(*inputs)

        # Shapes must match
        self.assertEqual(q_out.shape, fp_out.shape)

        # Values must be close (the wrapper delegates to original operations)
        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_no_quant_output_shape(self):
        """Check that the output has the expected shape (B, seq_len, text_hidden_size)."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()

        inputs = self._sample_inputs()
        with torch.no_grad():
            output = q_module(*inputs)

        expected_shape = (self.batch_size, self.seq_len, self.text_hidden_size)
        self.assertEqual(output.shape, expected_shape)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def test_mode_transitions(self):
        """Check the calibration lifecycle: NO_QUANT → CALIB → QUANT."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()

        self.assertIs(q_module._mode, Mode.NO_QUANT)

        q_module.enable_calibration()
        self.assertIs(q_module._mode, Mode.CALIB)

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(*inputs)

        q_module.freeze_qparams()
        self.assertIs(q_module._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns no direct observers (delegated to sub-wrappers)."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()

        all_obs = list(q_module._all_observers())
        # The multimodal embedder has no own observers; quantization is
        # handled by the sub-wrappers (QuantGemma4RMSNorm, QuantLinear).
        self.assertEqual(len(all_obs), 0)

    # ------------------------------------------------------------------
    # Calibration and fake quantization
    # ------------------------------------------------------------------

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()
        q_module.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(*inputs)
        q_module.freeze_qparams()

        with torch.no_grad():
            output = q_module(*inputs)

        expected_shape = (self.batch_size, self.seq_len, self.text_hidden_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    # ------------------------------------------------------------------
    # Submodule wrapping
    # ------------------------------------------------------------------

    def test_submodules_are_wrapped(self):
        """Check that embedding_pre_projection_norm and embedding_projection are wrapped."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()

        self.assertIsInstance(q_module.embedding_pre_projection_norm, PTQWrapper)
        self.assertIsInstance(q_module.embedding_projection, PTQWrapper)

    # ------------------------------------------------------------------
    # Config attributes
    # ------------------------------------------------------------------

    def test_config_attributes_are_stored(self):
        """Check that multimodal_hidden_size and text_hidden_size are accessible."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()

        self.assertEqual(fp_module.multimodal_hidden_size, self.multimodal_hidden_size)
        self.assertEqual(fp_module.text_hidden_size, self.text_hidden_size)

    # ------------------------------------------------------------------
    # as_export_module
    # ------------------------------------------------------------------

    def test_as_export_module_returns_self(self):
        """as_export_module should return self (this wrapper is already exportable)."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()
        q_module.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(*inputs)
        q_module.freeze_qparams()

        export_module = q_module.as_export_module(mode="prefill")
        self.assertIs(export_module, q_module)

    def test_as_export_module_forward_matches_quant_forward(self):
        """Export module forward should produce the same output as quant forward."""
        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        q_module = QuantGemma4MultimodalEmbedder(fp_module).eval()
        q_module.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(*inputs)
        q_module.freeze_qparams()

        export_module = q_module.as_export_module(mode="prefill")

        with torch.no_grad():
            quant_out = q_module(*inputs)
            export_out = export_module(*inputs)

        self.assertTrue(torch.allclose(quant_out, export_out, atol=1e-6))

    # ------------------------------------------------------------------
    # prepare integration
    # ------------------------------------------------------------------

    def test_prepare_wraps_multimodal_embedder_when_registered(self):
        """Check that registry-based prepare wraps Gemma4MultimodalEmbedder."""
        from tico.quantization import prepare

        fp_module = _make_multimodal_embedder(self.vision_cfg, self.text_cfg)
        prepared = prepare(fp_module, PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4MultimodalEmbedder)


if __name__ == "__main__":
    unittest.main()

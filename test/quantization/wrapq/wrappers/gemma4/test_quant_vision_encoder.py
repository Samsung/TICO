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

"""Unit tests for the Gemma4 vision encoder PTQ wrapper."""

import unittest
from unittest import mock

import torch
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder import (
    QuantGemma4VisionEncoder,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


_SKIP_MSG = "required transformers Gemma4 vision modules are not installed"


def _has_gemma4_vision() -> bool:
    """Return whether the installed transformers package provides Gemma4 vision."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionEncoder,
        )
    except Exception:
        return False
    return True


def _make_vision_config(**overrides):
    """Create a tiny Gemma4 vision config for synthetic encoder tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    kwargs = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
    )
    kwargs.update(overrides)
    cfg = Gemma4VisionConfig(**kwargs)
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _make_encoder(cfg=None):
    """Create a tiny Gemma4 vision encoder in eval mode."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoder

    cfg = cfg if cfg is not None else _make_vision_config()
    return Gemma4VisionEncoder(cfg).eval()


def _vision_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = 4
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _vision_position_ids_with_padding(
    batch_size: int, seq_len: int, num_valid: int
) -> torch.Tensor:
    """Create pixel position ids where positions >= num_valid are padding (-1)."""
    ids = _vision_position_ids(batch_size, seq_len).clone()
    ids[:, num_valid:, :] = -1
    return ids


@unittest.skipUnless(_has_gemma4_vision(), _SKIP_MSG)
class TestQuantGemma4VisionEncoder(unittest.TestCase):
    """Validate Gemma4 vision encoder wrapper behavior."""

    def setUp(self):
        """Create deterministic test inputs."""
        torch.manual_seed(2026)
        self.cfg = _make_vision_config()
        self.hidden_size = self.cfg.hidden_size
        self.seq_len = 8
        self.batch_size = 1

    def _sample_inputs(self, batch_size=None, seq_len=None):
        """Create synthetic encoder inputs."""
        batch_size = batch_size or self.batch_size
        seq_len = seq_len or self.seq_len
        return {
            "inputs_embeds": torch.randn(batch_size, seq_len, self.hidden_size),
            "attention_mask": torch.ones(batch_size, seq_len),
            "pixel_position_ids": _vision_position_ids(batch_size, seq_len),
        }

    # ------------------------------------------------------------------
    # NO_QUANT mode
    # ------------------------------------------------------------------

    def test_no_quant_forward_matches_fp(self):
        """In NO_QUANT mode the wrapper should match the floating-point module."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        self.assertIs(q_encoder._mode, Mode.NO_QUANT)

        inputs = self._sample_inputs()
        with torch.no_grad():
            q_out = q_encoder(**inputs)
            fp_out = fp_encoder(**inputs)

        fp_hidden = fp_out.last_hidden_state
        self.assertEqual(q_out.shape, fp_hidden.shape)
        self.assertTrue(torch.allclose(q_out, fp_hidden, atol=1e-5, rtol=1e-5))

    def test_no_quant_output_shape(self):
        """Check that the encoder output has the expected static shape."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        inputs = self._sample_inputs()
        with torch.no_grad():
            out = q_encoder(**inputs)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def test_mode_transitions(self):
        """Check the calibration lifecycle: NO_QUANT → CALIB → QUANT."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        self.assertIs(q_encoder._mode, Mode.NO_QUANT)

        q_encoder.enable_calibration()
        self.assertIs(q_encoder._mode, Mode.CALIB)

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)

        q_encoder.freeze_qparams()
        self.assertIs(q_encoder._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns the encoder-level observers."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        all_obs = q_encoder._all_observers()
        self.assertEqual(
            len(all_obs), 5
        )  # act_in, attention_mask, position_cos, position_sin, encoder_out

    # ------------------------------------------------------------------
    # Calibration and fake quantization
    # ------------------------------------------------------------------

    def test_input_is_observed_in_calib_mode(self):
        """In CALIB mode the input should be observed through obs_act_in."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with mock.patch.object(
            q_encoder.obs_act_in,
            "collect",
            wraps=q_encoder.obs_act_in.collect,
        ) as mock_collect:
            with torch.no_grad():
                _ = q_encoder(**inputs)
            mock_collect.assert_called()

    def test_output_is_fake_quantized_in_quant_mode(self):
        """In QUANT mode the encoder output should be fake-quantized through obs_encoder_out."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        with mock.patch.object(
            q_encoder.obs_encoder_out,
            "fake_quant",
            wraps=q_encoder.obs_encoder_out.fake_quant,
        ) as mock_fq:
            with torch.no_grad():
                _ = q_encoder(**inputs)
            mock_fq.assert_called()

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        with torch.no_grad():
            out = q_encoder(**inputs)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_attention_mask_is_observed_in_calib_mode(self):
        """In CALIB mode the attention mask should be observed through obs_attention_mask."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with mock.patch.object(
            q_encoder.obs_attention_mask,
            "collect",
            wraps=q_encoder.obs_attention_mask.collect,
        ) as mock_collect:
            with torch.no_grad():
                _ = q_encoder(**inputs)
            mock_collect.assert_called()

    def test_position_cos_sin_are_observed_in_calib_mode(self):
        """In CALIB mode the position embeddings should be observed through obs_position_cos/sin."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with mock.patch.object(
            q_encoder.obs_position_cos,
            "collect",
            wraps=q_encoder.obs_position_cos.collect,
        ) as mock_cos_collect, mock.patch.object(
            q_encoder.obs_position_sin,
            "collect",
            wraps=q_encoder.obs_position_sin.collect,
        ) as mock_sin_collect:
            with torch.no_grad():
                _ = q_encoder(**inputs)
            mock_cos_collect.assert_called()
            mock_sin_collect.assert_called()

    def test_attention_mask_is_fake_quantized_in_quant_mode(self):
        """In QUANT mode the attention mask should be fake-quantized through obs_attention_mask."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        with mock.patch.object(
            q_encoder.obs_attention_mask,
            "fake_quant",
            wraps=q_encoder.obs_attention_mask.fake_quant,
        ) as mock_fq:
            with torch.no_grad():
                _ = q_encoder(**inputs)
            mock_fq.assert_called()

    def test_position_embeddings_are_fake_quantized_in_quant_mode(self):
        """In QUANT mode position embeddings should be fake-quantized through obs_position_cos/sin."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        with mock.patch.object(
            q_encoder.obs_position_cos,
            "fake_quant",
            wraps=q_encoder.obs_position_cos.fake_quant,
        ) as mock_cos_fq, mock.patch.object(
            q_encoder.obs_position_sin,
            "fake_quant",
            wraps=q_encoder.obs_position_sin.fake_quant,
        ) as mock_sin_fq:
            with torch.no_grad():
                _ = q_encoder(**inputs)
            mock_cos_fq.assert_called()
            mock_sin_fq.assert_called()

    # ------------------------------------------------------------------
    # dtype override
    # ------------------------------------------------------------------

    def test_dtype_override(self):
        """Check that PTQConfig overrides propagate to encoder observers."""
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "encoder_out": {"dtype": DType.uint(4)},
            },
        )
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder, qcfg=cfg).eval()

        self.assertIsInstance(q_encoder.obs_act_in, AffineObserverBase)
        self.assertIsInstance(q_encoder.obs_encoder_out, AffineObserverBase)
        self.assertEqual(q_encoder.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_encoder.obs_encoder_out.dtype, DType.uint(4))

    # ------------------------------------------------------------------
    # Precomputed RoPE lookup tables
    # ------------------------------------------------------------------

    def test_gather_position_embeddings_matches_hf_rotary(self):
        """Check that _gather_position_embeddings matches Gemma4VisionRotaryEmbedding.forward()."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4VisionRotaryEmbedding,
        )

        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        seq_len = 8
        pixel_position_ids = _vision_position_ids(1, seq_len)
        dummy_hidden = torch.zeros(1, seq_len, self.cfg.hidden_size)

        # Reference: HuggingFace rotary embedding
        rotary_emb = Gemma4VisionRotaryEmbedding(self.cfg)
        ref_cos, ref_sin = rotary_emb(dummy_hidden, pixel_position_ids)

        # Our implementation: gather from lookup tables
        our_cos, our_sin = q_encoder._gather_position_embeddings(
            pixel_position_ids, dtype=torch.float32
        )

        self.assertEqual(our_cos.shape, ref_cos.shape)
        self.assertEqual(our_sin.shape, ref_sin.shape)
        self.assertTrue(torch.allclose(our_cos, ref_cos, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(our_sin, ref_sin, atol=1e-5, rtol=1e-5))

    def test_gather_position_embeddings_with_padding(self):
        """Check that _gather_position_embeddings handles padding (-1) positions gracefully."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        seq_len = 8
        num_valid = 5
        pixel_position_ids = _vision_position_ids_with_padding(1, seq_len, num_valid)

        cos, sin = q_encoder._gather_position_embeddings(pixel_position_ids)
        self.assertEqual(cos.shape, (1, seq_len, self.cfg.head_dim))
        self.assertEqual(sin.shape, (1, seq_len, self.cfg.head_dim))
        # Padding positions (clamped to 0) should have same values as position 0
        self.assertTrue(
            torch.allclose(cos[:, num_valid:, :], cos[:, 0:1, :], atol=1e-6)
        )

    # ------------------------------------------------------------------
    # Bidirectional attention mask with padding
    # ------------------------------------------------------------------

    def test_make_bidirectional_mask_no_padding(self):
        """Check that _make_bidirectional_mask produces all-zeros mask when no padding."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        seq_len = 6
        pixel_position_ids = _vision_position_ids(1, seq_len)
        mask = q_encoder._make_bidirectional_mask(pixel_position_ids)

        self.assertEqual(mask.shape, (1, 1, seq_len, seq_len))
        self.assertTrue(torch.all(mask == 0.0))

    def test_make_bidirectional_mask_with_padding(self):
        """Check that _make_bidirectional_mask masks out padding positions."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        seq_len = 6
        num_valid = 4
        pixel_position_ids = _vision_position_ids_with_padding(1, seq_len, num_valid)
        mask = q_encoder._make_bidirectional_mask(pixel_position_ids)

        self.assertEqual(mask.shape, (1, 1, seq_len, seq_len))
        # Valid↔Valid block should be 0.0
        self.assertTrue(torch.all(mask[0, 0, :num_valid, :num_valid] == 0.0))
        # Padding rows should be masked
        self.assertTrue(torch.all(mask[0, 0, num_valid:, :] < 0))
        # Padding cols should be masked
        self.assertTrue(torch.all(mask[0, 0, :num_valid, num_valid:] < 0))

    def test_make_bidirectional_mask_fill_value(self):
        """Check that _make_bidirectional_mask uses qcfg.attention_mask_fill_value."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        seq_len = 4
        num_valid = 2
        pixel_position_ids = _vision_position_ids_with_padding(1, seq_len, num_valid)
        mask = q_encoder._make_bidirectional_mask(pixel_position_ids)

        # Fill value comes from PTQConfig, not config.attention_invalid_logits_value
        fill_value = float(q_encoder.qcfg.attention_mask_fill_value)
        self.assertAlmostEqual(
            mask[0, 0, num_valid, num_valid].item(), fill_value, places=5
        )

    # ------------------------------------------------------------------
    # Dynamic forward with padding
    # ------------------------------------------------------------------

    def test_dynamic_forward_with_padding_produces_valid_output(self):
        """Check that dynamic forward with padding produces valid finite output."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        seq_len = 8
        num_valid = 5
        pixel_position_ids = _vision_position_ids_with_padding(1, seq_len, num_valid)
        inputs_embeds = torch.randn(1, seq_len, self.hidden_size)

        with torch.no_grad():
            out = q_encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones(1, seq_len),
                pixel_position_ids=pixel_position_ids,
            )

        self.assertEqual(out.shape, (1, seq_len, self.hidden_size))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.norm(out[:, :num_valid, :]) > 0)

    # ------------------------------------------------------------------
    # forward_export
    # ------------------------------------------------------------------

    def test_forward_export_uses_template_buffers(self):
        """forward_export reads position_embeddings_cos/sin_template and attention_mask_template."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        # Register template buffers like as_export_module does
        seq_len = self.seq_len
        pixel_position_ids = _vision_position_ids(1, seq_len)
        cos, sin = q_encoder._gather_position_embeddings(pixel_position_ids)
        q_encoder.register_buffer("position_embeddings_cos_template", cos)
        q_encoder.register_buffer("position_embeddings_sin_template", sin)
        attention_mask = q_encoder._make_bidirectional_mask(pixel_position_ids)
        q_encoder.register_buffer("attention_mask_template", attention_mask)

        with torch.no_grad():
            out = q_encoder.forward_export(inputs["inputs_embeds"])

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, seq_len, self.hidden_size))
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_export_matches_forward(self):
        """forward_export should produce similar results to forward."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        pixel_position_ids = _vision_position_ids(1, self.seq_len)
        inputs = {
            "inputs_embeds": torch.randn(1, self.seq_len, self.hidden_size),
            "attention_mask": torch.ones(1, self.seq_len),
            "pixel_position_ids": pixel_position_ids,
        }
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        # Register template buffers for forward_export
        cos, sin = q_encoder._gather_position_embeddings(pixel_position_ids)
        q_encoder.register_buffer("position_embeddings_cos_template", cos)
        q_encoder.register_buffer("position_embeddings_sin_template", sin)
        attention_mask = q_encoder._make_bidirectional_mask(pixel_position_ids)
        q_encoder.register_buffer("attention_mask_template", attention_mask)

        with torch.no_grad():
            forward_out = q_encoder(**inputs)
            export_out = q_encoder.forward_export(inputs["inputs_embeds"])

        self.assertEqual(export_out.shape, forward_out.shape)
        self.assertTrue(
            torch.allclose(export_out, forward_out, atol=1e-4, rtol=1e-4),
            f"Max diff: {(export_out - forward_out).abs().max().item()}",
        )

    # ------------------------------------------------------------------
    # as_export_module
    # ------------------------------------------------------------------

    def test_as_export_module_requires_quant_mode(self):
        """as_export_module should assert that mode is QUANT."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()

        pixel_position_ids = _vision_position_ids(1, self.seq_len)

        # Should fail in NO_QUANT mode
        with self.assertRaises(AssertionError):
            q_encoder.as_export_module("prefill", pixel_position_ids=pixel_position_ids)

    def test_as_export_module_returns_adapter(self):
        """as_export_module should return a Gemma4VisionEncoderPrefillExportAdapter."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionEncoderPrefillExportAdapter,
        )

        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        pixel_position_ids = _vision_position_ids(1, self.seq_len)
        adapter = q_encoder.as_export_module(
            "prefill", pixel_position_ids=pixel_position_ids
        )
        self.assertIsInstance(adapter, Gemma4VisionEncoderPrefillExportAdapter)

    def test_as_export_module_registers_template_buffers(self):
        """as_export_module should register template buffers on the wrapper."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        pixel_position_ids = _vision_position_ids(1, self.seq_len)
        q_encoder.as_export_module("prefill", pixel_position_ids=pixel_position_ids)

        # as_export_module should have registered template buffers
        self.assertTrue(hasattr(q_encoder, "attention_mask_template"))
        self.assertTrue(hasattr(q_encoder, "position_embeddings_cos_template"))
        self.assertTrue(hasattr(q_encoder, "position_embeddings_sin_template"))

    def test_as_export_module_with_padding_produces_valid_output(self):
        """Check that as_export_module with padding pixel_position_ids produces valid output."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        # Use position IDs with some padding
        seq_len = self.seq_len
        num_valid = seq_len - 2
        pixel_position_ids = _vision_position_ids_with_padding(1, seq_len, num_valid)

        adapter = q_encoder.as_export_module(
            "prefill", pixel_position_ids=pixel_position_ids
        )

        inputs_embeds = torch.randn(1, seq_len, self.hidden_size)
        with torch.no_grad():
            output = adapter(inputs_embeds)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, seq_len, self.hidden_size))
        self.assertTrue(torch.isfinite(output).all())

    def test_unsupported_export_mode_raises(self):
        """Check that vision encoder exposes only a prefill export graph."""
        fp_encoder = _make_encoder(self.cfg)
        q_encoder = QuantGemma4VisionEncoder(fp_encoder).eval()
        q_encoder.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_encoder(**inputs)
        q_encoder.freeze_qparams()

        pixel_position_ids = _vision_position_ids(1, self.seq_len)
        with self.assertRaisesRegex(ValueError, "Unsupported Gemma4 vision encoder"):
            q_encoder.as_export_module("decode", pixel_position_ids=pixel_position_ids)


if __name__ == "__main__":
    unittest.main()

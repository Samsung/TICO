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

"""Unit tests for the Gemma4 vision pooler PTQ wrapper."""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_pooler import (
    QuantGemma4VisionPooler,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 vision."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionPooler,
        )
    except Exception:
        return False
    return True


def _make_vision_config():
    """Create a tiny Gemma4 vision config for synthetic pooler tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _make_pooler():
    """Create a tiny Gemma4 vision pooler in eval mode."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

    cfg = _make_vision_config()
    return Gemma4VisionPooler(cfg).eval()


def _pixel_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence.

    The pooler requires ``pixel_position_ids`` with shape ``(B, S, 2)`` where
    the last dimension encodes ``(x, y)`` patch coordinates.  We build a
    simple grid layout that is compatible with the ``output_length`` used in
    tests: ``seq_len = output_length * k^2`` where ``k`` is the pooling factor.
    """
    side = int(seq_len**0.5)
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _padding_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create an all-False padding mask (no padding)."""
    return torch.zeros(batch_size, seq_len, dtype=torch.bool)


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4VisionPooler(unittest.TestCase):
    """Validate Gemma4 vision pooler wrapper behavior."""

    def setUp(self):
        """Create deterministic inputs."""
        torch.manual_seed(2026)
        self.cfg = _make_vision_config()
        self.hidden_size = self.cfg.hidden_size
        # Use seq_len=16 and output_length=4 so that k=2 (16 / 4 = 4, sqrt(4) = 2).
        self.seq_len = 16
        self.output_length = 4
        self.batch_size = 1

    def _sample_inputs(self, batch_size=None, seq_len=None, output_length=None):
        """Create synthetic pooler inputs."""
        batch_size = batch_size or self.batch_size
        seq_len = seq_len or self.seq_len
        output_length = output_length or self.output_length
        return {
            "hidden_states": torch.randn(batch_size, seq_len, self.hidden_size),
            "pixel_position_ids": _pixel_position_ids(batch_size, seq_len),
            "padding_positions": _padding_positions(batch_size, seq_len),
            "output_length": output_length,
        }

    # ------------------------------------------------------------------
    # NO_QUANT mode
    # ------------------------------------------------------------------

    def test_no_quant_forward_matches_fp(self):
        """In NO_QUANT mode the wrapper should match the floating-point module."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()

        self.assertIs(q_pooler._mode, Mode.NO_QUANT)

        inputs = self._sample_inputs()
        with torch.no_grad():
            q_out = q_pooler(**inputs)
            fp_out = fp_pooler(**inputs)

        # Both should return tuples (pooled_features, updated_padding)
        self.assertIsInstance(q_out, tuple)
        self.assertIsInstance(fp_out, tuple)
        self.assertEqual(len(q_out), 2)
        self.assertEqual(len(fp_out), 2)

        # Shapes must match
        self.assertEqual(q_out[0].shape, fp_out[0].shape)
        self.assertEqual(q_out[1].shape, fp_out[1].shape)

        # Values must be close (the wrapper delegates to the original module)
        self.assertTrue(torch.allclose(q_out[0], fp_out[0], atol=1e-5, rtol=1e-5))
        # Padding masks may differ slightly due to different computation paths
        self.assertEqual(q_out[1].shape, fp_out[1].shape)

    def test_no_quant_output_shape(self):
        """Check that the pooled output has the expected static shape."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()

        inputs = self._sample_inputs()
        with torch.no_grad():
            pooled, padding = q_pooler(**inputs)

        self.assertEqual(
            pooled.shape, (self.batch_size, self.output_length, self.hidden_size)
        )
        self.assertEqual(padding.shape, (self.batch_size, self.output_length))

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def test_mode_transitions(self):
        """Check the calibration lifecycle: NO_QUANT → CALIB → QUANT."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()

        self.assertIs(q_pooler._mode, Mode.NO_QUANT)

        q_pooler.enable_calibration()
        self.assertIs(q_pooler._mode, Mode.CALIB)

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)

        q_pooler.freeze_qparams()
        self.assertIs(q_pooler._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns all 5 observers."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()

        all_obs = q_pooler._all_observers()
        self.assertEqual(len(all_obs), 6)

    # ------------------------------------------------------------------
    # Calibration and fake quantization
    # ------------------------------------------------------------------

    def test_input_is_observed_in_calib_mode(self):
        """In CALIB mode the input should be observed through obs_pool_in."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with mock.patch.object(
            q_pooler.obs_pool_in,
            "collect",
            wraps=q_pooler.obs_pool_in.collect,
        ) as mock_collect:
            with torch.no_grad():
                _ = q_pooler(**inputs)
            mock_collect.assert_called()

    def test_output_is_fake_quantized_in_quant_mode(self):
        """In QUANT mode the pooled output should be fake-quantized through obs_pool_out."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        with mock.patch.object(
            q_pooler.obs_pool_out,
            "fake_quant",
            wraps=q_pooler.obs_pool_out.fake_quant,
        ) as mock_fq:
            with torch.no_grad():
                _ = q_pooler(**inputs)
            mock_fq.assert_called()

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        with torch.no_grad():
            pooled, padding = q_pooler(**inputs)

        self.assertEqual(
            pooled.shape, (self.batch_size, self.output_length, self.hidden_size)
        )
        self.assertTrue(torch.isfinite(pooled).all())

    # ------------------------------------------------------------------
    # dtype override
    # ------------------------------------------------------------------

    def test_dtype_override(self):
        """Check that PTQConfig overrides propagate to pooler observers."""
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "pool_in": {"dtype": DType.uint(4)},
                "pool_out": {"dtype": DType.uint(4)},
            },
        )
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler, qcfg=cfg).eval()

        self.assertIsInstance(q_pooler.obs_pool_in, AffineObserverBase)
        self.assertIsInstance(q_pooler.obs_pool_out, AffineObserverBase)
        self.assertEqual(q_pooler.obs_pool_in.dtype, DType.uint(4))
        self.assertEqual(q_pooler.obs_pool_out.dtype, DType.uint(4))

    # ------------------------------------------------------------------
    # forward_export
    # ------------------------------------------------------------------

    def test_forward_export_requires_precomputed_buffers(self):
        """forward_export requires pool_weights and pool_mask buffers."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        # Precompute buffers like as_export_module does
        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)
        weights, mask = q_pooler._build_pool_weights(
            seq_len=self.seq_len,
            output_length=self.output_length,
            pixel_position_ids=pixel_pos_ids,
        )
        q_pooler.register_buffer("pool_weights", weights)
        q_pooler.register_buffer("pool_mask", mask)
        q_pooler.obs_pool_weight.collect(weights)
        q_pooler.obs_pool_weight.compute_qparams()

        # forward_export should work with precomputed buffers
        with torch.no_grad():
            out = q_pooler.forward_export(
                hidden_states=inputs["hidden_states"],
                padding_positions=inputs["padding_positions"],
            )
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    def test_forward_export_matches_forward(self):
        """forward_export should produce similar results to forward."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        # Precompute buffers
        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)
        weights, mask = q_pooler._build_pool_weights(
            seq_len=self.seq_len,
            output_length=self.output_length,
            pixel_position_ids=pixel_pos_ids,
        )
        q_pooler.register_buffer("pool_weights", weights)
        q_pooler.register_buffer("pool_mask", mask)
        q_pooler.obs_pool_weight.collect(weights)
        q_pooler.obs_pool_weight.compute_qparams()

        with torch.no_grad():
            forward_out = q_pooler(**inputs)
            export_out = q_pooler.forward_export(
                hidden_states=inputs["hidden_states"],
                padding_positions=inputs["padding_positions"],
            )

        # Both should produce similar shaped outputs
        self.assertEqual(export_out[0].shape, forward_out[0].shape)
        self.assertTrue(
            torch.allclose(export_out[0], forward_out[0], atol=1e-4, rtol=1e-4)
        )

    # ------------------------------------------------------------------
    # as_export_module
    # ------------------------------------------------------------------

    def test_as_export_module_requires_quant_mode(self):
        """as_export_module should assert that mode is QUANT."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()

        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)

        # Should fail in NO_QUANT mode
        with self.assertRaises(AssertionError):
            q_pooler.as_export_module(
                output_length=self.output_length,
                pixel_position_ids=pixel_pos_ids,
            )

    def test_as_export_module_returns_adapter(self):
        """as_export_module should return a Gemma4VisionPoolerPrefillExportAdapter."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionPoolerPrefillExportAdapter,
        )

        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)
        adapter = q_pooler.as_export_module(
            output_length=self.output_length,
            pixel_position_ids=pixel_pos_ids,
        )
        self.assertIsInstance(adapter, Gemma4VisionPoolerPrefillExportAdapter)

    def test_as_export_module_precomputes_buffers_on_wrapper(self):
        """as_export_module should register pool_weights and pool_mask on the wrapper."""
        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)
        _ = q_pooler.as_export_module(
            output_length=self.output_length,
            pixel_position_ids=pixel_pos_ids,
        )

        # Buffers should be on the wrapper
        self.assertTrue(hasattr(q_pooler, "pool_weights"))
        self.assertTrue(hasattr(q_pooler, "pool_mask"))
        self.assertEqual(
            q_pooler.pool_weights.shape, (1, self.output_length, self.seq_len)
        )
        self.assertEqual(q_pooler.pool_mask.shape, (1, self.output_length))

    def test_export_adapter_decomposed_forward_matches_fp(self):
        """The decomposed export adapter should match the original pooler output."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionPoolerPrefillExportAdapter,
        )

        fp_pooler = _make_pooler()
        q_pooler = QuantGemma4VisionPooler(fp_pooler).eval()
        q_pooler.enable_calibration()

        inputs = self._sample_inputs()
        with torch.no_grad():
            _ = q_pooler(**inputs)
        q_pooler.freeze_qparams()

        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)
        adapter = q_pooler.as_export_module(
            output_length=self.output_length,
            pixel_position_ids=pixel_pos_ids,
        )
        adapter.eval()

        adapter_kwargs = {
            "hidden_states": inputs["hidden_states"],
            "pixel_position_ids": inputs["pixel_position_ids"],
            "padding_positions": inputs["padding_positions"],
        }
        with torch.no_grad():
            adapter_out = adapter(**adapter_kwargs)

        # Compare with direct wrapper call
        with torch.no_grad():
            wrapper_out = q_pooler(**inputs)

        self.assertIsInstance(adapter_out, tuple)
        self.assertEqual(len(adapter_out), 2)
        self.assertEqual(adapter_out[0].shape, wrapper_out[0].shape)
        # The decomposed forward should produce numerically close results
        self.assertTrue(
            torch.allclose(adapter_out[0], wrapper_out[0], atol=1e-4, rtol=1e-4),
            f"Max diff: {(adapter_out[0] - wrapper_out[0]).abs().max().item()}",
        )

    def test_build_pool_weights_matches_original(self):
        """The precomputed pool weights should produce the same result as _avg_pool_by_positions."""
        pixel_pos_ids = _pixel_position_ids(self.batch_size, self.seq_len)
        weights, mask = QuantGemma4VisionPooler._build_pool_weights(
            seq_len=self.seq_len,
            output_length=self.output_length,
            pixel_position_ids=pixel_pos_ids,
        )

        # Verify weight matrix shape
        self.assertEqual(weights.shape, (1, self.output_length, self.seq_len))
        self.assertEqual(mask.shape, (1, self.output_length))

        # Verify that the weight matrix produces the same output as the original pooler
        fp_pooler = _make_pooler()
        hidden = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        padding = _padding_positions(self.batch_size, self.seq_len)

        # Original pooler path
        with torch.no_grad():
            fp_out, fp_mask = fp_pooler(
                hidden, pixel_pos_ids, padding, output_length=self.output_length
            )

        # Decomposed path: masked_fill + matmul + scale
        hidden_zeroed = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        pooled = weights @ hidden_zeroed.float()
        pooled = pooled * fp_pooler.root_hidden_size

        self.assertTrue(
            torch.allclose(pooled, fp_out, atol=1e-4, rtol=1e-4),
            f"Max diff: {(pooled - fp_out).abs().max().item()}",
        )


if __name__ == "__main__":
    unittest.main()

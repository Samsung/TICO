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

"""Smoke tests for Gemma4 vision encoder prepare-calibrate-convert flow."""

import copy
import os
import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper


IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
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


def _make_vision_config():
    """Create a tiny Gemma4 vision config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    cfg = Gemma4VisionConfig(
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
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _vision_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = 4
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4_vision(), _SKIP_MSG)
class TestGemma4VisionEncoderSmoke(unittest.TestCase):
    """Exercise Gemma4 vision encoder wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4 vision encoder modules."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoder

        self.cfg = _make_vision_config()
        self.fp_encoder = Gemma4VisionEncoder(self.cfg).eval()
        self.fp_ref = copy.deepcopy(self.fp_encoder).eval()

    def _sample(self):
        """Create one synthetic Gemma4 vision encoder sample for dynamic forward."""
        batch_size, seq_len = 1, 8
        return {
            "inputs_embeds": torch.randn(batch_size, seq_len, self.cfg.hidden_size),
            "pixel_position_ids": _vision_position_ids(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }

    def test_no_quant_vision_encoder_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder import (
            QuantGemma4VisionEncoder,
        )

        wrapped = QuantGemma4VisionEncoder(self.fp_encoder, qcfg=PTQConfig()).eval()
        sample = self._sample()

        with torch.no_grad():
            quant_out = wrapped(**sample)

        # Compare against HF reference
        with torch.no_grad():
            fp_out = self.fp_ref(**sample)

        fp_hidden = fp_out.last_hidden_state
        self.assertEqual(quant_out.shape, fp_hidden.shape)
        self.assertTrue(torch.allclose(quant_out, fp_hidden, atol=1e-5, rtol=1e-5))

    def test_prepare_convert_vision_encoder_flow(self):
        """Quantize Gemma4 vision encoder and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder import (
            QuantGemma4VisionEncoder,
        )

        prepared = prepare(self.fp_encoder, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4VisionEncoder)

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._sample()
        with torch.no_grad():
            quant_out = quantized(**sample)

        with torch.no_grad():
            fp_out = self.fp_ref(**sample)

        fp_hidden = fp_out.last_hidden_state
        self.assertEqual(quant_out.shape, fp_hidden.shape)
        self.assertTrue(torch.isfinite(quant_out).all())

    def test_prefill_export_adapter_after_convert(self):
        """Check the converted wrapper exposes the static prefill adapter contract."""
        prepared = prepare(self.fp_encoder, PTQConfig())
        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample())

        quantized = convert(prepared)

        # Provide pixel_position_ids matching the calibration seq_len.
        pixel_position_ids = _vision_position_ids(1, 8)
        seq_len = pixel_position_ids.shape[1]
        adapter = quantized.as_export_module(
            "prefill", pixel_position_ids=pixel_position_ids
        ).eval()

        # Adapter forward only needs inputs_embeds.
        inputs_embeds = torch.randn(1, seq_len, self.cfg.hidden_size)
        with torch.no_grad():
            output = adapter(inputs_embeds)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, seq_len, self.cfg.hidden_size))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

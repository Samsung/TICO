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

"""Smoke tests for Gemma4 vision attention prepare-calibrate-convert flow."""

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
    """Return whether the installed transformers package provides Gemma4 vision."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionAttention,
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


def _vision_rope(batch_size: int, seq_len: int, head_dim: int):
    """Create synthetic Gemma4 vision RoPE tables."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


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
@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4VisionAttentionSmoke(unittest.TestCase):
    """Exercise Gemma4 vision attention wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4 vision attention modules."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionAttention

        self.cfg = _make_vision_config()
        self.fp_attn = Gemma4VisionAttention(self.cfg, layer_idx=0).eval()
        self.fp_ref = copy.deepcopy(self.fp_attn).eval()

    def _sample(self):
        """Create one synthetic Gemma4 vision attention sample."""
        batch_size, seq_len = 1, 8
        return {
            "hidden_states": torch.randn(batch_size, seq_len, self.cfg.hidden_size),
            "position_embeddings": _vision_rope(
                batch_size,
                seq_len,
                self.cfg.head_dim,
            ),
            "attention_mask": torch.zeros(batch_size, 1, seq_len, seq_len),
            "position_ids": _vision_position_ids(batch_size, seq_len),
        }

    def test_no_quant_vision_attention_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_clippable_linear import (  # noqa: F401
            QuantGemma4ClippableLinear,
        )
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_attention import (
            QuantGemma4VisionAttention,
        )

        wrapped = QuantGemma4VisionAttention(self.fp_attn, qcfg=PTQConfig()).eval()
        sample = self._sample()

        with torch.no_grad():
            quant_out = wrapped(**sample)[0]
            fp_out = self.fp_ref(**sample)[0]

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_prepare_convert_vision_attention_flow(self):
        """Quantize Gemma4 vision attention and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_clippable_linear import (  # noqa: F401
            QuantGemma4ClippableLinear,
        )
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_attention import (
            QuantGemma4VisionAttention,
        )

        prepared = prepare(self.fp_attn, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4VisionAttention)

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample())

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._sample()
        with torch.no_grad():
            quant_out = quantized(**sample)[0]
            fp_out = self.fp_ref(**sample)[0]

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()

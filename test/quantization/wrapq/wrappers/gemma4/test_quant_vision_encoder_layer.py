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

"""Unit tests for the Gemma4 vision encoder-layer PTQ wrapper."""

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
            Gemma4VisionEncoderLayer,
        )
    except Exception:
        return False
    return True


def _make_vision_config(**overrides):
    """Create a tiny Gemma4 vision config for synthetic encoder-layer tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    kwargs = dict(
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
    kwargs.update(overrides)
    cfg = Gemma4VisionConfig(**kwargs)
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _vision_rope(batch_size: int, seq_len: int, head_dim: int):
    """Create synthetic RoPE tables shaped like Gemma4 vision RoPE output."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _vision_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = 4
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _zero_mask(batch_size: int, q_len: int, k_len: int) -> torch.Tensor:
    """Return an additive attention mask that keeps every patch token."""
    return torch.zeros(batch_size, 1, q_len, k_len)


@unittest.skipUnless(_has_gemma4_vision(), _SKIP_MSG)
class TestQuantGemma4VisionEncoderLayer(unittest.TestCase):
    """Validate Gemma4 vision encoder-layer wrapper behavior."""

    def setUp(self):
        """Create deterministic test inputs."""
        torch.manual_seed(2026)

    @staticmethod
    def _make_layer(cfg=None, layer_idx: int = 0):
        """Create a floating-point Gemma4 vision encoder layer."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoderLayer

        cfg = cfg if cfg is not None else _make_vision_config()
        return Gemma4VisionEncoderLayer(cfg, layer_idx=layer_idx).eval()

    def _sample(self, cfg, batch_size: int = 2, seq_len: int = 8):
        """Create one synthetic Gemma4 vision encoder-layer sample."""
        return {
            "hidden_states": torch.randn(batch_size, seq_len, cfg.hidden_size),
            "position_embeddings": _vision_rope(batch_size, seq_len, cfg.head_dim),
            "attention_mask": _zero_mask(batch_size, seq_len, seq_len),
            "position_ids": _vision_position_ids(batch_size, seq_len),
        }

    def test_00_prepare_wraps_vision_encoder_layer_when_registered(self):
        """Check that registry-based prepare wraps Gemma4VisionEncoderLayer."""
        from tico.quantization import prepare
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder_layer import (
            QuantGemma4VisionEncoderLayer,
        )
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        fp_layer = self._make_layer(_make_vision_config())
        prepared = prepare(fp_layer, PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4VisionEncoderLayer)

    def test_no_quant_forward_matches_hf_vision_encoder_layer(self):
        """Check that the wrapper matches Hugging Face eager encoder-layer output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder_layer import (
            QuantGemma4VisionEncoderLayer,
        )

        cfg = _make_vision_config()
        fp_layer = self._make_layer(cfg)
        qlayer = QuantGemma4VisionEncoderLayer(fp_layer).eval()
        sample = self._sample(cfg)

        with torch.no_grad():
            quant_out = qlayer(**sample)
            fp_out = fp_layer(**sample)

        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertTrue(torch.allclose(quant_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_mode_transitions_and_child_override(self):
        """Check lifecycle transitions and nested clippable-linear overrides."""
        from tico.quantization.wrapq.dtypes import DType
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder_layer import (
            QuantGemma4VisionEncoderLayer,
        )
        from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

        qcfg = PTQConfig(
            overrides={
                "self_attn": {
                    "q_proj": {
                        "linear": {
                            "act_in": {"dtype": DType.uint(4)},
                            "act_out": {"dtype": DType.uint(4)},
                        }
                    }
                }
            }
        )
        qlayer = QuantGemma4VisionEncoderLayer(self._make_layer(), qcfg=qcfg)

        self.assertIs(qlayer._mode, Mode.NO_QUANT)
        qlayer.enable_calibration()
        self.assertIs(qlayer._mode, Mode.CALIB)
        qlayer(**self._sample(qlayer.config, batch_size=1, seq_len=4))
        qlayer.freeze_qparams()

        self.assertIs(qlayer._mode, Mode.QUANT)
        inner_q_proj = qlayer.self_attn.wrapped.q_proj.wrapped.linear.wrapped
        self.assertIsInstance(inner_q_proj, QuantLinear)
        self.assertEqual(inner_q_proj.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(inner_q_proj.obs_act_out.dtype, DType.uint(4))

    def test_prefill_export_adapter_contract(self):
        """Validate the static prefill export adapter output contract."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder_layer import (
            QuantGemma4VisionEncoderLayer,
        )

        cfg = _make_vision_config()
        qlayer = QuantGemma4VisionEncoderLayer(self._make_layer(cfg)).eval()
        adapter = qlayer.as_export_module("prefill").eval()
        sample = self._sample(cfg, batch_size=1, seq_len=4)

        with torch.no_grad():
            output = adapter(
                sample["hidden_states"],
                sample["attention_mask"],
                sample["position_embeddings"],
                position_ids=sample["position_ids"],
            )

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 4, cfg.hidden_size))
        self.assertTrue(torch.isfinite(output).all())

    def test_unsupported_export_mode_raises(self):
        """Check that vision encoder layers expose only a prefill export graph."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder_layer import (
            QuantGemma4VisionEncoderLayer,
        )

        qlayer = QuantGemma4VisionEncoderLayer(self._make_layer()).eval()
        with self.assertRaisesRegex(ValueError, "Unsupported Gemma4 vision"):
            qlayer.as_export_module("decode")

    def test_missing_position_embeddings_raises(self):
        """Check that callers must pass static Gemma4 vision RoPE tables."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_encoder_layer import (
            QuantGemma4VisionEncoderLayer,
        )

        cfg = _make_vision_config()
        qlayer = QuantGemma4VisionEncoderLayer(self._make_layer(cfg)).eval()
        hidden = torch.randn(1, 4, cfg.hidden_size)
        with self.assertRaisesRegex(ValueError, "position_embeddings"):
            qlayer(hidden)


if __name__ == "__main__":
    unittest.main()

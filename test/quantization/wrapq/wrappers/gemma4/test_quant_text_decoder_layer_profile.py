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

"""Unit tests for Gemma4 decoder export-profile validation."""

import unittest

from tico.quantization.config.ptq import PTQConfig


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextDecoderLayer,
        )
    except Exception:
        return False
    return True


def _make_text_config():
    """Create a tiny dense Gemma4 configuration for export tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=8,
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            }
        },
        hidden_size_per_layer_input=0,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4TextDecoderExportProfile(unittest.TestCase):
    """Validate NPU-profile enforcement at the decoder export boundary."""

    @staticmethod
    def _make_layer():
        """Create a floating-point Gemma4 decoder layer."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        return Gemma4TextDecoderLayer(_make_text_config(), layer_idx=0).eval()

    def test_default_npu_profile_is_accepted(self):
        """The default unrolled profile should produce an export adapter."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        layer = QuantGemma4TextDecoderLayer(self._make_layer()).eval()

        adapter = layer.as_export_module("prefill")

        self.assertIsNotNone(adapter)

    def test_reference_profile_is_rejected_by_default(self):
        """A batched reference graph should not silently enter NPU export."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        layer = QuantGemma4TextDecoderLayer(
            self._make_layer(),
            qcfg=PTQConfig(model_args={"profile": "reference_eval"}),
        ).eval()

        with self.assertRaisesRegex(ValueError, "npu_export"):
            layer.as_export_module("prefill")

    def test_profile_validation_can_be_disabled_explicitly(self):
        """Reference experiments may opt out of the NPU boundary check."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
            QuantGemma4TextDecoderLayer,
        )

        layer = QuantGemma4TextDecoderLayer(
            self._make_layer(),
            qcfg=PTQConfig(model_args={"profile": "reference_eval"}),
        ).eval()

        adapter = layer.as_export_module(
            "prefill",
            require_npu_profile=False,
        )

        self.assertIsNotNone(adapter)


if __name__ == "__main__":
    unittest.main()

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

"""Smoke tests for Gemma4 vision model prepare-calibrate-convert flow."""

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
            Gemma4VisionModel,
        )
    except Exception:
        return False
    return True


def _make_vision_config(**overrides):
    """Create a tiny Gemma4 vision config for synthetic smoke tests."""
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
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _vision_position_ids(batch_size: int, num_patches: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = int(num_patches**0.5)
    coords = torch.arange(num_patches)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


@unittest.skipIf(
    not IS_INTERNAL_TEST,
    "Internal smoke test — set RUN_INTERNAL_TESTS=1 to enable it.",
)
@unittest.skipUnless(_has_gemma4_vision(), _SKIP_MSG)
class TestGemma4VisionModelSmoke(unittest.TestCase):
    """Exercise Gemma4 vision model wrapper parity and PTQ flow."""

    def setUp(self):
        """Create deterministic tiny Gemma4 vision model modules."""
        torch.manual_seed(2026)
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        self.cfg = _make_vision_config()
        self.fp_model = Gemma4VisionModel(self.cfg).eval()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        # For 16 patches with pooling_kernel_size=2: output_length = 16 / 4 = 4
        self.num_patches = 16
        self.patch_size = self.cfg.patch_size
        self.batch_size = 1

    def _sample(self):
        """Create one synthetic Gemma4 vision model sample.

        The HF Gemma4VisionModel expects pre-flattened patches:
            pixel_values: (B, num_patches, 3*patch_size^2)
            pixel_position_ids: (B, num_patches, 2)
        """
        patch_dim = 3 * self.patch_size**2
        return {
            "pixel_values": torch.randn(self.batch_size, self.num_patches, patch_dim),
            "pixel_position_ids": _vision_position_ids(
                self.batch_size, self.num_patches
            ),
        }

    def test_no_quant_vision_model_matches_reference(self):
        """The wrapper should match the floating-point module before quantization."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        wrapped = QuantGemma4VisionModel(self.fp_model, qcfg=PTQConfig()).eval()
        sample = self._sample()

        with torch.no_grad():
            quant_out = wrapped(**sample, return_dict=True)
            fp_out = self.fp_ref(**sample, return_dict=True)

        self.assertEqual(
            quant_out.last_hidden_state.shape, fp_out.last_hidden_state.shape
        )
        self.assertTrue(
            torch.allclose(
                quant_out.last_hidden_state,
                fp_out.last_hidden_state,
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_prepare_convert_vision_model_flow(self):
        """Quantize Gemma4 vision model and validate a synthetic output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )

        prepared = prepare(self.fp_model, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4VisionModel)

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample(), return_dict=True)

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        sample = self._sample()
        with torch.no_grad():
            quant_out = quantized(**sample, return_dict=True)
            fp_out = self.fp_ref(**sample, return_dict=True)

        self.assertEqual(
            quant_out.last_hidden_state.shape, fp_out.last_hidden_state.shape
        )
        self.assertTrue(torch.isfinite(quant_out.last_hidden_state).all())

    def test_as_export_module_flow(self):
        """Test the as_export_module flow for Circle export."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionModelPrefillExportAdapter,
        )

        prepared = prepare(self.fp_model, PTQConfig())

        with torch.no_grad():
            for _ in range(3):
                prepared(**self._sample(), return_dict=True)

        quantized = convert(prepared)

        sample = self._sample()
        export_module = quantized.wrapped.as_export_module(
            mode="prefill",
            pixel_position_ids=sample["pixel_position_ids"],
        )

        # as_export_module returns Gemma4VisionModelPrefillExportAdapter
        self.assertIsInstance(export_module, Gemma4VisionModelPrefillExportAdapter)

        # Verify forward works (adapter delegates to wrapped_model.forward_export)
        with torch.no_grad():
            output = export_module(**sample)

        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_vision_model_with_standardize_false(self):
        """Test vision model when standardize=False."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        cfg = _make_vision_config(standardize=False)
        fp_model = Gemma4VisionModel(cfg).eval()
        wrapped = QuantGemma4VisionModel(fp_model, qcfg=PTQConfig()).eval()

        sample = self._sample()
        with torch.no_grad():
            quant_out = wrapped(**sample, return_dict=True)
            fp_out = self.fp_ref(**sample, return_dict=True)

        self.assertEqual(
            quant_out.last_hidden_state.shape, fp_out.last_hidden_state.shape
        )
        self.assertTrue(torch.isfinite(quant_out.last_hidden_state).all())


if __name__ == "__main__":
    unittest.main()

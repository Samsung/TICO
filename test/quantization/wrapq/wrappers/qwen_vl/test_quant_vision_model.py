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

import math
import unittest
from typing import Tuple
from unittest import mock

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLVisionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_model import (
    QuantQwen3VLVisionModel,
)


skip_msg = "transformers not installed — skipping Qwen3VLVisionModel tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLVisionModel(unittest.TestCase):
    fp_model: torch.nn.Module
    hidden_size: int
    num_heads: int
    head_dim: int
    theta: float

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        # Use smaller sizes for testing
        cfg = Qwen3VLVisionConfig(
            hidden_size=64,
            num_heads=4,
            depth=2,  # Smaller depth for faster testing
            temporal_patch_size=2,
            patch_size=16,
        )

        # Ensure eager attention implementation so outputs are deterministic
        # and do not require GPU flash attention kernels.
        # Some versions use `_attn_implementation`, others expose `attn_implementation`.
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        cls.fp_model = Qwen3VLVisionModel(cfg)
        cls.hidden_size = cfg.hidden_size
        cls.num_heads = cfg.num_heads
        cls.head_dim = cls.hidden_size // cls.num_heads
        cls.theta = (
            cls.fp_model.rotary_pos_emb.theta
            if hasattr(cls.fp_model.rotary_pos_emb, "theta")
            else 10000.0
        )

    @staticmethod
    def _make_ptq_config(grid_thw: Tuple[int, int, int]) -> PTQConfig:
        return PTQConfig(
            model_args={
                "vision": {
                    "grid_thw": grid_thw,
                }
            }
        )

    def _create_test_inputs(
        self, grid_thw: Tuple[int, int, int] = (1, 8, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to create test inputs for VisionModel."""
        t, h, w = grid_thw
        num_patches = t * h * w
        vision_cfg = self.fp_model.config
        patch_dim = (
            vision_cfg.in_channels
            * vision_cfg.temporal_patch_size
            * vision_cfg.patch_size
            * vision_cfg.patch_size
        )
        # Explicit batch-one ABI used by the static NPU export path.
        hidden_states = torch.randn(1, num_patches, patch_dim)
        grid_tensor = torch.tensor([grid_thw])
        return hidden_states, grid_tensor

    @staticmethod
    def _normalize_vision_output(
        output,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Normalize version-dependent vision outputs for direct comparison."""
        if hasattr(output, "pooler_output"):
            image_embeds = output.pooler_output
            deepstack_features = getattr(output, "deepstack_features", None)
        else:
            image_embeds, deepstack_features = output

        if deepstack_features is None:
            normalized_features: Tuple[torch.Tensor, ...] = ()
        else:
            normalized_features = tuple(deepstack_features)
        return image_embeds, normalized_features

    def _assert_vision_outputs_equal(self, expected, actual) -> None:
        """Assert exact parity for merged and DeepStack vision outputs."""
        expected_image, expected_deepstack = self._normalize_vision_output(expected)
        actual_image, actual_deepstack = self._normalize_vision_output(actual)

        torch.testing.assert_close(expected_image, actual_image, rtol=0.0, atol=0.0)
        self.assertEqual(len(expected_deepstack), len(actual_deepstack))
        for expected_feature, actual_feature in zip(
            expected_deepstack, actual_deepstack
        ):
            torch.testing.assert_close(
                expected_feature,
                actual_feature,
                rtol=0.0,
                atol=0.0,
            )

    def test_get_vision_grid_thw_from_config(self):
        """Test _get_vision_grid_thw static method with valid config."""
        ptq_config = self._make_ptq_config((1, 8, 8))

        grid_thw = QuantQwen3VLVisionModel._get_vision_grid_thw(ptq_config)
        expected = torch.tensor([[1, 8, 8]])
        self.assertTrue(torch.equal(grid_thw, expected))
        self.assertEqual(grid_thw.shape, (1, 3))

    def test_get_vision_grid_thw_missing_config(self):
        """Test _get_vision_grid_thw raises error when config is missing."""
        # Test with None config
        with self.assertRaises(ValueError) as context:
            QuantQwen3VLVisionModel._get_vision_grid_thw(None)
        self.assertIn("model_args", str(context.exception))

        # Test with config without vision_grid_thw
        ptq_config = PTQConfig()
        with self.assertRaises(ValueError) as context:
            QuantQwen3VLVisionModel._get_vision_grid_thw(ptq_config)
        self.assertIn("vision.grid_thw", str(context.exception))

    def test_precompute_rope_inv_freq(self):
        """Test _precompute_rope_inv_freq static method."""
        dim = 32
        theta = 10000.0
        inv_freq = QuantQwen3VLVisionModel._precompute_rope_inv_freq(dim, theta)

        self.assertEqual(inv_freq.shape, (dim // 2,))
        self.assertTrue(torch.all(inv_freq > 0))
        # Check that frequencies are decreasing
        self.assertTrue(torch.all(inv_freq[:-1] >= inv_freq[1:]))

    def test_precompute_cu_seqlens(self):
        """Test _precompute_cu_seqlens static method."""
        grid_thw = torch.tensor(
            [[1, 8, 8], [2, 4, 4]]
        )  # 1*8*8 + 2*4*4 = 96 total patches
        cu_seqlens = QuantQwen3VLVisionModel._precompute_cu_seqlens(grid_thw)

        self.assertEqual(cu_seqlens.shape, (4,))  # 3 images + 1 padding
        self.assertEqual(cu_seqlens[0].item(), 0)
        self.assertEqual(cu_seqlens[1].item(), 64)  # 1st image: 1*8*8 = 64 patches
        self.assertEqual(cu_seqlens[2].item(), 80)  # 2nd image: 1*4*4 = 16 patches
        self.assertEqual(
            cu_seqlens[3].item(), 96
        )  # 3rd image: 1*4*4 = 16 patches, total 96

    def test_precompute_rope_position_embeddings(self):
        """Test _precompute_rope_position_embeddings static method."""
        grid_thw = torch.tensor([[1, 8, 8]])
        inv_freq = QuantQwen3VLVisionModel._precompute_rope_inv_freq(
            dim=self.head_dim // 2,
            theta=self.theta,
        )

        cos_t, sin_t = QuantQwen3VLVisionModel._precompute_rope_position_embeddings(
            merge_size=2,
            rope_inv_freq=inv_freq,
            grid_thw=grid_thw,
        )

        expected_patches = math.prod(grid_thw[0].tolist())  # t * h * w = 1 * 8 * 8 = 64
        self.assertEqual(cos_t.shape, (expected_patches, self.head_dim))
        self.assertEqual(sin_t.shape, (expected_patches, self.head_dim))

    def test_rot_pos_emb(self):
        """Test _rot_pos_emb static method."""
        grid_thw = torch.tensor([[1, 8, 8]])
        inv_freq = QuantQwen3VLVisionModel._precompute_rope_inv_freq(
            dim=self.head_dim // 2,
            theta=self.theta,
        )

        rotary_pos_emb = QuantQwen3VLVisionModel._rot_pos_emb(2, inv_freq, grid_thw)

        expected_patches = math.prod(grid_thw[0].tolist())  # t * h * w = 1 * 8 * 8 = 64
        self.assertEqual(rotary_pos_emb.shape, (expected_patches, self.head_dim // 2))

    def test_create_freq_table(self):
        """Test _create_freq_table static method."""
        seqlen = 64
        inv_freq = torch.randn(16)  # dim//2 = 32//2 = 16
        freq_table = QuantQwen3VLVisionModel._create_freq_table(seqlen, inv_freq)

        self.assertEqual(freq_table.shape, (seqlen, inv_freq.shape[0]))

    def test_fast_pos_embed_interpolate(self):
        """Test _fast_pos_embed_interpolate static method."""
        grid_thw = torch.tensor([[1, 8, 8]])
        pos_embeds = QuantQwen3VLVisionModel._fast_pos_embed_interpolate(
            merge_size=2,
            num_grid_per_side=48,  # From model config
            pos_embedder=self.fp_model.pos_embed,
            grid_thw=grid_thw,
        )

        expected_patches = math.prod(grid_thw[0].tolist())  # t * h * w = 1 * 8 * 8 = 64
        self.assertEqual(pos_embeds.shape, (expected_patches, self.hidden_size))

    def test_init_with_valid_config(self):
        """Test successful initialization with valid config."""
        ptq_config = self._make_ptq_config((1, 8, 8))

        q_model = QuantQwen3VLVisionModel(
            self.fp_model, qcfg=ptq_config, fp_name="test_model"
        )

        # Check that buffers are registered
        self.assertTrue(hasattr(q_model, "cu_seqlens_template"))
        self.assertTrue(hasattr(q_model, "pos_embed_template"))
        self.assertTrue(hasattr(q_model, "rope_inv_freq"))
        self.assertTrue(hasattr(q_model, "rope_cos_template"))
        self.assertTrue(hasattr(q_model, "rope_sin_template"))
        self.assertFalse(q_model.pos_embed_template.requires_grad)

        # Check submodule wrapping
        self.assertIsNotNone(q_model.patch_embed)
        self.assertEqual(len(q_model.blocks), len(self.fp_model.blocks))
        self.assertIsNotNone(q_model.merger)
        self.assertEqual(
            len(q_model.deepstack_merger_list), len(self.fp_model.deepstack_merger_list)
        )

    def test_as_export_module_returns_static_adapter(self):
        """The vision wrapper should expose its fixed-grid export adapter."""
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config((1, 8, 8)),
            fp_name="test_export_adapter",
        ).eval()

        export_module = q_model.as_export_module(mode="prefill")

        self.assertIsInstance(export_module, Qwen3VLVisionPrefillExportAdapter)
        self.assertIs(export_module.wrapped, q_model)

    def test_static_adapter_unwraps_transparent_wrapper(self):
        """The adapter should resolve a vision model behind a transparent wrapper."""
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config((1, 8, 8)),
            fp_name="test_export_unwrap",
        ).eval()
        transparent = torch.nn.Module()
        transparent.wrapped = q_model

        export_module = Qwen3VLVisionPrefillExportAdapter(transparent)

        self.assertIs(export_module.wrapped, q_model)

    def test_wrapper_smoke_case_uses_static_adapter(self):
        """The vision-model smoke export should select the static adapter."""
        from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import (
            QwenVisionModelCase,
        )

        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config((1, 8, 8)),
            fp_name="test_smoke_export_adapter",
        ).eval()

        export_module = QwenVisionModelCase().export_module(q_model, {})

        self.assertIsInstance(export_module, Qwen3VLVisionPrefillExportAdapter)

    def test_as_export_module_supports_quant_mode(self):
        """A calibrated vision wrapper should expose the static adapter."""
        grid_thw = (1, 8, 8)
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config(grid_thw),
            fp_name="test_quant_export_adapter",
        ).eval()
        hidden_states, grid_tensor = self._create_test_inputs(grid_thw)
        q_model.enable_calibration()
        with torch.no_grad():
            q_model(hidden_states, grid_tensor)
        q_model.freeze_qparams()

        export_module = q_model.as_export_module(mode="prefill")

        self.assertIsInstance(export_module, Qwen3VLVisionPrefillExportAdapter)

    def test_as_export_module_rejects_calibration_mode(self):
        """Static export should reject a wrapper with incomplete qparams."""
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config((1, 8, 8)),
            fp_name="test_export_calibration",
        ).eval()
        q_model.enable_calibration()

        with self.assertRaisesRegex(RuntimeError, "NO_QUANT or QUANT"):
            q_model.as_export_module(mode="prefill")

    def test_as_export_module_rejects_unsupported_mode(self):
        """The vision export adapter should support prefill mode only."""
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config((1, 8, 8)),
            fp_name="test_export_mode",
        ).eval()

        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3-VL vision"):
            q_model.as_export_module(mode="decode")

    def test_static_adapter_bypasses_dynamic_forward(self):
        """The static adapter should call ``forward_export`` directly."""
        grid_thw = (1, 8, 8)
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config(grid_thw),
            fp_name="test_static_dispatch",
        ).eval()
        export_module = Qwen3VLVisionPrefillExportAdapter(q_model).eval()
        hidden_states, grid_tensor = self._create_test_inputs(grid_thw)

        with mock.patch.object(
            q_model,
            "forward",
            side_effect=AssertionError("dynamic forward was used"),
        ):
            output = export_module(hidden_states, grid_tensor)

        image_embeds, _ = self._normalize_vision_output(output)
        self.assertEqual(image_embeds.shape[0], math.prod(grid_thw) // 4)

    def test_dynamic_forward_uses_runtime_grid(self):
        """Eager execution should derive metadata from the runtime grid."""
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config((1, 8, 8)),
            fp_name="test_dynamic_runtime_grid",
        ).eval()
        hidden_states, runtime_grid = self._create_test_inputs((1, 4, 4))

        with torch.no_grad():
            output = q_model(hidden_states, runtime_grid)

        image_embeds, _ = self._normalize_vision_output(output)
        self.assertEqual(image_embeds.shape[0], 4)

    def test_static_adapter_matches_dynamic_forward_for_fixed_grid(self):
        """Dynamic and fixed-profile metadata should produce identical outputs."""
        grid_thw = (1, 8, 8)
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config(grid_thw),
            fp_name="test_static_parity",
        ).eval()
        export_module = q_model.as_export_module(mode="prefill").eval()
        hidden_states, grid_tensor = self._create_test_inputs(grid_thw)

        with torch.no_grad():
            dynamic_output = q_model(hidden_states, grid_tensor)
            static_output = export_module(hidden_states, grid_tensor)

        self._assert_vision_outputs_equal(dynamic_output, static_output)

    def test_non_strict_export_with_temporal_grid(self):
        """Fixed temporal grids should export without data-dependent split sizes."""
        grid_thw = (2, 4, 4)
        q_model = QuantQwen3VLVisionModel(
            self.fp_model,
            qcfg=self._make_ptq_config(grid_thw),
            fp_name="test_temporal_export",
        ).eval()
        export_module = q_model.as_export_module(mode="prefill").eval()
        hidden_states, grid_tensor = self._create_test_inputs(grid_thw)

        with torch.no_grad():
            expected = export_module(hidden_states, grid_tensor)
        exported_program = torch.export.export(
            export_module,
            (hidden_states, grid_tensor),
            strict=False,
        )
        with torch.no_grad():
            actual = exported_program.module()(hidden_states, grid_tensor)

        self._assert_vision_outputs_equal(expected, actual)

    def test_init_missing_vision_grid_thw(self):
        """Test initialization fails without vision_grid_thw."""
        ptq_config = PTQConfig()

        with self.assertRaises(ValueError) as context:
            QuantQwen3VLVisionModel(
                self.fp_model, qcfg=ptq_config, fp_name="test_model"
            )
        self.assertIn("vision.grid_thw", str(context.exception))

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        ptq_config = self._make_ptq_config((1, 8, 8))
        q_model = QuantQwen3VLVisionModel(
            self.fp_model, qcfg=ptq_config, fp_name="test_model"
        )
        self.assertIs(q_model._mode, Mode.NO_QUANT)

        q_model.enable_calibration()
        self.assertIs(q_model._mode, Mode.CALIB)

        # Run forward pass during calibration
        hidden_states, grid_thw = self._create_test_inputs((1, 8, 8))
        _ = q_model(hidden_states, grid_thw)

        q_model.freeze_qparams()
        self.assertIs(q_model._mode, Mode.QUANT)

    def test_observer_count(self):
        """Test that the wrapper has the correct number of observers."""
        ptq_config = self._make_ptq_config((1, 8, 8))
        q_model = QuantQwen3VLVisionModel(
            self.fp_model, qcfg=ptq_config, fp_name="test_model"
        )

        observers = list(q_model._all_observers())
        # Should have 4 local observers: pos_embeds, pos_add, rope_cos, rope_sin
        self.assertEqual(len(observers), 4)

    def test_precomputed_embeddings_shape(self):
        """Test that precomputed embeddings have correct shapes."""
        ptq_config = self._make_ptq_config((1, 8, 8))
        q_model = QuantQwen3VLVisionModel(
            self.fp_model, qcfg=ptq_config, fp_name="test_model"
        )

        expected_patches = math.prod(
            ptq_config.get_model_arg("vision")["grid_thw"]
        )  # t * h * w = 1 * 8 * 8 = 64

        # Check position embeddings
        self.assertEqual(
            q_model.pos_embed_template.shape, (expected_patches, self.hidden_size)
        )

        # Check RoPE embeddings
        self.assertEqual(
            q_model.rope_cos_template.shape,
            (expected_patches, self.head_dim),
        )
        self.assertEqual(
            q_model.rope_sin_template.shape,
            (expected_patches, self.head_dim),
        )

        # Check cumulative sequence lengths
        self.assertEqual(q_model.cu_seqlens_template.shape, (2,))  # 1 image + 1 padding

    def test_registration_in_registry(self):
        """Test that Qwen3VLVisionModel is properly registered."""
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        wrapper_cls = lookup(Qwen3VLVisionModel)
        self.assertIs(wrapper_cls, QuantQwen3VLVisionModel)

    def test_output_structure(self):
        """Test that output has correct structure."""
        ptq_config = self._make_ptq_config((1, 8, 8))
        q_model = QuantQwen3VLVisionModel(
            self.fp_model, qcfg=ptq_config, fp_name="test_model"
        )
        q_model.enable_calibration()

        hidden_states, grid_thw = self._create_test_inputs((1, 8, 8))
        _ = q_model(hidden_states, grid_thw)

        q_model.freeze_qparams()

        with torch.no_grad():
            q_out = q_model(hidden_states, grid_thw)

        # Check shapes
        expected_patches = math.prod(
            ptq_config.get_model_arg("vision")["grid_thw"]
        )  # t * h * w = 1 * 8 * 8

        # The structure of q_out depends on transformers version
        merged_hidden_states = (
            q_out.pooler_output if q_model.has_deepstack_model_output else q_out[0]
        )

        self.assertEqual(merged_hidden_states.shape[0], expected_patches // 4)

    def test_different_grid_sizes(self):
        """Test with different grid sizes."""
        test_cases = [
            ((1, 4, 4), "small_image"),
            ((1, 6, 6), "medium_image"),
            ((1, 8, 8), "large_image"),
        ]

        grid_thw_list: tuple[int, int, int]
        description: str
        for grid_thw_list, description in test_cases:
            with self.subTest(description=description):
                ptq_config = self._make_ptq_config(grid_thw_list)
                q_model = QuantQwen3VLVisionModel(
                    self.fp_model, qcfg=ptq_config, fp_name=f"test_model_{description}"
                )

                hidden_states, grid_thw = self._create_test_inputs(grid_thw_list)

                q_model.enable_calibration()
                _ = q_model(hidden_states, grid_thw)
                q_model.freeze_qparams()

                with torch.no_grad():
                    q_out = q_model(hidden_states, grid_thw)

                # The structure of q_out depends on transformers version
                merged_hidden_states = (
                    q_out.pooler_output
                    if q_model.has_deepstack_model_output
                    else q_out[0]
                )

                expected_patches = math.prod(grid_thw_list)  # t * h * w
                self.assertEqual(merged_hidden_states.shape[0], expected_patches // 4)

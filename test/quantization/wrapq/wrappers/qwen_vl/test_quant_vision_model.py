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

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
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
        # Input shape: (seq_len, in_channels * temporal_patch_size * patch_size * patch_size)
        hidden_states = torch.randn(
            num_patches, 3 * 2 * 16 * 16
        )  # 3 channels, 2 temporal, 16x16 patches
        grid_tensor = torch.tensor([grid_thw])
        return hidden_states, grid_tensor

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

        # Check submodule wrapping
        self.assertIsNotNone(q_model.patch_embed)
        self.assertEqual(len(q_model.blocks), len(self.fp_model.blocks))
        self.assertIsNotNone(q_model.merger)
        self.assertEqual(
            len(q_model.deepstack_merger_list), len(self.fp_model.deepstack_merger_list)
        )

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

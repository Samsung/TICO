# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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

import copy
import unittest
from typing import Tuple

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.qwen_vl.quant_model import QuantQwen3VLModel


skip_msg = "transformers not installed — skipping Qwen3VLModel tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLModel(unittest.TestCase):
    fp_model: torch.nn.Module
    hidden_size: int
    vocab_size: int
    patch_size: int
    temporal_patch_size: int
    video_token_id: int
    image_token_id: int
    spatial_merge_size: int
    ptq_config: PTQConfig

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        # Use smaller sizes for testing
        cfg = Qwen3VLConfig(
            vision_config={
                "hidden_size": 64,
                "num_heads": 4,
                "depth": 2,  # Smaller depth for faster testing
                "temporal_patch_size": 2,
                "patch_size": 16,
                "out_hidden_size": 64,
                "deepstack_visual_indexes": [0, 1],
            },
            text_config={
                "hidden_size": 64,
                "intermediate_size": 256,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "num_hidden_layers": 2,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "max_position_embeddings": 1024,
                "vocab_size": 1000,
                "use_cache": False,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            image_token_id=998,
            video_token_id=999,
        )

        assert cfg.image_token_id < cfg.text_config.vocab_size
        assert cfg.video_token_id < cfg.text_config.vocab_size
        assert cfg.vision_config.out_hidden_size == cfg.text_config.hidden_size

        cls.fp_model = Qwen3VLModel(cfg)
        cls.patch_size = cfg.vision_config.patch_size
        cls.temporal_patch_size = cfg.vision_config.temporal_patch_size
        cls.hidden_size = cfg.text_config.hidden_size
        cls.vocab_size = cfg.text_config.vocab_size
        cls.video_token_id = cfg.video_token_id
        cls.image_token_id = cfg.image_token_id
        cls.spatial_merge_size = cfg.vision_config.spatial_merge_size

    @staticmethod
    def _make_ptq_config(grid_thw: Tuple[int, int, int]) -> PTQConfig:
        return PTQConfig(
            model_args={
                "vision": {
                    "grid_thw": grid_thw,
                    "visual_start_idx": 0,
                    "spatial_merge_size": 2,
                }
            }
        )

    @staticmethod
    def _compute_3d_position_ids(
        input_ids: torch.Tensor,
        thw: Tuple[int, int, int],
        spatial_merge_size: int,
        image_token_id: int,
    ) -> torch.Tensor:
        """
        Compute 3D position IDs for multimodal RoPE.
        This function pre-computes position_ids to avoid tracing issues during model export.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        position_ids = torch.ones(
            3, batch_size, seq_len, dtype=input_ids.dtype, device=device
        )

        for i in range(batch_size):
            # Find positions of image tokens
            image_mask = input_ids[i] == image_token_id
            image_positions = torch.nonzero(image_mask, as_tuple=True)[0]

            llm_pos_ids_list: list[torch.tensor] = []
            st = 0

            # Process visual tokens
            if len(image_positions) > 0:
                # Group consecutive placeholder tokens into a single visual object
                # All consecutive image tokens represent ONE image/video
                start_pos = image_positions[0].item()

                # Text position IDs (before first visual token)
                text_len = start_pos - st
                if text_len > 0:
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                # Vision position IDs (3D)
                llm_grid_t = 1  # Always 1 for images
                llm_grid_h = thw[1] // spatial_merge_size
                llm_grid_w = thw[2] // spatial_merge_size

                t_index = (
                    torch.arange(llm_grid_t, device=device)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h, device=device)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w, device=device)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + st_idx
                )

                # Update st to after all visual placeholder tokens
                # The number of visual tokens is (thw[1] // spatial_merge_size) * (thw[2] // spatial_merge_size)
                num_visual_tokens = (thw[1] // spatial_merge_size) * (
                    thw[2] // spatial_merge_size
                )
                st = start_pos + num_visual_tokens

            # Trailing text
            if st < seq_len:
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = seq_len - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, :] = llm_positions

        return position_ids

    def _create_text_only_input(self, batch_size=1, seq_len=10):
        """Helper to create text-only input without images/videos."""
        input_ids = torch.randint(
            low=0, high=self.vocab_size, size=(batch_size, seq_len), dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def _create_visual_input(
        self,
        visual_token_id: int,
        batch_size: int,
        seq_len: int,
        thw: Tuple[int, int, int],
    ):
        """Helper to create input with videos or images."""
        assert visual_token_id in (self.video_token_id, self.image_token_id)

        # Calculate number of visual placeholder tokens needed
        # Each video is represented by multiple tokens after spatial merge
        # Spatial merge reduces the grid size by spatial_merge_size in each dimension
        num_video_tokens = (thw[1] // self.spatial_merge_size) * (
            thw[2] // self.spatial_merge_size
        )
        assert (
            num_video_tokens <= seq_len
        ), f"{num_video_tokens} video tokens can't fit into input sequence of length {seq_len}"

        # Create input_ids with random text tokens
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size - 2,
            size=(batch_size, seq_len),
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)

        # Replace first tokens with video placeholder tokens
        # This marks where the video features should be inserted
        for i in range(batch_size):
            input_ids[i, :num_video_tokens] = visual_token_id

        num_temporal_patches, num_spatial_patches_h, num_spatial_patches_w = thw

        # Create pixel values for videos
        pixel_values = torch.randn(
            batch_size,
            3,
            num_temporal_patches * self.temporal_patch_size,
            num_spatial_patches_h * self.patch_size,
            num_spatial_patches_w * self.patch_size,
        )
        video_grid_thw = torch.tensor([thw])

        position_ids = self._compute_3d_position_ids(
            input_ids=input_ids,
            thw=thw,
            spatial_merge_size=self.spatial_merge_size,
            image_token_id=visual_token_id,
        )

        return input_ids, attention_mask, pixel_values, video_grid_thw, position_ids

    def _create_video_input(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        thw: Tuple[int, int, int] = (1, 8, 8),
    ):
        return self._create_visual_input(self.video_token_id, batch_size, seq_len, thw)

    def _create_image_input(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        thw: Tuple[int, int, int] = (1, 8, 8),
    ):
        return self._create_visual_input(self.image_token_id, batch_size, seq_len, thw)

    # -------------------------------------------------------------------------
    # Initialization tests
    # -------------------------------------------------------------------------

    def test_wraps_submodules(self):
        """Test that __init__ wraps all submodules with PTQWrapper."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)

        # Check that submodules are wrapped
        self.assertTrue(hasattr(q_model, "visual"))
        self.assertIsInstance(q_model.visual, PTQWrapper)

        self.assertTrue(hasattr(q_model, "language_model"))
        self.assertIsInstance(q_model.language_model, PTQWrapper)

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        self.assertIs(q_model._mode, Mode.NO_QUANT)

        q_model.enable_calibration()
        self.assertIs(q_model._mode, Mode.CALIB)

        # Run forward pass during calibration (text-only)
        input_ids, attention_mask = self._create_text_only_input()
        _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        q_model.freeze_qparams()
        self.assertIs(q_model._mode, Mode.QUANT)

    # -------------------------------------------------------------------------
    # Forward pass tests
    # -------------------------------------------------------------------------

    def test_forward_text_only(self):
        """Test forward pass with text-only input."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        input_ids, attention_mask = self._create_text_only_input()
        _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output structure
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(
            output.last_hidden_state.shape, (batch_size, seq_len, self.hidden_size)
        )

    def test_forward_with_images(self):
        """Test forward pass with image input."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        (
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            position_ids,
        ) = self._create_image_input(thw=thw)

        _ = q_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(
            output.last_hidden_state.shape, (batch_size, seq_len, self.hidden_size)
        )

    def test_forward_with_videos(self):
        """Test forward pass with video input."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        (
            input_ids,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
            position_ids,
        ) = self._create_video_input(thw=thw)

        _ = q_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(
            output.last_hidden_state.shape, (batch_size, seq_len, self.hidden_size)
        )

    def test_forward_with_both_images_and_videos(self):
        """Test forward pass with both image and video inputs (tests deepstack feature combination)."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Calculate visual token count
        num_visual_tokens = (thw[1] // self.spatial_merge_size) * (
            thw[2] // self.spatial_merge_size
        )  # 16 tokens

        # Create input with both images and videos
        batch_size = 1
        seq_len = 64

        # Start with image tokens
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size - 2,
            size=(batch_size, seq_len),
            dtype=torch.long,
        )
        input_ids[0, 0:num_visual_tokens] = self.image_token_id

        # Add video tokens later in the sequence (with some text in between)
        video_start = num_visual_tokens + 10
        input_ids[
            0, video_start : video_start + num_visual_tokens
        ] = self.video_token_id

        # Create pixel values for images
        pixel_values = torch.randn(
            batch_size,
            3,
            thw[0] * self.temporal_patch_size,
            thw[1] * self.patch_size,
            thw[2] * self.patch_size,
        )
        image_grid_thw = torch.tensor([thw])

        # Create pixel values for videos
        pixel_values_videos = torch.randn(
            batch_size,
            3,
            thw[0] * self.temporal_patch_size,
            thw[1] * self.patch_size,
            thw[2] * self.patch_size,
        )
        video_grid_thw = torch.tensor([thw])

        # Run forward pass with both images and videos
        _ = q_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(
            output.last_hidden_state.shape, (batch_size, seq_len, self.hidden_size)
        )

    def test_forward_with_inputs_embeds(self):
        """Test forward pass with inputs_embeds (triggers embedding comparison logic)."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        batch_size = 1
        seq_len = 20

        # Create inputs_embeds with some tokens matching image/video token embeddings
        inputs_embeds = torch.randn(batch_size, seq_len, self.hidden_size)

        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)

        # Run forward pass with inputs_embeds (not input_ids)
        # This should trigger the embedding comparison logic in _get_placeholder_mask
        _ = q_model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        self.assertEqual(
            output.last_hidden_state.shape, (batch_size, seq_len, self.hidden_size)
        )

    def test_forward_with_inputs_embeds_and_images(self):
        """Test forward pass with inputs_embeds and images (triggers QuantQwen3VLModel._get_placeholder_mask)."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        batch_size = 1
        seq_len = 64

        # Create inputs_embeds with some tokens matching image/video token embeddings
        inputs_embeds = torch.randn(batch_size, seq_len, self.hidden_size)

        # Calculate number of visual placeholder tokens needed
        # Each video is represented by multiple tokens after spatial merge
        # Spatial merge reduces the grid size by spatial_merge_size in each dimension
        num_video_tokens = (thw[1] // self.spatial_merge_size) * (
            thw[2] // self.spatial_merge_size
        )

        # Replace first tokens with video placeholder tokens
        # This marks where the video features should be inserted
        embedder = q_model.language_model.wrapped.embed_tokens
        img_tkn = torch.tensor(
            self.image_token_id,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        img_tkn_emb = embedder(img_tkn)
        for i in range(batch_size):
            inputs_embeds[i, :num_video_tokens] = img_tkn_emb

        pixel_values = torch.randn(
            batch_size,
            3,
            thw[0] * self.temporal_patch_size,
            thw[1] * self.patch_size,
            thw[2] * self.patch_size,
        )

        grid_thw = torch.tensor([thw])

        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_len)

        # Run forward pass with inputs_embeds (not input_ids)
        # This should trigger the embedding comparison logic in _get_placeholder_mask
        _ = q_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        )

        q_model.freeze_qparams()

        # Recompute image token embedding as quantization may produce a different result for it
        img_tkn_emb = embedder(img_tkn)
        for i in range(batch_size):
            inputs_embeds[i, :num_video_tokens] = img_tkn_emb

        with torch.no_grad():
            output = q_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        self.assertEqual(
            output.last_hidden_state.shape, (batch_size, seq_len, self.hidden_size)
        )

    def test_forward_input_validation(self):
        """Test that forward validates input requirements."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)

        # Test: neither input_ids nor inputs_embeds
        with self.assertRaises(ValueError) as context:
            _ = q_model()
        self.assertIn(
            "exactly one of input_ids or inputs_embeds", str(context.exception)
        )

        # Test: both input_ids and inputs_embeds
        input_ids, _ = self._create_text_only_input()
        inputs_embeds = torch.randn(1, 10, self.hidden_size)
        with self.assertRaises(ValueError) as context:
            _ = q_model(input_ids=input_ids, inputs_embeds=inputs_embeds)
        self.assertIn(
            "exactly one of input_ids or inputs_embeds", str(context.exception)
        )

    # -------------------------------------------------------------------------
    # Output comparison tests
    # -------------------------------------------------------------------------

    def test_forward_diff_text_only(self):
        """
        Test that quantized output is acceptably close to FP reference for text-only input.
        """
        torch.manual_seed(42)
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Calibrate with multiple inputs
        for _ in range(4):
            input_ids, attention_mask = self._create_text_only_input()
            _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        q_model.freeze_qparams()

        input_ids, attention_mask = self._create_text_only_input()
        with torch.no_grad():
            q_out = q_model(input_ids=input_ids, attention_mask=attention_mask)
            fp_out = self.fp_model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(q_out.last_hidden_state.shape, fp_out.last_hidden_state.shape)
        diff = (fp_out.last_hidden_state - q_out.last_hidden_state).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.7)  # acceptably close

    # -------------------------------------------------------------------------
    # Registration tests
    # -------------------------------------------------------------------------

    def test_registration_in_registry(self):
        """Test that Qwen3VLModel is properly registered."""
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_model import (
            QuantQwen3VLModel,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        wrapper_cls = lookup(Qwen3VLModel)
        self.assertIs(wrapper_cls, QuantQwen3VLModel)

    # -------------------------------------------------------------------------
    # Observer tests
    # -------------------------------------------------------------------------

    def test_observer_count(self):
        """Test that the wrapper has the correct number of observers."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)

        observers = list(q_model._all_observers())
        # Should have 1 local observer (obs_mm_fusion)
        self.assertEqual(len(observers), 1)

    def test_activation_stats_collected_text_only(self):
        """Test that activation statistics are collected during calibration (text-only)."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Run forward pass to collect stats
        input_ids, attention_mask = self._create_text_only_input()
        _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        # Freeze and check qparams exist for multimodal fusion observer
        q_model.freeze_qparams()
        self.assertTrue(q_model.obs_mm_fusion.has_qparams)

    def test_activation_stats_collected_with_images(self):
        """Test that activation statistics are collected during calibration (with images)."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Run forward pass with images
        (
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            position_ids,
        ) = self._create_image_input(thw=thw)
        _ = q_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Freeze and check qparams exist
        q_model.freeze_qparams()
        self.assertTrue(q_model.obs_mm_fusion.has_qparams)

    # -------------------------------------------------------------------------
    # Multiple calibration steps tests
    # -------------------------------------------------------------------------

    def test_multiple_calibration_steps_text_only(self):
        """Test that running multiple calibration iterations works correctly."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Run multiple calibration steps
        for i in range(5):
            input_ids, attention_mask = self._create_text_only_input()
            _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        q_model.freeze_qparams()

        # Verify that observer has quantization parameters
        self.assertTrue(q_model.obs_mm_fusion.has_qparams)

    # -------------------------------------------------------------------------
    # Config override tests
    # -------------------------------------------------------------------------

    def test_dtype_override(self):
        """
        PTQConfig overrides should propagate to observers created by QuantQwen3VLModel.
        """
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        ptq_config.overrides = {"mm_fusion": {"dtype": DType.uint(4)}}
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)

        # Check that overrides were applied
        self.assertEqual(q_model.obs_mm_fusion.dtype, DType.uint(4))

    # -------------------------------------------------------------------------
    # Batch size tests
    # -------------------------------------------------------------------------

    def test_different_batch_sizes_text_only(self):
        """Test that quantization works correctly with different batch sizes."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Calibrate with one batch size
        input_ids, attention_mask = self._create_text_only_input(batch_size=2)
        for _ in range(3):
            _ = q_model(input_ids=input_ids, attention_mask=None)
        q_model.freeze_qparams()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            input_ids, attention_mask = self._create_text_only_input(batch_size)
            with torch.no_grad():
                output = q_model(input_ids=input_ids, attention_mask=None)

            expected_shape = (batch_size, input_ids.shape[1], self.hidden_size)
            self.assertEqual(output.last_hidden_state.shape, expected_shape)

    # -------------------------------------------------------------------------
    # Rope deltas tests
    # -------------------------------------------------------------------------

    def test_rope_deltas_computed_after_forward(self):
        """Test that rope_deltas are computed after forward pass."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # Initially None
        self.assertIsNone(q_model.rope_deltas)

        # After forward pass (text-only), rope_deltas should be computed
        input_ids, attention_mask = self._create_text_only_input()
        output = q_model(input_ids=input_ids, attention_mask=attention_mask)

        # rope_deltas should now be set (even for text-only input)
        self.assertIsNotNone(q_model.rope_deltas)
        self.assertIsNotNone(output.rope_deltas)

    # -------------------------------------------------------------------------
    # _get_rope_index tests
    # -------------------------------------------------------------------------

    def test_get_rope_index_with_images_and_videos(self):
        """Test _get_rope_index generates correct 3D position IDs for mixed image/video input."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)

        # Create input with vision_start_token_id followed by image/video tokens
        batch_size = 1
        seq_len = 64
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        # Add vision_start_token_id, image tokens, then text
        idx = 0
        input_ids[0, idx] = self.fp_model.config.vision_start_token_id
        idx += 1
        input_ids[0, idx] = self.image_token_id
        idx += 1
        input_ids[0, idx : idx + 16] = self.image_token_id  # 16 image tokens
        idx += 16
        input_ids[0, idx : idx + 10] = torch.randint(
            0, self.vocab_size - 2, (10,), dtype=torch.long
        )  # Text tokens
        idx += 10
        input_ids[0, idx] = self.fp_model.config.vision_start_token_id
        idx += 1
        input_ids[0, idx] = self.video_token_id
        idx += 1
        input_ids[0, idx : idx + 32] = self.video_token_id  # 32 video tokens
        idx += 32
        input_ids[0, idx:] = torch.randint(
            0, self.vocab_size - 2, (seq_len - idx,), dtype=torch.long
        )

        attention_mask = torch.ones_like(input_ids)

        # Grid dimensions for images/videos
        # 1 image: (1, 8, 8) -> after spatial merge: (1, 4, 4) -> 16 tokens
        # 2 videos: (1, 4, 8) -> after spatial merge: (1, 2, 4) -> 16 tokens each
        image_grid_thw = torch.tensor([[1, 8, 8]])
        video_grid_thw = torch.tensor([[1, 4, 8], [1, 4, 8]])

        # Call _get_rope_index
        position_ids, mrope_position_deltas = q_model._get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

        # Verify output structure
        self.assertEqual(position_ids.shape, (3, batch_size, seq_len))
        self.assertEqual(mrope_position_deltas.shape, (batch_size, 1))

        # Check that position_ids are 3D (t, h, w) for vision tokens
        # First 16 tokens should be image (t=0, h in [0,3], w in [0,3])
        image_pos_range = position_ids[:, :, 0:16]
        self.assertEqual(image_pos_range.shape, (3, batch_size, 16))

        # Next 10 text tokens should have 3D IDs (t=0, h=w=0 since no visual)
        text_pos_range = position_ids[:, :, 16:26]
        self.assertEqual(text_pos_range.shape, (3, batch_size, 10))

        # Last 32 video tokens should have 3D IDs
        video_pos_range = position_ids[:, :, 26:58]
        self.assertEqual(video_pos_range.shape, (3, batch_size, 32))

    def test_compute_3d_position_ids_reuses_cached_rope_deltas(self):
        """Test that _compute_3d_position_ids reuses cached rope_deltas for subsequent passes."""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLModel(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        # First forward pass to compute rope_deltas
        input_ids, attention_mask = self._create_text_only_input(
            batch_size=1, seq_len=10
        )
        _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        # rope_deltas should now be cached
        self.assertIsNotNone(q_model.rope_deltas)
        assert q_model.rope_deltas is not None  # for mypy
        cached_rope_deltas = q_model.rope_deltas.clone()

        # Simulate autoregressive generation with past_key_values
        # This should trigger the else branch that reuses cached rope_deltas
        seq_len_new = 15  # Generate 5 more tokens
        input_ids_new = torch.randint(
            0, self.vocab_size, size=(1, seq_len_new), dtype=torch.long
        )
        attention_mask_new = torch.ones_like(input_ids_new)

        # Create mock past_key_values that simulates previous cache
        class MockPastKeyValues:
            def __init__(self, seq_length):
                self._seq_length = seq_length

            def get_seq_length(self):
                return self._seq_length  # Previous sequence length

        past_key_values = MockPastKeyValues(seq_length=10)
        inputs_embeds_new = torch.randn(1, seq_len_new, self.hidden_size)

        # Call _compute_3d_position_ids with past_key_values (should reuse cached deltas)
        position_ids = q_model._compute_3d_position_ids(
            input_ids=None,
            inputs_embeds=inputs_embeds_new,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            cache_position=None,
            past_key_values=past_key_values,
        )

        # Verify rope_deltas were reused (not recomputed)
        self.assertTrue(torch.equal(q_model.rope_deltas, cached_rope_deltas))

        # Verify position_ids shape
        assert position_ids is not None  # for mypy
        self.assertEqual(position_ids.shape, (3, 1, seq_len_new))

        # Verify position_ids are monotonic (increasing) and properly offset
        pos_ids_first = position_ids[0, 0, 0].item()
        pos_ids_last = position_ids[0, 0, -1].item()
        self.assertGreater(pos_ids_last, pos_ids_first)
        self.assertGreaterEqual(pos_ids_first, past_key_values.get_seq_length())

    # -------------------------------------------------------------------------
    # Circle conversion tests
    # -------------------------------------------------------------------------

    def test_graph_tracing_behavior_with_images(self):
        """Test that QuantQwen3VLModel behavior in graph tracing mode."""
        import tico

        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)

        # Prepare the model for quantization
        prepared_model = tico.quantization.prepare(
            copy.deepcopy(self.fp_model), ptq_config, inplace=False
        )
        prepared_model.eval()

        # Create example input
        (
            input_ids,
            attention_mask,
            pixel_values,
            grid_thw,
            position_ids,
        ) = self._create_image_input(batch_size=1, seq_len=64, thw=thw)

        # Calibrate with text-only input
        with torch.no_grad():
            prepared_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                position_ids=position_ids,
            )

        # Convert to quantized model
        quantized_model = tico.quantization.convert(prepared_model, inplace=False)

        # Create example input as namedtuple
        from collections import namedtuple

        ModelInput = namedtuple(
            "ModelInput",
            [
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values",
                "inputs_embeds",
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "cache_position",
            ],
        )

        example_input = ModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=pixel_values,
            pixel_values_videos=None,
            image_grid_thw=grid_thw,
            video_grid_thw=None,
            cache_position=None,
        )

        # Compute quantization error
        quantized_model.wrapped.force_export = True
        with torch.no_grad():
            test_input = example_input._asdict()
            quant_out = quantized_model(**test_input).last_hidden_state
            fp_out = self.fp_model(**test_input).last_hidden_state

        err = (quant_out - fp_out).abs().mean().item()
        self.assertLess(err, 1.0)

    def test_graph_tracing_behavior_with_videos(self):
        """Test that QuantQwen3VLModel behavior in graph tracing mode."""
        import tico

        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)

        # Prepare the model for quantization
        prepared_model = tico.quantization.prepare(
            copy.deepcopy(self.fp_model), ptq_config, inplace=False
        )
        prepared_model.eval()

        # Create example input
        (
            input_ids,
            attention_mask,
            pixel_values_videos,
            grid_thw,
            position_ids,
        ) = self._create_video_input(batch_size=1, seq_len=64, thw=thw)

        # Calibrate with text-only input
        with torch.no_grad():
            prepared_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=grid_thw,
                position_ids=position_ids,
            )

        # Convert to quantized model
        quantized_model = tico.quantization.convert(prepared_model, inplace=False)

        # Create example input as namedtuple
        from collections import namedtuple

        ModelInput = namedtuple(
            "ModelInput",
            [
                "input_ids",
                "attention_mask",
                "position_ids",
                "past_key_values",
                "inputs_embeds",
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "cache_position",
            ],
        )

        example_input = ModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=None,
            video_grid_thw=grid_thw,
            cache_position=None,
        )

        # Compute quantization error
        quantized_model.wrapped.force_export = True
        with torch.no_grad():
            test_input = example_input._asdict()
            quant_out = quantized_model(**test_input).last_hidden_state
            fp_out = self.fp_model(**test_input).last_hidden_state

        err = (quant_out - fp_out).abs().mean().item()
        self.assertLess(err, 1.0)


if __name__ == "__main__":
    unittest.main()

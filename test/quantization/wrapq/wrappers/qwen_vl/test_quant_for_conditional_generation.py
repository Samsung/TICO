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
from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import (
    QuantQwen3VLForConditionalGeneration,
)


skip_msg = "transformers not installed — skipping Qwen3VLForConditionalGeneration tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLForConditionalGeneration(unittest.TestCase):
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
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

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

        cls.fp_model = Qwen3VLForConditionalGeneration(cfg)
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
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)

        # Check that submodules are wrapped
        self.assertTrue(hasattr(q_model, "model"))
        self.assertIsInstance(q_model.model, PTQWrapper)

        self.assertTrue(hasattr(q_model, "lm_head"))
        self.assertIsInstance(q_model.lm_head, PTQWrapper)

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        ptq_config = self._make_ptq_config(grid_thw=(1, 8, 8))
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)
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
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)
        q_model.enable_calibration()

        input_ids, attention_mask = self._create_text_only_input()
        _ = q_model(input_ids=input_ids, attention_mask=attention_mask)

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output structure
        self.assertTrue(hasattr(output, "logits"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(output.logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_forward_with_images(self):
        """Test forward pass with image input."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)
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
            position_ids=position_ids,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "logits"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(output.logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_forward_with_videos(self):
        """Test forward pass with video input."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)
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
            position_ids=position_ids,
        )

        q_model.freeze_qparams()

        with torch.no_grad():
            output = q_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
            )

        # Check output structure
        self.assertTrue(hasattr(output, "logits"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(output.logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_forward_with_both_images_and_videos(self):
        """Test forward pass with both image and video inputs (tests deepstack feature combination)."""
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)
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
        self.assertTrue(hasattr(output, "logits"))
        self.assertTrue(hasattr(output, "past_key_values"))
        self.assertTrue(hasattr(output, "rope_deltas"))

        # Check output shape
        batch_size, seq_len = input_ids.shape
        self.assertEqual(output.logits.shape, (batch_size, seq_len, self.vocab_size))

    # -------------------------------------------------------------------------
    # Registration tests
    # -------------------------------------------------------------------------

    def test_registration_in_registry(self):
        """Test that Qwen3VLForConditionalGeneration is properly registered."""
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import (
            QuantQwen3VLForConditionalGeneration,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        wrapper_cls = lookup(Qwen3VLForConditionalGeneration)
        self.assertIs(wrapper_cls, QuantQwen3VLForConditionalGeneration)


if __name__ == "__main__":
    unittest.main()

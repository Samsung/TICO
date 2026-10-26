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
from typing import cast, TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

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
    fp_model: "Qwen3VLForConditionalGeneration"
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
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig,
            Qwen3VLTextConfig,
            Qwen3VLVisionConfig,
        )
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

        assert isinstance(cfg.text_config, Qwen3VLTextConfig)
        assert isinstance(cfg.vision_config, Qwen3VLVisionConfig)
        assert cfg.image_token_id < cfg.text_config.vocab_size
        assert cfg.video_token_id < cfg.text_config.vocab_size
        assert cfg.vision_config.out_hidden_size == cfg.text_config.hidden_size

        cls.fp_model = Qwen3VLForConditionalGeneration(cfg)
        assert isinstance(cfg.vision_config.patch_size, int)
        cls.patch_size = cfg.vision_config.patch_size
        assert isinstance(cfg.vision_config.temporal_patch_size, int)
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

            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0

            # Process visual tokens
            if len(image_positions) > 0:
                # Group consecutive placeholder tokens into a single visual object
                # All consecutive image tokens represent ONE image/video
                start_pos = int(image_positions[0].item())

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

    def _create_patchified_pixel_values(
        self,
        thw: Tuple[int, int, int],
        *,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Create processor-shaped patches for a batch-one visual input."""
        if batch_size != 1:
            raise ValueError(
                "Synthetic Qwen3-VL visual tests support batch size 1 only."
            )
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )

        vision_cfg = self.fp_model.config.vision_config
        assert isinstance(vision_cfg, Qwen3VLVisionConfig)
        num_patches = thw[0] * thw[1] * thw[2]
        assert isinstance(vision_cfg.patch_size, int)
        assert isinstance(vision_cfg.temporal_patch_size, int)
        patch_dim = (
            vision_cfg.in_channels
            * vision_cfg.temporal_patch_size
            * vision_cfg.patch_size
            * vision_cfg.patch_size
        )
        return torch.randn(1, num_patches, patch_dim)

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

        pixel_values = self._create_patchified_pixel_values(thw, batch_size=batch_size)
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

    def _require_position_id_generation_hook(self):
        if not callable(
            getattr(self.fp_model, "_prepare_position_ids_for_generation", None)
        ):
            self.skipTest(
                "This Transformers version does not use the position-ID generation hook"
            )

    def test_generation_config_assignment_updates_underlying_model(self):
        """Keep the writable GenerationMixin configuration contract."""
        q_model = QuantQwen3VLForConditionalGeneration(
            copy.deepcopy(self.fp_model),
            qcfg=self._make_ptq_config(grid_thw=(1, 8, 8)),
        )
        generation_config = copy.deepcopy(q_model.generation_config)
        generation_config.max_new_tokens = 5

        q_model.generation_config = generation_config

        self.assertIs(q_model.generation_config, generation_config)
        self.assertIs(q_model.module.generation_config, generation_config)
        self.assertEqual(q_model.generation_config.max_new_tokens, 5)

    def test_generation_hook_forwards_current_transformers_arguments(self):
        """Forward positional generation inputs to the underlying HF model."""
        q_model = QuantQwen3VLForConditionalGeneration(
            self.fp_model, qcfg=self._make_ptq_config(grid_thw=(1, 8, 8))
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        expected = {"input_ids": input_ids}
        with patch.object(
            q_model.module, "prepare_inputs_for_generation", return_value=expected
        ) as prepare_inputs:
            actual = q_model.prepare_inputs_for_generation(
                input_ids, 3, None, attention_mask, None, True
            )

        self.assertIs(actual, expected)
        prepare_inputs.assert_called_once()
        self.assertIs(prepare_inputs.call_args.kwargs["input_ids"], input_ids)
        self.assertIs(prepare_inputs.call_args.kwargs["attention_mask"], attention_mask)
        self.assertEqual(prepare_inputs.call_args.kwargs["next_sequence_length"], 3)
        self.assertTrue(prepare_inputs.call_args.kwargs["is_first_iteration"])

    def test_prepare_position_ids_for_generation_returns_3d_layout(self):
        """generate() must inject 3D mrope position ids, not 1D sequential ones.

        ``GenerationMixin.generate`` auto-creates position ids through
        ``self._prepare_position_ids_for_generation`` whenever the forward
        signature declares ``position_ids``. The generic implementation
        yields 1D sequential ids, which bypass the 3D mrope computation in
        the vision-language wrapper and corrupt image-token RoPE. The
        wrapper must delegate to the wrapped Qwen3-VL model, which returns
        the [4, batch, seq] (text + 3 mrope) layout.
        """
        self._require_position_id_generation_hook()
        thw = (1, 8, 8)
        ptq_config = self._make_ptq_config(grid_thw=thw)
        q_model = QuantQwen3VLForConditionalGeneration(self.fp_model, qcfg=ptq_config)

        (
            input_ids,
            attention_mask,
            _pixel_values,
            image_grid_thw,
            _position_ids,
        ) = self._create_image_input(thw=thw)

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        position_ids = q_model._prepare_position_ids_for_generation(
            input_ids, model_kwargs
        )

        batch_size, seq_len = input_ids.shape
        self.assertEqual(position_ids.shape, (4, batch_size, seq_len))

    def test_generation_position_ids_expand_along_batch_axis(self):
        """Input expansion during generate() must preserve the 3D mrope layout.

        Beam search and multi-sample decoding expand the batch through
        ``_expand_inputs_for_generation``. Position ids prepared for
        generation use the [4, batch, seq] mrope layout, where the batch
        lives on axis 1 -- unlike ordinary tensors, whose batch is axis 0.
        The generic ``GenerationMixin`` expansion (``repeat_interleave`` on
        dim=0) would interleave the four rope rows with batch copies and
        silently corrupt positions; the wrapped Qwen3-VL model expands them
        along dim=1 instead. The wrapper must delegate to it so that
        expanded position ids stay aligned with the expanded batch for any
        ``expand_size`` (including the no-op case ``expand_size=1``).

        A text-only batch is sufficient: the mis-expansion is a pure
        layout/batching bug and does not require vision inputs. The FP
        model is deep-copied so that rope-delta state mutated during the
        hook call does not leak into other tests via the shared fixture.
        """
        self._require_position_id_generation_hook()

        # Avoid changing the class-level FP model's mutable generation state.
        fp_model = copy.deepcopy(self.fp_model).eval()
        q_model = QuantQwen3VLForConditionalGeneration(
            fp_model,
            qcfg=self._make_ptq_config(grid_thw=(1, 8, 8)),
        ).eval()

        # Text-only input is enough to expose the position-layout regression.
        input_ids = torch.tensor([[0, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long)
        batch_size, seq_len = input_ids.shape

        position_ids = q_model._prepare_position_ids_for_generation(
            input_ids,
            {"attention_mask": attention_mask},
        )
        self.assertEqual(position_ids.shape, (4, batch_size, seq_len))

        for expand_size in (1, 2, 3):
            with self.subTest(expand_size=expand_size):
                expanded_ids, expanded_kwargs = q_model._expand_inputs_for_generation(
                    expand_size=expand_size,
                    is_encoder_decoder=False,
                    input_ids=input_ids.clone(),
                    attention_mask=attention_mask.clone(),
                    position_ids=position_ids.clone(),
                )

                self.assertEqual(
                    expanded_kwargs["position_ids"].shape,
                    (4, batch_size * expand_size, seq_len),
                )
                torch.testing.assert_close(
                    expanded_kwargs["position_ids"],
                    position_ids.repeat_interleave(expand_size, dim=1),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    expanded_ids,
                    input_ids.repeat_interleave(expand_size, dim=0),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    expanded_kwargs["attention_mask"],
                    attention_mask.repeat_interleave(expand_size, dim=0),
                    rtol=0,
                    atol=0,
                )

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
        input_ids[0, video_start : video_start + num_visual_tokens] = (
            self.video_token_id
        )

        # Create pixel values for images
        pixel_values = self._create_patchified_pixel_values(thw, batch_size=batch_size)
        image_grid_thw = torch.tensor([thw])

        # Create pixel values for videos
        pixel_values_videos = self._create_patchified_pixel_values(
            thw, batch_size=batch_size
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

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

from collections import namedtuple
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import get_model_arg
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel",
)
class QuantQwen3VLModel(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLModel module.

    This is the main multimodal model that combines vision and language processing:
    - Vision model (Qwen3VLVisionModel): Processes images/videos
    - Language model (Qwen3VLTextModel): Processes text and generates outputs
    - Multimodal fusion: Combines text and visual embeddings
    """

    # This boolean flag enforces model behavior that is only activated during model graph tracing (torch.export.export).
    # This flag is used in unit tests only in order to check the behavior without actually exporting the model.
    force_export: bool = False

    rope_deltas: Optional[torch.Tensor]  # Type annotation for registered buffer

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config

        # Extract visual_start_idx from config for the insertion
        # of visual embddings into the embedded input prompt
        self.visual_start_idx = self._get_visual_start_idx(qcfg)

        self.image_token_id = fp_model.config.image_token_id
        self.video_token_id = fp_model.config.video_token_id

        # Wrap vision model
        self.visual = PTQWrapper(
            fp_model.visual,
            qcfg=qcfg.child("visual") if qcfg else None,
            fp_name=f"{fp_name}.visual",
        )

        # Wrap language model
        self.language_model = PTQWrapper(
            fp_model.language_model,
            qcfg=qcfg.child("language_model") if qcfg else None,
            fp_name=f"{fp_name}.language_model",
        )

        # Cache for rope_deltas - register as buffer for proper export handling
        # persistent=False means it won't be saved in state_dict
        self.register_buffer("rope_deltas", None, persistent=False)

        # Multimodal fusion observers (masked scatter results)
        self.obs_mm_fusion = self._make_obs("mm_fusion")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass with fake quantization.

        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask of shape (batch_size, sequence_length)
            position_ids: Position IDs for RoPE
            past_key_values: Past key-value caches for autoregressive generation
            inputs_embeds: Input embeddings of shape (batch_size, sequence_length, hidden_size)
            pixel_values: Image pixel values of shape (batch_size, C, H, W)
            pixel_values_videos: Video pixel values
            image_grid_thw: Grid dimensions for images of shape (num_images, 3)
            video_grid_thw: Grid dimensions for videos
            cache_position: Cache positions for generation
            **kwargs: Additional keyword arguments

        Returns:
            Model output containing last hidden state, past key values, etc.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        if torch.compiler.is_compiling() or self.force_export:
            assert (
                position_ids is not None
            ), "position_ids must be provided as an argument since it's computation cannot be converted to Circle"

        # Validate input
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # Generate input embeddings from token IDs
        if inputs_embeds is None:
            assert hasattr(self.language_model.wrapped, "embed_tokens") and isinstance(
                self.language_model.wrapped.embed_tokens, PTQWrapper
            )
            inputs_embeds = self.language_model.wrapped.embed_tokens(input_ids)

        deepstack_image_embeds: list = None  # type: ignore[assignment]
        deepstack_video_embeds: list = None  # type: ignore[assignment]

        image_mask = None
        video_mask = None

        # Process images
        if pixel_values is not None:
            # Get image features from vision model
            image_outputs = self._get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features

            # Concatenate all image features
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # Create mask for image placeholder tokens
            image_mask, _ = self._get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )

            # Replace image placeholders with actual visual features
            if torch.compiler.is_compiling() or self.force_export:
                # If we are exporting the model as a static graph.
                # This assumes visual tokens are starting at visual_start_idx in the prompt
                # Violating this requirement will lead to a corrupted prompt
                inputs_embeds = self._fuse_text_n_image(
                    inputs_embeds, image_embeds, visual_start_idx=self.visual_start_idx
                )
            else:
                # This operation cannot be converted to Circle because it produces data-dependent dynamic shapes
                inputs_embeds = self._masked_scatter(
                    inputs_embeds, image_mask, image_embeds
                )

            # Quantize multimodal fusion result
            inputs_embeds = self._fq(inputs_embeds, self.obs_mm_fusion)

        # Process videos
        if pixel_values_videos is not None:
            # Get video features from vision model
            video_outputs = self._get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features

            # Concatenate all video features
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # Create mask for video placeholder tokens
            _, video_mask = self._get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )

            # Replace video placeholders with actual visual features
            if torch.compiler.is_compiling() or self.force_export:
                # If we are exporting the model as a static graph
                # This assumes visual tokens are starting at visual_start_idx in the prompt
                # Violating this requirement will lead to a corrupted prompt
                inputs_embeds = self._fuse_text_n_image(
                    inputs_embeds, video_embeds, visual_start_idx=self.visual_start_idx
                )
            else:
                # This operation cannot be converted to Circle because it produces data-dependent dynamic shapes
                inputs_embeds = self._masked_scatter(
                    inputs_embeds, video_mask, video_embeds
                )

            # Quantize multimodal fusion result
            inputs_embeds = self._fq(inputs_embeds, self.obs_mm_fusion)

        # Combine deepstack features from images and videos
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # Aggregate visual masks and deepstack features
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Compute 3D position IDs if not provided
        # Note: This involves only integer operations, no quantization needed
        if position_ids is None:
            position_ids = self._compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
            )

        # Pass through language model (wrapped with PTQWrapper)
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        # Return output with rope_deltas
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLModelOutputWithPast,
        )

        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    @staticmethod
    def _get_visual_start_idx(qcfg: Optional[PTQConfig]) -> int | None:
        visual_start_idx = get_model_arg(
            qcfg, "vision", "visual_start_idx", default=None
        )
        if visual_start_idx is None:
            raise ValueError(
                "vision.visual_start_idx must be specified in PTQConfig.model_args for "
                "QuantQwen3VLVisionModel.\n"
                "Example:\n"
                "PTQConfig(\n"
                "    model_args={\n"
                "        'vision': {\n"
                "            'visual_start_idx': 4,\n"
                "        }\n"
                "    }\n"
                ")"
            )

        visual_start_idx = int(visual_start_idx)
        if visual_start_idx < 0:
            raise ValueError(
                f"vision.visual_start_idx must be greater than zero, but got {visual_start_idx}."
            )
        return visual_start_idx

    @staticmethod
    def _fuse_text_n_image(inputs_embeds, visual_embeds, visual_start_idx):
        num_visual_tokens = visual_embeds.shape[0]
        flat_inputs = inputs_embeds.view(-1, inputs_embeds.shape[-1])
        flat_inputs[
            visual_start_idx : visual_start_idx + num_visual_tokens
        ] = visual_embeds
        inputs_embeds = flat_inputs.view_as(inputs_embeds)
        return inputs_embeds

    @staticmethod
    def _masked_scatter(inputs_embeds, visual_mask, visual_embeds):
        # Use indexing assignment instead of masked_scatter for better Circle support
        # (TICO can't convert torch.masked_scatter operator)
        flat_inputs = inputs_embeds.view(-1, inputs_embeds.shape[-1])
        mask_2d = visual_mask[..., 0]  # Get mask for the first dimension only
        _, indices = torch.nonzero(mask_2d, as_tuple=True)
        flat_inputs[indices] = visual_embeds
        inputs_embeds = flat_inputs.view_as(inputs_embeds)
        return inputs_embeds

    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ):
        """Get image features from vision model."""
        # Convert to vision model dtype
        pixel_values = pixel_values.type(self.visual.wrapped.module.dtype)

        # Process through vision model
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)

        # Get pooled output
        image_embeds, deepstack_features = (
            (vision_output.pooler_output, vision_output.deepstack_features)
            if self.visual.wrapped.has_deepstack_model_output
            else vision_output
        )

        # Split by image based on grid_thw
        spatial_merge_size = self.visual.wrapped.module.spatial_merge_size
        split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)

        OutputWithDeepstackFeatures = namedtuple(
            "OutputWithDeepstackFeatures", ["pooler_output", "deepstack_features"]
        )
        return OutputWithDeepstackFeatures(
            pooler_output=image_embeds, deepstack_features=deepstack_features
        )

    def _get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs,
    ):
        """Get video features from vision model (same as image processing)."""
        return self._get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    def _get_placeholder_mask(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
    ):
        """
        Obtain multimodal placeholder mask from input_ids or inputs_embeds.
        Validates that placeholder token count matches feature length.
        """
        if input_ids is None:
            # Compare embeddings directly
            embedder = self.language_model.wrapped.embed_tokens
            img_tkn = torch.tensor(
                self.image_token_id,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            img_tkn_emb = embedder(img_tkn)
            special_image_mask = (inputs_embeds == img_tkn_emb).all(-1)

            vid_tkn = torch.tensor(
                self.video_token_id,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            vid_tkn_emb = embedder(vid_tkn)
            special_video_mask = (inputs_embeds == vid_tkn_emb).all(-1)
        else:
            # Compare token IDs
            special_image_mask = input_ids == self.image_token_id
            special_video_mask = input_ids == self.video_token_id

        # Count image tokens
        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if image_features is not None:
            # Validate image tokens count matches features
            if not torch.compiler.is_compiling() and not self.force_export:
                assert (
                    inputs_embeds[special_image_mask].numel() == image_features.numel()
                ), f"Image features ({image_features.shape[0]}) and image tokens ({n_image_tokens}) do not match"

        # Count video tokens
        n_video_tokens = special_video_mask.sum()
        special_video_mask = (
            special_video_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if video_features is not None:
            # Validate video tokens count matches features
            assert (
                inputs_embeds[special_video_mask].numel() == video_features.numel()
            ), f"Video features ({video_features.shape[0]}) and video tokens ({n_video_tokens}) do not match"

        return special_image_mask, special_video_mask

    def _compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        past_key_values=None,
    ) -> torch.Tensor | None:
        """
        Compute 3D position IDs for multimodal RoPE.
        Note: This involves only integer operations, no quantization needed.
        """
        past_key_values_length = (
            0 if past_key_values is None else past_key_values.get_seq_length()
        )

        if self.rope_deltas is None or past_key_values_length == 0:
            position_ids, rope_deltas = self._get_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            # Type narrowing for mypy: assign to local non-None variable
            assert inputs_embeds is not None
            _rope_deltas: torch.Tensor = self.rope_deltas  # type: ignore[assignment]
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (past_key_values_length + _rope_deltas).to(inputs_embeds.device)
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        return position_ids

    def _get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        video_grid_thw: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Calculate 3D rope index based on image and video sizes."""
        # Since we use timestamps to separate videos, video_grid_thw should be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.visual.wrapped.module.spatial_merge_size
        image_token_id = self.module.config.image_token_id
        video_token_id = self.module.config.video_token_id
        vision_start_token_id = self.module.config.vision_start_token_id

        mrope_position_deltas = []

        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)

            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, current_input_ids in enumerate(total_input_ids):
                input_ids = current_input_ids[attention_mask[i] == 1]

                # Count images and videos
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    # Find next image or video token
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1

                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1

                    if ed_image < ed_video:
                        # This is an image
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        # This is a video
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )

                    text_len = ed - st

                    # Text position IDs
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )

                    # Vision position IDs (3D)
                    # t_index is always 0 because llm_grid_t is always 1
                    # (we use timestamps to encode temporal information for videos)
                    t_index = (
                        torch.arange(llm_grid_t, device=input_ids.device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=input_ids.device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )

                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Trailing text after all images/videos
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            # Fallback for text-only input
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module and wrapped submodules."""
        # Local observers
        yield self.obs_mm_fusion

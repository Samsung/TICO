#!/usr/bin/env python3
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

"""
Example script for quantizing and converting Qwen3VLForConditionalGeneration to Circle format.

This script demonstrates:
1. Loading a Qwen3VL vision-language model (with lm_head for generation)
2. Preparing calibration data with text, images, and videos
3. Configuring PTQ (Post-Training Quantization)
4. Calibrating the model to collect statistics
5. Converting to quantized model
6. Evaluating quantization accuracy
7. Converting to Circle format for deployment

Usage:
    python quantize_for_conditional_generation.py
"""

import copy
import sys
from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


# Check if transformers is available
if not has_transformers_for("qwen3-vl"):
    print(
        "Error: Required transformers package not installed. Cannot test Qwen3VLForConditionalGeneration."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

ModelInput = namedtuple(
    "ModelInput",
    [
        "input_ids",
        "attention_mask",
        "position_ids",
        "past_key_values",
        "inputs_embeds",
        "labels",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "cache_position",
        "logits_to_keep",
    ],
)


def create_visual_input(
    seq_len: int,
    thw: Tuple[int, int, int],
    spatial_merge_size: int,
    temporal_patch_size: int,
    spatial_patch_size: int,
    vocab_size: int,
    image_token_id: int,
):
    """Helper to create input with videos or images."""
    assert (
        image_token_id >= vocab_size - 2
    ), f"Visual token Id {image_token_id} must be outside text vocabulary range 0...{vocab_size-2}."

    batch_size = 1

    # Calculate number of visual placeholder tokens needed
    # Each video is represented by multiple tokens after spatial merge
    # Spatial merge reduces the grid size by spatial_merge_size in each dimension
    num_video_tokens = (thw[1] // spatial_merge_size) * (thw[2] // spatial_merge_size)
    assert (
        num_video_tokens <= seq_len
    ), f"{num_video_tokens} video tokens can't fit into input sequence of length {seq_len}"

    # Create input_ids with random text tokens
    input_ids = torch.randint(
        low=0,
        high=vocab_size - 2,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    # Replace first tokens with video placeholder tokens
    # This marks where the video features should be inserted
    for i in range(batch_size):
        input_ids[i, :num_video_tokens] = image_token_id

    num_temporal_patches, num_spatial_patches_h, num_spatial_patches_w = thw

    # Create pixel values for videos
    pixel_values = torch.randn(
        batch_size,
        3,
        num_temporal_patches * temporal_patch_size,
        num_spatial_patches_h * spatial_patch_size,
        num_spatial_patches_w * spatial_patch_size,
    )
    grid_thw = torch.tensor([thw])

    # Compute position_ids for 3D RoPE
    # This replicates the logic from _get_rope_index but pre-computes it
    position_ids = compute_3d_position_ids(
        input_ids=input_ids,
        thw=thw,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
    )

    return ModelInput(
        input_ids=input_ids,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=pixel_values,
        pixel_values_videos=None,
        image_grid_thw=grid_thw,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
    )


def compute_3d_position_ids(
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
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
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
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

            # Update st to after all visual placeholder tokens
            # The number of visual tokens is (thw[1] // spatial_merge_size) * (thw[2] // spatial_merge_size)
            num_visual_tokens = (thw[1] // spatial_merge_size) * (
                thw[2] // spatial_merge_size
            )
            st = start_pos + num_visual_tokens

        # Trailing text
        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = seq_len - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, :] = llm_positions

    return position_ids


def generate_calibration_data(
    batch_size: int,
    seq_len: int,
    thw: Tuple[int, int, int],
    spatial_merge_size: int,
    temporal_patch_size: int,
    spatial_patch_size: int,
    vocab_size: int,
    image_token_id: int,
):
    calibration_data = []
    for i in range(batch_size):
        x = create_visual_input(
            seq_len,
            thw,
            spatial_merge_size,
            temporal_patch_size,
            spatial_patch_size,
            vocab_size,
            image_token_id,
        )
        calibration_data.append(x)
    return calibration_data


def main():
    # Create Qwen3VL configuration
    cfg = Qwen3VLConfig(
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,  # Smaller depth for faster testing
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
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
    thw = (1, 8, 8)

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig(
        model_args={
            "vision": {
                "grid_thw": thw,
                "visual_start_idx": 0,
                "spatial_merge_size": 2,
            }
        }
    )

    # Load the model
    model = Qwen3VLForConditionalGeneration(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Generate calibration data
    calibration_data = generate_calibration_data(
        batch_size=10,
        seq_len=50,
        thw=thw,
        spatial_merge_size=cfg.vision_config.spatial_merge_size,
        temporal_patch_size=cfg.vision_config.temporal_patch_size,
        spatial_patch_size=cfg.vision_config.patch_size,
        vocab_size=cfg.text_config.vocab_size,
        image_token_id=cfg.image_token_id,
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for calibration_input in calibration_data:
            prepared_model(**calibration_input._asdict())

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute quantization error metrics
    with torch.no_grad():
        test_input = calibration_data[0]._asdict()
        test_input["position_ids"] = None
        quant_out = quantized_model(**test_input, return_dict=False)[0]
        fp_out = orig_model(**test_input).logits

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format

    example_input = calibration_data[0]
    quantized_model.wrapped.config.return_dict = False
    circle_model = tico.convert(quantized_model.eval(), example_input)

    # Save the Circle model
    filename = "qwen3vl_for_conditional_generation.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

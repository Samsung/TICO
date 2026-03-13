#!/usr/bin/env python3
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

import copy
import sys
from collections import namedtuple

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
    print("Error: transformers package not installed. Cannot test Qwen3VLVisionModel.")
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def generate_calibration_data(batch_size: int, sample_shape: tuple) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(batch_size):
        x = torch.randn(sample_shape)
        calibration_data.append(x)
    return calibration_data


def main():
    # Create the vision model configuration
    # Based on Qwen3VLVisionModel structure:
    # (patch_embed): Qwen3VLVisionPatchEmbed(
    #     (proj): Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    # )
    # (pos_embed): Embedding(2304, 1024)
    cfg = Qwen3VLVisionConfig(
        hidden_size=1024,
        num_position_embeddings=2304,  # 48x48 spatial grid
        temporal_patch_size=2,
        patch_size=16,
        depth=2,  # Number of transformer blocks (reduced for example)
    )
    model = Qwen3VLVisionModel(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Define grid_thw for fixed input size
    # grid_thw: (num_images, 3) with (temporal, height, width)
    # Example: [1, 24, 24] means 1 video with 1 temporal patch, 24 vertical, 24 horizontal
    # Total patches: 1 * 24 * 24 = 576
    THW = namedtuple(
        "THW", ["num_temporal_patches", "num_height_patches", "num_width_patches"]
    )
    vision_grid_thw = THW(1, 24, 24)
    grid_thw = torch.tensor([vision_grid_thw])

    # Input to patch_embed: (batch_size, in_channels, depth, height, width)
    # Example: (1, 3, 16, 384, 384)
    # - batch_size: 1
    # - in_channels: 3 (RGB)
    # - depth: frames = num_temporal_patches * temporal_patch_size = 1 * 2 = 2 frames
    # - height: num_height_patches * patch_size = 24 * 16 = 384
    # - width: num_width_patches * patch_size = 24 * 16 = 384
    num_frames = vision_grid_thw.num_temporal_patches * cfg.temporal_patch_size
    frame_height = vision_grid_thw.num_height_patches * cfg.patch_size
    frame_width = vision_grid_thw.num_width_patches * cfg.patch_size
    input_shape = (1, cfg.in_channels, num_frames, frame_height, frame_width)

    print(f"Input shape: {input_shape}")
    print(f"grid_thw: {grid_thw.tolist()}")

    # Generate calibration data
    calibration_data = generate_calibration_data(
        batch_size=20, sample_shape=input_shape
    )

    # Configure PTQ with vision_grid_thw override
    # This is required for QuantQwen3VLVisionModel to precompute RoPE embeddings
    ptq_config = tico.quantization.config.ptq.PTQConfig()
    setattr(ptq_config, "vision_grid_thw", vision_grid_thw)

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            prepared_model(batch, grid_thw)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Detect transformers version
    try:
        # transformers version 5.3.0
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            BaseModelOutputWithDeepstackFeatures,
        )
    except ImportError:
        # transformers version 4.57.0
        transformers_version = "old"
    else:
        # transformers version 5.3.0
        transformers_version = "new"

    # Compute PEIR (Peak Error-to-Input Ratio) between quantized model and original model
    with torch.no_grad():
        test_input = calibration_data[0]
        quant_out = quantized_model(test_input, grid_thw)
        fp_out = orig_model(test_input, grid_thw)

        # The structure of quant_out depends on transformers version
        if transformers_version == "new":
            quant_out = quant_out.pooler_output
            fp_out = fp_out.pooler_output
        else:
            quant_out = quant_out[0]
            fp_out = fp_out[0]

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format
    # example_inputs: (hidden_states, grid_thw)
    example_input = (calibration_data[0], grid_thw)
    circle_model = tico.convert(quantized_model.eval(), example_input)

    # Save the Circle model
    filename = "qwen3vl_vision_model.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

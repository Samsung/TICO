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

import torch
import torch.nn as nn

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

# Check if transformers is available

if not has_transformers_for("qwen3-vl"):
    print(
        "Error: Required transformers package not installed. Cannot test Qwen3VLVisionPatchEmbed."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed


def generate_calibration_data(batch_size: int, sample_shape) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(batch_size):
        x = torch.randn(sample_shape)
        calibration_data.append(x)
    return calibration_data


def main():
    # Create the vision patch embed model
    cfg = Qwen3VLVisionConfig(
        in_channels=3,
        hidden_size=1024,
        temporal_merge_size=2,
        patch_size=16,
    )
    model = Qwen3VLVisionPatchEmbed(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Qwen3VLVisionPatchEmbed(
    #     (proj): Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    # )
    assert model.proj.in_channels == 3
    assert model.proj.out_channels == 1024
    assert model.proj.kernel_size == (2, 16, 16)
    assert model.proj.stride == (2, 16, 16)

    # Generate calibration data
    # Input shape: (batch_size, in_channels, depth, height, width)
    # Example: (2, 3, 8, 224, 224) - 2 videos, RGB, 8 frames, 224x224 resolution
    calibration_data = generate_calibration_data(
        batch_size=20, sample_shape=(2, 3, 8, 224, 224)
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            prepared_model(batch)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Input Ratio) between quantized model and original model
    with torch.no_grad():
        quant_out = quantized_model(calibration_data[0])
        fp_out = orig_model(calibration_data[0])

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format
    # example_inputs shape: (batch_size, in_channels, depth, height, width)
    example_inputs = (torch.randn(2, 3, 8, 224, 224),)
    circle_model = tico.convert(quantized_model, example_inputs)

    # Save the Circle model
    filename = "quantized_vision_patch_embed.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

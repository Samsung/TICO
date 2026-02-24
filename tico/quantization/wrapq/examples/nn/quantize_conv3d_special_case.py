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

import tico
import tico.quantization
import tico.quantization.config.ptq

import torch
import torch.nn as nn
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs


def generate_calibration_data(
    num_batches: int,
    batch_size: int,
    in_channels: int,
    depth: int,
    height: int,
    width: int,
) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(num_batches):
        x = torch.randn(batch_size, in_channels, depth, height, width)
        calibration_data.append(x)
    return calibration_data


def main():
    # Create a Conv3d that meets the special case conditions
    # Input: (N=2, C=3, T=2, H=16, W=16)
    # Kernel: (2, 16, 16) - matches temporal and spatial dimensions
    # Stride: (2, 16, 16) - equals kernel size
    # Padding: 0
    model = nn.Conv3d(
        in_channels=3,
        out_channels=1024,
        kernel_size=(2, 16, 16),
        stride=(2, 16, 16),
        padding=0,
        bias=True,
        groups=1,
    )
    orig_model = copy.deepcopy(model)
    model.eval()

    # Model architecture:
    # Conv3d(
    #     (weight): Parameter [1024, 3, 2, 16, 16]
    #     (bias): Parameter [1024]
    # )

    print(f"Input channels:  {model.in_channels}")
    print(f"Output channels: {model.out_channels}")
    print(f"Kernel size:     {model.kernel_size}")
    print(f"Stride:          {model.stride}")
    print(f"Padding:         {model.padding}")

    # Generate calibration data that matches the kernel size in temporal and spatial dimensions.
    # Input shape: (batch_size, in_channels, depth, height, width)
    # Example: (10, 3, 2, 16, 16) - 10 samples, 3 channels (RGB), 2 frames, 16×16 pixels
    batch_size = 10
    in_channels = 3
    depth = 2
    height = 16
    width = 16
    calibration_data = generate_calibration_data(
        num_batches=2,
        batch_size=batch_size,
        in_channels=in_channels,
        depth=depth,
        height=height,
        width=width,
    )
    example_input = calibration_data[0]

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
        quant_out = quantized_model(example_input)
        fp_out = orig_model(example_input)

    print(f"Input shape:              {example_input.shape}")
    print(f"Output shape (FP32):      {fp_out.shape}")
    print(f"Output shape (Quantized): {quant_out.shape}")
    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format
    print("\nConverting to Circle format...")
    circle_model = tico.convert(quantized_model.eval(), (example_input,))

    # Save the Circle model
    filename = "quantized_conv3d.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

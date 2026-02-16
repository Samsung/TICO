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

import importlib.util
import sys

import torch
import torch.nn as nn

import tico
import tico.quantization
import tico.quantization.config.ptq

# Check if transformers is available
trans_spec = importlib.util.find_spec("transformers")
if trans_spec is None:
    print(
        "Error: transformers package not installed. Cannot test Qwen3VLVisionRotaryEmbedding."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding


def generate_calibration_data(batch_size: int, sequence_lengths: list) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for _ in range(batch_size):
        for seqlen in sequence_lengths:
            calibration_data.append(seqlen)
    return calibration_data


def main():
    # Create the vision rotary embedding model
    # dim=128 is typical for head_dim=64 in Qwen3-VL
    dim = 128
    theta = 10000.0
    model = Qwen3VLVisionRotaryEmbedding(dim=dim, theta=theta)
    model.eval()

    # Qwen3VLVisionRotaryEmbedding(
    #   (inv_freq): Buffer [64]  # dim/2 frequency bands
    # )
    assert model.dim == dim
    assert model.theta == theta
    assert model.inv_freq.shape == (dim // 2,)

    # Generate calibration data
    # Calibrate with various sequence lengths to capture full dynamic range
    calibration_data = generate_calibration_data(
        batch_size=20, sequence_lengths=[64, 128, 256, 512]
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for i, seqlen in enumerate(calibration_data):
            _ = prepared_model(seqlen)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Convert to Circle format
    # example_inputs: seqlen as an integer
    example_seqlen = 256
    circle_model = tico.convert(quantized_model, (example_seqlen,))

    # Save the Circle model
    filename = "quantized_vision_rotary_embedding.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

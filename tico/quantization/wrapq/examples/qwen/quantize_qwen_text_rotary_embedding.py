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
        "Error: transformers package not installed. Cannot test Qwen3VLTextRotaryEmbedding."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding


def generate_calibration_data(batch_size: int, sequence_lengths: list, head_dim: int):
    """Generate calibration data for PTQ"""
    calibration_data = []
    for _ in range(batch_size):
        for seq_len in sequence_lengths:
            # x tensor: shape (batch_size, seq_len, head_dim)
            x = torch.randn(2, seq_len, head_dim)
            # position_ids: shape (batch_size, seq_len)
            position_ids = torch.arange(seq_len).unsqueeze(0).expand(2, -1)
            calibration_data.append((x, position_ids))
    return calibration_data


def main():
    # Create the text rotary embedding model
    # Use typical Qwen3-VL dimensions
    cfg = Qwen3VLTextConfig(
        hidden_size=2048,  # Typical for Qwen3-VL 2B
        num_attention_heads=16,
        max_position_embeddings=4096,
    )
    model = Qwen3VLTextRotaryEmbedding(cfg)
    model.eval()

    # Qwen3VLTextRotaryEmbedding(
    #   (inv_freq): Buffer [dim/2]  # dim=128 for head_dim=64
    # )
    head_dim = (
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    )
    print(
        f"Config: hidden_size={cfg.hidden_size}, num_attention_heads={cfg.num_attention_heads}"
    )
    print(f"head_dim={head_dim}, inv_freq.shape={model.inv_freq.shape}")

    # Generate calibration data
    # Calibrate with various sequence lengths to capture full dynamic range
    # Important: Use maximum sequence length that will be used at inference
    calibration_data = generate_calibration_data(
        batch_size=20,
        sequence_lengths=[128, 256, 512, 1024, 2048, 4096],
        head_dim=head_dim,
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for i, (x, position_ids) in enumerate(calibration_data):
            _ = prepared_model(x, position_ids)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Convert to Circle format
    # example_inputs: tuple containing (x, position_ids)
    example_seq_len = 256
    example_x = torch.randn(2, example_seq_len, head_dim)
    example_position_ids = torch.arange(example_seq_len).unsqueeze(0).expand(2, -1)
    circle_model = tico.convert(quantized_model, (example_x, example_position_ids))

    # Save the Circle model
    filename = "quantized_text_rotary_embedding.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

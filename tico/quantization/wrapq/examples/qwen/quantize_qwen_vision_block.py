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
import importlib.util
import sys

import torch
import torch.nn as nn

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs

# Check if transformers is available
trans_spec = importlib.util.find_spec("transformers")
if trans_spec is None:
    print("Error: transformers package not installed. Cannot test Qwen3VLVisionBlock.")
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock


def generate_calibration_data(
    batch_size: int, num_patches: int, hidden_size: int
) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(batch_size):
        hidden_states = torch.randn(num_patches, hidden_size)
        cu_seqlens = torch.arange(0, num_patches + 1, 8)
        calibration_data.append((hidden_states, cu_seqlens))
    return calibration_data


def main():
    # Create the vision block model
    cfg = Qwen3VLVisionConfig(
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
    )
    model = Qwen3VLVisionBlock(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Qwen3VLVisionBlock(
    #     (norm1): LayerNorm(1024, eps=1e-06, elementwise_affine=True)
    #     (norm2): LayerNorm(1024, eps=1e-06, elementwise_affine=True)
    #     (attn): Qwen3VLVisionAttention(...)
    #     (mlp): Qwen3VLVisionMLP(...)
    # )
    assert cfg.hidden_size == 1024
    assert cfg.num_attention_heads == 16

    # Generate calibration data
    # Input shape: (num_patches, hidden_size)
    # Example: (256, 1024) - 256 patches from 2 videos (2*8*16=256)
    # cu_seqlens: Cumulative sequence lengths for handling variable-length sequences
    num_patches = 256
    hidden_size = cfg.hidden_size
    calibration_data = generate_calibration_data(
        batch_size=20, num_patches=num_patches, hidden_size=hidden_size
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for i, (hidden_states, cu_seqlens) in enumerate(calibration_data):
            prepared_model(hidden_states, cu_seqlens)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Input Ratio) between quantized model and original model
    with torch.no_grad():
        test_hidden, test_cu = calibration_data[0]
        quant_out = quantized_model(test_hidden, test_cu)
        fp_out = orig_model(test_hidden, test_cu)

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format
    # example_inputs: tuple containing (hidden_states, cu_seqlens)
    example_hidden = torch.randn(num_patches, hidden_size)
    example_cu = torch.arange(0, num_patches + 1, 8)
    example_inputs = (example_hidden, example_cu)
    circle_model = tico.convert(quantized_model, example_inputs)

    # Save the Circle model
    filename = "quantized_vision_block.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

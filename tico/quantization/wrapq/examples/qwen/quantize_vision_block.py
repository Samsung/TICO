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
        calibration_data.append(hidden_states)
    return calibration_data


def rand_rope(seq_len, head_dim):
    """Helper to create dummy rotary position embeddings"""
    emb = torch.randn(seq_len, head_dim)
    return emb.cos(), emb.sin()


def main():
    # Create the vision block model
    cfg = Qwen3VLVisionConfig(
        hidden_size=1024,
        num_heads=16,
    )
    # Ensure eager attention implementation so outputs are deterministic
    # and do not require GPU flash attention kernels.
    # Some versions use `_attn_implementation`, others expose `attn_implementation`.
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"

    model = Qwen3VLVisionBlock(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Generate calibration data
    # Input shape: (num_patches, hidden_size)
    # Example: (256, 1024) - 256 patches from 2 videos (2*8*16=256)
    # cu_seqlens: Cumulative sequence lengths for handling variable-length sequences
    num_patches = 256
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // cfg.num_heads
    cu_seqlens = torch.tensor([0, num_patches])
    calibration_data = generate_calibration_data(
        batch_size=20, num_patches=num_patches, hidden_size=hidden_size
    )
    pos_emb = rand_rope(num_patches, head_dim)

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for i, hidden_states in enumerate(calibration_data):
            prepared_model(hidden_states, cu_seqlens, position_embeddings=pos_emb)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Input Ratio) between quantized model and original model
    with torch.no_grad():
        test_hidden = calibration_data[0]
        quant_out = quantized_model(
            test_hidden, cu_seqlens, position_embeddings=pos_emb
        )
        fp_out = orig_model(test_hidden, cu_seqlens, position_embeddings=pos_emb)

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format
    # example_inputs: tuple containing (hidden_states, cu_seqlens)
    example_input = (calibration_data[0], cu_seqlens, None, pos_emb)
    circle_model = tico.convert(quantized_model, example_input)

    # Save the Circle model
    filename = "quantized_vision_block.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

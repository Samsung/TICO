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
    print(
        "Error: transformers package not installed. Cannot test Qwen3VLVisionPatchMerger."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger


def generate_calibration_data(
    batch_size: int, num_patches: int, hidden_size: int
) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(batch_size):
        x = torch.randn(num_patches, hidden_size)
        calibration_data.append(x)
    return calibration_data


def main():
    # Create the vision patch merger model
    cfg = Qwen3VLVisionConfig(
        hidden_size=1024,
        spatial_merge_size=2,
        out_hidden_size=2048,
    )
    model = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=False)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Qwen3VLVisionPatchMerger(
    #     (norm): LayerNorm(4096, eps=1e-06, elementwise_affine=True)
    #     (linear_fc1): Linear(in_features=4096, out_features=4096, bias=True)
    #     (act_fn): GELU(approximation='none')
    #     (linear_fc2): Linear(in_features=4096, out_features=2048, bias=True)
    # )
    assert (
        model.hidden_size == 4096
    )  # cfg.hidden_size * (cfg.spatial_merge_size**2) = 1024 * 2**2
    assert model.linear_fc1.in_features == 4096
    assert model.linear_fc1.out_features == 4096
    assert model.linear_fc2.in_features == 4096
    assert model.linear_fc2.out_features == 2048

    # Generate calibration data
    # Input shape: (num_patches, hidden_size)
    # Example: input.shape=(num_patches=32, hidden_size=1024)
    #     num_patches=32 can come from e.g. two 8-frame videos 32x32 pixels RGB channels after they are embedded by Qwen3VLVisionPatchEmbed (Conv3d):
    #     (Batch, Channels, Time, Height, Width) = (2, 3, 4, 32, 32) --> Qwen3VLVisionPatchEmbed --> (num_patches, hidden_size) = (2*4*4, 1024),
    #     where 2*4*4 means (2 videos) times (4 spatial patches) times (4 temporal patches).
    #     4 spatial patches can come from 32x32 frame with stride 16: 32/16 * 32/16 = 2*2 = 4.
    #     4 temporal patches can come from 8 frames with stride 2: 8 / 2 = 4.
    num_patches = 32
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
    # example_inputs shape: (num_patches, hidden_size)
    example_inputs = (torch.randn(num_patches, hidden_size),)
    circle_model = tico.convert(quantized_model, example_inputs)

    # Save the Circle model
    filename = "quantized_vision_patch_merger.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

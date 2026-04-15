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


torch.manual_seed(123)


def generate_calibration_data(
    num_batches: int,
    batch_size: int,
    normalized_shape: tuple,
) -> list:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(num_batches):
        x = torch.randn(batch_size, *normalized_shape)
        calibration_data.append(x)
    return calibration_data


def main():
    # Create LayerNorm model
    # Using a common configuration for transformer models
    normalized_shape = (768,)  # Hidden dimension size
    model = nn.LayerNorm(
        normalized_shape=normalized_shape,
        eps=1e-5,
        elementwise_affine=True,
    )
    orig_model = copy.deepcopy(model)
    model.eval()

    # Generate calibration data
    # Input shape: (batch_size, *normalized_shape)
    # Example: (10, 768) - 10 samples, 768 features
    batch_size = 10
    calibration_data = generate_calibration_data(
        num_batches=5,
        batch_size=batch_size,
        normalized_shape=normalized_shape,
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

    # Compute PEIR (Peak Error-to-Interval Ratio) between quantized model and original model
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
    circle_model = tico.convert(quantized_model.eval(), (example_input,))

    # Save the Circle model
    filename = "quantized_layernorm.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

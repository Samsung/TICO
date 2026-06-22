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

"""Example: PTQ quantization of Gemma4TextScaledWordEmbedding.

The Gemma4 text scaled word embedding module extends nn.Embedding by multiplying
the embeddings with a scalar scale factor (embed_scale). This script demonstrates
the full PTQ flow:

1. Create a tiny Gemma4TextScaledWordEmbedding with random weights (no download needed).
2. Prepare the model for quantization.
3. Calibrate with synthetic input IDs.
4. Convert to a fake-quantized model.
5. Compare FP vs. quantized outputs.
6. Export and convert to Circle format.
"""

import copy
import sys

import torch

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


# Check if transformers is available
if not has_transformers_for("gemma4"):
    print(
        "Error: transformers package with Gemma4 support not installed. "
        "Cannot test Gemma4TextScaledWordEmbedding."
    )
    sys.exit(1)

from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding


def generate_calibration_data(
    vocab_size: int,
    num_samples: int = 20,
    seq_len: int = 16,
) -> list[torch.Tensor]:
    """Generate calibration data for PTQ.

    Each sample is a tensor of input IDs with shape (1, seq_len).
    Values are randomly sampled from [0, vocab_size).
    """
    calibration_data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        calibration_data.append(input_ids)
    return calibration_data


def main():
    # Create the embedding module with a tiny config (no download needed).
    vocab_size = 1000  # Small vocabulary for testing
    embedding_dim = 64  # Small embedding dimension for testing
    padding_idx = 0
    embed_scale = 0.125  # Typical scale value

    model = Gemma4TextScaledWordEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        embed_scale=embed_scale,
    )
    orig_model = copy.deepcopy(model)
    model.eval()

    print(f"Gemma4TextScaledWordEmbedding:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  padding_idx: {padding_idx}")
    print(f"  embed_scale: {embed_scale}")

    # Generate calibration data
    calibration_data = generate_calibration_data(
        vocab_size=vocab_size,
        num_samples=20,
        seq_len=16,
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    print("\nCalibrating...")
    with torch.no_grad():
        for input_ids in calibration_data:
            prepared_model(input_ids)

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Interval Ratio) between quantized model and original model
    eval_input = calibration_data[0]
    with torch.no_grad():
        quant_out = quantized_model(eval_input)
        fp_out = orig_model(eval_input)

    print(f"\n┌───────────── Quantization Error Summary ─────────────")
    print(f"│ FP output shape    : {tuple(fp_out.shape)}")
    print(f"│ Quant output shape : {tuple(quant_out.shape)}")
    print(f"│ Mean |diff|        : {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR               : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Export and convert to Circle format.
    # The wrapper is already exportable, so as_export_module returns self.
    export_module = quantized_model.as_export_module(mode="prefill").eval()

    example_inputs = (torch.randint(0, vocab_size, (1, 16), dtype=torch.long),)

    print("\nConverting to Circle format...")
    circle_model = tico.convert(export_module, example_inputs)

    filename = "gemma4_text_scaled_word_embedding.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

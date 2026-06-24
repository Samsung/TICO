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

"""Example: PTQ quantization of Gemma4MultimodalEmbedder.

The Gemma4 multimodal embedder projects multimodal soft tokens (e.g. visual
features from the vision model) into the text model's hidden space. It applies:

1. RMS normalization (``embedding_pre_projection_norm``)
2. Linear projection (``embedding_projection``)

It accepts:

- ``inputs_embeds``: Soft token embeddings of shape ``(B, seq_len, multimodal_hidden_size)``

and produces text-hidden-space embeddings of shape ``(B, seq_len, text_hidden_size)``.

This script demonstrates the full PTQ flow:

1. Create a tiny Gemma4MultimodalEmbedder with random weights (no download needed).
2. Prepare the model for quantization.
3. Calibrate with synthetic data.
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
        "Cannot test Gemma4MultimodalEmbedder."
    )
    sys.exit(1)

from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder


def generate_calibration_data(
    batch_size: int,
    seq_len: int,
    multimodal_hidden_size: int,
    num_samples: int = 20,
) -> list[dict]:
    """Generate calibration data for PTQ.

    Each sample is a dict of keyword arguments matching the multimodal embedder's
    forward signature: ``inputs_embeds``.
    """
    calibration_data = []
    for _ in range(num_samples):
        sample = {
            "inputs_embeds": torch.randn(batch_size, seq_len, multimodal_hidden_size),
        }
        calibration_data.append(sample)
    return calibration_data


def main():
    # Create tiny configs for the multimodal embedder (no download needed).
    # Gemma4MultimodalEmbedder requires a multimodal_config and a text_config.
    vision_cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        patch_size=4,
        position_embedding_size=8,
        pooling_kernel_size=2,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
    )
    if not hasattr(vision_cfg, "_attn_implementation"):
        setattr(vision_cfg, "_attn_implementation", "eager")
    else:
        vision_cfg._attn_implementation = "eager"

    text_cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0}
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        enable_moe_block=False,
    )
    if not hasattr(text_cfg, "_attn_implementation"):
        setattr(text_cfg, "_attn_implementation", "eager")
    else:
        text_cfg._attn_implementation = "eager"

    model = Gemma4MultimodalEmbedder(vision_cfg, text_cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Gemma4MultimodalEmbedder(
    #   (embedding_pre_projection_norm): Gemma4RMSNorm(32, eps=1e-06)
    #   (embedding_projection): Linear(32, 64, bias=False)
    # )
    multimodal_hidden_size = model.multimodal_hidden_size
    text_hidden_size = model.text_hidden_size
    assert multimodal_hidden_size == 32
    assert text_hidden_size == 64

    # Generate calibration data
    batch_size = 1
    seq_len = 16

    calibration_data = generate_calibration_data(
        batch_size=batch_size,
        seq_len=seq_len,
        multimodal_hidden_size=multimodal_hidden_size,
        num_samples=20,
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    print("Calibrating...")
    with torch.no_grad():
        for sample in calibration_data:
            prepared_model(**sample)

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Interval Ratio) between quantized model and original model
    eval_sample = calibration_data[0]
    with torch.no_grad():
        quant_out = quantized_model(**eval_sample)
        fp_out = orig_model(**eval_sample)

    print(f"\n┌───────────── Quantization Error Summary ─────────────")
    print(f"│ FP output shape    : {tuple(fp_out.shape)}")
    print(f"│ Quant output shape : {tuple(quant_out.shape)}")
    print(f"│ Mean |diff|        : {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR               : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Export and convert to Circle format.
    # The multimodal embedder is a simple sequential module (RMSNorm + Linear),
    # so as_export_module returns self.
    wrapped = getattr(quantized_model, "wrapped", quantized_model)
    if hasattr(wrapped, "as_export_module"):
        export_module = wrapped.as_export_module(mode="prefill").eval()

        example_inputs = (
            torch.randn(batch_size, seq_len, multimodal_hidden_size),  # inputs_embeds
        )

        print("\nConverting to Circle format...")
        circle_model = tico.convert(export_module, example_inputs)

        filename = "gemma4_multimodal_embedder.q.circle"
        circle_model.save(filename)
        print(f"Circle model saved as '{filename}'")
    else:
        print("Note: as_export_module not available; skipping Circle export.")


if __name__ == "__main__":
    main()

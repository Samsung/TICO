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

"""Example: PTQ quantization of Gemma4VisionModel.

The Gemma4 vision model encodes image pixels into visual soft tokens through
a pipeline of patch embedding, transformer encoding, spatial pooling, and
optional standardization. It accepts:

- ``pixel_values``: Pre-flattened image patches of shape ``(B, num_patches, 3*patch_size^2)``
- ``pixel_position_ids``: 2D grid coordinates of shape ``(B, num_patches, 2)``

The output is a ``BaseModelOutputWithPast`` whose ``last_hidden_state`` contains
visual soft tokens after pooling and standardization.

This script demonstrates the full PTQ flow:

1. Create a tiny Gemma4VisionModel with random weights (no download needed).
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
        "Cannot test Gemma4VisionModel."
    )
    sys.exit(1)

from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel


def _pixel_position_ids(batch_size: int, num_patches: int) -> torch.Tensor:
    """Create deterministic 2D pixel position ids for a patch grid.

    The vision model requires ``pixel_position_ids`` with shape ``(B, num_patches, 2)``
    where the last dimension encodes ``(x, y)`` patch coordinates. We build a
    simple square grid layout.
    """
    side = int(num_patches**0.5)
    coords = torch.arange(num_patches)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def generate_calibration_data(
    batch_size: int,
    num_patches: int,
    patch_size: int,
    num_samples: int = 20,
) -> list[dict]:
    """Generate calibration data for PTQ.

    Each sample is a dict of keyword arguments matching the vision model's forward
    signature: ``pixel_values``, ``pixel_position_ids``.
    """
    patch_dim = 3 * patch_size**2
    calibration_data = []
    for _ in range(num_samples):
        sample = {
            "pixel_values": torch.randn(batch_size, num_patches, patch_dim),
            "pixel_position_ids": _pixel_position_ids(batch_size, num_patches),
            "return_dict": True,
        }
        calibration_data.append(sample)
    return calibration_data


def main():
    # Create the vision model with a tiny config (no download needed).
    cfg = Gemma4VisionConfig(
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
        standardize=True,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"

    model = Gemma4VisionModel(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Generate calibration data
    batch_size = 1
    num_patches = 16
    patch_size = cfg.patch_size

    calibration_data = generate_calibration_data(
        batch_size=batch_size,
        num_patches=num_patches,
        patch_size=patch_size,
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
        quant_out = quantized_model(**eval_sample).last_hidden_state
        fp_out = orig_model(**eval_sample).last_hidden_state

    print(f"\n┌───────────── Quantization Error Summary ─────────────")
    print(f"│ FP output shape    : {tuple(fp_out.shape)}")
    print(f"│ Quant output shape : {tuple(quant_out.shape)}")
    print(f"│ Mean |diff|        : {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR               : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Export and convert to Circle format.
    # The vision model's as_export_module requires pixel_position_ids to
    # precompute the pooler's static weight matrix.
    wrapped = getattr(quantized_model, "wrapped", quantized_model)
    if hasattr(wrapped, "as_export_module"):
        pixel_pos_ids = _pixel_position_ids(batch_size, num_patches)
        export_module = wrapped.as_export_module(
            mode="prefill",
            pixel_position_ids=pixel_pos_ids,
        ).eval()

        example_inputs = (
            torch.randn(batch_size, num_patches, 3 * patch_size**2),  # pixel_values
            _pixel_position_ids(batch_size, num_patches),  # pixel_position_ids
        )

        print("\nConverting to Circle format...")
        circle_model = tico.convert(export_module, example_inputs)

        filename = "gemma4_vision_model.q.circle"
        circle_model.save(filename)
        print(f"Circle model saved as '{filename}'")
    else:
        print("Note: as_export_module not available; skipping Circle export.")


if __name__ == "__main__":
    main()

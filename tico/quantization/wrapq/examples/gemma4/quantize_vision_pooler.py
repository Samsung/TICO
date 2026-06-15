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

"""Example: PTQ quantization of Gemma4VisionPooler.

The Gemma4 vision pooler performs spatial pooling of vision patch tokens and
scales the result by ``sqrt(hidden_size)`` in float32.  It returns a tuple
``(pooled_features, updated_padding)`` where ``pooled_features`` has shape
``(B, V, D)`` with ``V`` equal to the fixed ``output_length`` (number of
visual soft tokens).

This script demonstrates the full PTQ flow:

1. Create a tiny Gemma4VisionPooler with random weights (no download needed).
2. Prepare the model for quantization.
3. Calibrate with synthetic data.
4. Convert to a fake-quantized model.
5. Compare FP vs. quantized outputs.
6. Export the static-shape export adapter and convert to Circle format.
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
        "Cannot test Gemma4VisionPooler."
    )
    sys.exit(1)

from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler


def _pixel_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a patch grid.

    The pooler requires ``pixel_position_ids`` with shape ``(B, S, 2)`` where
    the last dimension encodes ``(x, y)`` patch coordinates.  We build a
    simple square grid layout.
    """
    side = int(seq_len**0.5)
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _padding_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create an all-False padding mask (no padding)."""
    return torch.zeros(batch_size, seq_len, dtype=torch.bool)


def generate_calibration_data(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    output_length: int,
    num_samples: int = 20,
) -> list[dict]:
    """Generate calibration data for PTQ.

    Each sample is a dict of keyword arguments matching the pooler's forward
    signature: ``hidden_states``, ``pixel_position_ids``,
    ``padding_positions``, ``output_length``.
    """
    calibration_data = []
    for _ in range(num_samples):
        sample = {
            "hidden_states": torch.randn(batch_size, seq_len, hidden_size),
            "pixel_position_ids": _pixel_position_ids(batch_size, seq_len),
            "padding_positions": _padding_positions(batch_size, seq_len),
            "output_length": output_length,
        }
        calibration_data.append(sample)
    return calibration_data


def main():
    # Create the vision pooler model with a tiny config (no download needed).
    # Use seq_len=16 and output_length=4 so that k=2 (16 / 4 = 4, sqrt(4) = 2).
    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"

    model = Gemma4VisionPooler(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Gemma4VisionPooler(
    #   (hidden_size): 32
    #   (root_hidden_size): 5.6568...
    # )
    assert model.hidden_size == 32

    # Generate calibration data
    batch_size = 1
    seq_len = 16
    output_length = 4
    hidden_size = cfg.hidden_size

    calibration_data = generate_calibration_data(
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        output_length=output_length,
        num_samples=20,
    )

    # Configure PTQ
    ptq_config = tico.quantization.config.ptq.PTQConfig()

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        for sample in calibration_data:
            prepared_model(**sample)

    # Convert to quantized model
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Interval Ratio) between quantized model and original model
    eval_sample = calibration_data[0]
    with torch.no_grad():
        quant_out = quantized_model(**eval_sample)
        fp_out = orig_model(**eval_sample)

    # Both return (pooled_features, updated_padding)
    quant_pooled, quant_padding = quant_out
    fp_pooled, fp_padding = fp_out

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ FP pooled shape    : {tuple(fp_pooled.shape)}")
    print(f"│ Quant pooled shape : {tuple(quant_pooled.shape)}")
    print(
        f"│ Mean |diff|        : {(quant_pooled - fp_pooled).abs().mean().item():.6f}"
    )
    print(f"│ PEIR               : {compute_peir(fp_pooled, quant_pooled) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_pooled, quant_pooled))

    # Export the static-shape export adapter and convert to Circle format.
    # The export adapter bakes output_length as a construction-time constant,
    # so the forward signature is (hidden_states, pixel_position_ids, padding_positions).
    wrapped = getattr(quantized_model, "wrapped", quantized_model)
    if hasattr(wrapped, "as_export_module"):
        export_module = wrapped.as_export_module(
            output_length=output_length,
            pixel_position_ids=_pixel_position_ids(batch_size, seq_len),
        ).eval()

        example_inputs = (
            torch.randn(batch_size, seq_len, hidden_size),  # hidden_states
            _pixel_position_ids(batch_size, seq_len),  # pixel_position_ids
            _padding_positions(batch_size, seq_len),  # padding_positions
        )
        circle_model = tico.convert(export_module, example_inputs)

        filename = "gemma4_vision_pooler.q.circle"
        circle_model.save(filename)
        print(f"Circle model saved as '{filename}'")
    else:
        print("Note: as_export_module not available; skipping Circle export.")


if __name__ == "__main__":
    main()

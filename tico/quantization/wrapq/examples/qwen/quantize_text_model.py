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

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


# Check if transformers is available

if not has_transformers_for("qwen3-vl"):
    print("Error: transformers package not installed. Cannot test Qwen3VLTextModel.")
    sys.exit(1)

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLTextModel,
)


def generate_calibration_data(
    batch_size: int, vocab_size: int, seq_len: int
) -> list[torch.Tensor]:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for _ in range(batch_size):
        # Generate random input IDs
        input_ids = torch.randint(0, vocab_size, (1, seq_len))
        calibration_data.append(input_ids)
    return calibration_data


def main():
    # Create a sample config (small model for faster testing)
    cfg = Qwen3VLTextConfig(
        vocab_size=10000,  # Smaller vocab for testing
        hidden_size=256,  # Smaller hidden size
        intermediate_size=1024,  # Smaller intermediate size
        num_hidden_layers=2,  # Only 2 layers for testing
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=2048,
        use_cache=False,
    )

    # Ensure eager attention implementation so outputs are deterministic
    # and do not require GPU flash attention kernels.
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"

    # Create the text model
    model = Qwen3VLTextModel(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Prepare for quantization
    qcfg = tico.quantization.config.ptq.PTQConfig()
    prepared_model = tico.quantization.prepare(model, qcfg)

    # Calibrate the model (collect statistics)
    batch_size = 10
    seq_len = 128
    vocab_size = cfg.vocab_size

    calibration_data = generate_calibration_data(
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )

    for input_ids in calibration_data:
        prepared_model(
            input_ids=input_ids,
            use_cache=False,  # Disable caching for calibration
        )

    # Convert to quantized version
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Interval Ratio) between quantized model and original model
    with torch.no_grad():
        test_input_ids = calibration_data[0]
        quant_out = quantized_model(
            input_ids=test_input_ids,
            use_cache=False,
        )
        fp_out = orig_model(
            input_ids=test_input_ids,
            use_cache=False,
        )

        # Extract last_hidden_state for comparison
        quant_hidden = quant_out.last_hidden_state
        fp_hidden = fp_out.last_hidden_state

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_hidden - fp_hidden).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_hidden, quant_hidden) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_hidden, quant_hidden))

    # Convert to Circle format
    example_input = (calibration_data[0],)
    circle_model = tico.convert(
        quantized_model.eval(), example_input, kwargs={"use_cache": False}
    )

    # Save the Circle model
    filename = "qwen3vl_text_model.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

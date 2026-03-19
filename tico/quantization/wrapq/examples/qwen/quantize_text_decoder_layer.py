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

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


# Check if transformers is available

if not has_transformers_for("qwen3-vl"):
    print(
        "Error: transformers package not installed. Cannot test Qwen3VLTextDecoderLayer."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLTextDecoderLayer,
)


def generate_calibration_data(
    batch_size: int, hidden_size: int, seq_len: int, head_dim: int
) -> tuple:
    """Generate calibration data for PTQ"""
    calibration_data = []
    for i in range(batch_size):
        hidden_states = torch.randn(1, seq_len, hidden_size)
        calibration_data.append(hidden_states)

    position_embeddings = (
        torch.randn(1, seq_len, head_dim),
        torch.randn(1, seq_len, head_dim),
    )
    attention_mask = torch.ones(1, 1, seq_len, seq_len)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    return calibration_data, position_embeddings, attention_mask, position_ids


def main():
    # Create a sample config
    cfg = Qwen3VLTextConfig(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=2048,
        intermediate_size=1024,
    )

    # Ensure eager attention implementation so outputs are deterministic
    # and do not require GPU flash attention kernels.
    # Some versions use `_attn_implementation`, others expose `attn_implementation`.
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"

    # Create a decoder layer
    model = Qwen3VLTextDecoderLayer(cfg, layer_idx=0)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Prepare for quantization
    qcfg = tico.quantization.config.ptq.PTQConfig()
    prepared_model = tico.quantization.prepare(model, qcfg)

    # Calibrate the model (collect statistics)
    batch_size = 10
    seq_len = 128
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // cfg.num_attention_heads

    (
        calibration_data,
        position_embeddings,
        attention_mask,
        position_ids,
    ) = generate_calibration_data(
        batch_size=batch_size,
        hidden_size=hidden_size,
        seq_len=seq_len,
        head_dim=head_dim,
    )
    for hidden_states in calibration_data:
        prepared_model(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    # Convert to quantized version
    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    # Compute PEIR (Peak Error-to-Input Ratio) between quantized model and original model
    with torch.no_grad():
        test_input = calibration_data[0]
        quant_out = quantized_model(
            hidden_states=test_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        fp_out = orig_model(
            hidden_states=test_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    print(f"┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Convert to Circle format
    example_input = (
        calibration_data[0],
        position_embeddings,
        attention_mask,
        position_ids,
    )
    circle_model = tico.convert(quantized_model.eval(), example_input)

    # Save the Circle model
    filename = "qwen3vl_text_decoder_layer.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

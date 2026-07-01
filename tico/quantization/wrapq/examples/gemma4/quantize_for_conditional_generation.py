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

"""Example: PTQ quantization of Gemma4ForConditionalGeneration.

The ``Gemma4ForConditionalGeneration`` is the top-level model that wraps
``Gemma4Model`` (vision + text decoder) and adds an ``lm_head`` linear layer
to produce vocabulary logits.  It also applies logit softcapping when
``final_logit_softcapping`` is set in the text config.

This script demonstrates the full PTQ flow:

1. Create a tiny Gemma4ForConditionalGeneration with random weights.
2. Prepare the model for quantization with a Gemma4-specific PTQ config.
3. Calibrate with synthetic text-only data.
4. Convert to a fake-quantized model.
5. Compare FP vs. quantized logits.
6. Export to Circle format via the export adapter.
"""

import copy
import sys

import torch

import tico
import tico.quantization
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


# Check if transformers is available
if not has_transformers_for("gemma4"):
    print(
        "Error: transformers package with Gemma4 support not installed. "
        "Cannot test Gemma4ForConditionalGeneration."
    )
    sys.exit(1)

from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration


def _make_vision_config() -> Gemma4VisionConfig:
    """Create a tiny Gemma4 vision config for the example."""
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
    return cfg


def _make_text_config() -> Gemma4TextConfig:
    """Create a tiny Gemma4 text config for the example."""
    cfg = Gemma4TextConfig(
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
        # Enable logit softcapping to exercise that code path.
        final_logit_softcapping=30.0,
    )
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _make_gemma4_config() -> Gemma4Config:
    """Create a tiny Gemma4 top-level config for the example."""
    return Gemma4Config(
        text_config=_make_text_config(),
        vision_config=_make_vision_config(),
        audio_config=None,
        image_token_id=10,
        video_token_id=11,
        audio_token_id=12,
    )


def generate_text_only_calibration_data(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 20,
) -> list[dict]:
    """Generate text-only calibration data for PTQ."""
    calibration_data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Avoid image placeholder token IDs to keep this text-only.
        input_ids = input_ids.clamp(0, 9)
        sample = {
            "input_ids": input_ids,
        }
        calibration_data.append(sample)
    return calibration_data


def main():
    # Create the model with a tiny config (no download needed).
    config = _make_gemma4_config()
    text_config = config.get_text_config()
    vision_config = config.vision_config

    model = Gemma4ForConditionalGeneration(config)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Dimensions
    batch_size = 1
    seq_len = 16
    num_visual_tokens = 4

    # Build a Gemma4-specific PTQ config.
    ptq_config = build_gemma4_e2b_ptq_config(
        num_text_layers=int(text_config.num_hidden_layers),
        num_vision_layers=int(vision_config.num_hidden_layers),
        model_args={
            "vision": {
                "visual_start_idx": 0,
                "num_visual_tokens": num_visual_tokens,
            }
        },
    )

    # Prepare the model for quantization
    print("Preparing model for quantization...")
    prepared_model = tico.quantization.prepare(model, ptq_config)

    # Calibrate with text-only data
    print("Calibrating (text-only)...")
    calibration_data = generate_text_only_calibration_data(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=text_config.vocab_size,
        num_samples=20,
    )
    with torch.no_grad():
        for sample in calibration_data:
            prepared_model(**sample)

    # Convert to quantized model
    print("Converting to quantized model...")
    quantized_model = tico.quantization.convert(prepared_model)

    # Compute PEIR between quantized model and original model
    eval_sample = calibration_data[0]
    with torch.no_grad():
        quant_out = quantized_model(**eval_sample)
        fp_out = orig_model(**eval_sample).logits

    print(f"\n┌───────────── Quantization Error Summary ─────────────")
    print(f"│ FP output shape    : {tuple(fp_out.shape)}")
    print(f"│ Quant output shape : {tuple(quant_out.shape)}")
    print(f"│ Mean |diff|        : {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR               : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))


if __name__ == "__main__":
    main()

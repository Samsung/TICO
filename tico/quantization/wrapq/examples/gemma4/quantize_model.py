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

"""Example: PTQ quantization of Gemma4Model (image-text).

The Gemma4Model is the top-level multimodal model that combines a vision
tower, a vision-to-text projection (embed_vision), and a text decoder
(language_model).  It accepts:

- ``input_ids``: Token IDs of shape ``(B, S)`` including optional image
  placeholder tokens (``config.image_token_id``).
- ``pixel_values``: Pre-flattened image patches of shape
  ``(B, num_patches, 3*patch_size^2)``.
- ``image_position_ids``: 2D patch coordinates of shape ``(B, num_patches, 2)``.

The output is a ``Gemma4ModelOutputWithPast`` whose ``last_hidden_state``
contains the decoder hidden states.

This script demonstrates the full PTQ flow:

1. Create a tiny Gemma4Model with random weights (no download needed).
2. Prepare the model for quantization with Gemma4-specific PTQ config.
3. Calibrate with synthetic text-only data (image placeholders replaced
   with pad_token_id before embedding, matching the production path).
4. Convert to a fake-quantized model.
5. Compare FP vs. quantized outputs.
"""

import copy
import sys

import torch

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


# Check if transformers is available
if not has_transformers_for("gemma4"):
    print(
        "Error: transformers package with Gemma4 support not installed. "
        "Cannot test Gemma4Model."
    )
    sys.exit(1)

from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model


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
        # Use small token IDs that fit within vocab_size=256
        image_token_id=10,
        video_token_id=11,
        audio_token_id=12,
    )


def _pixel_position_ids(batch_size: int, num_patches: int) -> torch.Tensor:
    """Create deterministic 2D pixel position ids for a patch grid."""
    side = int(num_patches**0.5)
    coords = torch.arange(num_patches)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def generate_text_only_calibration_data(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_samples: int = 20,
) -> list[dict]:
    """Generate text-only calibration data for PTQ.

    Each sample contains ``input_ids`` with no image placeholder tokens.
    The QuantGemma4Model wrapper will embed them directly.
    """
    calibration_data = []
    for _ in range(num_samples):
        # Use token IDs in range [0, vocab_size) but avoid the image
        # placeholder token ID to keep this text-only.
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Ensure no accidental image placeholder tokens
        input_ids = input_ids.clamp(0, 9)
        sample = {
            "input_ids": input_ids,
        }
        calibration_data.append(sample)
    return calibration_data


def generate_image_text_calibration_data(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_patches: int,
    patch_size: int,
    image_token_id: int,
    num_visual_tokens: int,
    num_samples: int = 20,
) -> list[dict]:
    """Generate image-text calibration data for PTQ.

    Each sample contains ``input_ids`` with image placeholder tokens at
    fixed positions, plus ``pixel_values`` and ``image_position_ids`` for
    the vision tower.
    """
    patch_dim = 3 * patch_size**2
    calibration_data = []
    for _ in range(num_samples):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Place image placeholder tokens at the start
        input_ids[0, :num_visual_tokens] = image_token_id
        sample = {
            "input_ids": input_ids,
            "pixel_values": torch.randn(batch_size, num_patches, patch_dim),
            "image_position_ids": _pixel_position_ids(batch_size, num_patches),
        }
        calibration_data.append(sample)
    return calibration_data


def main():
    # Create the model with a tiny config (no download needed).
    config = _make_gemma4_config()
    text_config = config.get_text_config()
    vision_config = config.vision_config

    model = Gemma4Model(config)
    orig_model = copy.deepcopy(model)
    model.eval()

    # Dimensions
    batch_size = 1
    seq_len = 16
    num_patches = 16
    patch_size = vision_config.patch_size
    num_visual_tokens = 4

    # Build a Gemma4-specific PTQ config with proper overrides.
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
        quant_out = quantized_model(**eval_sample).last_hidden_state
        fp_out = orig_model(**eval_sample).last_hidden_state

    print(f"\n┌───────────── Quantization Error Summary ─────────────")
    print(f"│ FP output shape    : {tuple(fp_out.shape)}")
    print(f"│ Quant output shape : {tuple(quant_out.shape)}")
    print(f"│ Mean |diff|        : {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR               : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Also test with image-text data
    print("\nCalibrating with image-text data...")
    model2 = Gemma4Model(_make_gemma4_config())
    orig_model2 = copy.deepcopy(model2)
    model2.eval()

    ptq_config2 = build_gemma4_e2b_ptq_config(
        num_text_layers=int(text_config.num_hidden_layers),
        num_vision_layers=int(vision_config.num_hidden_layers),
        model_args={
            "vision": {
                "visual_start_idx": 0,
                "num_visual_tokens": num_visual_tokens,
            }
        },
    )

    prepared_model2 = tico.quantization.prepare(model2, ptq_config2)

    image_text_data = generate_image_text_calibration_data(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=text_config.vocab_size,
        num_patches=num_patches,
        patch_size=patch_size,
        image_token_id=config.image_token_id,
        num_visual_tokens=num_visual_tokens,
        num_samples=20,
    )
    with torch.no_grad():
        for sample in image_text_data:
            prepared_model2(**sample)

    quantized_model2 = tico.quantization.convert(prepared_model2)

    eval_sample2 = image_text_data[0]
    with torch.no_grad():
        quant_out2 = quantized_model2(**eval_sample2).last_hidden_state
        fp_out2 = orig_model2(**eval_sample2).last_hidden_state

    print(f"\n┌───────────── Image-Text Quantization Error ──────────")
    print(f"│ FP output shape    : {tuple(fp_out2.shape)}")
    print(f"│ Quant output shape : {tuple(quant_out2.shape)}")
    print(f"│ Mean |diff|        : {(quant_out2 - fp_out2).abs().mean().item():.6f}")
    print(f"│ PEIR               : {compute_peir(fp_out2, quant_out2) * 100:.6f} %")
    print(f"└──────────────────────────────────────────────────────")

    # ------------------------------------------------------------------
    # Export and convert to Circle format.
    #
    # The QuantGemma4Model.as_export_module() returns an export adapter
    # whose forward_export() takes precomputed inputs:
    #   - inputs_embeds:  Pre-fused text+image embeddings (B, S, H)
    #   - per_layer_inputs: PLE tensor (B, S, L, P) or None
    #   - attention_masks:  Dict mapping layer type to additive mask
    #   - position_embeddings: Dict mapping layer type to (cos, sin)
    #
    # In a real deployment the CPU runtime computes these; here we
    # synthesize them to demonstrate the Circle conversion path.
    # ------------------------------------------------------------------
    print("\nExporting to Circle format...")
    wrapped = getattr(quantized_model2, "wrapped", quantized_model2)
    export_module = wrapped.as_export_module(mode="prefill").eval()

    # Build precomputed example inputs (simulating CPU runtime output).
    hidden_size = int(text_config.hidden_size)
    head_dim = int(text_config.head_dim)
    num_layers = int(text_config.num_hidden_layers)
    ple_dim = int(getattr(text_config, "hidden_size_per_layer_input", 0) or 0)
    layer_types = list(text_config.layer_types)

    ex_inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)

    ex_per_layer_inputs = None
    if ple_dim > 0:
        ex_per_layer_inputs = torch.randn(batch_size, seq_len, num_layers, ple_dim)

    # Static causal masks and identity-like RoPE for each layer type.
    ex_attention_masks = {}
    ex_position_embeddings = {}
    for layer_type in layer_types:
        ex_attention_masks[layer_type] = torch.zeros(batch_size, 1, seq_len, seq_len)
        cos = torch.ones(batch_size, seq_len, head_dim)
        sin = torch.zeros(batch_size, seq_len, head_dim)
        ex_position_embeddings[layer_type] = (cos, sin)

    # Verify the export module produces finite output.
    with torch.no_grad():
        export_out = export_module(
            inputs_embeds=ex_inputs_embeds,
            per_layer_inputs=ex_per_layer_inputs,
            attention_masks=ex_attention_masks,
            position_embeddings=ex_position_embeddings,
        )
    print(f"Export output shape: {tuple(export_out.shape)}")
    assert torch.isfinite(export_out).all(), "Export output contains non-finite values!"

    # Convert to Circle format.
    example_inputs = (
        ex_inputs_embeds,  # inputs_embeds
        ex_per_layer_inputs,  # per_layer_inputs
        ex_attention_masks,  # attention_masks
        ex_position_embeddings,  # position_embeddings
    )

    print("Converting to Circle format...")
    circle_model = tico.convert(export_module, example_inputs)

    filename = "gemma4_model.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

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

"""
Example of using QuantQwen3VLForConditionalGeneration wrapper.

This script demonstrates how to:
1. Create a small Qwen3VLForConditionalGeneration model.
2. Wrap it with QuantQwen3VLForConditionalGeneration using `prepare`.
3. Perform calibration with synthetic data.
4. Freeze quantization parameters using `convert`.
5. Run forward pass and compare results.
6. Export the quantized model to .circle format.
"""

import pathlib

import torch

# Import necessary modules from tico
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode

# Check if transformers library is available for Qwen3-VL
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.utils.utils import SuppressWarning

# Set random seed for reproducibility
torch.manual_seed(123)


def main():
    if not has_transformers_for("qwen3-vl"):
        print("Required transformers not installed — skipping example")
        return

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    # Import the original model class
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    print("Creating a small Qwen3VLForConditionalGeneration model for testing...")

    # Create a small config for testing to make the example lightweight
    config = Qwen3VLConfig(
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 2,
            "num_heads": 4,
            "patch_size": 14,
            "temporal_patch_size": 1,
            "in_channels": 3,
            "num_position_embeddings": 144,  # 12*12
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [],
        },
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "vocab_size": 1000,
            "pad_token_id": 0,
        },
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
    )

    # Create the original model
    model_fp = Qwen3VLForConditionalGeneration(config)
    print(f"Original model created: {type(model_fp).__name__}")

    # Wrap the model with QuantQwen3VLForConditionalGeneration using `prepare`
    # This is the standard way to wrap models in tico
    print("\nWrapping the model with QuantQwen3VLForConditionalGeneration...")
    qmodel = prepare(model_fp, PTQConfig())
    qmodel.eval()  # Set to evaluation mode

    print(f"Quantized model created: {type(qmodel).__name__}")
    print(f"Wrapped module type: {type(qmodel.wrapped).__name__}")
    print(f"Initial mode: {qmodel._mode.name}")

    # Check that the model is in NO_QUANT mode
    assert qmodel._mode is Mode.NO_QUANT

    # Enable calibration mode (this is done internally by `prepare`, but we can check)
    print("\nModel is ready for calibration.")

    # -------------------------------------------------------------------------
    # 2. Calibration with synthetic data
    # -------------------------------------------------------------------------
    print("\nPerforming calibration with synthetic data...")

    # Create dummy inputs for calibration
    # For simplicity, we will not provide pixel_values, so no vision processing
    # This means the vision components will not be calibrated, but the text part will be
    BATCH_SIZE = 2
    SEQ_LEN = 10
    VOCAB_SIZE = config.text_config.vocab_size

    CALIB_INPUTS = []
    for _ in range(5):  # 5 calibration samples
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        CALIB_INPUTS.append({"input_ids": input_ids})

    # Run calibration
    with torch.no_grad():
        for inp in CALIB_INPUTS:
            _ = qmodel(**inp)

    print("Calibration completed.")

    # -------------------------------------------------------------------------
    # 3. Freeze quantization parameters
    # -------------------------------------------------------------------------
    print("\nFreezing quantization parameters...")
    convert(qmodel)

    print(f"Mode after convert: {qmodel._mode.name}")
    assert qmodel._mode is Mode.QUANT, "Quantization mode should be active now."
    print("Quantization parameters frozen.")

    # -------------------------------------------------------------------------
    # 4. Quick diff check (INT-sim vs FP32)
    # -------------------------------------------------------------------------
    print("\nComparing quantized and original model outputs...")
    test_input = {"input_ids": torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))}

    with torch.no_grad():
        q_out = qmodel(**test_input)
        fp_out = model_fp(**test_input)

    # For Qwen3VLForConditionalGeneration, the output is typically a CausalLMOutputWithPast
    # which has a `logits` attribute
    logits_quant = q_out.logits
    logits_fp = fp_out.logits

    diff_mean = (logits_quant - logits_fp).abs().mean().item()
    peir = compute_peir(logits_fp, logits_quant) * 100

    print("┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {diff_mean:.6f}")
    print(f"│ PEIR       : {peir:.6f} %")
    print("└──────────────────────────────────────────────────────")

    # Optionally, plot the outputs (this might not be very informative for high-dim tensors)
    # print(plot_two_outputs(logits_fp, logits_quant))

    # -------------------------------------------------------------------------
    # 5. Export the quantized model
    # -------------------------------------------------------------------------
    print("\nExporting the quantized model...")
    save_path = pathlib.Path("qwen3vl_conditional_generation.q.circle")

    # Example input for export
    example_input = (test_input["input_ids"],)

    with SuppressWarning(UserWarning, ".*"):
        try:
            import tico

            cm = tico.convert(qmodel, example_input)
            cm.save(save_path)
            print(f"Quantized Circle model saved to {save_path.resolve()}")
        except Exception as e:
            print(f"Export failed: {e}")
            print(
                "This might be expected if the model is not fully supported for export yet."
            )


if __name__ == "__main__":
    main()

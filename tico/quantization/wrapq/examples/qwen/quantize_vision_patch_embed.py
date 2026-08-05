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

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


if not has_transformers_for("qwen3-vl"):
    print(
        "Error: Required transformers package not installed. "
        "Cannot test Qwen3VLVisionPatchEmbed."
    )
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed


def generate_calibration_data(batch_size: int, sample_shape: tuple) -> list:
    """Generate processor-style flattened patch calibration data."""
    return [torch.randn(sample_shape) for _ in range(batch_size)]


def main():
    cfg = Qwen3VLVisionConfig(
        in_channels=3,
        hidden_size=1024,
        temporal_merge_size=2,
        patch_size=16,
    )
    model = Qwen3VLVisionPatchEmbed(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    assert model.proj.in_channels == 3
    assert model.proj.out_channels == 1024
    assert model.proj.kernel_size == (2, 16, 16)
    assert model.proj.stride == (2, 16, 16)

    num_patches = 16
    patch_dim = (
        cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    )
    sample_shape = (1, num_patches, patch_dim)
    calibration_data = generate_calibration_data(
        batch_size=20,
        sample_shape=sample_shape,
    )

    ptq_config = tico.quantization.config.ptq.PTQConfig()
    prepared_model = tico.quantization.prepare(model, ptq_config, inplace=True)

    with torch.no_grad():
        for batch in calibration_data:
            prepared_model(batch)

    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    with torch.no_grad():
        quant_out = quantized_model(calibration_data[0])
        fp_out = orig_model(calibration_data[0])

    print("┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print("└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # The exported ABI is [batch=1, num_patches, patch_dim]. The wrapper folds
    # the patch sequence into a batch-one rank-3 Linear input and returns
    # [num_patches, hidden_size].
    example_inputs = (torch.randn(sample_shape),)
    circle_model = tico.convert(quantized_model.eval(), example_inputs)

    filename = "qwen3vl_vision_patch_embed.q.circle"
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

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
from collections import namedtuple

import torch

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.utils.version import has_transformers_for

torch.manual_seed(123)


if not has_transformers_for("qwen3-vl"):
    print("Error: transformers package not installed. Cannot test Qwen3VLVisionModel.")
    sys.exit(1)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_model import (
    QuantQwen3VLVisionModel,
)
from tico.quantization.wrapq.wrappers.qwen_vl.vision_profile import Qwen3VLVisionProfile


def generate_calibration_data(batch_size: int, sample_shape: tuple) -> list:
    """Generate processor-style flattened patch calibration data."""
    return [torch.randn(sample_shape) for _ in range(batch_size)]


def main():
    cfg = Qwen3VLVisionConfig(
        hidden_size=1024,
        num_position_embeddings=2304,
        temporal_patch_size=2,
        patch_size=16,
        depth=2,
    )
    model = Qwen3VLVisionModel(cfg)
    orig_model = copy.deepcopy(model)
    model.eval()

    # The processor supplies one flattened vector for each temporal/spatial patch.
    THW = namedtuple(
        "THW", ["num_temporal_patches", "num_height_patches", "num_width_patches"]
    )
    vision_grid_thw = THW(1, 24, 24)
    grid_thw = torch.tensor([vision_grid_thw])

    num_patches = (
        vision_grid_thw.num_temporal_patches
        * vision_grid_thw.num_height_patches
        * vision_grid_thw.num_width_patches
    )
    patch_dim = (
        cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    )
    input_shape = (1, num_patches, patch_dim)

    print(f"Input shape: {input_shape}")
    print(f"grid_thw: {grid_thw.tolist()}")

    calibration_data = generate_calibration_data(
        batch_size=20,
        sample_shape=input_shape,
    )

    # Calibration and eager evaluation are profile-agnostic. Each sample supplies
    # its actual grid to the vision wrapper at call time.
    ptq_config = tico.quantization.config.ptq.PTQConfig()
    prepared_model = tico.quantization.prepare(model, ptq_config, inplace=True)

    with torch.no_grad():
        for batch in calibration_data:
            prepared_model(batch, grid_thw)

    quantized_model = tico.quantization.convert(prepared_model, inplace=True)

    with torch.no_grad():
        test_input = calibration_data[0]
        quant_out = quantized_model(test_input, grid_thw)
        fp_out = orig_model(test_input, grid_thw)

        if QuantQwen3VLVisionModel.has_deepstack_model_output:
            quant_out = quant_out.pooler_output
            fp_out = fp_out.pooler_output
        else:
            quant_out = quant_out[0]
            fp_out = fp_out[0]

    print("┌───────────── Quantization Error Summary ─────────────")
    print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
    print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
    print("└──────────────────────────────────────────────────────")
    print(plot_two_outputs(fp_out, quant_out))

    # Static export owns one explicit deployment profile and exposes a pixel-only
    # graph ABI. The CPU runtime uses the profile to select the matching artifact.
    profile = Qwen3VLVisionProfile.from_grid_thw(grid_thw)
    profile.validate_spatial_merge_size(int(cfg.spatial_merge_size))
    export_model = quantized_model.as_export_module(
        mode="prefill",
        grid_thw=profile,
    ).eval()
    circle_model = tico.convert(export_model, (calibration_data[0],))

    filename = profile.circle_filename("q")
    circle_model.save(filename)
    print(f"Circle model saved as '{filename}'")


if __name__ == "__main__":
    main()

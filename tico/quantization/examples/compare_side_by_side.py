#!/usr/bin/env python3
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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
Compare Intermediate Outputs Between FP and Quantized Models.

This script quantizes a model using a recipe configuration and compares the
intermediate outputs of each submodule between the original (floating-point)
and quantized models. It uses forward hooks to capture outputs at each layer
and computes the Percentage Error in Range (PEIR) metric.

The PEIR metric measures how much the quantized output deviates from the
original output relative to the output range, providing layer-by-layer
insight into quantization error propagation.

Usage:
    python -m tico.quantization.examples.compare_side_by_side \
        --config path/to/config.yaml \
        [--set key=value ...]

Example:
    python -m tico.quantization.examples.compare_side_by_side \
        --config tico/quantization/examples/configs/llama_universal_gptq_quantize.yaml \
        --set "pipeline.1.gptq.weight_bits=8"

The script requires:
    - A recipe YAML config with model, calibration, and pipeline settings
    - Sufficient GPU memory to hold both FP and quantized models
    - The model family adapter registered in TICO
"""

import argparse
import gc
from typing import Any

import torch.nn as nn

from tico.quantization.recipes.adapters import resolve_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import (  # noqa: F401
    merge_diff_stats,
    trace_parity,
)
from tico.quantization.recipes.extensions import load_recipe_extensions
from tico.quantization.recipes.runner import QuantizationRunner
from tico.utils.utils import print_gpu_memory


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Trace and evaluate PEIR for GPTQ-quantized models."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to recipe YAML/JSON config for quantization.",
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted paths.",
    )

    return parser.parse_args()


def get_fp_model_context(cfg: dict[str, Any]) -> RecipeContext:
    adapter = resolve_adapter(cfg)
    ctx = RecipeContext(cfg=cfg, adapter=adapter)
    ctx = adapter.load_model(ctx)
    return ctx


def get_quanized_model_context(cfg: dict[str, Any]) -> RecipeContext:
    ctx: RecipeContext = QuantizationRunner().run(cfg)
    return ctx


def main() -> None:
    """Main entry point for the trace_gptq script."""
    args = parse_arguments()

    # Load config with overrides
    overrides = list(args.set)
    cfg = load_recipe_config(args.config, overrides=overrides)
    load_recipe_extensions(cfg)

    # Disable evaluation and export stages (we only want quantization)
    if "evaluation" in cfg:
        cfg["evaluation"]["enabled"] = False
    if "export" in cfg:
        cfg["export"]["enabled"] = False

    print_gpu_memory("Init")

    q_ctx: RecipeContext = get_quanized_model_context(cfg)
    print_gpu_memory("After get_quanized_model_context")

    fp_ctx: RecipeContext = get_fp_model_context(cfg)
    print_gpu_memory("After get_fp_model_context")

    q_model: nn.Module = q_ctx.model.eval()
    fp_model: nn.Module = fp_ctx.model.eval()

    adapter = resolve_adapter(cfg)
    fp_ctx.calibration_inputs = adapter.build_calibration_inputs(fp_ctx)

    gc.collect()
    print_gpu_memory("Before trace_parity")

    trace_parity(
        fp_model=fp_model,
        q_model=q_model,
        ctx=fp_ctx,
    )


if __name__ == "__main__":
    main()

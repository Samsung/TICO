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
import math
from typing import Any, Mapping

import torch
import torch.nn as nn

from tqdm import tqdm

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import (
    collect_forward_outputs,
    compare_outputs,
)
from tico.quantization.recipes.runner import QuantizationRunner
from tico.quantization.wrapq.utils.introspection import DifferenceStatistics


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
    adapter = get_adapter(cfg["model"]["family"])
    ctx = RecipeContext(cfg=cfg, adapter=adapter)
    ctx = adapter.load_model(ctx)
    return ctx


def get_quanized_model_context(cfg: dict[str, Any]) -> RecipeContext:
    ctx: RecipeContext = QuantizationRunner().run(cfg)
    return ctx


def merge_diff_stats(
    accumulated_diff: DifferenceStatistics,
    new_diff: DifferenceStatistics,
    n: int,
) -> DifferenceStatistics:
    assert n > 1
    squared_stddev = accumulated_diff.stddev**2
    return DifferenceStatistics(
        min=min(accumulated_diff.min, new_diff.min),
        max=max(accumulated_diff.max, new_diff.max),
        mean=accumulated_diff.mean + (new_diff.mean - accumulated_diff.mean) / n,
        stddev=math.sqrt(squared_stddev + (new_diff.stddev**2 - squared_stddev) / n),
        peir=accumulated_diff.peir + (new_diff.peir - accumulated_diff.peir) / n,
    )


def trace_parity(
    fp_model: nn.Module,
    q_model: nn.Module,
    ctx: RecipeContext,
) -> None:
    """
    Compare intermediate outputs between FP and quantized models.

    This function runs both models on the same calibration input and collects
    the output of each submodule using forward hooks. It then computes and
    displays the PEIR (Percentage Error in Range) for each layer.

    Parameters:
        fp_model: The original floating-point model.
        q_model: The quantized model to compare against.
        ctx: Recipe context containing calibration inputs and device info.

    Raises:
        RuntimeError: If no calibration inputs are available in the context.
    """
    if not ctx.calibration_inputs:
        raise RuntimeError("Trace requires at least one calibration input.")

    accumulated_diff_stats: dict[str, DifferenceStatistics] = {}

    i = 0
    for sample in tqdm(
        ctx.calibration_inputs, unit="batch", desc="side-by-side difference"
    ):
        sample = (
            sample if not isinstance(sample, torch.Tensor) else sample.to(ctx.device)
        )
        if isinstance(sample, Mapping):
            sample = {
                k: v.to(ctx.device) if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }

        fp_outputs = collect_forward_outputs(
            fp_model,
            sample,
            print_trace=False,
            skip_wrappers=False,
        )

        q_outputs = collect_forward_outputs(
            q_model,
            sample,
            print_trace=False,
            skip_wrappers=False,
        )

        new_diff: dict[str, DifferenceStatistics] = compare_outputs(
            fp_outputs, q_outputs, verbose=False
        )

        if i == 0:
            accumulated_diff_stats = new_diff
        else:
            for layer_name, accumulated_diff in accumulated_diff_stats.items():
                accumulated_diff_stats[layer_name] = merge_diff_stats(
                    accumulated_diff=accumulated_diff,
                    new_diff=new_diff[layer_name],
                    n=i + 1,
                )

        i += 1

    for layer_name, delta_stats in accumulated_diff_stats.items():
        print(
            f"{layer_name}: mean|diff|={delta_stats.mean:.8f}, max|diff|={delta_stats.max:.8f}, peir={delta_stats.peir * 100:.6f}%"
        )


def main() -> None:
    """Main entry point for the trace_gptq script."""
    args = parse_arguments()

    # Load config with overrides
    overrides = list(args.set)
    cfg = load_recipe_config(args.config, overrides=overrides)

    # Disable evaluation and export stages (we only want quantization)
    if "evaluation" in cfg:
        cfg["evaluation"]["enabled"] = False
    if "export" in cfg:
        cfg["export"]["enabled"] = False

    q_ctx: RecipeContext = get_quanized_model_context(cfg)
    fp_ctx: RecipeContext = get_fp_model_context(cfg)
    q_model: nn.Module = q_ctx.model.eval()
    fp_model: nn.Module = fp_ctx.model.eval()

    adapter = get_adapter(cfg["model"]["family"])
    fp_ctx.calibration_inputs = adapter.build_calibration_inputs(fp_ctx)

    trace_parity(
        fp_model=fp_model,
        q_model=q_model,
        ctx=fp_ctx,
    )


if __name__ == "__main__":
    main()

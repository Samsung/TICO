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
import gc
import math
from collections import OrderedDict
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from tqdm import tqdm

from tico.quantization import convert, prepare
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.override_policies import apply_ptq_override_policies
from tico.quantization.wrapq.utils.introspection import (
    DifferenceStatistics,
    get_tensor_statistics,
    TensorStatistics,
)
from tico.utils.utils import print_gpu_memory


def _summarize(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, empty)"
        data = value.detach().float()
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"mean={data.mean().item():.5f}, min={data.min().item():.5f}, "
            f"max={data.max().item():.5f}, std={data.std().item():.5f})"
        )
    if isinstance(value, (tuple, list)):
        return (
            "["
            + ", ".join(_summarize(v) for v in value[:3])
            + (", ..." if len(value) > 3 else "")
            + "]"
        )
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(f"{k}: {_summarize(v)}" for k, v in list(value.items())[:5])
            + "}"
        )
    return repr(value)


def collect_forward_outputs(
    model: torch.nn.Module,
    inputs: Any,
    *,
    interesting_modules: Iterable[str] = (),
    print_trace: bool = True,
    skip_wrappers: bool = True,
) -> OrderedDict[str, Any]:
    """
    Run ``model`` once and collect each selected submodule's forward output.

    Forward hooks registered by this call are removed before returning or
    propagating an exception, including failures during hook registration,
    the model forward, or the trace printing callback.
    """
    outputs: OrderedDict[str, Any] = OrderedDict()
    interesting = set(interesting_modules)

    hooks: list[torch.utils.hooks.RemovableHandle] = []
    try:
        for name, module in model.named_modules():
            name = name.replace(".wrapped", "").replace("wrapped.", "")
            if skip_wrappers and module.__class__.__name__.startswith("Quant"):
                continue
            if (
                interesting
                and name not in interesting
                and not any(name.startswith(prefix) for prefix in interesting)
            ):
                continue

            def make_hook(module_name: str):
                def hook(_module, args, out):
                    outputs[module_name] = out
                    if print_trace:
                        print(f"[{module_name}] {_summarize(out)}")

                return hook

            hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            if isinstance(inputs, Mapping):
                model(**inputs)
            elif isinstance(inputs, tuple):
                model(*inputs)
            else:
                model(inputs)
    finally:
        for hook in hooks:
            hook.remove()

    return outputs


def compare_outputs(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    verbose: bool = True,
) -> dict[str, DifferenceStatistics]:
    if verbose:
        print("\n=== Side-by-side diff ===")
    common = [name for name in left.keys() if name in right]
    result: dict[str, DifferenceStatistics] = {}
    for name in common:
        lval, rval = left[name], right[name]
        if (
            isinstance(lval, torch.Tensor)
            and isinstance(rval, torch.Tensor)
            and lval.shape == rval.shape
        ):
            diff: torch.Tensor = (lval.detach().float() - rval.detach().float()).abs()
            delta_stats: TensorStatistics = get_tensor_statistics(diff)
            interval = (lval.detach().max() - lval.detach().min()).item()
            peir = delta_stats.max / interval if interval != 0.0 else float("nan")
            result[name] = DifferenceStatistics(**delta_stats._asdict(), peir=peir)
            if verbose:
                print(
                    f"{name}: mean|diff|={delta_stats.mean:.8f}, max|diff|={delta_stats.max:.8f}, peir={peir * 100:.6f}%"
                )
        elif verbose:
            print(f"{name}: non-tensor or shape-mismatched output")
    return result


def trace_ptq_parity(
    ctx: RecipeContext,
    *,
    enable_quantization: bool = False,
    interesting_modules: Iterable[str] = (),
) -> None:
    """Trace FP model and a PTQ-prepared copy on the first calibration sample."""
    if not ctx.calibration_inputs:
        raise RuntimeError("Trace requires at least one calibration input.")

    sample = ctx.calibration_inputs[0]
    sample = sample if not isinstance(sample, torch.Tensor) else sample.to(ctx.device)
    if isinstance(sample, Mapping):
        sample = {
            k: v.to(ctx.device) if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }

    fp_model = ctx.model.eval()
    q_model = copy.deepcopy(ctx.model).eval()
    ptq_stage_cfg = next(
        (s for s in ctx.cfg.get("pipeline", []) if s.get("name") == "ptq"), None
    )
    if ptq_stage_cfg is None:
        raise RuntimeError("Trace mode requires a PTQ stage in the recipe config.")

    qcfg = ctx.adapter.build_ptq_config(ctx, ptq_stage_cfg)
    # Apply the recipe's selector/raw overrides through the same helper as
    # PTQStage so the traced copy is prepared with the effective PTQ policy.
    # The copy is both the structure source and the prepare target, and the
    # selected adapter declares that structure exactly as in PTQStage.
    qcfg = apply_ptq_override_policies(
        qcfg,
        ptq_stage_cfg,
        family=ctx.adapter.family,
        model=q_model,
        adapter=ctx.adapter,
    )
    q_model = prepare(q_model, qcfg)
    ctx.adapter.calibrate_prepared_model(ctx, q_model, ptq_stage_cfg)
    if enable_quantization:
        q_model = convert(q_model)

    print("\n=== FP trace ===")
    fp_outputs = collect_forward_outputs(
        fp_model,
        sample,
        interesting_modules=interesting_modules,
        print_trace=True,
        skip_wrappers=False,
    )

    print("\n=== PTQ trace ===")
    q_outputs = collect_forward_outputs(
        q_model,
        sample,
        interesting_modules=interesting_modules,
        print_trace=True,
        skip_wrappers=False,
    )

    compare_outputs(fp_outputs, q_outputs)


def merge_diff_stats(
    accumulated_diff: DifferenceStatistics,
    new_diff: DifferenceStatistics,
    n: int,
) -> DifferenceStatistics:
    """
    Fold the ``n``-th batch summary into the running per-batch aggregate.

    ``min``/``max`` take the extreme of the two summaries; ``mean``, ``peir``,
    and the squared ``stddev`` are running averages over batch summaries, so
    every batch has equal weight regardless of its element count. This is an
    aggregate of per-batch statistics, not a pooled statistic over all elements.
    """
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
    Compare intermediate outputs between two already prepared models.

    Unlike :func:`trace_ptq_parity`, this does not load, prepare, calibrate, or
    convert anything. Both models are run on every calibration input in
    ``ctx.calibration_inputs`` (FP model first, then ``q_model``), each
    submodule's output is collected with forward hooks, and the per-batch
    difference summaries are accumulated with :func:`merge_diff_stats` before
    the final PEIR (Percentage Error in Range) summary is printed.

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

        fp_outputs.clear()
        q_outputs.clear()
        gc.collect()
        print_gpu_memory(f"Batch {i}")

        i += 1

    for layer_name, delta_stats in accumulated_diff_stats.items():
        print(
            f"{layer_name}: mean|diff|={delta_stats.mean:.8f}, max|diff|={delta_stats.max:.8f}, peir={delta_stats.peir * 100:.6f}%"
        )

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
from collections import OrderedDict
from typing import Any, Iterable, Mapping

import torch

from tico.quantization import convert, prepare
from tico.quantization.recipes.context import RecipeContext


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
    outputs: OrderedDict[str, Any] = OrderedDict()
    interesting = set(interesting_modules)

    hooks = []
    for name, module in model.named_modules():
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

    for hook in hooks:
        hook.remove()

    return outputs


def compare_outputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    print("\n=== Side-by-side diff ===")
    common = [name for name in left.keys() if name in right]
    for name in common:
        lval, rval = left[name], right[name]
        if (
            isinstance(lval, torch.Tensor)
            and isinstance(rval, torch.Tensor)
            and lval.shape == rval.shape
        ):
            diff = (lval.detach().float() - rval.detach().float()).abs()
            print(
                f"{name}: mean|diff|={diff.mean().item():.8f}, max|diff|={diff.max().item():.8f}"
            )
        else:
            print(f"{name}: non-tensor or shape-mismatched output")


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

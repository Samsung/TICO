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

"""Runner for wrapper-level quantization smoke checks."""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.registry import case_names, get_case
from tico.quantization.recipes.debug.wrapper_smoke.result import WrapperSmokeResult
from tico.quantization.recipes.debug.wrapper_smoke.utils import (
    calibration_iters,
    detach_cpu_float,
    ensure_output_dir,
    finite_all,
    safe_peir,
    smoke_section,
    suppress_tico_warnings,
)


def _limit_inputs(inputs: list[ForwardInput], limit: int) -> list[ForwardInput]:
    """Limit calibration inputs while preserving at least one sample."""
    if limit <= 0:
        return inputs
    return inputs[: max(1, min(len(inputs), limit))]


def _write_report(result: WrapperSmokeResult, report_json: str | Path | None) -> None:
    """Write a JSON report when requested."""
    if report_json is None:
        return
    path = Path(report_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    result.artifacts["report_json"] = str(path)


def _print_plot(
    case_name: str,
    fp_tensor: torch.Tensor,
    quant_tensor: torch.Tensor,
    output_dir: Path | None,
    emit_plot: bool,
    result: WrapperSmokeResult,
) -> None:
    """Render and optionally persist the existing plot_two_outputs visualization."""
    if not emit_plot:
        return
    try:
        from tico.quantization.evaluation.utils import plot_two_outputs

        plot = plot_two_outputs(fp_tensor, quant_tensor)
    except Exception as exc:
        result.messages.append(f"plot_two_outputs failed: {exc}")
        return

    print(plot)
    if output_dir is not None:
        plot_path = output_dir / f"{case_name}.plot.txt"
        plot_path.write_text(str(plot), encoding="utf-8")
        result.artifacts["plot"] = str(plot_path)


def _export_circle(
    case: WrapperSmokeCase,
    quantized: torch.nn.Module,
    eval_sample: ForwardInput,
    cfg: Mapping[str, Any],
    output_dir: Path,
    result: WrapperSmokeResult,
) -> None:
    """Export a quantized smoke module to Circle."""
    try:
        import tico

        export_module = case.export_module(quantized, cfg).eval()
        export_input = case.export_input(eval_sample, cfg)
        args = tuple(export_input.args)
        kwargs = dict(export_input.kwargs)
        with torch.no_grad(), suppress_tico_warnings():
            if kwargs:
                circle_model = tico.convert(export_module, args, kwargs)
            else:
                circle_model = tico.convert(export_module, args)
        path = output_dir / case.export_filename(cfg)
        circle_model.save(path)
        result.artifacts["circle"] = str(path)
    except Exception as exc:
        result.passed = False
        result.messages.append(f"Circle export failed: {exc}")


def run_wrapper_smoke(
    case_name: str,
    *,
    cfg: Mapping[str, Any] | None = None,
    export: str = "none",
    output_dir: str | Path | None = None,
    strict: bool = False,
    emit_plot: bool = True,
    report_json: str | Path | None = None,
    calibration_limit: int | None = None,
) -> WrapperSmokeResult:
    """Run one wrapper smoke case and return a structured result."""
    cfg = cfg or {}
    section = smoke_section(cfg)
    case = get_case(case_name)
    availability = case.availability()
    if not availability.available:
        result = WrapperSmokeResult(
            case=case.name,
            passed=False,
            messages=[availability.reason or "case is not available"],
        )
        if strict:
            result.raise_if_failed()
        return result

    result = WrapperSmokeResult(case=case.name, passed=True)
    export_requested = export != "none" or bool(
        section.get("export", {}).get("enabled", False)
    )
    output_path = (
        ensure_output_dir(output_dir or section.get("output_dir"))
        if export_requested or report_json or emit_plot
        else None
    )

    with torch.no_grad():
        model, reference = case.build(cfg)
        prepared = case.prepare_model(model, cfg)
        case.after_prepare(prepared, cfg)

        limit = calibration_limit
        if limit is None:
            limit = calibration_iters(cfg, default=3)
        cal_inputs = _limit_inputs(case.calibration_inputs(prepared, cfg), int(limit))
        for sample in cal_inputs:
            case.forward(prepared, sample)

        eval_sample = case.eval_input(prepared, cfg)
        if case.compare_reference_source == "prepared":
            fp_output = case.forward(prepared, eval_sample)
        else:
            fp_output = case.reference_forward(reference, eval_sample)

        quantized = case.convert_model(prepared, cfg)
        quant_output = case.forward(quantized, eval_sample)

    fp_tensor = detach_cpu_float(case.output_tensor(fp_output))
    quant_tensor = detach_cpu_float(case.output_tensor(quant_output))
    shape_match = tuple(fp_tensor.shape) == tuple(quant_tensor.shape)
    quant_finite = finite_all(quant_tensor)
    fp_finite = finite_all(fp_tensor)

    if shape_match:
        diff = (quant_tensor - fp_tensor).abs()
        mean_abs_diff = float(diff.mean().item())
        max_abs_diff = float(diff.max().item())
        peir = safe_peir(fp_tensor, quant_tensor)
    else:
        mean_abs_diff = None
        max_abs_diff = None
        peir = None

    result.metrics.update(
        {
            "shape_match": shape_match,
            "fp_shape": list(fp_tensor.shape),
            "quant_shape": list(quant_tensor.shape),
            "fp_finite": fp_finite,
            "quant_finite": quant_finite,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "peir": peir,
        }
    )

    if not shape_match:
        result.passed = False
        result.messages.append("output shape mismatch")
    if not quant_finite:
        result.passed = False
        result.messages.append("quantized output has non-finite values")
    if not fp_finite:
        result.passed = False
        result.messages.append("floating-point reference output has non-finite values")
    if mean_abs_diff is not None and case.max_mean_abs_diff is not None:
        if mean_abs_diff > case.max_mean_abs_diff:
            result.passed = False
            result.messages.append(
                f"mean_abs_diff {mean_abs_diff:.6f} exceeds {case.max_mean_abs_diff:.6f}"
            )
    if mean_abs_diff is not None and case.min_mean_abs_diff is not None:
        if mean_abs_diff < case.min_mean_abs_diff:
            result.passed = False
            result.messages.append(
                f"mean_abs_diff {mean_abs_diff:.6f} is below {case.min_mean_abs_diff:.6f}"
            )
    if peir is not None and case.max_peir is not None:
        if peir > case.max_peir:
            result.passed = False
            result.messages.append(f"PEIR {peir:.6f} exceeds {case.max_peir:.6f}")

    if export_requested:
        assert output_path is not None
        export_kind = (
            export
            if export != "none"
            else section.get("export", {}).get("artifact", "circle")
        )
        if export_kind != "circle":
            result.passed = False
            result.messages.append(f"unsupported export artifact: {export_kind}")
        else:
            _export_circle(case, quantized, eval_sample, cfg, output_path, result)

    _write_report(result, report_json)
    print(result.format_text())
    if shape_match:
        _print_plot(case.name, fp_tensor, quant_tensor, output_path, emit_plot, result)
    if strict:
        result.raise_if_failed()
    return result


def run_wrapper_smoke_suite(
    cases: Sequence[str] | None = None,
    *,
    cfg: Mapping[str, Any] | None = None,
    export: str = "none",
    output_dir: str | Path | None = None,
    strict: bool = False,
    emit_plot: bool = True,
    calibration_limit: int | None = None,
) -> list[WrapperSmokeResult]:
    """Run multiple smoke cases and return all results."""
    selected = list(cases or case_names())
    results: list[WrapperSmokeResult] = []
    for name in selected:
        result = run_wrapper_smoke(
            name,
            cfg=cfg,
            export=export,
            output_dir=output_dir,
            strict=False,
            emit_plot=emit_plot,
            calibration_limit=calibration_limit,
        )
        results.append(result)
    if strict:
        failed = [result.case for result in results if not result.passed]
        if failed:
            raise RuntimeError(f"Wrapper smoke suite failed: {', '.join(failed)}")
    return results

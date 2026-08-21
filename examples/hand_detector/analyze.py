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

"""Run numerical quantization analyses for the MediaPipe palm detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from examples.hand_detector import (
    adaround as adaround_cli,
    block_reconstruction as block_reconstruction_cli,
    precision_ablation as precision_ablation_cli,
)
from examples.hand_detector._support.analysis import (
    output_boundaries,
    OUTPUT_NAMES,
    summarize_percentile_observers,
)
from examples.hand_detector._support.cumulative_sensitivity import (
    build_activation_sensitivity_path_report,
    print_activation_sensitivity_path,
    select_activation_sensitivity_groups,
)
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.group_observer_sweep import (
    activation_group_override_paths,
    build_group_observer_policies,
    build_group_observer_sweep_result,
    evaluate_internal_full,
    EvaluatedGroupObserverPolicy,
    make_group_observer_overrides,
    print_group_observer_sweep,
    validate_group_observer_overrides,
)
from examples.hand_detector._support.observer_sweep import (
    build_observer_sweep_result,
    evaluate_observer_profiles,
    OBSERVER_SWEEP_PROFILES,
    print_observer_sweep,
)
from examples.hand_detector._support.quantization import (
    export_quantized_circle,
    quantization_name,
    quantize_candidate,
)
from examples.hand_detector._support.sensitivity import (
    build_activation_sensitivity_groups,
    build_activation_sensitivity_report,
    print_activation_sensitivity,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector
from tico.quantization.analysis import (
    AffineQuantizationPolicy,
    build_clipping_candidates,
    collect_output_calibration_data,
    evaluate_clipping_candidates,
    make_output_adapter,
    QuantizationAblation,
    QuantizationProfile,
    QuantizationSensitivity,
    SensitivityMode,
)
from tico.quantization.wrapq.control import iter_quantization_sites
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver


DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PERCENTILES = (99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999)
DEFAULT_GROUP_PERCENTILES = (99.9, 99.95, 99.99, 99.995, 99.999)
DEFAULT_TAIL_PERCENTAGES = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def parse_args() -> argparse.Namespace:
    """Parse the analysis subcommand and its arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ablation = subparsers.add_parser(
        "ablation",
        help=(
            "Run output-only, weight-only, activation-only, full, and "
            "internal-full PTQ."
        ),
    )
    _add_model_arguments(ablation)
    _add_dataset_arguments(ablation, evaluation=True)
    ablation.add_argument("--bits", type=int, default=8, choices=[8, 16])
    ablation.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "ablation.json",
    )

    clipping = subparsers.add_parser(
        "output-clipping",
        help="Compare MinMax, percentile, and L1 output clipping ranges.",
    )
    _add_model_arguments(clipping)
    _add_dataset_arguments(clipping, evaluation=True)
    clipping.add_argument("--bits", type=int, default=8, choices=[8, 16])
    clipping.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_PERCENTILES),
    )
    clipping.add_argument(
        "--tail-percentages",
        type=float,
        nargs="+",
        default=list(DEFAULT_TAIL_PERCENTAGES),
    )
    clipping.add_argument("--skip-l1-search", action="store_true")
    clipping.add_argument("--max-values-per-output", type=int, default=1_000_000)
    clipping.add_argument("--sampling-seed", type=int, default=20260803)
    clipping.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "output_clipping.json",
    )

    observer_sweep = subparsers.add_parser(
        "observer-sweep",
        help=(
            "Compare MinMax and percentile activation observers across "
            "activation-only, full, and internal-full PTQ."
        ),
    )
    _add_model_arguments(observer_sweep)
    _add_dataset_arguments(observer_sweep, evaluation=True)
    observer_sweep.add_argument("--bits", type=int, default=8, choices=[8, 16])
    observer_sweep.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_PERCENTILES),
    )
    observer_sweep.add_argument("--max-samples", type=int, default=131_072)
    observer_sweep.add_argument("--samples-per-batch", type=int, default=4_096)
    observer_sweep.add_argument("--sampling-seed", type=int, default=20260803)
    observer_sweep.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "observer_sweep.json",
    )

    group_observer_sweep = subparsers.add_parser(
        "group-observer-sweep",
        help=(
            "Keep a global percentile policy and sweep observer overrides for "
            "selected activation-domain groups."
        ),
    )
    _add_model_arguments(group_observer_sweep)
    _add_dataset_arguments(group_observer_sweep, evaluation=True)
    group_observer_sweep.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[8, 16],
    )
    group_observer_sweep.add_argument(
        "--global-percentile",
        type=float,
        default=99.99,
    )
    group_observer_sweep.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_GROUP_PERCENTILES),
    )
    group_observer_sweep.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Activation sensitivity group names evaluated independently.",
    )
    group_observer_sweep.add_argument(
        "--score-output",
        choices=OUTPUT_NAMES,
        default="regressors",
    )
    group_observer_sweep.add_argument("--skip-minmax", action="store_true")
    group_observer_sweep.add_argument("--max-samples", type=int, default=131_072)
    group_observer_sweep.add_argument(
        "--samples-per-batch",
        type=int,
        default=4_096,
    )
    group_observer_sweep.add_argument(
        "--sampling-seed",
        type=int,
        default=20260803,
    )
    group_observer_sweep.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "group_observer_sweep.json",
    )

    block_reconstruction_cli.add_subparser(subparsers)
    adaround_cli.add_subparser(subparsers)
    precision_ablation_cli.add_subparser(subparsers)

    sensitivity = subparsers.add_parser(
        "activation-sensitivity",
        help=(
            "Analyze independent, cumulative, or greedy activation-domain "
            "paths from a percentile-calibrated E:internal-full baseline."
        ),
    )
    _add_model_arguments(sensitivity)
    _add_dataset_arguments(sensitivity, evaluation=True)
    sensitivity.add_argument("--bits", type=int, default=8, choices=[8, 16])
    sensitivity.add_argument("--percentile", type=float, default=99.99)
    sensitivity.add_argument("--max-samples", type=int, default=131_072)
    sensitivity.add_argument("--samples-per-batch", type=int, default=4_096)
    sensitivity.add_argument("--sampling-seed", type=int, default=20260803)
    sensitivity.add_argument(
        "--score-output",
        choices=OUTPUT_NAMES,
        default="regressors",
    )
    sensitivity.add_argument(
        "--strategy",
        choices=("independent", "cumulative", "ranked", "greedy"),
        default="independent",
        help=(
            "Independent leave-one-float ranking, an explicitly ordered "
            "cumulative path, a fixed initial ranking, or greedy re-ranking "
            "after every selected group."
        ),
    )
    sensitivity.add_argument(
        "--groups",
        nargs="+",
        help=(
            "Optional ordered group names. Required for cumulative strategy; "
            "for ranked or greedy, restricts the candidate pool."
        ),
    )
    sensitivity.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help=(
            "Maximum ranked or greedy selections; zero evaluates only the " "baseline."
        ),
    )
    sensitivity.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.0,
        help=(
            "Stop greedy search when the best incremental score improvement "
            "is not greater than this value. Ignored by other strategies."
        ),
    )
    sensitivity.add_argument(
        "--top-k",
        type=int,
        default=20,
        help=(
            "Number of independent groups printed to the console; zero prints "
            "all groups."
        ),
    )
    sensitivity.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "activation_sensitivity.json",
    )
    return parser.parse_args()


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights",
        type=Path,
        default=DIRECTORY / "hand_detector_float.pt",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DIRECTORY / "hand_detector_spec.json",
    )


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    evaluation: bool,
) -> None:
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--synthetic-calibration-samples", type=int, default=32)
    if evaluation:
        parser.add_argument("--evaluation-dir", type=Path)
        parser.add_argument("--evaluation-offset", type=int, default=0)
        parser.add_argument("--evaluation-limit", type=int)
        parser.add_argument("--synthetic-evaluation-samples", type=int, default=8)
        parser.add_argument(
            "--require-disjoint",
            action="store_true",
            help="Reject overlapping calibration and evaluation file selections.",
        )


def main() -> None:
    """Dispatch the selected quantization analysis."""
    args = parse_args()
    if args.command == "ablation":
        _run_ablation(args)
    elif args.command == "output-clipping":
        _run_output_clipping(args)
    elif args.command == "observer-sweep":
        _run_observer_sweep(args)
    elif args.command == "group-observer-sweep":
        _run_group_observer_sweep(args)
    elif args.command == "block-reconstruction":
        block_reconstruction_cli.run(args)
    elif args.command == "adaround":
        adaround_cli.run(args)
    elif args.command == "precision-ablation":
        precision_ablation_cli.run(args)
    elif args.command == "activation-sensitivity":
        _run_activation_sensitivity(args)
    else:
        raise RuntimeError(f"Unhandled analysis command: {args.command}")


def _run_ablation(args: argparse.Namespace) -> None:
    float_model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    calibration, evaluation, data_metadata = _load_datasets(args)
    candidate = quantize_candidate(float_model, args.bits, calibration)
    runner = QuantizationAblation(
        float_model,
        candidate,
        boundaries=output_boundaries(candidate),
        output_adapter=OUTPUT_ADAPTER,
    )
    report = runner.run(
        evaluation,
        metadata={
            **data_metadata,
            "dtype": quantization_name(args.bits),
        },
    )
    _print_ablation(report.to_dict())
    output = report.write_json(args.report_json)
    print(f"\nWrote {output}")


def _run_output_clipping(args: argparse.Namespace) -> None:
    float_model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    calibration, evaluation, data_metadata = _load_datasets(args)
    policy = (
        AffineQuantizationPolicy.uint8()
        if args.bits == 8
        else AffineQuantizationPolicy.int16()
    )
    calibration_data = collect_output_calibration_data(
        float_model,
        calibration,
        output_adapter=OUTPUT_ADAPTER,
        max_values_per_output=args.max_values_per_output,
        seed=args.sampling_seed,
    )
    candidates = {
        data.name: build_clipping_candidates(
            data,
            policy,
            percentiles=args.percentiles,
            tail_percentages=args.tail_percentages,
            include_l1_search=not args.skip_l1_search,
        )
        for data in calibration_data
    }
    evaluated = evaluate_clipping_candidates(
        float_model,
        evaluation,
        calibration_data,
        candidates,
        policy,
        output_adapter=OUTPUT_ADAPTER,
    )
    _print_output_clipping(calibration_data, evaluated, policy.name)
    report = {
        "analysis": "output_clipping",
        "metadata": {
            **data_metadata,
            "dtype": policy.name,
            "max_values_per_output": args.max_values_per_output,
            "sampling_seed": args.sampling_seed,
        },
        "calibration_outputs": [
            {
                "name": data.name,
                "observed_minimum": data.observed_minimum,
                "observed_maximum": data.observed_maximum,
                "total_value_count": data.total_value_count,
                "sampled_value_count": data.sampled_value_count,
            }
            for data in calibration_data
        ],
        "outputs": {
            name: [candidate.to_dict() for candidate in output_candidates]
            for name, output_candidates in evaluated.items()
        },
    }
    _write_json(args.report_json, report)


def _run_observer_sweep(args: argparse.Namespace) -> None:
    _validate_percentiles(args.percentiles)
    float_model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    calibration, evaluation, data_metadata = _load_datasets(args)
    results: dict[str, dict[str, Any]] = {}

    minmax = quantize_candidate(
        float_model,
        args.bits,
        calibration,
        activation_observer=MinMaxObserver,
    )
    minmax_profiles = evaluate_observer_profiles(
        float_model,
        minmax,
        evaluation,
        boundaries=output_boundaries(minmax),
        output_adapter=OUTPUT_ADAPTER,
    )
    results["minmax"] = build_observer_sweep_result(
        observer="MinMaxObserver",
        profiles=minmax_profiles,
        observer_details=(),
    )

    for percentile in args.percentiles:
        name = f"percentile_{percentile:g}".replace(".", "_")
        candidate = quantize_candidate(
            float_model,
            args.bits,
            calibration,
            activation_observer=PercentileObserver,
            activation_observer_kwargs={
                "percentile": percentile,
                "max_samples": args.max_samples,
                "samples_per_batch": args.samples_per_batch,
                "seed": args.sampling_seed,
            },
        )
        profiles = evaluate_observer_profiles(
            float_model,
            candidate,
            evaluation,
            boundaries=output_boundaries(candidate),
            output_adapter=OUTPUT_ADAPTER,
        )
        results[name] = build_observer_sweep_result(
            observer="PercentileObserver",
            percentile=percentile,
            profiles=profiles,
            observer_details=summarize_percentile_observers(candidate),
        )

    print_observer_sweep(results, quantization_name(args.bits))
    _write_json(
        args.report_json,
        {
            "analysis": "activation_observer_sweep",
            "metadata": {
                **data_metadata,
                "dtype": quantization_name(args.bits),
                "percentiles": args.percentiles,
                "max_samples": args.max_samples,
                "samples_per_batch": args.samples_per_batch,
                "sampling_seed": args.sampling_seed,
                "profiles": [profile.value for profile in OBSERVER_SWEEP_PROFILES],
                "ranking_profile": QuantizationProfile.INTERNAL_FULL.value,
                "outputs_alias_profile": QuantizationProfile.FULL.value,
            },
            "results": results,
        },
    )


def _run_group_observer_sweep(args: argparse.Namespace) -> None:
    _validate_percentiles([args.global_percentile, *args.percentiles])
    float_model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    calibration, evaluation, data_metadata = _load_datasets(args)
    global_observer_kwargs = {
        "percentile": args.global_percentile,
        "max_samples": args.max_samples,
        "samples_per_batch": args.samples_per_batch,
        "seed": args.sampling_seed,
    }
    baseline_candidate = quantize_candidate(
        float_model,
        args.bits,
        calibration,
        activation_observer=PercentileObserver,
        activation_observer_kwargs=global_observer_kwargs,
    )
    baseline_boundaries = output_boundaries(baseline_candidate)
    all_groups = build_activation_sensitivity_groups(
        baseline_candidate,
        baseline_boundaries,
    )
    groups = select_activation_sensitivity_groups(all_groups, args.groups)
    override_paths_by_group = {
        group.name: activation_group_override_paths(baseline_candidate, group)
        for group in groups
    }
    baseline_outputs, baseline_site_count = evaluate_internal_full(
        float_model,
        baseline_candidate,
        evaluation,
        boundaries=baseline_boundaries,
        output_adapter=OUTPUT_ADAPTER,
    )
    policies = build_group_observer_policies(
        percentiles=args.percentiles,
        global_percentile=args.global_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        seed=args.sampling_seed,
        include_minmax=not args.skip_minmax,
    )
    del baseline_candidate

    total_evaluations = len(groups) * len(policies)
    evaluation_index = 0
    group_results: list[dict[str, object]] = []
    for group in groups:
        override_paths = override_paths_by_group[group.name]
        evaluations: list[EvaluatedGroupObserverPolicy] = []
        for policy in policies:
            evaluation_index += 1
            print(
                f"[{evaluation_index}/{total_evaluations}] "
                f"{group.name}: {policy.name}"
            )
            candidate = quantize_candidate(
                float_model,
                args.bits,
                calibration,
                activation_observer=PercentileObserver,
                activation_observer_kwargs=global_observer_kwargs,
                overrides=make_group_observer_overrides(
                    policy,
                    bit_width=args.bits,
                    override_paths=override_paths,
                ),
            )
            validate_group_observer_overrides(
                candidate,
                policy,
                override_paths,
            )
            outputs, enabled_site_count = evaluate_internal_full(
                float_model,
                candidate,
                evaluation,
                boundaries=output_boundaries(candidate),
                output_adapter=OUTPUT_ADAPTER,
            )
            evaluations.append(
                EvaluatedGroupObserverPolicy(
                    policy=policy,
                    outputs=outputs,
                    enabled_site_count=enabled_site_count,
                )
            )
            del candidate
        group_results.append(
            build_group_observer_sweep_result(
                group=group,
                override_paths=override_paths,
                global_percentile=args.global_percentile,
                baseline_outputs=baseline_outputs,
                baseline_site_count=baseline_site_count,
                evaluations=evaluations,
                score_output=args.score_output,
            )
        )

    print_group_observer_sweep(
        dtype_name=quantization_name(args.bits),
        global_percentile=args.global_percentile,
        baseline_outputs=baseline_outputs,
        baseline_site_count=baseline_site_count,
        group_results=group_results,
        score_output=args.score_output,
    )
    _write_json(
        args.report_json,
        {
            "analysis": "activation_group_observer_sweep",
            "metadata": {
                **data_metadata,
                "dtype": quantization_name(args.bits),
                "global_activation_observer": "PercentileObserver",
                "global_percentile": args.global_percentile,
                "candidate_percentiles": args.percentiles,
                "candidate_policies": [policy.to_dict() for policy in policies],
                "include_minmax": not args.skip_minmax,
                "max_samples": args.max_samples,
                "samples_per_batch": args.samples_per_batch,
                "sampling_seed": args.sampling_seed,
                "baseline_profile": QuantizationProfile.INTERNAL_FULL.value,
                "baseline_site_count": baseline_site_count,
                "score_output": args.score_output,
                "score_metric": "mae",
                "group_count": len(groups),
                "groups": [group.name for group in groups],
                "candidate_policy_count": len(policies),
                "candidate_evaluation_count": total_evaluations,
            },
            "baseline": baseline_outputs,
            "groups": group_results,
        },
    )


def _run_activation_sensitivity(args: argparse.Namespace) -> None:
    _validate_percentiles([args.percentile])
    if args.top_k < 0:
        raise ValueError("--top-k must be nonnegative.")
    if args.max_steps < 0:
        raise ValueError("--max-steps must be nonnegative.")
    if args.strategy == "cumulative" and args.groups is None:
        raise ValueError("--groups is required for cumulative sensitivity.")

    float_model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    calibration, evaluation, data_metadata = _load_datasets(args)
    candidate = quantize_candidate(
        float_model,
        args.bits,
        calibration,
        activation_observer=PercentileObserver,
        activation_observer_kwargs={
            "percentile": args.percentile,
            "max_samples": args.max_samples,
            "samples_per_batch": args.samples_per_batch,
            "seed": args.sampling_seed,
        },
    )
    boundaries = output_boundaries(candidate)
    all_groups = build_activation_sensitivity_groups(candidate, boundaries)
    groups = select_activation_sensitivity_groups(all_groups, args.groups)
    baseline_selector = boundaries.selector_for(QuantizationProfile.INTERNAL_FULL)
    all_sites = tuple(iter_quantization_sites(candidate))
    baseline_site_count = sum(baseline_selector(site) for site in all_sites)
    grouped_site_count = sum(group.site_count for group in all_groups)

    runner = QuantizationSensitivity(
        float_model,
        candidate,
        output_adapter=OUTPUT_ADAPTER,
    )
    quantization_groups = tuple(group.group for group in groups)
    common = {
        "mode": SensitivityMode.LEAVE_ONE_FLOAT,
        "score_output": args.score_output,
        "score_metric": "mae",
        "baseline_selector": baseline_selector,
    }

    report_body: dict[str, Any]
    strategy_metadata: dict[str, Any] = {}
    selected_group_count = 0
    if args.strategy == "independent":
        baseline, results = runner.run(
            evaluation,
            quantization_groups,
            **common,
        )
        print_activation_sensitivity(
            dtype_name=quantization_name(args.bits),
            percentile=args.percentile,
            baseline=baseline,
            results=results,
            top_k=args.top_k,
            baseline_site_count=baseline_site_count,
            score_output=args.score_output,
        )
        report_body = {
            "groups": build_activation_sensitivity_report(
                baseline=baseline,
                results=results,
                groups=groups,
            )
        }
    elif args.strategy == "cumulative":
        baseline, steps = runner.run_cumulative(
            evaluation,
            quantization_groups,
            **common,
        )
        print_activation_sensitivity_path(
            strategy=args.strategy,
            dtype_name=quantization_name(args.bits),
            percentile=args.percentile,
            baseline=baseline,
            results=steps,
            baseline_site_count=baseline_site_count,
            score_output=args.score_output,
        )
        selected_group_count = len(steps)
        report_body = {
            "steps": build_activation_sensitivity_path_report(
                baseline=baseline,
                results=steps,
                groups=groups,
            )
        }
    elif args.strategy == "ranked":
        if args.max_steps == 0:
            baseline, steps = runner.run_greedy(
                evaluation,
                quantization_groups,
                max_steps=0,
                **common,
            )
            initial_ranking = []
            ranked_groups = ()
        else:
            _, independent_results = runner.run(
                evaluation,
                quantization_groups,
                **common,
            )
            initial_ranking = [
                {
                    "rank": rank,
                    "group": result.group,
                    "score": result.score,
                    "sensitivity": result.sensitivity,
                }
                for rank, result in enumerate(independent_results, start=1)
            ]
            group_by_name = {group.name: group for group in groups}
            ranked_groups = tuple(
                group_by_name[result.group]
                for result in independent_results[: args.max_steps]
            )
            baseline, steps = runner.run_cumulative(
                evaluation,
                tuple(group.group for group in ranked_groups),
                **common,
            )
        print_activation_sensitivity_path(
            strategy=args.strategy,
            dtype_name=quantization_name(args.bits),
            percentile=args.percentile,
            baseline=baseline,
            results=steps,
            baseline_site_count=baseline_site_count,
            score_output=args.score_output,
        )
        selected_group_count = len(steps)
        strategy_metadata = {
            "max_steps": args.max_steps,
            "initial_ranking": initial_ranking,
        }
        report_body = {
            "steps": build_activation_sensitivity_path_report(
                baseline=baseline,
                results=steps,
                groups=ranked_groups,
            )
        }
    else:
        baseline, steps = runner.run_greedy(
            evaluation,
            quantization_groups,
            max_steps=args.max_steps,
            minimum_improvement=args.minimum_improvement,
            **common,
        )
        print_activation_sensitivity_path(
            strategy=args.strategy,
            dtype_name=quantization_name(args.bits),
            percentile=args.percentile,
            baseline=baseline,
            results=steps,
            baseline_site_count=baseline_site_count,
            score_output=args.score_output,
        )
        selected_group_count = len(steps)
        strategy_metadata = {
            "max_steps": args.max_steps,
            "minimum_improvement": args.minimum_improvement,
        }
        report_body = {
            "steps": build_activation_sensitivity_path_report(
                baseline=baseline,
                results=steps,
                groups=groups,
            )
        }

    _write_json(
        args.report_json,
        {
            "analysis": "activation_block_sensitivity",
            "metadata": {
                **data_metadata,
                "dtype": quantization_name(args.bits),
                "activation_observer": "PercentileObserver",
                "activation_percentile": args.percentile,
                "max_samples": args.max_samples,
                "samples_per_batch": args.samples_per_batch,
                "sampling_seed": args.sampling_seed,
                "baseline_profile": QuantizationProfile.INTERNAL_FULL.value,
                "baseline_site_count": baseline_site_count,
                "mode": SensitivityMode.LEAVE_ONE_FLOAT.value,
                "strategy": args.strategy,
                "score_output": args.score_output,
                "score_metric": "mae",
                "group_count": len(all_groups),
                "candidate_group_count": len(groups),
                "candidate_groups": [group.name for group in groups],
                "selected_group_count": selected_group_count,
                "grouped_activation_site_count": grouped_site_count,
                **strategy_metadata,
            },
            "baseline": baseline,
            **report_body,
        },
    )


def _load_datasets(
    args: argparse.Namespace,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if args.calibration_dir is None:
        calibration = make_synthetic_inputs(
            args.synthetic_calibration_samples,
            seed=20260806,
        )
        calibration_paths: set[Path] = set()
        print("Using synthetic calibration inputs for a smoke test only.")
    else:
        calibration = load_npy_inputs(
            args.calibration_dir,
            args.calibration_limit,
            offset=args.calibration_offset,
        )
        calibration_paths = _selected_paths(
            args.calibration_dir,
            args.calibration_offset,
            args.calibration_limit,
        )

    if args.evaluation_dir is None:
        evaluation = make_synthetic_inputs(
            args.synthetic_evaluation_samples,
            seed=20260807,
        )
        evaluation_paths: set[Path] = set()
        print("Using synthetic evaluation inputs for a smoke test only.")
    else:
        evaluation = load_npy_inputs(
            args.evaluation_dir,
            args.evaluation_limit,
            offset=args.evaluation_offset,
        )
        evaluation_paths = _selected_paths(
            args.evaluation_dir,
            args.evaluation_offset,
            args.evaluation_limit,
        )

    overlap = calibration_paths & evaluation_paths
    if overlap:
        message = (
            f"Calibration and evaluation selections overlap by {len(overlap)} files. "
            "This is acceptable for numerical floor analysis, but not for selecting "
            "or reporting a final quantization policy."
        )
        if args.require_disjoint:
            raise ValueError(message)
        print(f"WARNING: {message}")

    return (
        calibration,
        evaluation,
        {
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "synthetic_calibration": args.calibration_dir is None,
            "synthetic_evaluation": args.evaluation_dir is None,
            "overlapping_files": len(overlap),
        },
    )


def _selected_paths(
    directory: Path,
    offset: int,
    limit: int | None,
) -> set[Path]:
    paths = list_npy_inputs(directory)
    selected = paths[offset : None if limit is None else offset + limit]
    return {path.resolve() for path in selected}


def _validate_percentiles(percentiles: list[float]) -> None:
    if not percentiles or any(not 0.0 < value <= 100.0 for value in percentiles):
        raise ValueError("Percentiles must be in the interval (0, 100].")


def _print_ablation(report: dict[str, Any]) -> None:
    print("\nQuantization A/B/C/D/E ablation")
    print(
        f"{'profile':18s} {'REG_MAE':>13s} {'REG_COS':>13s} "
        f"{'CLS_MAE':>13s} {'CLS_COS':>13s} {'SITES':>7s}"
    )
    parity = report["float_parity"]
    _print_output_row("float-parity", parity, 0)
    for profile in (
        QuantizationProfile.OUTPUT_ONLY,
        QuantizationProfile.WEIGHT_ONLY,
        QuantizationProfile.ACTIVATION_ONLY,
        QuantizationProfile.FULL,
        QuantizationProfile.INTERNAL_FULL,
    ):
        result = report["profiles"][profile.value]
        _print_output_row(
            f"{profile.value}:{result['label']}",
            result["outputs"],
            result["enabled_site_count"],
        )


def _print_output_row(label: str, outputs: dict[str, Any], sites: int) -> None:
    regressors = outputs["regressors"]
    classifiers = outputs["classifiers"]
    print(
        f"{label:18s} "
        f"{float(regressors['mae']):13.6e} "
        f"{float(regressors['cosine_similarity']):13.9f} "
        f"{float(classifiers['mae']):13.6e} "
        f"{float(classifiers['cosine_similarity']):13.9f} "
        f"{sites:7d}"
    )


def _print_output_clipping(calibration_data, evaluated, dtype_name: str) -> None:
    print(f"\n{dtype_name.upper()} final-output clipping analysis")
    print("All internal model computation remains floating point.")
    data_by_name = {data.name: data for data in calibration_data}
    for name, candidates in evaluated.items():
        data = data_by_name[name]
        print(
            f"\n{name}: sampled {data.sampled_value_count:,} / "
            f"{data.total_value_count:,} calibration values; raw range "
            f"[{data.observed_minimum:.6e}, {data.observed_maximum:.6e}]"
        )
        print(
            f"{'candidate':18s} {'CLIP_MIN':>13s} {'CLIP_MAX':>13s} "
            f"{'SCALE':>11s} {'CAL_MAE':>11s} {'EVAL_MAE':>11s} {'SAT(%)':>10s}"
        )
        ranked = sorted(
            candidates,
            key=lambda item: float(item.evaluation_error["mae"]),
        )
        for index, item in enumerate(ranked):
            marker = "*" if index == 0 else " "
            print(
                f"{marker}{item.candidate.name:17s} "
                f"{item.candidate.minimum:13.4e} "
                f"{item.candidate.maximum:13.4e} "
                f"{float(item.quantizer['scale']):11.4e} "
                f"{float(item.candidate.calibration_error['mae']):11.4e} "
                f"{float(item.evaluation_error['mae']):11.4e} "
                f"{100.0 * float(item.quantizer['saturation_ratio']):10.5f}"
            )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

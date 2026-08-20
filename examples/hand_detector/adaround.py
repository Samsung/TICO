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

"""Run validation-aware Conv2d AdaRound after activation reconstruction."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import torch

from examples.hand_detector._support.adaround import (
    apply_activation_reconstruction_report,
    run_hand_detector_adaround,
)
from examples.hand_detector._support.analysis import output_boundaries, OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.multistart_reconstruction import (
    split_reconstruction_samples_three_way,
)
from examples.hand_detector._support.optimized_export import (
    build_export_manifest,
    default_manifest_path,
    evaluate_full_quantized_model,
    export_full_integer_circle,
    write_export_manifest,
)
from examples.hand_detector._support.quantization import (
    quantization_name,
    quantize_candidate,
)
from examples.hand_detector._support.reconstruction import build_reconstruction_windows
from examples.hand_detector.hand_detector import load_nhwc_hand_detector
from tico.quantization.algorithm.adaround import AdaRoundConfig
from tico.quantization.algorithm.block_reconstruction import (
    ReconstructionLoss,
    ValidationObjective,
)
from tico.quantization.analysis import make_output_adapter
from tico.quantization.wrapq.observers.percentile import PercentileObserver


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "adaround",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        command,
        help=(
            "Optimize hard Conv2d weight rounding with fixed activation and "
            "weight qparams."
        ),
    )
    _add_arguments(parser)
    return parser


def _add_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--synthetic-calibration-samples", type=int, default=200)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--synthetic-evaluation-samples", type=int, default=16)
    parser.add_argument("--require-disjoint", action="store_true")
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--max-samples", type=int, default=524_288)
    parser.add_argument("--samples-per-batch", type=int, default=4_096)
    parser.add_argument("--sampling-seed", type=int, default=20260803)
    parser.add_argument(
        "--activation-report",
        type=Path,
        help=(
            "Optional accepted activation-reconstruction JSON replayed before "
            "AdaRound. Without it, AdaRound starts from global percentile PTQ."
        ),
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--groups",
        nargs="+",
        help="Optimize Conv2d weights in each semantic group independently.",
    )
    target.add_argument(
        "--windows",
        nargs="+",
        help="Optimize Conv2d weights in contiguous joint windows joined by '+'.",
    )

    parser.add_argument("--selection-count", type=int, default=40)
    parser.add_argument("--acceptance-count", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=20260803)
    parser.add_argument(
        "--selection-score-output",
        choices=OUTPUT_NAMES,
        default="regressors",
    )
    parser.add_argument(
        "--selection-score-metric",
        choices=("mae", "mse", "rmse", "relative_mae"),
        default="mae",
    )
    parser.add_argument(
        "--auxiliary-tolerance",
        type=float,
        default=0.0,
        help="Maximum accepted non-primary output regression.",
    )
    parser.add_argument("--minimum-selection-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-acceptance-improvement", type=float, default=1e-3)

    parser.add_argument(
        "--reconstruction-loss",
        choices=tuple(value.value for value in ReconstructionLoss),
        default=ReconstructionLoss.NORMALIZED_L1.value,
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--alpha-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rounding-loss-weight", type=float, default=1e-2)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--beta-start", type=float, default=20.0)
    parser.add_argument("--beta-end", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=-0.1)
    parser.add_argument("--zeta", type=float, default=1.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--adaround-seed", type=int, default=20260820)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-circle", type=Path)
    parser.add_argument(
        "--export-manifest-json",
        type=Path,
        help=(
            "Optional optimized-export manifest. Defaults to a JSON sidecar "
            "next to --export-circle."
        ),
    )
    parser.add_argument(
        "--skip-export-verification",
        action="store_true",
        help=(
            "Skip static Circle quantization and layout verification. The "
            "full fake-quant deployment profile is still used for export."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "adaround.json",
    )


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    _validate_disjoint(args)
    device = torch.device(args.device)
    float_model = load_nhwc_hand_detector(args.weights, args.spec).to(device).eval()
    if args.calibration_dir is None:
        calibration = make_synthetic_inputs(
            args.synthetic_calibration_samples,
            seed=args.sampling_seed,
        )
    else:
        calibration = load_npy_inputs(
            args.calibration_dir,
            args.calibration_limit,
            offset=args.calibration_offset,
        )
    if args.evaluation_dir is None:
        evaluation = make_synthetic_inputs(
            args.synthetic_evaluation_samples,
            seed=args.sampling_seed + 1,
        )
    else:
        evaluation = load_npy_inputs(
            args.evaluation_dir,
            args.evaluation_limit,
            offset=args.evaluation_offset,
        )
    calibration = tuple(sample.to(device=device) for sample in calibration)
    evaluation = tuple(sample.to(device=device) for sample in evaluation)
    data_split = split_reconstruction_samples_three_way(
        calibration,
        args.selection_count,
        args.acceptance_count,
        seed=args.selection_seed,
    )
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
    activation_replay = None
    if args.activation_report is not None:
        activation_replay = apply_activation_reconstruction_report(
            candidate,
            args.activation_report,
            expected_percentile=args.percentile,
            expected_max_samples=args.max_samples,
        )

    boundaries = output_boundaries(candidate)
    windows = build_reconstruction_windows(
        candidate,
        boundaries,
        groups=args.groups,
        windows=args.windows,
    )
    config = AdaRoundConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_interval=args.evaluation_interval,
        alpha_learning_rate=args.alpha_learning_rate,
        reconstruction_loss=ReconstructionLoss(args.reconstruction_loss),
        rounding_loss_weight=args.rounding_loss_weight,
        warmup_fraction=args.warmup_fraction,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        gamma=args.gamma,
        zeta=args.zeta,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.adaround_seed,
    )
    auxiliary_output = _other_output(args.selection_score_output)
    selection_objective = ValidationObjective(
        primary_output=args.selection_score_output,
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_selection_improvement,
        output_tolerances={auxiliary_output: args.auxiliary_tolerance},
    )
    acceptance_objective = ValidationObjective(
        primary_output=args.selection_score_output,
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_acceptance_improvement,
        output_tolerances={auxiliary_output: args.auxiliary_tolerance},
    )
    report = run_hand_detector_adaround(
        float_model,
        candidate,
        data_split=data_split,
        evaluation_samples=evaluation,
        windows=windows,
        config=config,
        selection_objective=selection_objective,
        acceptance_objective=acceptance_objective,
        output_adapter=OUTPUT_ADAPTER,
        device=args.device,
    )
    payload = {
        "analysis": "validation_aware_adaround",
        "metadata": {
            "dtype": quantization_name(args.bits),
            "percentile": args.percentile,
            "max_samples": args.max_samples,
            "samples_per_batch": args.samples_per_batch,
            "sampling_seed": args.sampling_seed,
            "selection_seed": args.selection_seed,
            "selection_count": args.selection_count,
            "acceptance_count": args.acceptance_count,
            "selection_score_output": args.selection_score_output,
            "selection_score_metric": args.selection_score_metric,
            "auxiliary_output": auxiliary_output,
            "auxiliary_tolerance": args.auxiliary_tolerance,
            "minimum_selection_improvement": (args.minimum_selection_improvement),
            "minimum_acceptance_improvement": (args.minimum_acceptance_improvement),
            "reconstruction_loss": args.reconstruction_loss,
            "steps": args.steps,
            "alpha_learning_rate": args.alpha_learning_rate,
            "rounding_loss_weight": args.rounding_loss_weight,
            "warmup_fraction": args.warmup_fraction,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "gamma": args.gamma,
            "zeta": args.zeta,
            "adaround_seed": args.adaround_seed,
            "activation_replay": activation_replay,
            "windows": [window.to_dict() for window in windows],
        },
        **report,
    }
    _print_report(payload)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.report_json}")

    if args.export_circle is not None:
        final_full = evaluate_full_quantized_model(
            float_model,
            candidate,
            evaluation,
            output_adapter=OUTPUT_ADAPTER,
        )
        export_summary = export_full_integer_circle(
            candidate,
            args.export_circle,
            bit_width=args.bits,
            verify=not args.skip_export_verification,
        )
        payload["final_full_evaluation"] = final_full
        payload["export"] = export_summary
        args.report_json.write_text(
            json.dumps(payload, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        manifest = build_export_manifest(
            bit_width=args.bits,
            circle_summary=export_summary,
            optimization_report_path=args.report_json,
            activation_report_path=args.activation_report,
            optimization_metadata=payload["metadata"],
            baseline_internal_full=payload["baseline_evaluation"],
            final_internal_full=payload["final_evaluation"],
            final_full=final_full,
            steps=payload["steps"],
        )
        manifest_path = (
            args.export_manifest_json
            if args.export_manifest_json is not None
            else default_manifest_path(args.export_circle)
        )
        written_manifest = write_export_manifest(manifest_path, manifest)
        print(
            "Full deployment profile: "
            f"REG_MAE={float(final_full['regressors']['mae']):.6e}, "
            f"CLS_MAE={float(final_full['classifiers']['mae']):.6e}"
        )
        print(f"Wrote {export_summary['path']}")
        print(f"Wrote {written_manifest}")


def _print_report(report: dict[str, object]) -> None:
    baseline = report["baseline_evaluation"]
    assert isinstance(baseline, dict)
    print("\nValidation-aware AdaRound")
    print(
        "External baseline E:internal-full: "
        f"REG_MAE={float(baseline['regressors']['mae']):.6e}, "
        f"CLS_MAE={float(baseline['classifiers']['mae']):.6e}"
    )
    print(
        f"{'step':>4s} {'window':34s} {'result':>8s} {'best':>6s} "
        f"{'REG_MAE':>13s} {'GAIN_REG':>13s} {'CLS_MAE':>13s} "
        f"{'GAIN_CLS':>13s} {'WEIGHTS':>7s}"
    )
    baseline_reg = float(baseline["regressors"]["mae"])
    baseline_cls = float(baseline["classifiers"]["mae"])
    for step in report["steps"]:
        result = step["adaround"]
        outputs = step["evaluation_after"]
        outcome = "ACCEPT" if result["accepted"] else "ROLLBACK"
        print(
            f"{int(step['step']):4d} "
            f"{str(step['window']['name'])[:34]:34s} "
            f"{outcome:>8s} "
            f"{int(result['best_step']):6d} "
            f"{float(outputs['regressors']['mae']):13.6e} "
            f"{baseline_reg - float(outputs['regressors']['mae']):13.6e} "
            f"{float(outputs['classifiers']['mae']):13.6e} "
            f"{baseline_cls - float(outputs['classifiers']['mae']):13.6e} "
            f"{int(result['weight_group_count']):7d}"
        )
        print(f"     acceptance: {result['acceptance_reason']}")


def _other_output(primary: str) -> str:
    return next(output for output in OUTPUT_NAMES if output != primary)


def _validate_args(args: argparse.Namespace) -> None:
    if args.selection_count <= 0 or args.acceptance_count <= 0:
        raise ValueError("selection-count and acceptance-count must be positive.")
    if args.auxiliary_tolerance < 0.0:
        raise ValueError("auxiliary-tolerance must be nonnegative.")
    if args.minimum_selection_improvement < 0.0:
        raise ValueError("minimum-selection-improvement must be nonnegative.")
    if args.minimum_acceptance_improvement < 0.0:
        raise ValueError("minimum-acceptance-improvement must be nonnegative.")
    if args.export_manifest_json is not None and args.export_circle is None:
        raise ValueError("export-manifest-json requires --export-circle.")


def _validate_disjoint(args: argparse.Namespace) -> None:
    if not args.require_disjoint:
        return
    if args.calibration_dir is None or args.evaluation_dir is None:
        return
    calibration_paths = list_npy_inputs(args.calibration_dir)[
        args.calibration_offset : args.calibration_offset + args.calibration_limit
    ]
    evaluation_paths = list_npy_inputs(args.evaluation_dir)[
        args.evaluation_offset : args.evaluation_offset + args.evaluation_limit
    ]
    overlap = set(path.resolve() for path in calibration_paths).intersection(
        path.resolve() for path in evaluation_paths
    )
    if overlap:
        raise ValueError(
            f"Calibration and evaluation selections overlap by {len(overlap)} files."
        )


if __name__ == "__main__":
    main()

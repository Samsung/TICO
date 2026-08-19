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

"""Run validation-aware block or contiguous joint-window reconstruction."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import torch

from examples.hand_detector._support.analysis import output_boundaries, OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.quantization import (
    export_quantized_circle,
    quantization_name,
    quantize_candidate,
)
from examples.hand_detector._support.reconstruction import (
    build_reconstruction_windows,
    reconstruct_hand_detector_windows,
    split_reconstruction_samples,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector
from tico.quantization.algorithm.block_reconstruction import (
    BlockReconstructionConfig,
    ReconstructionLoss,
    ValidationObjective,
)
from tico.quantization.analysis import make_output_adapter
from tico.quantization.wrapq.observers.percentile import PercentileObserver


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone block-reconstruction parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "block-reconstruction",
) -> argparse.ArgumentParser:
    """Register this CLI under the shared hand-detector analyzer."""
    parser = subparsers.add_parser(
        command,
        help=(
            "Run held-out-selected single-block or joint-window activation "
            "qparam reconstruction."
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
    parser.add_argument("--synthetic-calibration-samples", type=int, default=64)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--synthetic-evaluation-samples", type=int, default=8)
    parser.add_argument("--require-disjoint", action="store_true")
    parser.add_argument("--bits", type=int, default=8, choices=[8, 16])
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--max-samples", type=int, default=524_288)
    parser.add_argument("--samples-per-batch", type=int, default=4_096)
    parser.add_argument("--sampling-seed", type=int, default=20260803)

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--groups",
        nargs="+",
        help="Reconstruct each named activation group independently.",
    )
    target.add_argument(
        "--windows",
        nargs="+",
        help=(
            "Reconstruct contiguous joint windows joined by '+', for example "
            "stem+feature_block_00."
        ),
    )

    parser.add_argument(
        "--selection-count",
        type=int,
        default=40,
        help=(
            "Calibration samples held out from gradient updates and used for "
            "checkpoint selection and whole-window rollback."
        ),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260803,
        help="Seed for the deterministic reconstruction/selection split.",
    )
    parser.add_argument(
        "--selection-score-output",
        choices=OUTPUT_NAMES,
        default="regressors",
    )
    parser.add_argument(
        "--selection-score-metric",
        choices=("mae", "mse", "rmse", "relative_mae"),
        default="mae",
        help="Lower-is-better held-out metric used for checkpoint selection.",
    )
    parser.add_argument(
        "--auxiliary-tolerance",
        "--classifier-tolerance",
        dest="auxiliary_tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum held-out MAE regression accepted on the non-primary "
            "detector output."
        ),
    )
    parser.add_argument(
        "--minimum-selection-improvement",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--reconstruction-loss",
        choices=tuple(value.value for value in ReconstructionLoss),
        default=ReconstructionLoss.NORMALIZED_MSE.value,
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--scale-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--zero-point-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--freeze-zero-point", action="store_true")
    parser.add_argument("--reconstruction-seed", type=int, default=20260816)
    parser.add_argument(
        "--qdrop-probability",
        type=float,
        default=0.0,
        help=(
            "Probability that each activation element bypasses quantization "
            "during reconstruction training. Zero preserves the existing "
            "fully quantized training path; the QDrop paper uses 0.5."
        ),
    )
    parser.add_argument("--qdrop-seed", type=int, default=20260817)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-circle", type=Path)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=(DIRECTORY / "reports" / "block_reconstruction_validated.json"),
    )


def main() -> None:
    """Run the standalone command."""
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    """Execute reconstruction from standalone or shared analyzer arguments."""
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
    train, selection = split_reconstruction_samples(
        calibration,
        args.selection_count,
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
    boundaries = output_boundaries(candidate)
    windows = build_reconstruction_windows(
        candidate,
        boundaries,
        groups=args.groups,
        windows=args.windows,
    )
    config = BlockReconstructionConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_interval=args.evaluation_interval,
        scale_learning_rate=args.scale_learning_rate,
        zero_point_learning_rate=args.zero_point_learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        optimize_scale=True,
        optimize_zero_point=not args.freeze_zero_point,
        loss=ReconstructionLoss(args.reconstruction_loss),
        seed=args.reconstruction_seed,
        qdrop_probability=args.qdrop_probability,
        qdrop_seed=args.qdrop_seed,
    )
    auxiliary_output = _other_output(args.selection_score_output)
    objective = ValidationObjective(
        primary_output=args.selection_score_output,
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_selection_improvement,
        output_tolerances={
            auxiliary_output: args.auxiliary_tolerance,
        },
    )
    report = reconstruct_hand_detector_windows(
        float_model,
        candidate,
        train_samples=train,
        selection_samples=selection,
        evaluation_samples=evaluation,
        windows=windows,
        config=config,
        objective=objective,
        output_adapter=OUTPUT_ADAPTER,
        device=args.device,
    )
    payload = {
        "analysis": "validation_aware_block_reconstruction",
        "metadata": {
            "dtype": quantization_name(args.bits),
            "percentile": args.percentile,
            "max_samples": args.max_samples,
            "samples_per_batch": args.samples_per_batch,
            "sampling_seed": args.sampling_seed,
            "reconstruction_seed": args.reconstruction_seed,
            "qdrop_probability": args.qdrop_probability,
            "qdrop_seed": args.qdrop_seed,
            "qdrop_granularity": "element",
            "selection_count": args.selection_count,
            "selection_seed": args.selection_seed,
            "selection_score_output": args.selection_score_output,
            "selection_score_metric": args.selection_score_metric,
            "auxiliary_output": auxiliary_output,
            "auxiliary_tolerance": args.auxiliary_tolerance,
            "minimum_selection_improvement": (args.minimum_selection_improvement),
            "reconstruction_loss": args.reconstruction_loss,
            "windows": [window.to_dict() for window in windows],
        },
        **report,
    }
    _print_report(payload)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.report_json}")
    if args.export_circle is not None:
        output = export_quantized_circle(candidate, args.export_circle)
        print(f"Wrote {output}")


def _print_report(report: dict[str, object]) -> None:
    baseline = report["baseline_evaluation"]
    assert isinstance(baseline, dict)
    print("\nValidation-aware block reconstruction")
    print(
        "External baseline E:internal-full: "
        f"REG_MAE={float(baseline['regressors']['mae']):.6e}, "
        f"CLS_MAE={float(baseline['classifiers']['mae']):.6e}"
    )
    print(
        f"{'step':>4s} {'window':38s} {'result':>8s} {'best':>6s} "
        f"{'REG_MAE':>13s} {'GAIN_REG':>13s} {'CLS_MAE':>13s} "
        f"{'GAIN_CLS':>13s}"
    )
    baseline_reg = float(baseline["regressors"]["mae"])
    baseline_cls = float(baseline["classifiers"]["mae"])
    for value in report["steps"]:
        reconstruction = value["reconstruction"]
        outputs = value["evaluation_after"]
        result = "ACCEPT" if reconstruction["accepted"] else "ROLLBACK"
        print(
            f"{int(value['step']):4d} "
            f"{str(value['window']['name'])[:38]:38s} "
            f"{result:>8s} "
            f"{int(reconstruction['best_step']):6d} "
            f"{float(outputs['regressors']['mae']):13.6e} "
            f"{baseline_reg - float(outputs['regressors']['mae']):13.6e} "
            f"{float(outputs['classifiers']['mae']):13.6e} "
            f"{baseline_cls - float(outputs['classifiers']['mae']):13.6e}"
        )
        print("     selection: " f"{reconstruction['acceptance_reason']}")


def _other_output(output_name: str) -> str:
    others = tuple(name for name in OUTPUT_NAMES if name != output_name)
    if len(others) != 1:
        raise ValueError(
            "Block reconstruction currently expects exactly two detector outputs."
        )
    return others[0]


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (0, 100].")
    if args.max_samples <= 0 or args.samples_per_batch <= 0:
        raise ValueError("Observer sample limits must be positive.")
    if args.auxiliary_tolerance < 0.0:
        raise ValueError("--auxiliary-tolerance must be nonnegative.")
    if args.minimum_selection_improvement < 0.0:
        raise ValueError("--minimum-selection-improvement must be nonnegative.")
    if not 0.0 <= args.qdrop_probability <= 1.0:
        raise ValueError("--qdrop-probability must be in [0, 1].")


def _validate_disjoint(args: argparse.Namespace) -> None:
    if not args.require_disjoint:
        return
    if args.calibration_dir is None or args.evaluation_dir is None:
        return
    calibration_end = (
        None
        if args.calibration_limit is None
        else args.calibration_offset + args.calibration_limit
    )
    evaluation_end = (
        None
        if args.evaluation_limit is None
        else args.evaluation_offset + args.evaluation_limit
    )
    calibration_paths = list_npy_inputs(args.calibration_dir)[
        args.calibration_offset : calibration_end
    ]
    evaluation_paths = list_npy_inputs(args.evaluation_dir)[
        args.evaluation_offset : evaluation_end
    ]
    overlap = set(path.resolve() for path in calibration_paths).intersection(
        path.resolve() for path in evaluation_paths
    )
    if overlap:
        raise ValueError(
            "Calibration and evaluation selections overlap by " f"{len(overlap)} files."
        )


if __name__ == "__main__":
    main()

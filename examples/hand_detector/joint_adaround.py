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

"""Run joint DW/PW learnable-scale AdaRound under the P2 W8/A16 profile."""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import torch

from examples.hand_detector._support.analysis import output_boundaries, OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.joint_adaround import (
    ALL_CONV_JOINT_GROUPS,
    apply_joint_adaround_checkpoint,
    PRIORITY_JOINT_GROUPS,
    run_hand_detector_joint_adaround,
    save_joint_adaround_checkpoint,
)
from examples.hand_detector._support.multistart_reconstruction import (
    split_reconstruction_samples_three_way,
)
from examples.hand_detector._support.reconstruction import build_reconstruction_windows
from examples.hand_detector._support.weight_precision_sensitivity import (
    build_w8a16_candidate,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector
from tico.quantization.algorithm.adaround import (
    JointAdaRoundConfig,
    JointAdaRoundObjective,
)
from tico.quantization.algorithm.block_reconstruction import ReconstructionLoss
from tico.quantization.analysis import make_output_adapter


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "joint-dwpw-adaround",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        command,
        help=(
            "Jointly optimize depthwise and regular Conv W8 scales and "
            "AdaRound decisions under A16."
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
    parser.add_argument("--synthetic-evaluation-samples", type=int, default=79)
    parser.add_argument("--require-disjoint", action="store_true")
    parser.add_argument("--uint8-percentile", type=float, default=99.99)
    parser.add_argument(
        "--int16-observer",
        choices=("minmax", "percentile"),
        default="minmax",
    )
    parser.add_argument("--int16-percentile", type=float, default=99.99)
    parser.add_argument("--max-samples", type=int, default=524_288)
    parser.add_argument("--samples-per-batch", type=int, default=4_096)
    parser.add_argument("--sampling-seed", type=int, default=20260803)

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--groups",
        nargs="+",
        help="Optimize one semantic DW/PW block per listed group.",
    )
    target.add_argument(
        "--windows",
        nargs="+",
        help="Optimize contiguous joint windows written as group_a+group_b.",
    )
    target.add_argument(
        "--priority-preset",
        action="store_true",
        help="Use the conditional-sensitivity priority Conv group sequence.",
    )
    target.add_argument(
        "--all-conv-preset",
        action="store_true",
        help=(
            "Optimize stem, all 30 feature blocks, and both regressor heads "
            "in execution order."
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="Restore a prior joint-AdaRound checkpoint before optimization.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        help="Save finalized hard weights and exact affine parameter qparams.",
    )
    parser.add_argument(
        "--allow-reoptimize",
        action="store_true",
        help="Allow selected windows already accepted in a loaded checkpoint.",
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
    parser.add_argument("--classifier-limit", type=float, default=0.1)
    parser.add_argument(
        "--relative-classifier-tolerance",
        type=float,
        help="Optional classifier regression limit relative to each entry state.",
    )
    parser.add_argument("--minimum-selection-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-acceptance-improvement", type=float, default=1e-3)

    parser.add_argument(
        "--reconstruction-loss",
        choices=tuple(value.value for value in ReconstructionLoss),
        default=ReconstructionLoss.NORMALIZED_L1.value,
    )
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--alpha-learning-rate", type=float, default=1e-3)
    parser.add_argument("--scale-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rounding-loss-weight", type=float, default=1e-2)
    parser.add_argument("--scale-loss-weight", type=float, default=1e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--beta-start", type=float, default=20.0)
    parser.add_argument("--beta-end", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=-0.1)
    parser.add_argument("--zeta", type=float, default=1.1)
    parser.add_argument("--max-scale-ratio", type=float, default=1.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--optimization-seed", type=int, default=20260830)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "joint_dwpw_adaround.json",
    )


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    _validate_disjoint(args)
    device = torch.device(args.device)
    float_model = load_nhwc_hand_detector(args.weights, args.spec).to(device).eval()
    calibration = _load_samples(
        args.calibration_dir,
        args.calibration_limit,
        args.calibration_offset,
        args.synthetic_calibration_samples,
        args.sampling_seed,
    )
    evaluation = _load_samples(
        args.evaluation_dir,
        args.evaluation_limit,
        args.evaluation_offset,
        args.synthetic_evaluation_samples,
        args.sampling_seed + 1,
    )
    calibration = tuple(sample.to(device=device) for sample in calibration)
    evaluation = tuple(sample.to(device=device) for sample in evaluation)
    data_split = split_reconstruction_samples_three_way(
        calibration,
        args.selection_count,
        args.acceptance_count,
        seed=args.selection_seed,
    )
    candidate, p2_metadata = build_w8a16_candidate(
        float_model,
        calibration,
        uint8_percentile=args.uint8_percentile,
        int16_observer=args.int16_observer,
        int16_percentile=args.int16_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        sampling_seed=args.sampling_seed,
    )
    checkpoint_replay = None
    if args.load_checkpoint is not None:
        checkpoint_replay = apply_joint_adaround_checkpoint(
            candidate,
            args.load_checkpoint,
        )
    boundaries = output_boundaries(candidate)
    if args.priority_preset:
        groups = list(PRIORITY_JOINT_GROUPS)
    elif args.all_conv_preset:
        groups = list(ALL_CONV_JOINT_GROUPS)
    else:
        groups = args.groups
    windows = build_reconstruction_windows(
        candidate,
        boundaries,
        groups=groups,
        windows=args.windows,
    )
    if checkpoint_replay is not None and not args.allow_reoptimize:
        previous = checkpoint_replay.get("metadata", {}).get(
            "accepted_windows",
            [],
        )
        previous_names = {str(name) for name in previous}
        overlap = tuple(
            window.name for window in windows if window.name in previous_names
        )
        if overlap:
            raise ValueError(
                "Selected windows were already accepted in the loaded "
                f"checkpoint: {overlap}. Pass --allow-reoptimize only for an "
                "intentional second optimization pass."
            )
    config = JointAdaRoundConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        evaluation_interval=args.evaluation_interval,
        alpha_learning_rate=args.alpha_learning_rate,
        scale_learning_rate=args.scale_learning_rate,
        reconstruction_loss=ReconstructionLoss(args.reconstruction_loss),
        rounding_loss_weight=args.rounding_loss_weight,
        scale_loss_weight=args.scale_loss_weight,
        warmup_fraction=args.warmup_fraction,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        gamma=args.gamma,
        zeta=args.zeta,
        max_scale_ratio=args.max_scale_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.optimization_seed,
    )
    relative = (
        {"classifiers": args.relative_classifier_tolerance}
        if args.relative_classifier_tolerance is not None
        else {}
    )
    absolute = {"classifiers": args.classifier_limit}
    selection_objective = JointAdaRoundObjective(
        primary_output=args.selection_score_output,
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_selection_improvement,
        absolute_output_limits=absolute,
        relative_output_tolerances=relative,
    )
    acceptance_objective = JointAdaRoundObjective(
        primary_output=args.selection_score_output,
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_acceptance_improvement,
        absolute_output_limits=absolute,
        relative_output_tolerances=relative,
    )
    report = run_hand_detector_joint_adaround(
        float_model,
        candidate,
        data_split=data_split,
        evaluation_samples=evaluation,
        windows=windows,
        config=config,
        selection_objective=selection_objective,
        acceptance_objective=acceptance_objective,
        output_adapter=OUTPUT_ADAPTER,
        device=device,
    )
    payload = {
        "analysis": "joint_dwpw_learnable_scale_adaround",
        "metadata": {
            **p2_metadata,
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "selection_count": args.selection_count,
            "acceptance_count": args.acceptance_count,
            "selection_seed": args.selection_seed,
            "selection_score_output": args.selection_score_output,
            "selection_score_metric": args.selection_score_metric,
            "classifier_limit": args.classifier_limit,
            "relative_classifier_tolerance": (args.relative_classifier_tolerance),
            "minimum_selection_improvement": (args.minimum_selection_improvement),
            "minimum_acceptance_improvement": (args.minimum_acceptance_improvement),
            "reconstruction_loss": args.reconstruction_loss,
            "steps": args.steps,
            "alpha_learning_rate": args.alpha_learning_rate,
            "scale_learning_rate": args.scale_learning_rate,
            "rounding_loss_weight": args.rounding_loss_weight,
            "scale_loss_weight": args.scale_loss_weight,
            "warmup_fraction": args.warmup_fraction,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "gamma": args.gamma,
            "zeta": args.zeta,
            "max_scale_ratio": args.max_scale_ratio,
            "optimization_seed": args.optimization_seed,
            "device": str(device),
            "priority_preset": bool(args.priority_preset),
            "all_conv_preset": bool(args.all_conv_preset),
            "checkpoint_replay": checkpoint_replay,
            "windows": [window.to_dict() for window in windows],
        },
        **report,
    }
    checkpoint = None
    if args.save_checkpoint is not None:
        checkpoint_path = save_joint_adaround_checkpoint(
            candidate,
            args.save_checkpoint,
            metadata={
                "source_report": str(args.report_json),
                "profile": payload["profile"],
                "final_evaluation": payload["final_evaluation"],
                "windows": payload["metadata"]["windows"],
                "accepted_windows": [
                    step["window"]["name"]
                    for step in payload["steps"]
                    if step["joint_adaround"]["accepted"]
                ],
            },
        )
        checkpoint = {"path": checkpoint_path}
        payload["checkpoint"] = checkpoint
    _print_report(payload)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nWrote {args.report_json}")
    if checkpoint is not None:
        print(f"Wrote {checkpoint['path']}")


def _print_report(report: dict[str, object]) -> None:
    baseline = report["baseline_evaluation"]
    assert isinstance(baseline, dict)
    print("\nJoint DW/PW learnable-scale AdaRound")
    print(
        "External P2 baseline: "
        f"REG_MAE={float(baseline['regressors']['mae']):.6e}, "
        f"CLS_MAE={float(baseline['classifiers']['mae']):.6e}"
    )
    print(
        f"{'step':>4s} {'window':34s} {'result':>8s} {'best':>6s} "
        f"{'REG_MAE':>13s} {'GAIN_REG':>13s} {'CLS_MAE':>13s} "
        f"{'GAIN_CLS':>13s} {'DW/PW':>7s}"
    )
    baseline_reg = float(baseline["regressors"]["mae"])
    baseline_cls = float(baseline["classifiers"]["mae"])
    for step in report["steps"]:
        result = step["joint_adaround"]
        outputs = step["evaluation_after"]
        outcome = "ACCEPT" if result["accepted"] else "ROLLBACK"
        families = result["weight_families"]
        depthwise = sum(value == "depthwise_conv" for value in families)
        regular = sum(value == "regular_conv" for value in families)
        print(
            f"{int(step['step']):4d} "
            f"{str(step['window']['name'])[:34]:34s} "
            f"{outcome:>8s} "
            f"{int(result['best_step']):6d} "
            f"{float(outputs['regressors']['mae']):13.6e} "
            f"{baseline_reg - float(outputs['regressors']['mae']):13.6e} "
            f"{float(outputs['classifiers']['mae']):13.6e} "
            f"{baseline_cls - float(outputs['classifiers']['mae']):13.6e} "
            f"{depthwise:1d}/{regular:1d}"
        )
        print(f"     acceptance: {result['acceptance_reason']}")


def _load_samples(
    directory: Path | None,
    limit: int | None,
    offset: int,
    synthetic_count: int,
    seed: int,
):
    if directory is None:
        return make_synthetic_inputs(synthetic_count, seed=seed)
    return load_npy_inputs(directory, limit, offset=offset)


def _validate_args(args: argparse.Namespace) -> None:
    if args.selection_count <= 0 or args.acceptance_count <= 0:
        raise ValueError("selection-count and acceptance-count must be positive.")
    for name, value in (
        ("classifier_limit", args.classifier_limit),
        ("minimum_selection_improvement", args.minimum_selection_improvement),
        ("minimum_acceptance_improvement", args.minimum_acceptance_improvement),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative.")
    if args.classifier_limit <= 0.0:
        raise ValueError("classifier-limit must be positive.")
    if args.relative_classifier_tolerance is not None and (
        not math.isfinite(args.relative_classifier_tolerance)
        or args.relative_classifier_tolerance < 0.0
    ):
        raise ValueError(
            "relative-classifier-tolerance must be finite and nonnegative."
        )


def _validate_disjoint(args: argparse.Namespace) -> None:
    if not args.require_disjoint:
        return
    if args.calibration_dir is None or args.evaluation_dir is None:
        return
    calibration_paths = list_npy_inputs(args.calibration_dir)[
        args.calibration_offset : (
            None
            if args.calibration_limit is None
            else args.calibration_offset + args.calibration_limit
        )
    ]
    evaluation_paths = list_npy_inputs(args.evaluation_dir)[
        args.evaluation_offset : (
            None
            if args.evaluation_limit is None
            else args.evaluation_offset + args.evaluation_limit
        )
    ]
    overlap = {path.resolve() for path in calibration_paths}.intersection(
        path.resolve() for path in evaluation_paths
    )
    if overlap:
        raise ValueError(
            f"Calibration and evaluation selections overlap by {len(overlap)} files."
        )


if __name__ == "__main__":
    main()

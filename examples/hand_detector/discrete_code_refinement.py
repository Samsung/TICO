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

"""Run gradient-ranked transactional refinement of fixed-scale W8 codes."""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import torch
from tico.quantization.algorithm.adaround import (
    DiscreteCodeRefinementConfig,
    JointAdaRoundObjective,
)
from tico.quantization.analysis import make_output_adapter

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.discrete_code_refinement import (
    run_hand_detector_discrete_code_refinement,
)
from examples.hand_detector._support.joint_adaround import (
    apply_joint_adaround_checkpoint,
    save_joint_adaround_checkpoint,
)
from examples.hand_detector._support.multistart_reconstruction import (
    split_reconstruction_samples_three_way,
)
from examples.hand_detector._support.weight_precision_sensitivity import (
    build_w8a16_candidate,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)
DEFAULT_PROPOSAL_SIZES = (2048, 1024, 512, 256, 128, 64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "discrete-code-refinement",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        command,
        help=(
            "Rank exact fixed-scale floor/ceil alternatives by final-output "
            "gradient, validate nested top-K proposals, and commit accepted rounds."
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
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        required=True,
        help="Entry hard-code checkpoint with fixed W8 scales and activation qparams.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        help="Save the accepted final codes and unchanged affine qparams.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Optional semantic group subset; all Conv sites are used by default.",
    )
    parser.add_argument("--require-accept", action="store_true")

    parser.add_argument("--selection-count", type=int, default=40)
    parser.add_argument("--acceptance-count", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=20260803)
    parser.add_argument(
        "--selection-score-metric",
        choices=("mae", "mse", "rmse", "relative_mae"),
        default="mae",
    )
    parser.add_argument("--classifier-limit", type=float, default=0.1)
    parser.add_argument(
        "--relative-classifier-tolerance",
        type=float,
        help="Optional classifier tolerance relative to each round entry.",
    )
    parser.add_argument("--minimum-selection-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-acceptance-improvement", type=float, default=1e-3)

    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument(
        "--proposal-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROPOSAL_SIZES),
        help="Nested top-K hard proposals evaluated in descending order.",
    )
    parser.add_argument(
        "--gradient-sample-count",
        type=int,
        default=0,
        help="Number of training samples used per round; zero uses all.",
    )
    parser.add_argument("--gradient-seed", type=int, default=20260901)
    parser.add_argument(
        "--gradient-auxiliary-weight",
        type=float,
        default=0.0,
        help="Classifier raw-MAE weight used only in the proposal gradient.",
    )
    parser.add_argument(
        "--training-loss",
        choices=("raw-mae", "normalized-l1"),
        default="raw-mae",
    )
    parser.add_argument(
        "--minimum-predicted-improvement",
        type=float,
        default=0.0,
        help="Discard individual alternatives below this first-order gain.",
    )
    parser.add_argument(
        "--target-regressor-mae",
        type=float,
        default=0.1,
        help="Stop after an accepted round reaches this acceptance-set REG MAE.",
    )
    parser.add_argument(
        "--initialization-metric-tolerance",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--initialization-metric-relative-tolerance",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "discrete_code_refinement.json",
    )


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    _validate_disjoint(args)
    if not args.load_checkpoint.is_file():
        raise FileNotFoundError(
            f"Discrete refinement checkpoint does not exist: {args.load_checkpoint}"
        )
    device = torch.device(args.device)
    float_model = (
        load_nhwc_hand_detector(
            args.weights,
            args.spec,
            map_location=device,
        )
        .to(device)
        .eval()
    )
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
    checkpoint_replay = apply_joint_adaround_checkpoint(
        candidate,
        args.load_checkpoint,
    )
    config = DiscreteCodeRefinementConfig(
        max_rounds=args.max_rounds,
        proposal_sizes=tuple(args.proposal_sizes),
        gradient_sample_count=args.gradient_sample_count,
        gradient_seed=args.gradient_seed,
        primary_output="regressors",
        auxiliary_output="classifiers",
        auxiliary_gradient_weight=args.gradient_auxiliary_weight,
        training_loss=args.training_loss.replace("-", "_"),
        minimum_predicted_improvement=args.minimum_predicted_improvement,
        target_primary_score=args.target_regressor_mae,
        initialization_metric_tolerance=args.initialization_metric_tolerance,
        initialization_metric_relative_tolerance=(
            args.initialization_metric_relative_tolerance
        ),
    )
    relative = (
        {"classifiers": args.relative_classifier_tolerance}
        if args.relative_classifier_tolerance is not None
        else {}
    )
    absolute = {"classifiers": args.classifier_limit}
    selection_objective = JointAdaRoundObjective(
        primary_output="regressors",
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_selection_improvement,
        absolute_output_limits=absolute,
        relative_output_tolerances=relative,
    )
    acceptance_objective = JointAdaRoundObjective(
        primary_output="regressors",
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_acceptance_improvement,
        absolute_output_limits=absolute,
        relative_output_tolerances=relative,
    )

    def progress(round_result) -> None:
        gradient = round_result.gradient_statistics
        print(
            f"\nround={round_result.round_index} "
            f"gradient_samples={len(gradient.sample_indices)} "
            f"reachable={gradient.reachable_candidate_count} "
            f"predicted_improving={gradient.predicted_improving_candidate_count}"
        )
        for proposal in round_result.proposal_evaluations:
            outputs = proposal.selection_outputs
            print(
                f"  K={proposal.applied_size:5d} "
                f"pred={proposal.predicted_improvement:+.6e} "
                f"REG={_mae(outputs, 'regressors'):.6e} "
                f"gain={proposal.selection_improvement:+.6e} "
                f"CLS={_mae(outputs, 'classifiers'):.6e} "
                f"selection={'yes' if proposal.selection_eligible else 'no'}"
            )
            if proposal.acceptance_attempted:
                print(
                    "      acceptance="
                    f"{'yes' if proposal.acceptance_eligible else 'no'} "
                    f"gain={proposal.acceptance_improvement:+.6e} "
                    f"reason={proposal.acceptance_reason}"
                )
        transition = round_result.transition_summary
        print(
            f"  result={'ACCEPT' if round_result.accepted else 'STOP'} "
            f"K={round_result.selected_size} "
            f"new={transition.newly_changed_count} "
            f"reverted={transition.reverted_count} "
            f"retained={transition.retained_count} "
            f"net={transition.net_changed_count}"
        )
        print("  reason=" + round_result.acceptance_reason)

    report = run_hand_detector_discrete_code_refinement(
        float_model,
        candidate,
        data_split=data_split,
        evaluation_samples=evaluation,
        config=config,
        selection_objective=selection_objective,
        acceptance_objective=acceptance_objective,
        output_adapter=OUTPUT_ADAPTER,
        requested_groups=args.groups,
        progress_callback=progress,
        device=device,
    )
    refinement = report["discrete_code_refinement"]
    accepted = bool(refinement["accepted"])
    payload = {
        "analysis": "gradient_ranked_discrete_code_refinement",
        "metadata": {
            **p2_metadata,
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "selection_count": args.selection_count,
            "acceptance_count": args.acceptance_count,
            "selection_seed": args.selection_seed,
            "selection_score_metric": args.selection_score_metric,
            "classifier_limit": args.classifier_limit,
            "relative_classifier_tolerance": args.relative_classifier_tolerance,
            "minimum_selection_improvement": (args.minimum_selection_improvement),
            "minimum_acceptance_improvement": (args.minimum_acceptance_improvement),
            "max_rounds": args.max_rounds,
            "proposal_sizes": args.proposal_sizes,
            "gradient_sample_count": args.gradient_sample_count,
            "gradient_seed": args.gradient_seed,
            "gradient_auxiliary_weight": args.gradient_auxiliary_weight,
            "training_loss": args.training_loss,
            "minimum_predicted_improvement": (args.minimum_predicted_improvement),
            "target_regressor_mae": args.target_regressor_mae,
            "initialization_metric_tolerance": (args.initialization_metric_tolerance),
            "initialization_metric_relative_tolerance": (
                args.initialization_metric_relative_tolerance
            ),
            "device": str(device),
            "requested_groups": args.groups,
            "checkpoint_replay": checkpoint_replay,
        },
        **report,
    }
    checkpoint = None
    if args.save_checkpoint is not None and (accepted or not args.require_accept):
        checkpoint_path = save_joint_adaround_checkpoint(
            candidate,
            args.save_checkpoint,
            metadata={
                "source_report": str(args.report_json),
                "source_checkpoint": str(args.load_checkpoint),
                "analysis": payload["analysis"],
                "accepted": accepted,
                "accepted_rounds": refinement["accepted_rounds"],
                "final_evaluation": payload["final_evaluation"],
                "final_code_change_count": refinement["final_code_change_count"],
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
    if args.require_accept and not accepted:
        raise RuntimeError(
            "Discrete code refinement accepted no round; the report was written, "
            "but no deployment checkpoint was produced."
        )


def _print_report(report: dict[str, object]) -> None:
    baseline = report["baseline_evaluation"]
    final = report["final_evaluation"]
    refinement = report["discrete_code_refinement"]
    assert isinstance(baseline, dict)
    assert isinstance(final, dict)
    assert isinstance(refinement, dict)
    print("\nGradient-ranked fixed-scale hard-code refinement")
    print(
        "Loaded entry:    "
        f"REG_MAE={_mae(baseline, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(baseline, 'classifiers'):.6e}"
    )
    print(
        "Final committed: "
        f"REG_MAE={_mae(final, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(final, 'classifiers'):.6e}"
    )
    print(
        f"result={'ACCEPT' if refinement['accepted'] else 'NO CHANGE'}, "
        f"accepted_rounds={int(refinement['accepted_rounds'])}, "
        f"final_code_changes={int(refinement['final_code_change_count'])}"
    )
    print("stop: " + str(refinement["stop_reason"]))


def _mae(outputs: dict[str, object], name: str) -> float:
    value = outputs[name]
    assert isinstance(value, dict)
    return float(value["mae"])


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
    sample_count = args.calibration_limit or args.synthetic_calibration_samples
    if args.selection_count + args.acceptance_count >= sample_count:
        raise ValueError(
            "selection-count + acceptance-count must be smaller than the "
            "calibration sample count."
        )
    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be nonnegative.")
    if not args.proposal_sizes or any(value <= 0 for value in args.proposal_sizes):
        raise ValueError("--proposal-sizes must contain positive integers.")
    if len(set(args.proposal_sizes)) != len(args.proposal_sizes):
        raise ValueError("--proposal-sizes must be unique.")
    if args.gradient_sample_count < 0:
        raise ValueError("--gradient-sample-count must be nonnegative.")
    for name in (
        "classifier_limit",
        "target_regressor_mae",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "gradient_auxiliary_weight",
        "minimum_predicted_improvement",
        "minimum_selection_improvement",
        "minimum_acceptance_improvement",
        "initialization_metric_tolerance",
        "initialization_metric_relative_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative.")
    if args.relative_classifier_tolerance is not None and (
        not math.isfinite(args.relative_classifier_tolerance)
        or args.relative_classifier_tolerance < 0.0
    ):
        raise ValueError("--relative-classifier-tolerance must be nonnegative.")
    if args.save_checkpoint is not None and args.report_json == args.save_checkpoint:
        raise ValueError("Report and checkpoint paths must differ.")


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

"""Benchmark O1 scheduling on one caller-provided full Circle artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tico.circle import CircleDocument
from tico.circle.passes import (
    CirclePassContext,
    CirclePassStrategy,
    ConstantFoldingProfile,
    create_o1_pipeline,
    O1LegacyCompatibilityOptions,
    O1LegalizationOptions,
    O1OptimizationOptions,
    O1PipelineOptions,
)


@dataclass(frozen=True)
class BenchmarkSample:
    """Record one complete O1 optimization run."""

    seconds: float
    executions: int
    changes: int
    output_bytes: int
    output_sha256: str


@dataclass(frozen=True)
class BenchmarkSummary:
    """Summarize repeated runs for one pass-manager strategy."""

    strategy: str
    mean_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    executions: int
    changes: int
    output_bytes: int
    output_sha256: str


def _pipeline(
    strategy: CirclePassStrategy,
    *,
    maximum_steps: int,
    options: O1PipelineOptions,
):
    """Create O1 and override only its optimize-phase scheduler."""

    pipeline = create_o1_pipeline(
        maximum_steps=maximum_steps,
        options=options,
    )
    optimize = next(phase for phase in pipeline.phases if phase.name == "optimize")
    optimize.manager.strategy = strategy
    return pipeline


def _run_once(
    source: CircleDocument,
    strategy: CirclePassStrategy,
    *,
    maximum_steps: int,
    options: O1PipelineOptions,
) -> tuple[BenchmarkSample, bytes]:
    """Run one isolated O1 optimization and return its serialized result."""

    document = source.clone()
    pipeline = _pipeline(
        strategy,
        maximum_steps=maximum_steps,
        options=options,
    )
    context = CirclePassContext(verify_after_each_pass=False)
    started = time.perf_counter()
    result = pipeline.run(document, context)
    elapsed = time.perf_counter() - started
    document.verify(raise_on_error=True)
    output = document.to_bytes()
    sample = BenchmarkSample(
        seconds=elapsed,
        executions=len(result.executions),
        changes=result.changes,
        output_bytes=len(output),
        output_sha256=hashlib.sha256(output).hexdigest(),
    )
    return sample, output


def _summarize(
    strategy: CirclePassStrategy,
    samples: list[BenchmarkSample],
) -> BenchmarkSummary:
    """Aggregate timings while requiring deterministic structural counters."""

    if not samples:
        raise ValueError("At least one benchmark sample is required.")
    first = samples[0]
    for sample in samples[1:]:
        if (
            sample.executions,
            sample.changes,
            sample.output_bytes,
            sample.output_sha256,
        ) != (
            first.executions,
            first.changes,
            first.output_bytes,
            first.output_sha256,
        ):
            raise AssertionError(
                f"{strategy.value} produced non-deterministic benchmark results."
            )
    timings = [sample.seconds for sample in samples]
    return BenchmarkSummary(
        strategy=strategy.value,
        mean_seconds=statistics.mean(timings),
        minimum_seconds=min(timings),
        maximum_seconds=max(timings),
        executions=first.executions,
        changes=first.changes,
        output_bytes=first.output_bytes,
        output_sha256=first.output_sha256,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Local full .circle artifact.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measured runs per scheduler (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Unmeasured runs per scheduler (default: 1).",
    )
    parser.add_argument(
        "--maximum-steps",
        type=int,
        default=1000,
        help="Maximum pass invocations per optimize phase.",
    )
    parser.add_argument(
        "--constant-folding-profile",
        choices=[profile.value for profile in ConstantFoldingProfile],
        default=ConstantFoldingProfile.BASIC.value,
    )
    parser.add_argument("--resolve-legacy-custom-ops", action="store_true")
    parser.add_argument("--legalize-dynamic-fully-connected", action="store_true")
    parser.add_argument("--fuse-transpose-conv-slice", action="store_true")
    parser.add_argument("--fuse-legacy-fc-gelu-fc", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one machine-readable JSON object.",
    )
    return parser.parse_args()


def main() -> int:
    """Compare the former restart scheduler with the O1 round scheduler."""

    args = _parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must not be negative.")

    source = CircleDocument.load(args.input)
    source.verify(raise_on_error=True)
    options = O1PipelineOptions(
        optimization=O1OptimizationOptions(
            constant_folding_profile=args.constant_folding_profile,
            fuse_transpose_conv_slice=args.fuse_transpose_conv_slice,
        ),
        legalization=O1LegalizationOptions(
            dynamic_fully_connected=args.legalize_dynamic_fully_connected,
        ),
        compatibility=O1LegacyCompatibilityOptions(
            resolve_custom_ops=args.resolve_legacy_custom_ops,
            fuse_fc_gelu_fc=args.fuse_legacy_fc_gelu_fc,
        ),
    )
    strategies = (
        CirclePassStrategy.RESTART,
        CirclePassStrategy.UNTIL_NO_CHANGE,
    )

    for _ in range(args.warmup):
        for strategy in strategies:
            _run_once(
                source,
                strategy,
                maximum_steps=args.maximum_steps,
                options=options,
            )

    samples: dict[CirclePassStrategy, list[BenchmarkSample]] = {
        strategy: [] for strategy in strategies
    }
    reference_output: bytes | None = None
    for _ in range(args.repeat):
        for strategy in strategies:
            sample, output = _run_once(
                source,
                strategy,
                maximum_steps=args.maximum_steps,
                options=options,
            )
            samples[strategy].append(sample)
            if reference_output is None:
                reference_output = output
            elif output != reference_output:
                raise AssertionError(
                    "O1 scheduler variants produced different Circle bytes."
                )

    restart = _summarize(CirclePassStrategy.RESTART, samples[strategies[0]])
    rounds = _summarize(CirclePassStrategy.UNTIL_NO_CHANGE, samples[strategies[1]])
    speedup = restart.mean_seconds / rounds.mean_seconds
    execution_reduction = 1.0 - (rounds.executions / restart.executions)
    report = {
        "input": str(args.input),
        "repeat": args.repeat,
        "warmup": args.warmup,
        "restart": asdict(restart),
        "until_no_change": asdict(rounds),
        "speedup": speedup,
        "execution_reduction": execution_reduction,
        "byte_identical": True,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Input: {args.input}")
        print(
            "RESTART: "
            f"{restart.mean_seconds:.6f}s mean, "
            f"{restart.executions} pass executions"
        )
        print(
            "UNTIL_NO_CHANGE: "
            f"{rounds.mean_seconds:.6f}s mean, "
            f"{rounds.executions} pass executions"
        )
        print(f"Speedup: {speedup:.3f}x")
        print(f"Pass execution reduction: {execution_reduction:.2%}")
        print(f"Output SHA-256: {rounds.output_sha256}")
        print("Outputs are byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

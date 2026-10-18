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

"""Compare complete bytes conversion with explicit file conversion on CPU.

Each measurement runs in a fresh, sequential subprocess. Module measurements
include touched source weights and export; PT2 measurements include loading a
fixture exported in an unmeasured subprocess. Elapsed time covers the conversion
API and save only. Streaming file hashing, structural
verification, constant/metadata equivalence, and reference numerical checks run
outside that measurement. The driver never loads a complete model binary.

Synthetic models sum embedding lookups from deterministic FP32 tables. Every
constant survives conversion, while activations and numerical probes stay small
as weight size increases. These are serialization-oriented full-conversion
measurements, not a prediction for every model architecture or target backend.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MODES = ("bytes", "file")
CHUNK_BYTES = 8 << 20


def _make_model(
    size_mib: int, tables: int, width: int, seed: int, identity_mul: bool = False
) -> tuple[Any, Any]:
    import torch

    rows = max(2, (size_mib << 20) // (tables * width * 4))

    class EmbeddingSum(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.tables = torch.nn.ModuleList(
                [
                    torch.nn.Embedding.from_pretrained(
                        # Fill all pages: uninitialized or zero-backed tensors would
                        # understate the physical RAM occupied by source weights.
                        torch.empty((rows, width), dtype=torch.float32).uniform_(
                            -1, 1, generator=generator
                        ),
                        freeze=True,
                    )
                    for _ in range(tables)
                ]
            )

        def forward(self, indices: Any) -> Any:
            result = self.tables[0](indices)
            for table in self.tables[1:]:
                result = result + table(indices)
            if identity_mul:
                return -(result * 1.0)
            return result

    model = EmbeddingSum().eval()
    probes = (
        torch.tensor([0, 1, rows // 2, rows - 1], dtype=torch.int64),
        torch.tensor([rows - 1, rows // 3, 0, rows // 2], dtype=torch.int64),
        torch.tensor([1, 1, rows - 2, 0], dtype=torch.int64),
    )
    return model, probes


def _configure_limits(limit: int | None) -> None:
    if limit is not None:
        from tico.circle import io as circle_io
        from tico.serialize import circle_binary

        circle_binary._FLATBUFFER_LIMIT = limit
        circle_io._STREAMING_INLINE_BUDGET = min(
            circle_io._STREAMING_INLINE_BUDGET, limit
        )


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _constant_bytes(model: Any) -> int:
    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )


def _write_probes(path: str, model: Any, probes: Any) -> None:
    import numpy as np
    import torch

    with torch.no_grad():
        expected = {
            f"output_{index}": model(probe).numpy()
            for index, probe in enumerate(probes)
        }
    np.savez(
        path,
        **expected,
        **{f"input_{index}": probe.numpy() for index, probe in enumerate(probes)},
    )


def _prepare_pt2(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.threads)
    model, probes = _make_model(
        args.size_mib, args.tables, args.width, args.seed, args.identity_mul
    )
    _write_probes(args.expected, model, probes)
    exported = torch.export.export(model, (probes[0],))
    torch.export.save(exported, args.fixture)
    return {
        "source_constant_bytes": _constant_bytes(model),
        "input_artifact_bytes": Path(args.fixture).stat().st_size,
        "fixture_parameters": _fixture_parameters(args, args.size_mib),
    }


def _measure(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    import tico
    import torch

    torch.set_num_threads(args.threads)
    _configure_limits(args.flatbuffer_limit)
    if args.input_kind == "pt2":
        source = json.loads(Path(args.fixture_info).read_text(encoding="utf-8"))
    else:
        model, probes = _make_model(
            args.size_mib, args.tables, args.width, args.seed, args.identity_mul
        )
        _write_probes(args.expected, model, probes)
        source = {"source_constant_bytes": _constant_bytes(model)}
    config = tico.CompileConfigV1(circle_optimize=args.optimization == "O1")
    peak_before_conversion = _peak_rss_bytes()
    started = time.perf_counter()
    if args.input_kind == "pt2":
        if args.mode == "bytes":
            converted = tico.convert_from_pt2(args.fixture, config=config)
            converted.save(args.output)
        else:
            tico.convert_from_pt2_to_file(args.fixture, args.output, config=config)
    elif args.mode == "bytes":
        converted = tico.convert(model, (probes[0],), config=config)
        converted.save(args.output)
    else:
        tico.convert_to_file(model, (probes[0],), args.output, config=config)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_before_conversion_bytes": peak_before_conversion,
        **source,
        "output_bytes": Path(args.output).stat().st_size,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tico_source": str(Path(tico.__file__).resolve()),
    }


def _hash_view(value: Any) -> str:
    digest = hashlib.sha256()
    view = memoryview(value).cast("B")
    for offset in range(0, len(view), CHUNK_BYTES):
        digest.update(view[offset : offset + CHUNK_BYTES])
    return digest.hexdigest()


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    from tico.circle._object import freeze_object
    from tico.circle.document import CircleDocument
    from tico.circle.runtime import CircleReferenceRuntime

    torch.set_num_threads(args.threads)
    with CircleDocument.load_mapped(args.output) as document:
        document.verify()
        operator_count = sum(
            len(subgraph.operators) if subgraph.operators is not None else 0
            for subgraph in document.model.subgraphs
        )
        # Compare every metadata field and buffer payload independently of inline
        # versus appended storage. Only physical offsets and sizes are normalized.
        metadata = copy.copy(document.model)
        metadata.buffers = [copy.copy(buffer) for buffer in document.model.buffers]
        payloads = []
        for buffer in metadata.buffers:
            data = buffer.data
            payloads.append(
                None
                if data is None
                else {"bytes": memoryview(data).nbytes, "sha256": _hash_view(data)}
            )
            buffer.data = None
            buffer.offset = 0
            buffer.size = 0
        metadata_digest = hashlib.sha256(
            repr(freeze_object(metadata)).encode("utf-8")
        ).hexdigest()
        runtime = CircleReferenceRuntime(document)
        with np.load(args.expected, allow_pickle=False) as expected:
            max_absolute_error = 0.0
            for index in range(3):
                outputs = runtime.run([expected[f"input_{index}"]]).outputs
                if len(outputs) != 1:
                    raise AssertionError(
                        f"Expected one output, received {len(outputs)}"
                    )
                actual = outputs[0]
                reference = np.asarray(expected[f"output_{index}"])
                if actual.shape != reference.shape or actual.dtype != reference.dtype:
                    raise AssertionError("Reference output shape or dtype differs")
                # Lookup is exact; a short FP32 sum can accumulate rounding error.
                np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
                max_absolute_error = max(
                    max_absolute_error, float(np.max(np.abs(actual - reference)))
                )
    return {
        "structural_verification": "passed",
        "reference_numerical_parity": "passed",
        "numerical_probes": 3,
        "max_absolute_error": max_absolute_error,
        "operator_count": operator_count,
        "metadata_sha256": metadata_digest,
        "buffer_payloads": payloads,
    }


def _child(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected", required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--input-kind", choices=("module", "pt2"), default="module")
    parser.add_argument("--identity-mul", action="store_true")
    parser.add_argument("--fixture")
    parser.add_argument("--fixture-info")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--size-mib", type=int)
    parser.add_argument("--tables", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--optimization", choices=("O0", "O1"))
    parser.add_argument("--flatbuffer-limit", type=int)
    args = parser.parse_args(argv)
    # The same benchmark script can exercise an unmodified main worktree using
    # this interpreter and its installed dependencies, without installing twice.
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    if args.prepare:
        report = _prepare_pt2(args)
    else:
        report = _validate(args) if args.validate else _measure(args)
    Path(args.report).write_text(json.dumps(report), encoding="utf-8")


def _spawn(argv: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", *argv],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _is_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def _measure_case(
    args: argparse.Namespace,
    directory: Path,
    size: int,
    label: str,
    mode: str,
    source_root: Path,
    run: int,
    fixture: tuple[Path, Path, Path] | None,
) -> dict[str, Any]:
    stem = f"{size}mib.{args.input_kind}.{label}.{run}"
    output = directory / f"{stem}.circle"
    expected = fixture[1] if fixture else directory / f"{stem}.expected.npz"
    report_path = directory / f"{stem}.measurement.json"
    validation_path = directory / f"{stem}.validation.json"
    common = ["--output", str(output), "--expected", str(expected)]
    child_args = [
        "--source-root",
        str(source_root),
        "--report",
        str(report_path),
        "--mode",
        mode,
        "--input-kind",
        args.input_kind,
        "--size-mib",
        str(size),
        "--tables",
        str(args.tables),
        "--width",
        str(args.width),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--optimization",
        args.optimization,
        *common,
    ]
    if args.flatbuffer_limit is not None:
        child_args += ["--flatbuffer-limit", str(args.flatbuffer_limit)]
    if fixture is not None:
        child_args += ["--fixture", str(fixture[0]), "--fixture-info", str(fixture[2])]
    if args.identity_mul:
        child_args.append("--identity-mul")
    print(
        f"Measuring {size} MiB {args.input_kind} {label}, run {run + 1}",
        file=sys.stderr,
        flush=True,
    )
    _spawn(child_args)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["output_sha256"] = _file_sha256(output)
    # Validation always uses the current source tree and runs after the measured
    # process has exited, so its allocations cannot inflate conversion peak RSS.
    _spawn(
        [
            "--validate",
            "--source-root",
            str(args.source_root),
            "--report",
            str(validation_path),
            "--threads",
            str(args.threads),
            *common,
        ]
    )
    report["validation"] = json.loads(validation_path.read_text(encoding="utf-8"))
    report.update(
        {
            "mode": label,
            "requested_mib": size,
            "run": run,
            "input_kind": args.input_kind,
            "identity_mul": args.identity_mul,
        }
    )
    if not args.keep_outputs:
        output.unlink()
        if fixture is None:
            expected.unlink()
    return report


def _fixture_parameters(args: argparse.Namespace, size: int) -> dict[str, int]:
    return {
        "size_mib": size,
        "tables": args.tables,
        "width": args.width,
        "seed": args.seed,
        "identity_mul": args.identity_mul,
    }


def _prepare_fixture(
    args: argparse.Namespace, directory: Path, size: int
) -> tuple[Path, Path, Path]:
    fixture = directory / f"{size}mib.input.pt2"
    expected = directory / f"{size}mib.probes.npz"
    info = directory / f"{size}mib.fixture.json"
    if all(path.is_file() for path in (fixture, expected, info)):
        prepared = json.loads(info.read_text(encoding="utf-8"))
        if (
            prepared.get("fixture_parameters") == _fixture_parameters(args, size)
            and prepared.get("input_artifact_bytes") == fixture.stat().st_size
        ):
            print(
                f"Reusing prepared {size} MiB PT2 fixture", file=sys.stderr, flush=True
            )
            return fixture, expected, info
    print(
        f"Preparing {size} MiB PT2 fixture outside measurement",
        file=sys.stderr,
        flush=True,
    )
    _spawn(
        [
            "--prepare",
            "--source-root",
            str(args.source_root),
            "--report",
            str(info),
            "--fixture",
            str(fixture),
            "--expected",
            str(expected),
            "--size-mib",
            str(size),
            "--tables",
            str(args.tables),
            "--width",
            str(args.width),
            "--seed",
            str(args.seed),
            "--threads",
            str(args.threads),
            *(["--identity-mul"] if args.identity_mul else []),
        ]
    )
    return fixture, expected, info


def _check_equivalence(results: list[dict[str, Any]]) -> None:
    reference = results[0]
    for report in results:
        validation = report["validation"]
        other = reference["validation"]
        if (
            validation["metadata_sha256"] != other["metadata_sha256"]
            or validation["buffer_payloads"] != other["buffer_payloads"]
        ):
            raise AssertionError(
                f"Constant or metadata mismatch: {reference['mode']} versus "
                f"{report['mode']} at {report['requested_mib']} MiB"
            )
        report["equivalent_to"] = reference["mode"]
        report["byte_identical_to_reference"] = (
            report["output_sha256"] == reference["output_sha256"]
        )


def _comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for size in sorted({report["requested_mib"] for report in results}):
        rows = [report for report in results if report["requested_mib"] == size]
        modes = {report["mode"] for report in rows}
        for baseline in ("baseline-bytes", "bytes"):
            if baseline not in modes or "file" not in modes:
                continue
            before = statistics.median(
                report["peak_rss_bytes"]
                for report in rows
                if report["mode"] == baseline
            )
            after = statistics.median(
                report["peak_rss_bytes"] for report in rows if report["mode"] == "file"
            )
            comparisons.append(
                {
                    "requested_mib": size,
                    "baseline": baseline,
                    "median_peak_rss_reduction_bytes": before - after,
                    "median_peak_rss_reduction_percent": 100
                    * (before - after)
                    / before,
                }
            )
    return comparisons


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mib", default="16,64,256")
    parser.add_argument("--input-kind", choices=("module", "pt2"), default="module")
    parser.add_argument("--identity-mul", action="store_true")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimization", choices=("O0", "O1"), default="O1")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--flatbuffer-limit", type=int)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument(
        "--json", action="store_true", help="Print JSON only to stdout."
    )
    args = parser.parse_args()
    try:
        sizes = [int(value) for value in args.sizes_mib.split(",")]
    except ValueError:
        parser.error("--sizes-mib must be comma-separated positive integers")
    if not sizes or min(sizes) <= 0 or len(set(sizes)) != len(sizes):
        parser.error("--sizes-mib must contain distinct positive integers")
    modes = args.modes.split(",")
    if not modes or any(mode not in MODES for mode in modes):
        parser.error(f"--modes must be comma-separated values from {MODES}")
    if len(set(modes)) != len(modes):
        parser.error("--modes must not contain duplicates")
    if min(args.tables, args.width, args.threads, args.repeat) <= 0:
        parser.error("--tables, --width, --threads, and --repeat must be positive")
    if args.flatbuffer_limit is not None and args.flatbuffer_limit <= 0:
        parser.error("--flatbuffer-limit must be positive")
    if args.keep_outputs and args.workdir is None:
        parser.error("--keep-outputs requires --workdir")
    args.source_root = Path(__file__).resolve().parents[2]
    variants = [(mode, mode, args.source_root) for mode in modes]
    revisions = {"current": _revision(args.source_root)}
    dirty = {"current": _is_dirty(args.source_root)}
    if args.baseline_root is not None:
        baseline_root = args.baseline_root.resolve()
        if not (baseline_root / "tico" / "__init__.py").is_file():
            parser.error("--baseline-root must be a TICO source checkout")
        variants.insert(0, ("baseline-bytes", "bytes", baseline_root))
        revisions["baseline"] = _revision(baseline_root)
        dirty["baseline"] = _is_dirty(baseline_root)
    if args.workdir is not None:
        args.workdir = args.workdir.resolve()
        args.workdir.mkdir(parents=True, exist_ok=True)
    context: Any = (
        contextlib.nullcontext(str(args.workdir))
        if args.workdir is not None
        else tempfile.TemporaryDirectory(prefix="tico-conversion-memory-")
    )
    results = []
    with context as workdir:
        for size in sizes:
            fixture = (
                _prepare_fixture(args, Path(workdir), size)
                if args.input_kind == "pt2"
                else None
            )
            case_results = []
            try:
                for label, mode, root in variants:
                    for run in range(args.repeat):
                        case_results.append(
                            _measure_case(
                                args,
                                Path(workdir),
                                size,
                                label,
                                mode,
                                root,
                                run,
                                fixture,
                            )
                        )
                _check_equivalence(case_results)
                results.extend(case_results)
            finally:
                if fixture is not None and not args.keep_outputs:
                    fixture[0].unlink(missing_ok=True)
    summary = {
        "revisions": revisions,
        "working_tree_dirty": dirty,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "optimization": args.optimization,
        "input_kind": args.input_kind,
        "identity_mul": args.identity_mul,
        "tables": args.tables,
        "width": args.width,
        "seed": args.seed,
        "threads": args.threads,
        "flatbuffer_limit": args.flatbuffer_limit,
        "rss_scope": "fresh process high-water mark including imports and source weights",
        "time_scope": "conversion API and file save; excludes setup and validation",
        "setup_scope": (
            "PT2 and probes prepared once per size in an unmeasured subprocess; "
            "measured API includes PT2 load and creates no source module"
            if args.input_kind == "pt2"
            else "source module and probes prepared in each measured process before timing"
        ),
        "comparisons": _comparisons(results),
        "results": results,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(
        f"input: {args.input_kind}; optimization: {args.optimization}; "
        "structure and reference parity passed"
    )
    print(
        f"{'source MiB':>10} {'mode':<15} {'peak RSS MiB':>12} "
        f"{'output MiB':>10} {'seconds':>9} {'same bytes':>10}"
    )
    for report in results:
        print(
            f"{report['source_constant_bytes'] / (1 << 20):>10.1f} "
            f"{report['mode']:<15} {report['peak_rss_bytes'] / (1 << 20):>12.1f} "
            f"{report['output_bytes'] / (1 << 20):>10.1f} "
            f"{report['elapsed_seconds']:>9.3f} "
            f"{str(report['byte_identical_to_reference']):>10}"
        )
    for comparison in summary["comparisons"]:
        print(
            f"{comparison['requested_mib']} MiB file versus {comparison['baseline']}: "
            f"median peak RSS reduction "
            f"{comparison['median_peak_rss_reduction_bytes'] / (1 << 20):.1f} MiB "
            f"({comparison['median_peak_rss_reduction_percent']:.1f}%)"
        )


if __name__ == "__main__":
    main()

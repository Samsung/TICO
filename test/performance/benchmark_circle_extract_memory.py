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

"""Measure peak RSS and elapsed time of ``tico-circle extract`` ownership paths.

The driver process writes one appended-layout fixture, then runs every
measurement in a fresh subprocess so that one run's peak RSS cannot leak into the
next. Three paths are compared on the same fixture:

- ``legacy``: eager ``bytes`` load, whole-document ``deepcopy`` before trimming,
  complete output ``bytes`` joined in memory, then written. This emulates the
  extraction path before metadata-first cloning and streaming saves.
- ``public``: eager ``bytes`` load, the public detached extraction API, and the
  streaming ``CircleDocument.save``.
- ``cli``: ``tico-circle extract`` itself: read-only mapping, borrowed payloads,
  streaming save.

Peak RSS comes from ``resource.getrusage`` and includes file-backed pages that
the mapping touched; it is not a Python-heap figure. ``--tracemalloc`` adds the
Python allocator's peak as a secondary number that excludes mapped pages.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

MODES = ("legacy", "public", "cli")


def _make_fixture(path: Path, buffers: int, buffer_mib: int, limit: int | None) -> None:
    import numpy as np
    from circle_schema import circle

    from tico.circle.document import CircleDocument
    from tico.serialize import circle_binary

    if limit is not None:
        circle_binary._FLATBUFFER_LIMIT = limit
    model = circle.Model.ModelT()
    model.version = 0
    model.description = "extract-memory-benchmark"
    code = circle.OperatorCode.OperatorCodeT()
    code.builtinCode = circle.BuiltinOperator.BuiltinOperator.ADD
    code.deprecatedBuiltinCode = code.builtinCode
    model.operatorCodes = [code]
    model.buffers = [circle.Buffer.BufferT()]
    graph = circle.SubGraph.SubGraphT()
    graph.name = "main"

    def tensor(name: str, buffer_index: int = 0) -> Any:
        value = circle.Tensor.TensorT()
        value.name = name
        value.shape = [1]
        value.type = circle.TensorType.TensorType.FLOAT32
        value.buffer = buffer_index
        return value

    tensors = [tensor("x")]
    operators = []
    previous = 0
    size = buffer_mib << 20
    for index in range(buffers):
        payload = circle.Buffer.BufferT()
        # Distinct, non-constant bytes so a bad offset is detected by digest.
        payload.data = (np.arange(size, dtype=np.uint32) + index).view(np.uint8)[:size]  # type: ignore[assignment]
        model.buffers.append(payload)
        tensors.append(tensor(f"w{index}", index + 1))
        tensors.append(tensor(f"y{index}"))
        operator = circle.Operator.OperatorT()
        operator.opcodeIndex = 0
        operator.inputs = [previous, len(tensors) - 2]
        operator.outputs = [len(tensors) - 1]
        operators.append(operator)
        previous = len(tensors) - 1
    graph.tensors = tensors
    graph.inputs = [0]
    graph.outputs = [previous]
    graph.operators = operators
    model.subgraphs = [graph]
    model.metadataBuffer = []
    model.metadata = []
    model.signatureDefs = []
    CircleDocument(model).save(path)


def _run_measurement(
    mode: str,
    source: Path,
    output: Path,
    ops: str,
    limit: int | None,
    trace: bool,
) -> dict[str, Any]:
    from tico.circle import io as circle_io
    from tico.serialize import circle_binary

    if limit is not None:
        # Keep the output layout identical across modes when the fixture used a
        # lowered private budget; the save path also honours it.
        circle_binary._FLATBUFFER_LIMIT = limit
        circle_io._STREAMING_INLINE_BUDGET = min(
            circle_io._STREAMING_INLINE_BUDGET, limit
        )

    if trace:
        tracemalloc.start()
    started = time.perf_counter()
    if mode == "cli":
        from tico.circle.cli.main import main

        status = main(["extract", str(source), "--ops", ops, "-o", str(output)])
        if status != 0:
            raise SystemExit(status)
    else:
        from tico.circle.document import CircleDocument
        from tico.circle.operations import extract_by_operator_indices, PayloadOwnership
        from tico.circle.selector import parse_operator_spec

        document = CircleDocument.load(source)
        if mode == "legacy":
            # Whole-document deepcopy, then in-place trimming on the copy, then
            # the complete binary joined in memory: the pre-change behaviour.
            trimmed = document.clone()
            result = extract_by_operator_indices(
                trimmed,
                parse_operator_spec(ops),
                payload_ownership=PayloadOwnership.BORROWED,
            )
            circle_io.write_circle_bytes(result.document.to_bytes(), output)
        else:
            result = extract_by_operator_indices(document, parse_operator_spec(ops))
            del document
            result.document.save(output)
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "mode": mode,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "output_bytes": output.stat().st_size,
    }
    if trace:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["tracemalloc_peak_bytes"] = peak
    return report


def _child(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output")
    parser.add_argument("--ops")
    parser.add_argument("--buffers", type=int)
    parser.add_argument("--buffer-mib", type=int)
    parser.add_argument("--flatbuffer-limit", type=int)
    parser.add_argument("--tracemalloc", action="store_true")
    args = parser.parse_args(argv)
    if args.mode is None:
        _make_fixture(
            Path(args.fixture), args.buffers, args.buffer_mib, args.flatbuffer_limit
        )
        return
    report = _run_measurement(
        args.mode,
        Path(args.fixture),
        Path(args.output),
        args.ops,
        args.flatbuffer_limit,
        args.tracemalloc,
    )
    print(json.dumps(report))


def _spawn(argv: list[str]) -> str:
    completed = subprocess.run(
        [sys.executable, __file__, "--child", *argv],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _format_mib(value: int) -> str:
    return f"{value / (1 << 20):.0f}"


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffers", type=int, default=8)
    parser.add_argument("--buffer-mib", type=int, default=128)
    parser.add_argument(
        "--flatbuffer-limit",
        type=int,
        default=None,
        help=(
            "Lower the private FlatBuffer budget so a mid-size fixture uses the "
            "appended layout. Omit it to benchmark real >2 GiB behaviour."
        ),
    )
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tracemalloc", action="store_true")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in MODES:
            parser.error(f"unknown mode {mode!r}; choose from {MODES}")
    if args.buffers < 2:
        parser.error("--buffers must be at least 2 so that a payload can be dropped")

    context: Any = (
        tempfile.TemporaryDirectory()
        if args.workdir is None
        else _NullDirectory(args.workdir)
    )
    with context as workdir:
        directory = Path(workdir)
        fixture = directory / "fixture.circle"
        fixture_args = [
            "--fixture",
            str(fixture),
            "--buffers",
            str(args.buffers),
            "--buffer-mib",
            str(args.buffer_mib),
        ]
        if args.flatbuffer_limit is not None:
            fixture_args += ["--flatbuffer-limit", str(args.flatbuffer_limit)]
        _spawn(fixture_args)
        scenarios = {
            "keep-one": "0",
            "keep-most": f"0-{args.buffers - 2}",
        }
        results: list[dict[str, Any]] = []
        for scenario, ops in scenarios.items():
            for mode in modes:
                for run in range(args.repeat):
                    output = directory / f"{scenario}.{mode}.{run}.circle"
                    child_args = [
                        "--fixture",
                        str(fixture),
                        "--mode",
                        mode,
                        "--ops",
                        ops,
                        "--output",
                        str(output),
                    ]
                    if args.flatbuffer_limit is not None:
                        child_args += ["--flatbuffer-limit", str(args.flatbuffer_limit)]
                    if args.tracemalloc:
                        child_args.append("--tracemalloc")
                    report = json.loads(_spawn(child_args))
                    report["scenario"] = scenario
                    report["run"] = run
                    results.append(report)
                    output.unlink()
        summary = {
            "fixture_bytes": fixture.stat().st_size,
            "buffers": args.buffers,
            "buffer_mib": args.buffer_mib,
            "flatbuffer_limit": args.flatbuffer_limit,
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "results": results,
        }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(
        f"fixture: {_format_mib(summary['fixture_bytes'])} MiB, "
        f"{args.buffers} x {args.buffer_mib} MiB payloads, "
        f"flatbuffer limit {args.flatbuffer_limit or 'default'}"
    )
    print(
        f"{'scenario':<10} {'mode':<8} {'peak RSS MiB':>13} {'output MiB':>11} {'seconds':>8}"
    )
    for report in results:
        print(
            f"{report['scenario']:<10} {report['mode']:<8} "
            f"{_format_mib(report['peak_rss_bytes']):>13} "
            f"{_format_mib(report['output_bytes']):>11} "
            f"{report['elapsed_seconds']:>8.2f}"
            + (
                f"  tracemalloc peak {_format_mib(report['tracemalloc_peak_bytes'])} MiB"
                if "tracemalloc_peak_bytes" in report
                else ""
            )
        )


class _NullDirectory:
    """Use a caller-provided directory without deleting it afterwards."""

    def __init__(self, path: str):
        self.path = path

    def __enter__(self) -> str:
        os.makedirs(self.path, exist_ok=True)
        return self.path

    def __exit__(self, *exc_info: object) -> None:
        return None


if __name__ == "__main__":
    main()

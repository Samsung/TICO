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

"""Run download-free CPU recipe checks, including real saved Circle execution.

Invoke with ``python -m tico.quantization.recipes.debug.cpu_validation`` from an
installed development checkout. Each invocation requires a new output directory.
All original child exit codes, configurations and logs remain available there.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    """Save strict JSON without dropping numerical failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _ptq_with_cache_evidence(ctx, path: Path) -> None:
    """Record real observer calls without changing calibration values or returns."""
    import torch

    from tico.quantization.recipes.stages.ptq import PTQStage

    visits: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {}
    original = ctx.adapter.calibrate_prepared_model

    def calibrate(context, prepared, stage):
        saved = []
        branch = ["prefill"]

        def trace_forward(_module, _args, kwargs):
            branch[0] = (
                "cache_decode"
                if kwargs.get("past_key_values") is not None
                else "prefill"
            )

        handle = prepared.register_forward_pre_hook(trace_forward, with_kwargs=True)
        try:
            for name, observer in prepared.named_modules():
                if name.rsplit(".", 1)[-1] not in {
                    "obs_past_key",
                    "obs_past_value",
                    "obs_new_k",
                    "obs_new_v",
                    "obs_present_key",
                    "obs_present_value",
                }:
                    continue
                old = observer.collect
                local = "collect" in vars(observer)
                saved.append((name, observer, old, local))

                def collect(value, *, old=old, name=name, observer=observer):
                    visits.append(
                        {
                            "module": name,
                            "branch": branch[0],
                            "shape": list(value.shape),
                            "valid_length": int(value.shape[-2]),
                            "nonempty": value.numel() > 0,
                            "tensor_dtype": str(value.dtype),
                            "quant_dtype": repr(observer.dtype),
                            "qscheme": repr(observer.qscheme),
                            "observer": type(observer).__name__,
                        }
                    )
                    return old(value)

                observer.collect = collect
            original(context, prepared, stage)
            for name, observer, _, _ in saved:
                statistics[name] = {
                    key: getattr(observer, key).detach().cpu().tolist()
                    for key in ("min_val", "max_val")
                    if isinstance(getattr(observer, key, None), torch.Tensor)
                }
        finally:
            handle.remove()
            for _, observer, old, local in saved:
                if local:
                    observer.collect = old
                else:
                    delattr(observer, "collect")

    ctx.adapter.calibrate_prepared_model = calibrate
    try:
        PTQStage().run(ctx, ctx.cfg["pipeline"][0])
    finally:
        delattr(ctx.adapter, "calibrate_prepared_model")
    frozen = {}
    for name, observer in ctx.model.named_modules():
        if name not in statistics:
            continue
        fields = {}
        for key in ("_cached_scale", "_cached_zp"):
            tensor = getattr(observer, key, None)
            if isinstance(tensor, torch.Tensor):
                fields[key] = {
                    "values": tensor.detach().cpu().tolist(),
                    "dtype": str(tensor.dtype),
                    "nonempty_finite": tensor.numel() > 0
                    and bool(torch.isfinite(tensor).all()),
                }
        frozen[name] = fields
    _write(
        path,
        {
            "visits": visits,
            "before_freeze": statistics,
            "frozen_qparams": frozen,
            "instrumentation": "Read-only collect/forward tracing; original callables restored before freeze.",
        },
    )


def _fixtures(root: Path, seed: int) -> None:
    """Calibrate/freeze both real HF families through the standard PTQ stage."""
    import torch

    from tico.quantization.recipes.debug.cpu_fixtures import fixture_context
    from tico.quantization.recipes.debug.export_regression import (
        _file_sha256,
        _state_contract,
    )

    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=False)
    for family in ("llama", "qwen3_vl"):
        ctx = fixture_context(family, seed=seed)
        geometry = ctx.model.config.to_dict()
        count = sum(parameter.numel() for parameter in ctx.model.parameters())
        print(
            f"{family}: {count} parameters, approximately {4 * count} FP32 parameter bytes",
            flush=True,
        )
        _ptq_with_cache_evidence(ctx, root / f"{family}.cache-coverage.json")
        checkpoint = root / f"{family}.pt"
        torch.save(ctx.model, checkpoint)
        _write(root / f"{family}.json", ctx.cfg)
        _write(
            root / f"{family}.provenance.json",
            {
                "seed": seed,
                "architecture": geometry,
                "parameters": count,
                "approximate_fp32_parameter_bytes": 4 * count,
                "checkpoint_sha256": _file_sha256(checkpoint),
                "config_sha256": _file_sha256(root / f"{family}.json"),
                "calibration_tokens_per_sample": 8,
                "calibration_samples": 2,
                "calibration_path": "standard adapter.calibrate_prepared_model + PTQStage",
                "quantization_state": _state_contract(ctx.model),
            },
        )


def _replay_array_keys(record: dict[str, Any]) -> set[str]:
    """Require explicit input/output counts before interpreting saved arrays."""
    inputs, outputs = record.get("input_count"), record.get("output_count")
    if type(inputs) is not int or inputs < 0:
        raise ValueError("Connected replay input_count must be a nonnegative integer.")
    if type(outputs) is not int or outputs < 1:
        raise ValueError("Connected replay output_count must be a positive integer.")
    return {
        f"{kind}_{index}"
        for kind, count in (("input", inputs), ("torch", outputs), ("circle", outputs))
        for index in range(count)
    }


def _load_replay_evidence(evidence: Path) -> dict[str, Any]:
    """Reject empty or malformed replay stages before loading the checkpoint."""
    source = json.loads((evidence / "summary.json").read_text())
    if not isinstance(source, dict):
        raise ValueError("Connected replay summary must be a mapping.")
    stages = source.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Connected replay requires nonempty stage evidence.")
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("Connected replay stage evidence must be a mapping.")
        _replay_array_keys(stage)
    return source


def _same_replay_arrays(
    previous, current: dict[str, Any], record: dict[str, Any]
) -> bool:
    """Compare complete saved evidence with exact numeric shape/dtype semantics."""
    from tico.quantization.recipes.debug.export_regression import compare_arrays

    expected = _replay_array_keys(record)
    if len(previous.files) != len(set(previous.files)):
        raise ValueError("Connected replay contains duplicate array keys.")
    if set(previous.files) != expected or set(current) != expected:
        raise ValueError(
            "Connected replay evidence array keys do not match the declared input/output counts."
        )
    return all(
        compare_arrays(current[key], previous[key], atol=0, rtol=0)["ok"]
        for key in sorted(expected)
    )


def _replay_connected(options: argparse.Namespace) -> int:
    """Replay saved connected host calls through the selected real source/runtime."""
    import re

    import numpy as np
    import torch

    import tico

    from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
    from tico.quantization.recipes.debug.export_regression import (
        _array,
        _file_sha256,
        _state_contract,
        _tree_arrays,
        compare_arrays,
    )
    from tico.quantization.recipes.export.qwen3_vl import _unwrap_qwen3_vl_components
    from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
    from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
        Qwen3VLDeepstackFusionExportAdapter,
        Qwen3VLLMHeadExportAdapter,
        Qwen3VLMultimodalEmbeddingExportAdapter,
        Qwen3VLTextEmbeddingExportAdapter,
    )
    from tico.utils.signature import ModelInputSpec

    repo = Path(options.repo).resolve()
    if not Path(tico.__file__).resolve().is_relative_to(repo):
        raise RuntimeError(
            "The requested replay checkout is not the imported tico package."
        )
    torch.set_num_threads(1)
    root, evidence = Path(options.output), Path(options.evidence)
    root.mkdir(parents=True, exist_ok=False)
    source = _load_replay_evidence(evidence)
    checkpoint = Path(options.checkpoint)
    if _file_sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("Connected replay checkpoint differs from the original run.")
    model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    initial = _state_contract(model)
    top, _, vision, text = _unwrap_qwen3_vl_components(model)
    stages: dict[str, torch.nn.Module] = {
        "vision_prefill_t1_h2_w2.q.circle": vision.as_export_module(
            "prefill", grid_thw=(1, 2, 2)
        ).eval(),
        "multimodal_embedding_prefill_t1_h2_w2.q.circle": Qwen3VLMultimodalEmbeddingExportAdapter(
            top, visual_start_idx=2
        ).eval(),
        "token_embedding.q.circle": Qwen3VLTextEmbeddingExportAdapter(top).eval(),
        "lm_head.q.circle": Qwen3VLLMHeadExportAdapter(top).eval(),
    }
    records = []
    with torch.no_grad():
        for index, original in enumerate(source["stages"]):
            name = original["stage"]
            if name not in stages:
                match = re.fullmatch(
                    r"decoder_layer_(prefill|decode)_(\d+)\.q\.circle", name
                )
                if match:
                    layer = text.layers[int(match[2])]
                    if not isinstance(layer, PTQWrapper):
                        raise TypeError("Connected replay requires PTQ decoder layers.")
                    stages[name] = layer.as_export_module(match[1], return_kv=True)
                elif re.fullmatch(r"deepstack_fusion_\d+_t1_h2_w2\.q\.circle", name):
                    stages[name] = Qwen3VLDeepstackFusionExportAdapter(
                        visual_start_idx=2
                    )
                else:
                    raise ValueError(f"Unknown connected replay stage: {name}")
            inputs = torch.load(
                evidence / original["structured_inputs"],
                map_location="cpu",
                weights_only=True,
            )
            binary = (Path(options.artifacts) / name).read_bytes()
            bound = ModelInputSpec(binary).bind(
                inputs["args"], inputs["kwargs"], check=False
            )
            runtime = CircleReferenceRuntime(binary)
            mode = (
                ExecutionMode.FAKE_QUANTIZE
                if runtime.has_quantized_activations()
                else ExecutionMode.NATIVE
            )
            golden = list(
                _tree_arrays(stages[name](*inputs["args"], **inputs["kwargs"])).values()
            )
            actual = runtime.run(bound, mode=mode).outputs
            if (
                len(golden) != original["output_count"]
                or len(actual) != original["output_count"]
            ):
                raise RuntimeError("Connected replay output count changed.")
            values: dict[str, Any] = {
                f"input_{i}": _array(value) for i, value in enumerate(bound)
            }
            for kind, tensors in (("torch", golden), ("circle", actual)):
                values.update(
                    {f"{kind}_{i}": _array(value) for i, value in enumerate(tensors)}
                )
            with np.load(
                evidence / original["evidence"], allow_pickle=False
            ) as previous:
                exact = _same_replay_arrays(previous, values, original)
                exact = exact and mode.value == original["mode"]
                outputs = []
                for i, (left, right) in enumerate(zip(golden, actual)):
                    left, right = _array(left), _array(right)
                    result = compare_arrays(left, right, atol=1e-5, rtol=1e-4)
                    if left.shape == right.shape and left.dtype == right.dtype:
                        mismatch = np.argwhere(
                            ~np.isclose(
                                left, right, atol=1e-5, rtol=1e-4, equal_nan=False
                            )
                        )
                        if len(mismatch):
                            coordinate = tuple(int(v) for v in mismatch[0])
                            result["first_difference"] = {
                                "index": list(coordinate),
                                "torch": float(left[coordinate]),
                                "circle": float(right[coordinate]),
                            }
                    outputs.append(result)
                if any(not np.isfinite(value).all() for value in values.values()):
                    raise ValueError("Connected replay contains nonfinite evidence.")
                np.savez(root / f"call-{index:03d}.npz", **values)
                records.append(
                    {
                        "stage": name,
                        "same_saved_host_inputs_torch_circle": exact,
                        "input_count": len(bound),
                        "output_count": len(actual),
                        "torch_circle": outputs,
                    }
                )
    preserved = initial == _state_contract(model)
    same = bool(records) and all(
        record["same_saved_host_inputs_torch_circle"] for record in records
    )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    _write(
        root / "summary.json",
        {
            "status": "PASS" if same and preserved else "FAIL",
            "purpose": "Exact same-input connected replay; strict Torch/Circle parity is reported separately.",
            "code_sha": git.stdout.strip(),
            "tico_import": str(Path(tico.__file__).resolve()),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "capacity": source["capacity"],
            "same_saved_host_inputs_torch_circle": same,
            "frozen_state_preserved": preserved,
            "strict_torch_circle_status": (
                "PASS"
                if all(
                    output["ok"]
                    for record in records
                    for output in record["torch_circle"]
                )
                else "FAIL"
            ),
            "source_evidence": str(evidence),
            "stages": records,
        },
    )
    return 0 if same and preserved else 1


def _command(root: Path, name: str, command: list[str], *, cwd: Path) -> dict[str, Any]:
    """Run one child sequentially, retaining its original return code."""
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    record: dict[str, Any] = {
        "command": command,
        "cwd": str(cwd),
        "log": str(root / f"{name}.log"),
        "stderr": "merged into stdout log",
        "environment": {
            key: environment[key]
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
    }
    started = time.monotonic()
    print(f"Running {name}", flush=True)
    with (root / f"{name}.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record.update(
        {
            "returncode": result.returncode,
            "seconds": time.monotonic() - started,
            "status": (
                "PASS"
                if result.returncode == 0
                else "ERROR" if result.returncode == 2 else "FAIL"
            ),
        }
    )
    _write(root / f"{name}.command.json", record)
    return record


def _run(options: argparse.Namespace) -> int:
    """Orchestrate selected stages without treating skipped stages as passing."""
    root = Path(options.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[4]
    versions = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "circle-schema", "numpy")
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    source_paths = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
        ],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout.split(b"\0")
    source_digest = hashlib.sha256()
    for raw_path in sorted(set(source_paths) - {b""}):
        source_digest.update(raw_path + b"\0")
        source_digest.update((repo / os.fsdecode(raw_path)).read_bytes())
    resources: dict[str, Any] = {
        name: Path(path).read_text().strip() if Path(path).is_file() else None
        for name, path in (
            ("cgroup_memory_max", "/sys/fs/cgroup/memory.max"),
            ("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"),
        )
    }
    resources["disk_free_bytes"] = shutil.disk_usage(root).free
    _write(
        root / "environment.json",
        {
            "command": sys.argv,
            "cwd": os.getcwd(),
            "source_sha256": source_digest.hexdigest(),
            "resources": resources,
            "versions": versions,
            "python": sys.version,
            "executable": sys.executable,
            "repo": str(repo),
            "start_sha": git.stdout.strip(),
            "working_tree": dirty.stdout,
            "seed": options.seed,
            "import_path": sys.path,
        },
    )
    records: dict[str, Any] = {}
    stages = (
        ("regression", "generate", "coverage", "evaluation", "circle")
        if options.stage == "all"
        else (options.stage,)
    )
    tests = {
        "regression": [
            "test.quantization.recipes.test_cpu_fixtures",
            "test.quantization.recipes.test_cpu_validation",
            "test.quantization.recipes.test_export_regression",
            "test.quantization.wrapq.utils.test_static_export_inputs",
            "test.quantization.wrapq.wrappers.llama.test_rope_consistency",
        ],
        "generate": ["test.quantization.recipes.test_cpu_generation"],
        "coverage": ["test.quantization.recipes.test_qwen_cache_coverage_cpu"],
        "evaluation": ["test.quantization.recipes.test_cpu_evaluation"],
    }
    for stage in stages:
        if stage in tests:
            command = [
                sys.executable,
                "-m",
                "tico.quantization.recipes.debug.cpu_validation",
                "--stage",
                "_tests",
                "--output",
                str(root / f"{stage}-tests"),
            ]
            for module in tests[stage]:
                command += ["--test-module", module]
            records[stage] = _command(root, stage, command, cwd=repo)
            continue
        fixture_dir = (
            Path(options.fixtures).resolve() if options.fixtures else root / "fixtures"
        )
        if not options.fixtures:
            records["fixtures"] = _command(
                root,
                "fixtures",
                [
                    sys.executable,
                    "-m",
                    "tico.quantization.recipes.debug.cpu_validation",
                    "--stage",
                    "_fixtures",
                    "--output",
                    str(fixture_dir),
                    "--seed",
                    str(options.seed),
                ],
                cwd=repo,
            )
            if records["fixtures"]["returncode"]:
                records[stage] = {
                    "status": "BLOCKED",
                    "reason": "Fixture calibration failed; see fixtures.log",
                }
                continue
        for family in ("llama", "qwen3_vl"):
            provenance = json.loads(
                (fixture_dir / f"{family}.provenance.json").read_text()
            )
            from tico.quantization.recipes.debug.export_regression import _file_sha256

            if provenance["checkpoint_sha256"] != _file_sha256(
                fixture_dir / f"{family}.pt"
            ):
                raise ValueError(
                    f"{family}: checkpoint hash differs from its calibration provenance."
                )
            if provenance["config_sha256"] != _file_sha256(
                fixture_dir / f"{family}.json"
            ):
                raise ValueError(
                    f"{family}: configuration hash differs from its calibration provenance."
                )
        for family, capacity in (("llama", 16), ("qwen3_vl", 16), ("qwen3_vl", 32)):
            name = f"{family}-{capacity}"
            command = [
                sys.executable,
                str(repo / "tico/quantization/recipes/debug/export_regression.py"),
                "run" if options.baseline_repo else "isolation",
                "--candidate-repo",
                str(repo),
                "--config",
                str(fixture_dir / f"{family}.json"),
                "--checkpoint",
                str(fixture_dir / f"{family}.pt"),
                "--seq-len",
                str(capacity),
                "--eager-seq-len",
                "48",
                "--decode-steps",
                "4",
                "--seed",
                str(options.seed),
                "--threads",
                "1",
                "--output",
                str(root / name),
                "--require-torch-parity",
            ]
            if family == "llama":
                command += ["--artifacts", "circle_bundle,circle_full"]
            if options.baseline_repo:
                command += [
                    "--baseline-repo",
                    str(Path(options.baseline_repo).resolve()),
                ]
            records[name] = _command(root, name, command, cwd=repo)
            if (
                family == "qwen3_vl"
                and (root / name / "candidate/before/snapshot.json").is_file()
            ):
                connected = f"{name}-connected"
                records[connected] = _command(
                    root,
                    connected,
                    [
                        sys.executable,
                        "-m",
                        "tico.quantization.recipes.debug.cpu_validation",
                        "--stage",
                        "_connected",
                        "--output",
                        str(root / connected),
                        "--checkpoint",
                        str(fixture_dir / f"{family}.pt"),
                        "--artifacts",
                        str(root / name / "candidate/before/artifacts"),
                        "--capacity",
                        str(capacity),
                    ],
                    cwd=repo,
                )
    failed = any(record["status"] != "PASS" for record in records.values())
    _write(
        root / "summary.json",
        {
            "status": "FAIL" if failed else "PASS",
            "stages": records,
            "not_in_scope": [
                "large pretrained quality",
                "2048-token full-model capacity",
                "NPU integer execution and performance",
            ],
            "strict_tolerances": {"atol": 1e-5, "rtol": 1e-4},
        },
    )
    return 1 if failed else 0


def main(argv=None) -> int:
    """Provide independently rerunnable stages and explicit failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "regression",
            "generate",
            "coverage",
            "evaluation",
            "circle",
            "replay",
            "_fixtures",
            "_connected",
            "_tests",
        ),
        default="all",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1073)
    parser.add_argument(
        "--fixtures", help="Reuse trusted local frozen fixtures without recalibrating."
    )
    parser.add_argument(
        "--baseline-repo",
        help="Baseline checkout for same-checkpoint exporter comparison.",
    )
    parser.add_argument(
        "--test-module", action="append", default=[], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--evidence", help="Original connected-run evidence directory for replay."
    )
    parser.add_argument(
        "--repo", help="Checkout providing the imported TICO package for replay."
    )
    parser.add_argument(
        "--checkpoint",
        help="Trusted frozen checkpoint for connected execution or replay.",
    )
    parser.add_argument(
        "--artifacts",
        help="Directory of saved Circle stages for connected execution or replay.",
    )
    parser.add_argument("--capacity", type=int, default=16, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if Path(options.output).exists():
        parser.error("Output already exists; choose a new directory.")
    if options.stage == "replay":
        required = ("repo", "checkpoint", "evidence", "artifacts")
        missing = ["--" + name for name in required if not getattr(options, name)]
        if missing:
            parser.error("Replay requires " + ", ".join(missing) + ".")
        for name in ("repo", "evidence", "artifacts"):
            if not Path(getattr(options, name)).is_dir():
                parser.error(f"Replay --{name} must be an existing directory.")
        if not Path(options.checkpoint).is_file():
            parser.error("Replay --checkpoint must be an existing file.")
    try:
        if options.stage == "replay":
            return _replay_connected(options)
        if options.stage == "_tests":
            import unittest

            import torch

            torch.set_num_threads(1)
            root = Path(options.output)
            root.mkdir(parents=True, exist_ok=False)
            suite = unittest.TestLoader().loadTestsFromNames(options.test_module)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            complete = (
                result.testsRun > 0 and not result.skipped and result.wasSuccessful()
            )
            _write(
                root / "summary.json",
                {
                    "status": (
                        "PASS"
                        if complete
                        else (
                            "INCOMPLETE"
                            if result.skipped or not result.testsRun
                            else "FAIL"
                        )
                    ),
                    "tests_run": result.testsRun,
                    "skips": [
                        {"test": test.id(), "reason": reason}
                        for test, reason in result.skipped
                    ],
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                    "modules": options.test_module,
                },
            )
            return 0 if complete else 1
        if options.stage == "_fixtures":
            _fixtures(Path(options.output), options.seed)
            return 0
        if options.stage == "_connected":
            from tico.quantization.recipes.debug.cpu_static_cache import connected_qwen

            return connected_qwen(
                Path(options.checkpoint),
                Path(options.artifacts),
                Path(options.output),
                options.capacity,
            )
        return _run(options)
    except Exception as error:
        traceback.print_exc()
        status = (
            "BLOCKED"
            if isinstance(
                error, (ModuleNotFoundError, importlib.metadata.PackageNotFoundError)
            )
            else "ERROR"
        )
        evidence = {
            "status": status,
            "type": type(error).__name__,
            "message": str(error),
            "command": sys.argv,
            "cwd": os.getcwd(),
        }
        _write(Path(options.output) / "error.json", evidence)
        _write(Path(options.output) / "summary.json", evidence)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

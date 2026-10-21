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

"""Run offline RoPE checks on a source checkout; missing tests never count as PASS."""

import argparse
import importlib.metadata
import json
import sys
import traceback
import unittest
from pathlib import Path


def main(argv=None) -> int:
    """Run tiny real-HF models and static-input tests without downloading weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Patched TICO source checkout, including test/.")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    if args.report.exists():
        parser.error("Report already exists; choose a new path.")
    repo = Path(args.repo).resolve()
    if not (repo / "test/quantization/wrapq/wrappers/llama/test_rope_consistency.py").is_file():
        parser.error("--repo must include the patched RoPE consistency test.")
    sys.path.insert(0, str(repo))
    report = {"status": "ERROR", "repo": str(repo), "scope": "Tiny real-HF models, actual wrappers, and Torch Export; not pretrained-model Circle or NPU validation."}
    code = 2
    try:
        import torch
        import tico

        if not Path(tico.__file__).resolve().is_relative_to(repo):
            raise RuntimeError("TICO was imported from a different checkout.")
        torch.set_num_threads(args.threads)
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "circle-schema")
        }
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            "test.quantization.wrapq.wrappers.llama.test_rope_consistency",
            "test.quantization.wrapq.utils.test_static_export_inputs",
        ])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        incomplete = bool(result.errors or result.skipped or result.testsRun == 0)
        code = 2 if incomplete else 0 if result.wasSuccessful() else 1
        report.update({
            "status": {0: "PASS", 1: "FAIL", 2: "ERROR"}[code],
            "tests_run": result.testsRun,
            "failures": [{"test": str(test), "traceback": text} for test, text in result.failures],
            "errors": [{"test": str(test), "traceback": text} for test, text in result.errors],
            "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        })
    except Exception as exc:
        traceback.print_exc()
        report.update({"error_type": type(exc).__name__, "message": str(exc)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())

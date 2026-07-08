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

"""CLI tests for inspector.py wrapper-smoke dispatch."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import io
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.examples.inspector as inspect_cli


class TestInspectWrapperSmokeCLI(unittest.TestCase):
    """Validate wrapper-smoke command-line dispatch without running quantization."""

    def test_wrapper_smoke_dispatches_single_case(self):
        """inspector.py should route wrapper-smoke arguments to the shared runner."""
        calls = {}
        argv = [
            "inspector.py",
            "--mode",
            "wrapper-smoke",
            "--config",
            "wrapper_smoke.yaml",
            "--case",
            "nn_linear",
            "--export",
            "circle",
            "--output-dir",
            "out",
            "--strict",
            "--no-plot",
            "--calibration-iters",
            "1",
        ]

        def fake_load_recipe_config(path, overrides):
            """Record config loading and return a minimal config."""
            calls["load"] = (path, overrides)
            return {"runtime": {}}

        with patch.object(
            inspect_cli,
            "load_recipe_config",
            fake_load_recipe_config,
        ), patch.object(
            inspect_cli, "set_seed", lambda seed: calls.setdefault("seed", seed)
        ), patch.object(
            inspect_cli,
            "run_wrapper_smoke",
            lambda *args, **kwargs: calls.setdefault("run", (args, kwargs)),
        ), patch.object(
            sys, "argv", argv
        ):
            inspect_cli.main()

        self.assertEqual(calls["run"][0], ("nn_linear",))
        self.assertEqual(calls["run"][1]["export"], "circle")
        self.assertEqual(calls["run"][1]["output_dir"], "out")
        self.assertTrue(calls["run"][1]["strict"])
        self.assertFalse(calls["run"][1]["emit_plot"])
        self.assertEqual(calls["run"][1]["calibration_limit"], 1)

    def test_wrapper_smoke_list_cases_does_not_require_config(self):
        """Case listing should not require loading a recipe config."""
        calls: dict[str, bool] = {}
        argv = ["inspector.py", "--mode", "wrapper-smoke", "--list-cases"]

        with patch.object(
            inspect_cli,
            "_print_wrapper_smoke_cases",
            lambda: calls.setdefault("listed", True),
        ), patch.object(sys, "argv", argv):
            inspect_cli.main()

        self.assertTrue(calls["listed"])

    def test_wrapper_smoke_list_cases_prints_grouped_case_names(self):
        """Case listing should print grouped case names without tag metadata."""
        fake_cases = [
            SimpleNamespace(name="nn_linear", tags=("nn", "linear")),
            SimpleNamespace(name="nn_conv3d", tags=("nn", "conv3d")),
            SimpleNamespace(
                name="llama_attention_prefill", tags=("llama", "attention")
            ),
            SimpleNamespace(name="qwen3_vl_text_attention", tags=("qwen3_vl", "text")),
            SimpleNamespace(name="custom_case", tags=("custom",)),
        ]

        stdout = io.StringIO()
        with patch.object(
            inspect_cli, "list_cases", lambda: fake_cases
        ), redirect_stdout(stdout):
            inspect_cli._print_wrapper_smoke_cases()

        self.assertEqual(
            stdout.getvalue().splitlines(),
            [
                "Available wrapper smoke cases:",
                "",
                "  nn_linear",
                "  nn_conv3d",
                "",
                "  llama_attention_prefill",
                "",
                "  qwen3_vl_text_attention",
                "",
                "  custom_case",
            ],
        )
        self.assertNotIn("[", stdout.getvalue())
        self.assertNotIn("]", stdout.getvalue())
        self.assertNotIn("nn,linear", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

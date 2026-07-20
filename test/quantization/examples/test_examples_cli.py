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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.examples.evaluate as evaluate_cli
import tico.quantization.examples.inspector as inspect_cli
import tico.quantization.examples.quantize as quantize_cli

import torch


class TestQuantizationExamplesCLI(unittest.TestCase):
    def test_quantize_cli_builds_overrides_and_runs_recipe(self):
        """quantize.py should translate CLI arguments into config overrides."""
        calls: dict[str, Any] = {}

        class FakeRunner:
            """Fake runner that records the loaded config."""

            def run(self, cfg):
                """Record the config passed by the CLI."""
                calls["run"] = cfg

        def fake_load_recipe_config(path, overrides):
            calls["load"] = (path, overrides)
            return {"loaded": True}

        argv = [
            "quantize.py",
            "--config",
            "recipe.yaml",
            "--model",
            "new-model",
            "--device",
            "cpu",
            "--output-dir",
            "out",
            "--set",
            "pipeline.0.enabled=false",
        ]

        with patch.object(
            quantize_cli, "load_recipe_config", fake_load_recipe_config
        ), patch.object(quantize_cli, "QuantizationRunner", FakeRunner), patch.object(
            sys, "argv", argv
        ):
            quantize_cli.main()

        self.assertEqual(
            calls["load"],
            (
                "recipe.yaml",
                [
                    "pipeline.0.enabled=false",
                    "model.name_or_path=new-model",
                    "runtime.device=cpu",
                    "export.output_dir=out",
                ],
            ),
        )
        self.assertEqual(calls["run"], {"loaded": True})

    def test_evaluate_cli_loads_checkpoint_and_overrides_tasks(self):
        """evaluate.py should load checkpoints and route task overrides to the adapter."""
        calls: dict[str, Any] = {}

        class FakeAdapter:
            """Fake adapter used by evaluate CLI tests."""

            family = "llama"

            def load_model(self, ctx):
                """Attach a floating-point model to the context."""
                ctx.model = SimpleNamespace(name="fp")
                return ctx

            def evaluate(self, ctx):
                """Record the evaluated context."""
                calls["ctx"] = ctx

        class FakeCheckpoint:
            """Fake checkpoint with an eval method."""

            def eval(self):
                """Return a marker representing the evaluated checkpoint."""
                return "checkpoint-eval"

        argv = [
            "evaluate.py",
            "--config",
            "recipe.yaml",
            "--checkpoint",
            "model.pt",
            "--tasks",
            "winogrande,arc_easy",
        ]

        with patch.object(
            evaluate_cli,
            "load_recipe_config",
            lambda path, overrides: {
                "model": {"family": "llama", "name_or_path": "model"},
                "runtime": {},
                "evaluation": {},
            },
        ), patch.object(
            evaluate_cli, "set_seed", lambda seed: calls.setdefault("seed", seed)
        ), patch.object(
            evaluate_cli, "get_adapter", lambda family: FakeAdapter()
        ), patch.object(
            evaluate_cli.torch,
            "load",
            lambda path, **kwargs: FakeCheckpoint(),
        ), patch.object(
            sys, "argv", argv
        ):
            evaluate_cli.main()

        self.assertEqual(calls["ctx"].model, "checkpoint-eval")
        self.assertEqual(
            calls["ctx"].cfg["evaluation"]["lm_eval_tasks"],
            "winogrande,arc_easy",
        )

    def test_inspect_cli_dispatches_static_llama_runtime(self):
        """inspector.py should dispatch static LLaMA runtime mode without loading adapters."""
        calls: dict[str, Any] = {}

        def fake_load_recipe_config(path, overrides):
            calls["load"] = (path, overrides)
            return {
                "runtime": {},
                "debug": {
                    "static_llama_runtime": {
                        "model": "new-model",
                        "device": "cuda",
                    }
                },
            }

        argv = [
            "inspector.py",
            "--config",
            "recipe.yaml",
            "--mode",
            "static-llama-runtime",
            "--model",
            "new-model",
            "--device",
            "cuda",
        ]

        with patch.object(
            inspect_cli,
            "load_recipe_config",
            fake_load_recipe_config,
        ), patch.object(inspect_cli, "set_seed", lambda seed: None), patch.object(
            inspect_cli,
            "run_static_llama_runtime",
            lambda cfg: calls.setdefault("cfg", cfg),
        ), patch.object(
            sys, "argv", argv
        ):
            inspect_cli.main()

        self.assertEqual(
            calls["load"],
            (
                "recipe.yaml",
                [
                    "debug.static_llama_runtime.model=new-model",
                    "debug.static_llama_runtime.device=cuda",
                ],
            ),
        )
        self.assertEqual(calls["cfg"].model, "new-model")
        self.assertEqual(calls["cfg"].device, "cuda")

    def test_inspect_cli_dispatches_tied_embedding_smoke(self):
        """inspector.py should dispatch tied embedding smoke mode."""
        calls: dict[str, Any] = {}

        argv = [
            "inspector.py",
            "--config",
            "recipe.yaml",
            "--mode",
            "tied-embedding-smoke",
        ]

        with patch.object(
            inspect_cli,
            "load_recipe_config",
            lambda path, overrides: {
                "runtime": {},
                "debug": {"tied_embedding": {"vocab_size": 7}},
            },
        ), patch.object(inspect_cli, "set_seed", lambda seed: None), patch.object(
            inspect_cli,
            "run_tied_embedding_smoke",
            lambda cfg: calls.setdefault("cfg", cfg),
        ), patch.object(
            sys, "argv", argv
        ):
            inspect_cli.main()

        self.assertEqual(calls["cfg"].vocab_size, 7)

    def test_inspect_cli_trace_mode_loads_adapter_and_runs_trace(self):
        """inspector.py should run the trace helper for trace mode."""
        calls: dict[str, Any] = {}

        class FakeAdapter:
            """Fake adapter for trace mode."""

            def load_model(self, ctx):
                """Attach a dummy model."""
                ctx.model = torch.nn.Linear(2, 2)
                return ctx

            def build_calibration_inputs(self, ctx):
                """Return one fake calibration input."""
                return [torch.ones(1, 2)]

        argv = [
            "inspector.py",
            "--config",
            "recipe.yaml",
            "--mode",
            "trace",
            "--enable-quantization",
            "--interesting-modules",
            "model.layers.0",
        ]

        with patch.object(
            inspect_cli,
            "load_recipe_config",
            lambda path, overrides: {"runtime": {}, "model": {"family": "llama"}},
        ), patch.object(inspect_cli, "set_seed", lambda seed: None), patch.object(
            inspect_cli, "get_adapter", lambda family: FakeAdapter()
        ), patch.object(
            inspect_cli,
            "trace_ptq_parity",
            lambda ctx, enable_quantization, interesting_modules: calls.setdefault(
                "trace", (ctx, enable_quantization, interesting_modules)
            ),
        ), patch.object(
            sys, "argv", argv
        ):
            inspect_cli.main()

        self.assertTrue(calls["trace"][1])
        self.assertEqual(calls["trace"][2], ["model.layers.0"])

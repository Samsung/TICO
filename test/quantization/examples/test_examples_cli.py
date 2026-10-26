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

import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import tico.quantization.examples.compare_side_by_side as side_by_side_cli
import tico.quantization.examples.evaluate as evaluate_cli
import tico.quantization.examples.inspector as inspect_cli
import tico.quantization.examples.quantize as quantize_cli
import tico.quantization.recipes.debug.trace as trace_mod

import torch
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.context import RecipeContext


class _Scale(torch.nn.Module):
    """Multiply by a constant so FP (x) and quantized (2x) outputs differ."""

    def __init__(self, factor: float):
        super().__init__()
        self.first = torch.nn.Identity()
        self.factor = factor

    def forward(self, x):
        return self.first(x) * self.factor


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

        with (
            patch.object(quantize_cli, "load_recipe_config", fake_load_recipe_config),
            patch.object(quantize_cli, "QuantizationRunner", FakeRunner),
            patch.object(sys, "argv", argv),
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

    def test_evaluate_cli_loads_checkpoint_and_selects_top_level_targets(self):
        """evaluate.py should preserve task details and select top-level targets."""
        calls: dict[str, Any] = {}

        class FakeAdapter:
            """Fake adapter used by evaluate CLI tests."""

            family = "llama"
            evaluation_targets = frozenset({"lm_eval", "ppl"})
            evaluation_target_requirements = {"lm_eval": "lm_eval_tasks"}

            def validate_evaluation_config(self, cfg):
                """Validate targets through the shared adapter implementation."""
                ModelAdapter.validate_evaluation_config(cast(ModelAdapter, self), cfg)

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
            "lm_eval,ppl",
        ]

        with (
            patch.object(
                evaluate_cli,
                "load_recipe_config",
                lambda path, overrides: {
                    "model": {"family": "llama", "name_or_path": "model"},
                    "runtime": {},
                    "evaluation": {"enabled": True, "lm_eval_tasks": "mmlu"},
                },
            ),
            patch.object(
                evaluate_cli,
                "set_seed",
                lambda seed: calls.setdefault("seed", seed),
            ),
            patch.object(
                evaluate_cli,
                "resolve_adapter",
                lambda cfg: FakeAdapter(),
            ),
            patch.object(
                evaluate_cli.torch,
                "load",
                lambda path, **kwargs: FakeCheckpoint(),
            ),
            patch.object(
                sys,
                "argv",
                argv,
            ),
        ):
            evaluate_cli.main()

        self.assertEqual(calls["ctx"].model, "checkpoint-eval")
        self.assertEqual(
            calls["ctx"].cfg["evaluation"]["selected_tasks"],
            ["lm_eval", "ppl"],
        )
        self.assertEqual(
            calls["ctx"].cfg["evaluation"]["lm_eval_tasks"],
            "mmlu",
        )

    def test_evaluate_cli_rejects_config_key_alias_as_target(self):
        """evaluate.py should reject non-canonical target names before model loading."""
        calls: list[str] = []

        class FakeAdapter:
            """Fake adapter used to verify early target validation."""

            family = "llama"
            evaluation_targets = frozenset({"lm_eval", "ppl"})
            evaluation_target_requirements = {"lm_eval": "lm_eval_tasks"}

            def validate_evaluation_config(self, cfg):
                """Validate targets through the shared adapter implementation."""
                ModelAdapter.validate_evaluation_config(cast(ModelAdapter, self), cfg)

            def load_model(self, ctx):
                """Record an unexpected model load."""
                calls.append("load")
                return ctx

        argv = [
            "evaluate.py",
            "--config",
            "recipe.yaml",
            "--tasks",
            "lm_eval_tasks",
        ]

        with (
            patch.object(
                evaluate_cli,
                "load_recipe_config",
                lambda path, overrides: {
                    "model": {"family": "llama", "name_or_path": "model"},
                    "runtime": {},
                    "evaluation": {"enabled": True, "lm_eval_tasks": "mmlu"},
                },
            ),
            patch.object(
                evaluate_cli,
                "resolve_adapter",
                lambda cfg: FakeAdapter(),
            ),
            patch.object(
                sys,
                "argv",
                argv,
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported evaluation target",
            ):
                evaluate_cli.main()

        self.assertEqual(calls, [])

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

        with (
            patch.object(
                inspect_cli,
                "load_recipe_config",
                fake_load_recipe_config,
            ),
            patch.object(inspect_cli, "set_seed", lambda seed: None),
            patch.object(
                inspect_cli,
                "run_static_llama_runtime",
                lambda cfg: calls.setdefault("cfg", cfg),
            ),
            patch.object(sys, "argv", argv),
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

    def test_inspect_cli_dispatches_static_gemma4_runtime(self):
        """inspector.py should apply Gemma4 runtime-specific CLI overrides."""
        calls: dict[str, Any] = {}

        def fake_load_recipe_config(path, overrides):
            calls["load"] = (path, overrides)
            return {
                "runtime": {},
                "debug": {
                    "static_gemma4_runtime": {
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
            "static-gemma4-runtime",
            "--model",
            "new-model",
            "--device",
            "cuda",
        ]

        with (
            patch.object(
                inspect_cli,
                "load_recipe_config",
                fake_load_recipe_config,
            ),
            patch.object(inspect_cli, "set_seed", lambda seed: None),
            patch.object(
                inspect_cli,
                "run_static_gemma4_runtime",
                lambda cfg: calls.setdefault("cfg", cfg),
            ),
            patch.object(sys, "argv", argv),
        ):
            inspect_cli.main()

        self.assertEqual(
            calls["load"],
            (
                "recipe.yaml",
                [
                    "debug.static_gemma4_runtime.model=new-model",
                    "debug.static_gemma4_runtime.device=cuda",
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

        with (
            patch.object(
                inspect_cli,
                "load_recipe_config",
                lambda path, overrides: {
                    "runtime": {},
                    "debug": {"tied_embedding": {"vocab_size": 7}},
                },
            ),
            patch.object(inspect_cli, "set_seed", lambda seed: None),
            patch.object(
                inspect_cli,
                "run_tied_embedding_smoke",
                lambda cfg: calls.setdefault("cfg", cfg),
            ),
            patch.object(sys, "argv", argv),
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

        with (
            patch.object(
                inspect_cli,
                "load_recipe_config",
                lambda path, overrides: {"runtime": {}, "model": {"family": "llama"}},
            ),
            patch.object(inspect_cli, "set_seed", lambda seed: None),
            patch.object(inspect_cli, "resolve_adapter", lambda cfg: FakeAdapter()),
            patch.object(
                inspect_cli,
                "trace_ptq_parity",
                lambda ctx, enable_quantization, interesting_modules: calls.setdefault(
                    "trace", (ctx, enable_quantization, interesting_modules)
                ),
            ),
            patch.object(sys, "argv", argv),
        ):
            inspect_cli.main()

        self.assertTrue(calls["trace"][1])
        self.assertEqual(calls["trace"][2], ["model.layers.0"])

    def test_compare_side_by_side_cli_prepares_models_and_runs_real_comparison(self):
        """compare_side_by_side.py should prepare both contexts, then run the real comparison."""
        events: list[str] = []
        calls: dict[str, Any] = {}
        fp_model = _Scale(1.0).train()
        q_model = _Scale(2.0).train()

        def fake_load_recipe_config(path, overrides):
            calls["load"] = (path, overrides)
            events.append("load_config")
            return {
                "model": {"family": "fake"},
                "evaluation": {"enabled": True},
                "export": {"enabled": True},
            }

        class FakeRunner:
            """Fake runner that returns a context holding the quantized model."""

            def run(self, cfg):
                """Record the stage flags seen by the runner and attach q_model."""
                events.append("runner.run")
                calls["runner_flags"] = (
                    cfg["evaluation"]["enabled"],
                    cfg["export"]["enabled"],
                )
                return RecipeContext(cfg=cfg, adapter=None, model=q_model)

        class FakeAdapter:
            """Fake adapter that loads the FP model and builds two calibration batches."""

            def load_model(self, ctx):
                """Attach the FP model to the context."""
                events.append("load_model")
                ctx.model = fp_model
                return ctx

            def build_calibration_inputs(self, ctx):
                """Return two batches with different element counts."""
                events.append("build_calibration_inputs")
                calls["calibration_ctx"] = ctx
                return [torch.tensor([0.0, 2.0]), torch.tensor([0.0, 0.0, 0.0, 8.0])]

        def fake_resolve_adapter(cfg):
            events.append("resolve_adapter")
            return FakeAdapter()

        def fake_print_gpu_memory(label):
            events.append(f"mem:{label}")

        argv = [
            "compare_side_by_side.py",
            "--config",
            "recipe.yaml",
            "--set",
            "pipeline.1.gptq.weight_bits=8",
            "--set",
            "calibration.n_samples=2",
        ]

        stdout = io.StringIO()
        with (
            patch.object(
                side_by_side_cli, "load_recipe_config", fake_load_recipe_config
            ),
            patch.object(
                side_by_side_cli,
                "load_recipe_extensions",
                lambda cfg: events.append("load_extensions"),
            ),
            patch.object(side_by_side_cli, "QuantizationRunner", FakeRunner),
            patch.object(side_by_side_cli, "resolve_adapter", fake_resolve_adapter),
            patch.object(side_by_side_cli, "print_gpu_memory", fake_print_gpu_memory),
            patch.object(trace_mod, "print_gpu_memory", fake_print_gpu_memory),
            patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(stdout),
        ):
            side_by_side_cli.main()

        self.assertEqual(
            calls["load"],
            (
                "recipe.yaml",
                ["pipeline.1.gptq.weight_bits=8", "calibration.n_samples=2"],
            ),
        )
        self.assertEqual(calls["runner_flags"], (False, False))
        self.assertEqual(
            events,
            [
                "load_config",
                "load_extensions",
                "mem:Init",
                "runner.run",
                "mem:After get_quanized_model_context",
                "resolve_adapter",
                "load_model",
                "mem:After get_fp_model_context",
                "resolve_adapter",
                "build_calibration_inputs",
                "mem:Before trace_parity",
                "mem:Batch 0",
                "mem:Batch 1",
            ],
        )
        self.assertFalse(fp_model.training)
        self.assertFalse(q_model.training)
        # Calibration inputs are built from the FP context, not the quantized one.
        self.assertIs(calls["calibration_ctx"].model, fp_model)
        # The CLI reaches the debug-owned comparison, which prints real statistics.
        self.assertIs(side_by_side_cli.trace_parity, trace_mod.trace_parity)
        self.assertEqual(
            [line for line in stdout.getvalue().splitlines() if "mean|diff|" in line],
            [
                "first: mean|diff|=0.00000000, max|diff|=0.00000000, peir=0.000000%",
                ": mean|diff|=1.50000000, max|diff|=8.00000000, peir=100.000000%",
            ],
        )

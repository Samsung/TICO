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

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.recipes.stages.gptq as gptq_mod
import tico.quantization.recipes.stages.ptq as ptq_mod

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.gptq import GPTQStage
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.wrapq.dtypes import DType


class DummyAdapter:
    """Adapter fake used by stage tests."""

    family = "llama"

    def __init__(self):
        self.calibrated: tuple[Any, ...] | None = None
        self.forwarded: tuple[Any, ...] | None = None

    def build_ptq_config(self, ctx, stage_cfg):
        """Return a sentinel PTQ config."""
        return {"config": "ptq", "stage": stage_cfg}

    def calibrate_prepared_model(self, ctx, model, stage_cfg):
        """Record the prepared model passed to PTQ calibration."""
        self.calibrated = (ctx, model, stage_cfg)

    def forward_calibration(self, ctx, model, calibration_inputs, *, desc):
        """Record GPTQ calibration arguments."""
        self.forwarded = (ctx, model, calibration_inputs, desc)


class TestRecipeStages(unittest.TestCase):
    def test_ptq_stage_prepares_calibrates_injects_and_converts(self):
        """PTQStage should run prepare, qparam injection, calibration, and convert."""
        adapter = DummyAdapter()
        source_model = SimpleNamespace(name="source")
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)
        prepared_model = SimpleNamespace(name="prepared")
        calls: dict[str, Any] = {}
        stdout = io.StringIO()

        def fake_prepare(model, config):
            calls["prepare"] = (model, config)
            return prepared_model

        def fake_convert(model):
            calls["convert"] = model
            return "converted"

        def fake_inject(owner, quantizers, verbose=False):
            calls["inject"] = (owner, quantizers, verbose)
            return {"matched": 1, "missed": 0, "unused": 0}

        with patch.object(ptq_mod, "prepare", fake_prepare), patch.object(
            ptq_mod, "convert", fake_convert
        ), patch.object(
            ptq_mod, "find_gptq_quantizers", lambda model: (model, {"linear": object()})
        ), patch.object(
            ptq_mod, "inject_gptq_qparams", fake_inject
        ), patch.object(
            ptq_mod,
            "clear_gptq_quantizers",
            lambda model: calls.setdefault("clear", model),
        ):
            with contextlib.redirect_stdout(stdout):
                result = PTQStage().run(ctx, {"name": "ptq", "verbose": True})

        self.assertEqual(result.model, "converted")
        self.assertIs(calls["prepare"][0], source_model)
        self.assertEqual(calls["prepare"][1]["config"], "ptq")
        self.assertIs(calls["inject"][0], prepared_model)
        self.assertIn("linear", calls["inject"][1])
        self.assertTrue(calls["inject"][2])
        self.assertIs(calls["clear"], prepared_model)
        calibrated = adapter.calibrated
        assert calibrated is not None
        self.assertIs(calibrated[1], prepared_model)
        self.assertIs(calls["convert"], prepared_model)
        self.assertNotIn("=== Effective PTQ overrides ===", stdout.getvalue())
        self.assertNotIn("=== Model after PTQ ===", stdout.getvalue())

    def test_ptq_stage_prints_final_overrides_and_converted_model(self):
        """PTQStage should print final overrides and the converted model on request."""

        class EffectiveOverrideAdapter(DummyAdapter):
            """Adapter fake that returns a real PTQ config."""

            def build_ptq_config(self, ctx, stage_cfg):
                """Return a PTQ config containing an unrelated adapter override."""
                config = PTQConfig()
                config.set_override(
                    (
                        "model",
                        "layers",
                        "0",
                        "self_attn",
                        "q_proj",
                        "weight",
                    ),
                    affine(DType.uint(4)),
                )
                return config

        adapter = EffectiveOverrideAdapter()
        source_model = SimpleNamespace(model=SimpleNamespace(layers=[object()]))
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)
        prepared_model = SimpleNamespace(name="prepared")
        calls: dict[str, Any] = {}

        def fake_prepare(model, config):
            calls["prepare"] = (model, config)
            return prepared_model

        def fake_convert(model):
            calls["convert"] = model
            return "converted-model"

        stage_cfg = {
            "name": "ptq",
            "print_overrides": True,
            "print_model": True,
            "specs": {
                "mx_int8": {
                    "kind": "mx",
                    "elem_format": "int8",
                    "axis": -1,
                }
            },
            "override_policies": [
                {
                    "name": "all_down_proj_outputs_mx",
                    "target": {
                        "component": "text",
                        "layers": "all",
                        "module": "mlp.down_proj",
                        "observers": ["act_out"],
                    },
                    "spec": "mx_int8",
                }
            ],
            "raw_overrides": {
                "model.layers.0.mlp.down_proj.act_out": "int16",
            },
        }

        stdout = io.StringIO()
        with patch.object(ptq_mod, "prepare", fake_prepare), patch.object(
            ptq_mod, "convert", fake_convert
        ), patch.object(ptq_mod, "find_gptq_quantizers", lambda model: (None, None)):
            with contextlib.redirect_stdout(stdout):
                result = PTQStage().run(ctx, stage_cfg)

        output = stdout.getvalue()
        override_path = "model.layers.0.mlp.down_proj.act_out"
        matching_lines = [line for line in output.splitlines() if override_path in line]

        self.assertEqual(result.model, "converted-model")
        self.assertIs(calls["convert"], prepared_model)
        self.assertIsInstance(calls["prepare"][1], PTQConfig)
        self.assertEqual(len(matching_lines), 1)
        self.assertIn("observer=MinMaxObserver", matching_lines[0])
        self.assertIn("dtype=int16", matching_lines[0])
        self.assertNotIn("MXObserver", matching_lines[0])
        self.assertNotIn("elem_format=int8", matching_lines[0])
        self.assertNotIn("__quant_spec_replace_role__", output)
        self.assertNotIn(
            "model.layers.0.self_attn.q_proj.weight",
            output,
        )
        self.assertIn("=== Model after PTQ ===\nconverted-model", output)
        self.assertLess(
            output.index("=== Effective PTQ overrides ==="),
            output.index("=== Model after PTQ ==="),
        )

    def test_ptq_stage_allows_missing_gptq_quantizers(self):
        """PTQStage should continue when no GPTQ quantizers are attached."""
        adapter = DummyAdapter()
        ctx = RecipeContext(cfg={}, adapter=adapter, model=object())
        prepared_model = SimpleNamespace(name="prepared")

        with patch.object(
            ptq_mod, "prepare", lambda model, config: prepared_model
        ), patch.object(ptq_mod, "convert", lambda model: model), patch.object(
            ptq_mod, "find_gptq_quantizers", lambda model: (None, None)
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                PTQStage().run(ctx, {"name": "ptq"})

        calibrated = adapter.calibrated
        assert calibrated is not None
        self.assertIs(calibrated[1], prepared_model)

    def test_ptq_stage_uses_runtime_verbose_for_gptq_injection_summary(self):
        """PTQStage should allow runtime.verbose to print injection summaries."""
        adapter = DummyAdapter()
        ctx = RecipeContext(
            cfg={"runtime": {"verbose": True}},
            adapter=adapter,
            model=object(),
        )
        prepared_model = SimpleNamespace(name="prepared")
        calls: dict[str, Any] = {}

        def fake_inject(owner, quantizers, verbose=False):
            calls["verbose"] = verbose
            return {"matched": 1, "missed": 0, "unused": 0}

        with patch.object(
            ptq_mod, "prepare", lambda model, config: prepared_model
        ), patch.object(ptq_mod, "convert", lambda model: model), patch.object(
            ptq_mod, "find_gptq_quantizers", lambda model: (model, {"linear": object()})
        ), patch.object(
            ptq_mod, "inject_gptq_qparams", fake_inject
        ), patch.object(
            ptq_mod, "clear_gptq_quantizers", lambda model: None
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                PTQStage().run(ctx, {"name": "ptq"})

        self.assertTrue(calls["verbose"])

    def test_gptq_stage_builds_generic_config_and_runs_calibration(self):
        """GPTQStage should build a generic GPTQConfig and run calibration/convert."""
        adapter = DummyAdapter()
        model = torch.nn.Linear(2, 2)
        ctx = RecipeContext(
            cfg={},
            adapter=adapter,
            model=model,
            calibration_inputs=[torch.randn(1, 2)],
        )
        prepared_model = SimpleNamespace(name="gptq-prepared")
        calls: dict[str, Any] = {}

        def fake_prepare(model_arg, config, inplace=False):
            calls["prepare"] = (model_arg, config, inplace)
            return prepared_model

        def fake_convert(model_arg, inplace=False):
            calls["convert"] = (model_arg, inplace)
            return "gptq-converted"

        stage_cfg = {
            "name": "gptq",
            "weight_bits": 4,
            "quantize_lm_head": True,
            "use_orig_model_inference": True,
            "unknown_key": "ignored",
        }

        with patch.object(gptq_mod, "prepare", fake_prepare), patch.object(
            gptq_mod, "convert", fake_convert
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                result = GPTQStage().run(ctx, stage_cfg)

        config = calls["prepare"][1]
        self.assertEqual(result.model, "gptq-converted")
        self.assertEqual(calls["prepare"], (model, config, True))
        self.assertEqual(config.weight_bits, 4)
        self.assertTrue(config.quantize_lm_head)
        self.assertTrue(config.use_orig_model_inference)
        self.assertEqual(
            adapter.forwarded,
            (ctx, prepared_model, ctx.calibration_inputs, "GPTQ calibration"),
        )
        self.assertEqual(calls["convert"], (prepared_model, True))

    def test_gptq_stage_loads_sensitivity_from_nested_path(self):
        """GPTQStage should load sensitivity tensors with mode=load."""
        adapter = DummyAdapter()
        ctx = RecipeContext(
            cfg={},
            adapter=adapter,
            model=torch.nn.Linear(2, 2),
            calibration_inputs=[],
        )
        calls: dict[str, Any] = {}

        def fake_prepare(model, config, inplace=False):
            calls["config"] = config
            return model

        with tempfile.TemporaryDirectory() as tmpdir:
            sensitivity_path = Path(tmpdir) / "sensitivity.pt"
            sensitivity = {"linear": torch.tensor([1.0])}
            torch.save(sensitivity, sensitivity_path)

            with patch.object(gptq_mod, "prepare", fake_prepare), patch.object(
                gptq_mod, "convert", lambda model, inplace=False: model
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    GPTQStage().run(
                        ctx,
                        {
                            "name": "gptq",
                            "mse": "smse",
                            "sensitivity": {
                                "mode": "load",
                                "path": str(sensitivity_path),
                            },
                        },
                    )

        self.assertTrue(
            calls["config"].sensitivity["linear"].equal(sensitivity["linear"])
        )

    def test_gptq_stage_saves_computed_sensitivity_with_mode_save(self):
        """GPTQStage should save computed SMSE sensitivity with mode=save."""
        adapter = DummyAdapter()
        ctx = RecipeContext(
            cfg={},
            adapter=adapter,
            model=torch.nn.Linear(2, 2),
            calibration_inputs=[torch.randn(1, 2)],
        )
        sensitivity = {"linear": torch.tensor([1.0, 2.0])}
        calls: dict[str, Any] = {}

        class FakeSensitivityCalibrator:
            """Fake sensitivity calibrator used by this stage test."""

            def __init__(self, model, calibration_inputs):
                """Record constructor arguments."""
                calls["calibrator"] = (model, calibration_inputs)

            def compute_sensitivity_info(self):
                """Return deterministic fake sensitivity tensors."""
                return sensitivity

        def fake_prepare(model, config, inplace=False):
            calls["config"] = config
            return model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gptq_sensitivity.pt"

            with patch.object(
                gptq_mod, "SensitivityCalibrator", FakeSensitivityCalibrator
            ), patch.object(gptq_mod, "prepare", fake_prepare), patch.object(
                gptq_mod, "convert", lambda model, inplace=False: model
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = GPTQStage().run(
                        ctx,
                        {
                            "name": "gptq",
                            "mse": "smse",
                            "sensitivity": {
                                "mode": "save",
                                "path": str(output_path),
                            },
                        },
                    )

            loaded = torch.load(output_path, map_location="cpu")

        self.assertTrue(loaded["linear"].equal(sensitivity["linear"]))
        self.assertTrue(
            calls["config"].sensitivity["linear"].equal(sensitivity["linear"])
        )
        self.assertEqual(result.artifacts["gptq_sensitivity_path"], str(output_path))
        self.assertEqual(calls["calibrator"], (ctx.model, ctx.calibration_inputs))

    def test_gptq_stage_caches_computed_sensitivity_on_cache_miss(self):
        """GPTQStage should compute and save sensitivity when mode=cache misses."""
        adapter = DummyAdapter()
        sensitivity = {"linear": torch.tensor([3.0])}
        calls: dict[str, Any] = {}

        class FakeSensitivityCalibrator:
            """Fake sensitivity calibrator used by this stage test."""

            def __init__(self, model, calibration_inputs):
                """Record constructor arguments."""
                calls["calibrator"] = (model, calibration_inputs)

            def compute_sensitivity_info(self):
                """Return deterministic fake sensitivity tensors."""
                return sensitivity

        def fake_prepare(model, config, inplace=False):
            calls["config"] = config
            return model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gptq_sensitivity.pt"
            ctx = RecipeContext(
                cfg={},
                adapter=adapter,
                model=torch.nn.Linear(2, 2),
                calibration_inputs=[torch.randn(1, 2)],
            )

            with patch.object(
                gptq_mod, "SensitivityCalibrator", FakeSensitivityCalibrator
            ), patch.object(gptq_mod, "prepare", fake_prepare), patch.object(
                gptq_mod, "convert", lambda model, inplace=False: model
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = GPTQStage().run(
                        ctx,
                        {
                            "name": "gptq",
                            "mse": "smse",
                            "sensitivity": {
                                "mode": "cache",
                                "path": str(output_path),
                            },
                        },
                    )

            loaded = torch.load(output_path, map_location="cpu")

        self.assertTrue(loaded["linear"].equal(sensitivity["linear"]))
        self.assertTrue(
            calls["config"].sensitivity["linear"].equal(sensitivity["linear"])
        )
        self.assertEqual(result.artifacts["gptq_sensitivity_path"], str(output_path))

    def test_gptq_stage_loads_cached_sensitivity_on_cache_hit(self):
        """GPTQStage should load sensitivity when mode=cache finds an existing file."""
        adapter = DummyAdapter()
        calls: dict[str, Any] = {}

        class FakeSensitivityCalibrator:
            """Calibrator that should not be called on cache hits."""

            def __init__(self, model, calibration_inputs):
                """Fail if a cache hit tries to compute sensitivity."""
                raise AssertionError("cache hit should not compute sensitivity")

        def fake_prepare(model, config, inplace=False):
            calls["config"] = config
            return model

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "gptq_sensitivity.pt"
            sensitivity = {"linear": torch.tensor([4.0])}
            torch.save(sensitivity, output_path)
            ctx = RecipeContext(
                cfg={},
                adapter=adapter,
                model=torch.nn.Linear(2, 2),
                calibration_inputs=[torch.randn(1, 2)],
            )

            with patch.object(
                gptq_mod, "SensitivityCalibrator", FakeSensitivityCalibrator
            ), patch.object(gptq_mod, "prepare", fake_prepare), patch.object(
                gptq_mod, "convert", lambda model, inplace=False: model
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    GPTQStage().run(
                        ctx,
                        {
                            "name": "gptq",
                            "mse": "smse",
                            "sensitivity": {
                                "mode": "cache",
                                "path": str(output_path),
                            },
                        },
                    )

        self.assertTrue(
            calls["config"].sensitivity["linear"].equal(sensitivity["linear"])
        )


if __name__ == "__main__":
    unittest.main()

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
import unittest
from unittest.mock import patch

from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.dataset_usage import DatasetUsageError
from tico.quantization.recipes.runner import QuantizationRunner


class _FakeAdapter:
    """Provide the minimal adapter surface required by QuantizationRunner."""

    family = "qwen3_vl"

    def __init__(self):
        self.load_calls = 0
        self.build_calls = 0

    def validate_evaluation_config(self, cfg):
        """Accept all evaluation settings used by these runner tests."""
        return None

    def requires_calibration_inputs(self, cfg):
        """Report no adapter-specific calibration requirement."""
        return False

    def load_model(self, ctx):
        """Record model loading without constructing a real model."""
        self.load_calls += 1
        ctx.model = object()
        return ctx

    def build_calibration_inputs(self, ctx):
        """Record calibration construction."""
        self.build_calls += 1
        return [{"input_ids": "sample"}]

    def evaluate(self, ctx):
        """Skip evaluation in the runner unit test."""
        return None

    def export(self, ctx):
        """Skip export in the runner unit test."""
        return None


class _FakeStage:
    """Expose configurable calibration metadata without running an algorithm."""

    def __init__(self, requires_calibration_inputs):
        self.requires_calibration_inputs = requires_calibration_inputs

    def run(self, ctx, stage_cfg):
        """Return the recipe context unchanged."""
        return ctx


class TestRunnerDatasetUsage(unittest.TestCase):
    """Test early data validation and lazy calibration construction."""

    @staticmethod
    def _base_config() -> dict:
        return {
            "model": {"family": "qwen3_vl", "name_or_path": "fake/model"},
            "runtime": {"print_config": False, "seed": 1},
            "calibration": {
                "datasets": [{"dataset": "vqav2", "split": "testdev", "n_samples": 1}],
                "seq_len": 16,
            },
            "pipeline": [],
            "evaluation": {"enabled": False},
            "export": {"enabled": False},
        }

    def test_evaluation_only_run_skips_calibration_loading(self):
        adapter = _FakeAdapter()
        cfg = self._base_config()

        with patch(
            "tico.quantization.recipes.runner.get_adapter",
            return_value=adapter,
        ), contextlib.redirect_stdout(io.StringIO()):
            result = QuantizationRunner().run(cfg)

        self.assertIsInstance(result, RecipeContext)
        self.assertEqual(adapter.load_calls, 1)
        self.assertEqual(adapter.build_calls, 0)
        self.assertEqual(result.calibration_inputs, [])
        self.assertEqual(cfg["calibration"]["resolved_sources"], [])

    def test_disabled_calibration_stage_does_not_build_inputs(self):
        """A disabled calibration-dependent stage must not load calibration data."""
        adapter = _FakeAdapter()
        cfg = self._base_config()
        cfg["pipeline"] = [{"name": "fake", "enabled": False}]
        stage = _FakeStage(requires_calibration_inputs=True)

        with patch(
            "tico.quantization.recipes.runner.get_adapter",
            return_value=adapter,
        ), patch(
            "tico.quantization.recipes.runner.get_stage",
            return_value=stage,
        ) as get_stage_mock, contextlib.redirect_stdout(
            io.StringIO()
        ):
            result = QuantizationRunner().run(cfg)

        get_stage_mock.assert_not_called()
        self.assertEqual(adapter.load_calls, 1)
        self.assertEqual(adapter.build_calls, 0)
        self.assertEqual(result.calibration_inputs, [])

    def test_unsafe_calibration_fails_before_model_loading(self):
        adapter = _FakeAdapter()
        cfg = self._base_config()
        cfg["pipeline"] = [{"name": "fake", "enabled": True}]
        stage = _FakeStage(requires_calibration_inputs=True)

        with patch(
            "tico.quantization.recipes.runner.get_adapter",
            return_value=adapter,
        ), patch(
            "tico.quantization.recipes.runner.get_stage",
            return_value=stage,
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            with self.assertRaises(DatasetUsageError):
                QuantizationRunner().run(cfg)

        self.assertEqual(adapter.load_calls, 0)
        self.assertEqual(adapter.build_calls, 0)

    def test_calibration_is_built_when_an_enabled_stage_requires_it(self):
        adapter = _FakeAdapter()
        cfg = self._base_config()
        cfg["calibration"]["datasets"][0]["split"] = "train"
        cfg["pipeline"] = [{"name": "fake", "enabled": True}]
        stage = _FakeStage(requires_calibration_inputs=True)

        with patch(
            "tico.quantization.recipes.runner.get_adapter",
            return_value=adapter,
        ), patch(
            "tico.quantization.recipes.runner.get_stage",
            return_value=stage,
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            result = QuantizationRunner().run(cfg)

        self.assertEqual(adapter.load_calls, 1)
        self.assertEqual(adapter.build_calls, 1)
        self.assertEqual(result.calibration_inputs, [{"input_ids": "sample"}])


if __name__ == "__main__":
    unittest.main()

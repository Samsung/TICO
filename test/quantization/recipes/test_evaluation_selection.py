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

import unittest
from typing import Any, Mapping, Sequence

import torch

from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.evaluation.selection import (
    get_mapping_evaluation_config,
    get_selected_evaluation_targets,
    parse_evaluation_targets,
    should_run_evaluation,
    should_run_mapping_evaluation,
    validate_adapter_evaluation_config,
)


class _FakeAdapter(ModelAdapter):
    """Minimal adapter used to test common evaluation target validation."""

    family = "fake"
    evaluation_targets = frozenset({"alpha", "beta"})
    evaluation_target_requirements = {"alpha": "alpha_tasks"}

    def load_model(self, ctx: RecipeContext) -> RecipeContext:
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError

    def build_calibration_inputs(self, ctx: RecipeContext) -> list[Any]:
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError

    def forward_calibration(
        self,
        ctx: RecipeContext,
        model: torch.nn.Module,
        calibration_inputs: Sequence[Any],
        *,
        desc: str,
    ) -> None:
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError

    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError

    def build_ptq_config(
        self,
        ctx: RecipeContext,
        stage_cfg: Mapping[str, Any],
    ):
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError

    def evaluate(self, ctx: RecipeContext) -> None:
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError

    def export(self, ctx: RecipeContext) -> None:
        """Satisfy the abstract adapter contract for validation-only tests."""
        raise NotImplementedError


class _LegacyDuckAdapter:
    """Legacy adapter that predates explicit evaluation target validation."""

    family = "legacy"


class TestEvaluationSelection(unittest.TestCase):
    """Tests for common top-level evaluation target selection."""

    def test_parse_evaluation_targets_preserves_order(self):
        """CLI parsing should trim names and preserve their declared order."""
        self.assertEqual(
            parse_evaluation_targets("mmmu, ppl"),
            ["mmmu", "ppl"],
        )

    def test_parse_evaluation_targets_rejects_empty_names(self):
        """CLI parsing should reject empty entries instead of silently skipping them."""
        for raw in ("", "mmmu,", "mmmu,,ppl"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "empty|at least one"):
                    parse_evaluation_targets(raw)

    def test_parse_evaluation_targets_rejects_duplicates(self):
        """CLI parsing should reject duplicate canonical target names."""
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_evaluation_targets("ppl,ppl")

    def test_selected_targets_distinguish_absent_and_empty(self):
        """An absent selector should preserve defaults while an empty list runs nothing."""
        self.assertIsNone(get_selected_evaluation_targets({}))
        self.assertEqual(
            get_selected_evaluation_targets({"selected_tasks": []}),
            (),
        )

    def test_should_run_evaluation_uses_defaults_without_selector(self):
        """Legacy enabled behavior should remain unchanged without selected_tasks."""
        self.assertTrue(should_run_evaluation({}, "ppl", default_enabled=True))
        self.assertFalse(should_run_evaluation({}, "ppl", default_enabled=False))

    def test_should_run_evaluation_uses_exclusive_allow_list(self):
        """A configured selector should disable every unlisted evaluation target."""
        eval_cfg = {"selected_tasks": ["mmmu", "ppl"]}

        self.assertTrue(should_run_evaluation(eval_cfg, "ppl", default_enabled=False))
        self.assertFalse(should_run_evaluation(eval_cfg, "vqa", default_enabled=True))

    def test_mapping_selection_overrides_nested_enabled(self):
        """An explicit target selection should override the nested enabled flag."""
        eval_cfg = {
            "selected_tasks": ["mmmu"],
            "mmmu": {"enabled": False},
            "ppl": {"enabled": True},
        }

        self.assertTrue(should_run_mapping_evaluation(eval_cfg, "mmmu"))
        self.assertFalse(should_run_mapping_evaluation(eval_cfg, "ppl"))

    def test_mapping_selection_preserves_legacy_enabled_behavior(self):
        """Mapping-backed targets should still use nested enabled without a selector."""
        eval_cfg = {
            "mmmu": {"enabled": True},
            "ppl": {"enabled": False},
        }

        self.assertTrue(should_run_mapping_evaluation(eval_cfg, "mmmu"))
        self.assertFalse(should_run_mapping_evaluation(eval_cfg, "ppl"))

    def test_missing_mapping_uses_default_configuration(self):
        """A selected mapping-backed target may use its implementation defaults."""
        self.assertEqual(
            get_mapping_evaluation_config({}, "mmmu"),
            {},
        )

    def test_adapter_rejects_unsupported_selected_target(self):
        """Adapters should reject unknown canonical targets before model execution."""
        cfg = {
            "evaluation": {
                "enabled": True,
                "selected_tasks": ["gamma"],
            }
        }

        with self.assertRaisesRegex(ValueError, "Unsupported evaluation target"):
            _FakeAdapter().validate_evaluation_config(cfg)

    def test_adapter_requires_selected_target_details(self):
        """Adapters should reject selected targets whose required details are empty."""
        cfg = {
            "evaluation": {
                "enabled": True,
                "selected_tasks": ["alpha"],
                "alpha_tasks": [],
            }
        }

        with self.assertRaisesRegex(
            ValueError,
            "requires non-empty evaluation.alpha_tasks",
        ):
            _FakeAdapter().validate_evaluation_config(cfg)

    def test_disabled_evaluation_ignores_selected_targets(self):
        """Disabled evaluation should not validate or execute selected targets."""
        cfg = {
            "evaluation": {
                "enabled": False,
                "selected_tasks": ["gamma"],
            }
        }

        _FakeAdapter().validate_evaluation_config(cfg)

    def test_legacy_adapter_remains_compatible_without_selector(self):
        """Legacy adapters should continue to run when no selector is configured."""
        validate_adapter_evaluation_config(
            _LegacyDuckAdapter(),
            {"evaluation": {"enabled": True}},
        )

    def test_legacy_adapter_rejects_explicit_selector(self):
        """Legacy adapters should fail clearly when selected_tasks is requested."""
        for selected_tasks in (["alpha"], []):
            with self.subTest(selected_tasks=selected_tasks):
                with self.assertRaisesRegex(
                    TypeError,
                    "must implement validate_evaluation_config",
                ):
                    validate_adapter_evaluation_config(
                        _LegacyDuckAdapter(),
                        {
                            "evaluation": {
                                "enabled": True,
                                "selected_tasks": selected_tasks,
                            }
                        },
                    )


if __name__ == "__main__":
    unittest.main()

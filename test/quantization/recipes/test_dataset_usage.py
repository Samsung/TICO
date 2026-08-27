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

import copy
import unittest
import warnings

from tico.quantization.recipes.data.dataset_usage import (
    CALIBRATION_ROLE,
    dataset_usages_overlap,
    DatasetUsageError,
    EVALUATION_ROLE,
    resolve_dataset_usage,
    validate_recipe_dataset_usage,
    validate_single_dataset_usage,
)


class TestRoleAwareDatasetResolution(unittest.TestCase):
    """Test canonical identities and role-specific defaults."""

    def test_vqav2_has_independent_role_defaults(self):
        calibration = resolve_dataset_usage(
            dataset="vqav2",
            role=CALIBRATION_ROLE,
            consumer="test",
        )
        evaluation = resolve_dataset_usage(
            dataset="HuggingFaceM4/VQAv2",
            role=EVALUATION_ROLE,
            consumer="test",
        )

        self.assertEqual(calibration.split, "train")
        self.assertEqual(evaluation.split, "validation")
        self.assertEqual(calibration.canonical_id, evaluation.canonical_id)
        self.assertEqual(calibration.canonical_id, "lmms-lab/VQAv2-FewShot")
        self.assertEqual(calibration.config, "full")

    def test_mmlu_calibration_defaults_to_auxiliary_train(self):
        usage = resolve_dataset_usage(
            dataset="cais/mmlu",
            role=CALIBRATION_ROLE,
            consumer="test",
        )

        self.assertEqual(usage.split, "auxiliary_train")
        self.assertEqual(usage.config, "all")

    def test_wikitext_aliases_share_one_identity(self):
        left = resolve_dataset_usage(
            dataset="wikitext2",
            role=CALIBRATION_ROLE,
            split="train",
            consumer="left",
        )
        right = resolve_dataset_usage(
            dataset="Salesforce/wikitext",
            role=EVALUATION_ROLE,
            split="train",
            config="wikitext-2-raw-v1",
            consumer="right",
        )

        self.assertTrue(dataset_usages_overlap(left, right))

    def test_unknown_dataset_requires_explicit_split(self):
        with self.assertRaisesRegex(ValueError, "specify the split explicitly"):
            resolve_dataset_usage(
                dataset="example/custom-corpus",
                role=CALIBRATION_ROLE,
                consumer="test",
            )

    def test_missing_role_default_requires_explicit_split(self):
        with self.assertRaisesRegex(
            DatasetUsageError,
            "no default split.*Specify the split explicitly",
        ):
            resolve_dataset_usage(
                dataset="videomme",
                role=CALIBRATION_ROLE,
                consumer="test",
            )

    def test_unregistered_calibration_dataset_fails_by_default(self):
        usage = resolve_dataset_usage(
            dataset="example/custom-corpus",
            role=CALIBRATION_ROLE,
            split="train",
            consumer="test",
        )

        with self.assertRaisesRegex(
            DatasetUsageError,
            "no registered calibration policy",
        ):
            validate_single_dataset_usage(usage)

    def test_unregistered_calibration_dataset_requires_explicit_opt_in(self):
        usage = resolve_dataset_usage(
            dataset="example/custom-corpus",
            role=CALIBRATION_ROLE,
            split="train",
            consumer="test",
        )

        with self.assertWarnsRegex(
            RuntimeWarning,
            "UNREGISTERED CALIBRATION DATASET ENABLED",
        ):
            validate_single_dataset_usage(
                usage,
                allow_unregistered_dataset=True,
            )

    def test_videomme_is_evaluation_only(self):
        usage = resolve_dataset_usage(
            dataset="lmms-lab/Video-MME",
            role=CALIBRATION_ROLE,
            split="test",
            consumer="test",
        )

        with self.assertRaises(DatasetUsageError):
            validate_single_dataset_usage(usage)

    def test_videomme_text_alias_is_evaluation_only(self):
        """The future text-only registry key must retain Video-MME policy."""
        usage = resolve_dataset_usage(
            dataset="videomme_text",
            role=CALIBRATION_ROLE,
            split="test",
            consumer="test",
        )

        self.assertEqual(usage.policy_key, "videomme")
        with self.assertRaises(DatasetUsageError):
            validate_single_dataset_usage(usage)

    def test_textvqa_encoder_alias_uses_canonical_identity(self):
        """The redirected TextVQA repository must share the registered policy."""
        usage = resolve_dataset_usage(
            dataset="lmms-lab-encoder/textvqa",
            role=EVALUATION_ROLE,
            consumer="test",
        )

        self.assertEqual(usage.policy_key, "textvqa")
        self.assertEqual(usage.split, "validation")

    def test_llava_bench_train_split_is_not_calibration_safe(self):
        usage = resolve_dataset_usage(
            dataset="lmms-lab/llava-bench-in-the-wild",
            role=CALIBRATION_ROLE,
            split="train",
            consumer="test",
        )

        with self.assertRaisesRegex(DatasetUsageError, "not allowed for role"):
            validate_single_dataset_usage(usage)

    def test_formatted_coco_source_is_evaluation_only(self):
        usage = resolve_dataset_usage(
            dataset="lmms-lab/COCO-Caption2017",
            role=CALIBRATION_ROLE,
            split="val",
            consumer="test",
        )

        with self.assertRaisesRegex(DatasetUsageError, "not allowed for role"):
            validate_single_dataset_usage(usage)


class TestRecipeDatasetValidation(unittest.TestCase):
    """Test complete-recipe safety validation and provenance."""

    @staticmethod
    def _qwen_config() -> dict:
        return {
            "model": {"family": "qwen3_vl", "name_or_path": "model"},
            "calibration": {
                "datasets": [
                    {"dataset": "vqav2", "split": "train", "n_samples": 8},
                    {"dataset": "wikitext2", "split": "train", "n_samples": 8},
                ],
                "seq_len": 128,
            },
            "pipeline": [{"name": "ptq", "enabled": True}],
            "evaluation": {
                "enabled": True,
                "vlm_tasks": ["vqav2"],
                "ppl": {"enabled": True, "dataset": "wikitext2", "split": "test"},
            },
        }

    def test_safe_vqav2_and_wikitext_combinations_pass(self):
        cfg = self._qwen_config()

        calibration, evaluation = validate_recipe_dataset_usage(
            cfg,
            include_calibration=True,
            emit_summary=False,
        )

        self.assertEqual(len(calibration), 2)
        self.assertEqual(len(evaluation), 2)
        self.assertFalse(cfg["calibration"]["transductive"])
        self.assertEqual(len(cfg["calibration"]["resolved_sources"]), 2)
        self.assertEqual(len(cfg["evaluation"]["resolved_sources"]), 2)

    def test_single_dataset_uses_calibration_role_default(self):
        cfg = self._qwen_config()
        cfg["calibration"] = {"dataset": "vqav2", "n_samples": 4}
        cfg["evaluation"]["enabled"] = False

        calibration, _ = validate_recipe_dataset_usage(
            cfg,
            include_calibration=True,
            emit_summary=False,
        )

        self.assertEqual(calibration[0].split, "train")

    def test_compact_mixed_dataset_specs_are_resolved(self):
        cfg = self._qwen_config()
        cfg["calibration"] = {
            "datasets": "vqav2:train:4,wikitext2:train:5",
            "n_samples": 8,
        }
        cfg["evaluation"]["enabled"] = False

        calibration, _ = validate_recipe_dataset_usage(
            cfg,
            include_calibration=True,
            emit_summary=False,
        )

        self.assertEqual(
            [(usage.policy_key, usage.split, usage.n_samples) for usage in calibration],
            [("vqav2", "train", 4), ("wikitext2", "train", 5)],
        )

    def test_vqav2_validation_calibration_fails(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"][0]["split"] = "validation"

        with self.assertRaisesRegex(
            DatasetUsageError,
            "validation.*not calibration-safe",
        ):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_vqav2_testdev_calibration_fails_without_evaluation(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {"dataset": "vqav2", "split": "testdev", "n_samples": 8}
        ]
        cfg["evaluation"]["enabled"] = False

        with self.assertRaisesRegex(DatasetUsageError, "testdev.*not calibration-safe"):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_wikitext_test_overlap_fails(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {"dataset": "Salesforce/wikitext", "split": "test", "n_samples": 8}
        ]
        cfg["evaluation"]["vlm_tasks"] = []

        with self.assertRaisesRegex(
            DatasetUsageError,
            "calibration/evaluation overlap",
        ):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_alias_overlap_is_detected(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {
                "dataset": "lmms-lab/VQAv2",
                "split": "validation",
                "n_samples": 8,
            }
        ]
        cfg["evaluation"]["ppl"]["enabled"] = False

        with self.assertRaisesRegex(
            DatasetUsageError,
            "calibration/evaluation overlap",
        ):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_mmmu_pro_vision_test_calibration_fails(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {
                "dataset": "mmmu_pro_vision",
                "split": "test",
                "n_samples": 8,
            }
        ]
        cfg["evaluation"]["enabled"] = False

        with self.assertRaisesRegex(DatasetUsageError, "not allowed for role"):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_mmlu_default_is_safe_for_mmlu_evaluation(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [{"dataset": "mmlu", "n_samples": 8}]
        cfg["evaluation"] = {
            "enabled": True,
            "mmlu": {"enabled": True, "n_samples": 16},
        }

        calibration, evaluation = validate_recipe_dataset_usage(
            cfg,
            include_calibration=True,
            emit_summary=False,
        )

        self.assertEqual(calibration[0].split, "auxiliary_train")
        self.assertEqual(evaluation[0].split, "test")
        self.assertEqual(evaluation[1].role, "few_shot")
        self.assertEqual(evaluation[1].split, "dev")
        self.assertEqual(evaluation[1].n_samples, 5)

    def test_global_evaluation_sample_count_is_recorded_for_vqa(self):
        cfg = self._qwen_config()
        cfg["evaluation"]["n_samples"] = 12
        cfg["evaluation"]["ppl"]["enabled"] = False

        _, evaluation = validate_recipe_dataset_usage(
            cfg,
            include_calibration=True,
            emit_summary=False,
        )

        self.assertEqual(evaluation[0].consumer, "evaluation.vlm_tasks:vqav2")
        self.assertEqual(evaluation[0].n_samples, 12)

    def test_evaluation_role_violation_fails_before_loading(self):
        """Central validation must reject a calibration-only source for evaluation."""
        cfg = self._qwen_config()
        cfg["evaluation"] = {
            "enabled": True,
            "ppl": {
                "enabled": True,
                "dataset": "alpaca",
                "split": "train",
            },
        }

        with self.assertRaisesRegex(DatasetUsageError, "not allowed for role"):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_overlap_override_does_not_waive_evaluation_role_errors(self):
        cfg = self._qwen_config()
        cfg["calibration"]["allow_benchmark_overlap"] = True
        cfg["evaluation"] = {
            "enabled": True,
            "ppl": {
                "enabled": True,
                "dataset": "alpaca",
                "split": "train",
            },
        }

        with self.assertRaisesRegex(DatasetUsageError, "not allowed for role"):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_selected_tasks_excludes_inactive_overlap(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {"dataset": "wikitext2", "split": "train", "n_samples": 8}
        ]
        cfg["evaluation"] = {
            "enabled": True,
            "selected_tasks": ["ppl"],
            "vlm_tasks": ["vqav2"],
            "ppl": {"enabled": True, "dataset": "wikitext2", "split": "test"},
        }

        _, evaluation = validate_recipe_dataset_usage(
            cfg,
            include_calibration=True,
            emit_summary=False,
        )

        self.assertEqual([usage.consumer for usage in evaluation], ["evaluation.ppl"])

    def test_skipped_calibration_is_not_validated_or_resolved(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {"dataset": "videomme", "split": "test", "n_samples": 8}
        ]
        cfg["pipeline"] = []
        cfg["evaluation"]["enabled"] = False

        calibration, evaluation = validate_recipe_dataset_usage(
            cfg,
            include_calibration=False,
            emit_summary=False,
        )

        self.assertEqual(calibration, [])
        self.assertEqual(evaluation, [])
        self.assertEqual(cfg["calibration"]["resolved_sources"], [])

    def test_transductive_override_warns_and_records_provenance(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {"dataset": "vqav2", "split": "validation", "n_samples": 8}
        ]
        cfg["calibration"]["allow_benchmark_overlap"] = True
        cfg["evaluation"]["ppl"]["enabled"] = False

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

        self.assertTrue(captured)
        self.assertIn("TRANSDUCTIVE DATASET USAGE ENABLED", str(captured[0].message))
        self.assertTrue(cfg["calibration"]["transductive"])
        self.assertTrue(cfg["calibration"]["dataset_usage_warnings"])

    def test_unregistered_recipe_dataset_fails_by_default(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {
                "dataset": "example/custom-corpus",
                "split": "train",
                "n_samples": 8,
            }
        ]
        cfg["evaluation"]["enabled"] = False

        with self.assertRaisesRegex(
            DatasetUsageError,
            "no registered calibration policy",
        ):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_unregistered_recipe_dataset_opt_in_records_unverified_provenance(self):
        cfg = self._qwen_config()
        cfg["calibration"] = {
            "datasets": [
                {
                    "dataset": "example/custom-corpus",
                    "split": "train",
                    "n_samples": 8,
                }
            ],
            "allow_unregistered_dataset": True,
        }
        cfg["evaluation"]["enabled"] = False

        with self.assertWarnsRegex(
            RuntimeWarning,
            "UNREGISTERED CALIBRATION DATASET ENABLED",
        ):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

        self.assertTrue(cfg["calibration"]["unverified"])
        self.assertFalse(cfg["calibration"]["transductive"])
        self.assertTrue(cfg["calibration"]["dataset_usage_warnings"])

    def test_target_inclusion_is_rejected_by_default(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"] = [
            {
                "dataset": "mmlu",
                "n_samples": 8,
                "include_targets": True,
            }
        ]
        cfg["evaluation"]["enabled"] = False

        with self.assertRaisesRegex(DatasetUsageError, "gold targets"):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_single_dataset_target_inclusion_is_rejected(self):
        cfg = self._qwen_config()
        cfg["calibration"] = {
            "dataset": "mmlu",
            "include_targets": True,
            "n_samples": 8,
        }
        cfg["evaluation"]["enabled"] = False

        with self.assertRaisesRegex(DatasetUsageError, "gold targets"):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_target_inclusion_cannot_be_enabled_by_overlap_override(self):
        cfg = self._qwen_config()
        cfg["calibration"] = {
            "datasets": [
                {
                    "dataset": "mmlu",
                    "n_samples": 8,
                    "include_targets": True,
                }
            ],
            "allow_benchmark_overlap": True,
        }
        cfg["evaluation"]["enabled"] = False

        with self.assertRaisesRegex(
            DatasetUsageError,
            "does not permit target leakage",
        ):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

    def test_validation_does_not_mutate_config_on_failure(self):
        cfg = self._qwen_config()
        cfg["calibration"]["datasets"][0]["split"] = "testdev"
        before = copy.deepcopy(cfg)

        with self.assertRaises(DatasetUsageError):
            validate_recipe_dataset_usage(
                cfg,
                include_calibration=True,
                emit_summary=False,
            )

        self.assertEqual(cfg, before)


if __name__ == "__main__":
    unittest.main()

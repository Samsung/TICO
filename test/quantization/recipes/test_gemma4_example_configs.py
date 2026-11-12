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

import copy
import unittest
from pathlib import Path

from tico.quantization.recipes.adapters import get_adapter, resolve_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.evaluation.selection import (
    validate_adapter_evaluation_config,
)
from tico.quantization.recipes.stages import get_stage

_CONFIG_DIR = (
    Path(__file__).resolve().parents[3]
    / "tico"
    / "quantization"
    / "examples"
    / "configs"
)


class TestGemma4ExampleConfigs(unittest.TestCase):
    """Tests for the canonical Gemma4 quantize/evaluate/export presets."""

    @staticmethod
    def _load(name: str):
        """Load one committed Gemma4 recipe config."""
        return load_recipe_config(_CONFIG_DIR / name)

    def test_all_configs_select_the_gemma4_adapter(self):
        """Every canonical preset should select the registered Gemma4 family."""
        for name in (
            "gemma4_quantize.yaml",
            "gemma4_eval_suite.yaml",
            "gemma4_export.yaml",
            "gemma4_e4b_quantize.yaml",
            "gemma4_e4b_eval_suite.yaml",
            "gemma4_e4b_export.yaml",
        ):
            with self.subTest(name=name):
                cfg = self._load(name)
                self.assertEqual(cfg["model"]["family"], "gemma4")
                self.assertEqual(cfg["runtime"]["seed"], 42)

    def test_quantize_config_runs_ptq_and_saves_a_checkpoint(self):
        """The quantize preset should use mixed calibration and export PTQ state."""
        cfg = self._load("gemma4_quantize.yaml")

        self.assertEqual([stage["name"] for stage in cfg["pipeline"]], ["gptq", "ptq"])
        self.assertTrue(cfg["pipeline"][1]["enabled"])
        self.assertEqual(
            [entry["dataset"] for entry in cfg["calibration"]["datasets"]],
            ["vqav2", "wikitext2"],
        )
        self.assertTrue(cfg["export"]["enabled"])
        self.assertIn("ptq_checkpoint", cfg["export"]["artifacts"])

    def test_eval_config_does_not_run_quantization_stages(self):
        """The evaluation preset should only dispatch the Gemma4 judge suite."""
        cfg = self._load("gemma4_eval_suite.yaml")
        llava_bench = cfg["evaluation"]["llava_bench"]

        self.assertEqual(cfg["pipeline"], [])
        self.assertTrue(cfg["evaluation"]["enabled"])
        self.assertTrue(llava_bench["enabled"])
        self.assertEqual(llava_bench["mode"], "judge")
        self.assertEqual(llava_bench["resized_height"], 896)
        self.assertEqual(llava_bench["resized_width"], 896)
        self.assertFalse(cfg["export"]["enabled"])

        self.assertEqual(cfg["evaluation"]["vlm_tasks"], ["vqav2"])
        self.assertTrue(cfg["evaluation"]["coco"])
        self.assertTrue(cfg["evaluation"]["videomme"]["enabled"])
        self.assertTrue(cfg["evaluation"]["mmlu"]["enabled"])
        self.assertTrue(cfg["evaluation"]["hellaswag"]["enabled"])
        self.assertTrue(cfg["evaluation"]["mmmu"]["enabled"])
        self.assertTrue(cfg["evaluation"]["ppl"]["enabled"])
        self.assertEqual(
            cfg["evaluation"]["mmmu"]["dataset"],
            "MMMU/MMMU_Pro",
        )
        self.assertEqual(cfg["evaluation"]["n_samples"], 1000)

    def test_export_config_requests_circle_per_layer_artifact(self):
        """The export preset should directly request static Circle artifacts."""
        cfg = self._load("gemma4_export.yaml")

        self.assertEqual(cfg["pipeline"], [])
        self.assertFalse(cfg["evaluation"]["enabled"])
        self.assertTrue(cfg["export"]["enabled"])
        self.assertEqual(cfg["export"]["artifacts"], ["circle_per_layer"])
        self.assertEqual(cfg["export"]["max_seq_len"], 2048)
        self.assertTrue(cfg["export"]["prefill_decode"])

    def test_e4b_presets_validate_with_the_shared_adapter(self):
        """E4B recipes must resolve the same adapter and registered stages as E2B."""
        for name in (
            "gemma4_e4b_quantize.yaml",
            "gemma4_e4b_eval_suite.yaml",
            "gemma4_e4b_export.yaml",
        ):
            with self.subTest(name=name):
                cfg = self._load(name)
                self.assertEqual(cfg["model"]["name_or_path"], "google/gemma-4-E4B-it")
                self.assertIs(resolve_adapter(cfg), get_adapter("gemma4"))
                validate_adapter_evaluation_config(resolve_adapter(cfg), cfg)
                for stage in cfg["pipeline"]:
                    self.assertIsNotNone(get_stage(stage["name"]))
                self.assertEqual(
                    cfg["model_args"]["vision"]["profile"], "gemma4_66x36_264"
                )

    def test_e4b_quantize_and_export_keep_the_existing_workflow(self):
        """The variant changes model dimensions without changing pipeline contracts."""
        quantize = self._load("gemma4_e4b_quantize.yaml")
        export = self._load("gemma4_e4b_export.yaml")
        e4b_pipeline = copy.deepcopy(quantize["pipeline"])
        # Cache placement is a memory setting, not a pipeline contract: E4B
        # float32 weights leave no room for calibration caches on a 48 GB GPU.
        self.assertIs(e4b_pipeline[0].pop("move_cache_to_cpu"), True)
        self.assertEqual(e4b_pipeline, self._load("gemma4_quantize.yaml")["pipeline"])
        self.assertEqual(quantize["export"]["artifacts"], ["ptq_checkpoint"])
        self.assertEqual(export["pipeline"], [])
        self.assertEqual(export["export"]["artifacts"], ["circle_per_layer"])
        self.assertTrue(export["export"]["prefill_decode"])
        self.assertEqual(export["export"]["vision"]["granularity"], "both")

    def test_gptq_and_ptq_weight_grids_match_for_reused_modules(self):
        """GPTQ qparams are reused by PTQ only when both use the same weight grid."""
        for name in ("gemma4_quantize.yaml", "gemma4_e4b_quantize.yaml"):
            with self.subTest(name=name):
                stages = {
                    stage["name"]: stage for stage in self._load(name)["pipeline"]
                }
                self.assertEqual(stages["gptq"]["weight_bits"], 4)
                self.assertEqual(stages["ptq"]["linear_weight"], "uint4")
                self.assertEqual(stages["ptq"]["vision_patch_embed_weight"], "uint4")

    def test_e4b_output_paths_are_separate_from_e2b(self):
        """Variant presets must never overwrite E2B checkpoints or quality reports."""
        for purpose in ("quantize", "export", "eval_suite"):
            with self.subTest(purpose=purpose):
                e2b = self._load(f"gemma4_{purpose}.yaml")
                e4b = self._load(f"gemma4_e4b_{purpose}.yaml")
                self.assertNotEqual(
                    e2b["export"]["output_dir"], e4b["export"]["output_dir"]
                )
                self.assertIn("gemma4_e4b", e4b["export"]["output_dir"])
        cfg = self._load("gemma4_e4b_eval_suite.yaml")
        self.assertEqual(
            cfg["evaluation"]["output_dir"], "./out/gemma4_e4b_eval_suite/evaluation"
        )
        llava = cfg["evaluation"]["llava_bench"]
        self.assertEqual(llava["candidate_label"], "gemma4_e4b")
        for field in ("dir", "answers", "reviews", "summary"):
            with self.subTest(field=field):
                value = llava["output"][field]
                self.assertTrue(value.startswith("./out/gemma4_e4b_eval_suite/"))
                self.assertNotIn("gemma4_e2b", value)

    def test_e4b_eval_and_static_runtime_share_vision_geometry(self):
        """E4B static debugging and eager judge inputs select the common image profile."""
        cfg = self._load("static_gemma4_e4b_runtime.yaml")["debug"][
            "static_gemma4_runtime"
        ]
        self.assertEqual(cfg["model"], "google/gemma-4-E4B-it")
        self.assertEqual(cfg["vision_profile"], "gemma4_66x36_264")
        self.assertEqual(cfg["max_seq"], 2048)
        llava = self._load("gemma4_e4b_eval_suite.yaml")["evaluation"]["llava_bench"]
        self.assertEqual((llava["resized_height"], llava["resized_width"]), (576, 1056))
        self.assertEqual(llava["visual_token_margin"], 264)


if __name__ == "__main__":
    unittest.main()

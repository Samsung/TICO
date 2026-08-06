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
from pathlib import Path

from tico.quantization.recipes.config import load_recipe_config


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


if __name__ == "__main__":
    unittest.main()

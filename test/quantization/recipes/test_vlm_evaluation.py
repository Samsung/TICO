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

import tico.quantization.evaluation.vlm_eval_utils as vlm_eval_utils
import tico.quantization.recipes.evaluation.vlm as vlm


class TestVlmEvaluation(unittest.TestCase):
    def test_evaluate_llava_bench_routes_dataset_name(self):
        """LLaVA-Bench evaluation should use the shared COCO-score helper."""
        captured = {}

        def fake_get_dataset(name, n):
            captured["get_dataset"] = {"name": name, "n": n}
            return ["sample"], object()

        def fake_get_coco_scores_on_dataset(**kwargs):
            captured["scores"] = kwargs
            return {"CIDEr": 1.5, "total_count": 1, "skipped_count": 0}

        with patch.object(vlm, "get_dataset", fake_get_dataset), patch.object(
            vlm, "get_coco_scores_on_dataset", fake_get_coco_scores_on_dataset
        ):
            result = vlm.evaluate_llava_bench(
                model=object(),
                processor=object(),
                device="cpu",
                n_samples=3,
                max_seq_len=128,
            )

        self.assertEqual(result["CIDEr"], 1.5)
        self.assertEqual(captured["get_dataset"], {"name": "llava_bench", "n": 3})
        self.assertEqual(captured["scores"]["dataset_name"], "llava_bench")
        self.assertEqual(captured["scores"]["ds"], ["sample"])
        self.assertEqual(captured["scores"]["max_seq_len"], 128)

    def test_print_coco_score_results_keeps_counts_readable(self):
        """COCO-style result printing should not format count fields as floats."""
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            vlm.print_coco_score_results(
                "Results",
                {"CIDEr": 1.23456, "total_count": 2, "skipped_count": 1},
            )

        output = buffer.getvalue()
        self.assertIn("CIDEr          1.235", output)
        self.assertIn("total_count    2", output)
        self.assertIn("skipped_count  1", output)
        self.assertNotIn("total_count    2.000", output)

    def test_coco_eval_dependencies_checked_before_consuming_dataset(self):
        """Missing COCO eval dependencies should fail before samples run."""
        consumed = False

        def dataset():
            nonlocal consumed
            consumed = True
            yield {}

        def fake_import_module(module_name):
            if module_name == "pycocotools.coco":
                raise ModuleNotFoundError(
                    "No module named 'pycocotools'", name="pycocotools"
                )
            return object()

        with patch.object(
            vlm_eval_utils.importlib,
            "import_module",
            side_effect=fake_import_module,
        ):
            with self.assertRaisesRegex(RuntimeError, "pycocotools\\.coco"):
                vlm_eval_utils.get_coco_scores_on_dataset(
                    model=object(),
                    processor=object(),
                    dataset_name="coco",
                    ds=dataset(),
                    device="cpu",
                    metrics=["CIDEr"],
                )

        self.assertFalse(consumed)


if __name__ == "__main__":
    unittest.main()

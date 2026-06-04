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

import unittest
from unittest.mock import patch

import tico.quantization.recipes.evaluation.vlm as vlm


class TestVlmLlavaBenchRouting(unittest.TestCase):
    """Tests for COCO-style VLM evaluation routing."""

    def test_evaluate_coco_forwards_dataset_name_override(self):
        """evaluate_coco should not hard-code the COCO dataset name."""
        captured = {}

        def fake_evaluate_coco_score_dataset(**kwargs):
            captured.update(kwargs)
            return {"CIDEr": 1.0}

        with patch.object(
            vlm, "evaluate_coco_score_dataset", fake_evaluate_coco_score_dataset
        ):
            result = vlm.evaluate_coco(
                model="model",
                processor="processor",
                device="cpu",
                n_samples=3,
                max_seq_len=128,
                dataset_name="llava_bench",
            )

        self.assertEqual(result, {"CIDEr": 1.0})
        self.assertEqual(captured["model"], "model")
        self.assertEqual(captured["processor"], "processor")
        self.assertEqual(captured["dataset_name"], "llava_bench")
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(captured["n_samples"], 3)
        self.assertEqual(captured["max_seq_len"], 128)

    def test_evaluate_llava_bench_uses_shared_helper(self):
        """evaluate_llava_bench should route through the shared COCO-score helper."""
        captured = {}

        def fake_get_dataset(name, n):
            captured["get_dataset"] = {"name": name, "n": n}
            return ["sample"], object()

        def fake_get_coco_scores_on_dataset(**kwargs):
            captured["scores"] = kwargs
            return {"CIDEr": 2.0, "total_count": 1, "skipped_count": 0}

        with patch.object(vlm, "get_dataset", fake_get_dataset), patch.object(
            vlm, "get_coco_scores_on_dataset", fake_get_coco_scores_on_dataset
        ):
            result = vlm.evaluate_llava_bench(
                model="model",
                processor="processor",
                device="cpu",
                n_samples=1,
                max_seq_len=128,
            )

        self.assertEqual(result["CIDEr"], 2.0)
        self.assertEqual(captured["get_dataset"], {"name": "llava_bench", "n": 1})
        self.assertEqual(captured["scores"]["dataset_name"], "llava_bench")
        self.assertEqual(captured["scores"]["ds"], ["sample"])
        self.assertEqual(captured["scores"]["max_seq_len"], 128)


if __name__ == "__main__":
    unittest.main()

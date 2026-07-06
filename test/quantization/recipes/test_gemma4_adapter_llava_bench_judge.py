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
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod

import torch
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext


class TestGemma4AdapterLlavaBenchJudge(unittest.TestCase):
    """Tests for Gemma4 adapter dispatch into LLaVA-Bench judge evaluation."""

    def test_evaluate_dispatches_nested_llava_bench_judge_config(self):
        """Nested llava_bench config should call the judge recipe wrapper."""
        calls = []
        adapter = Gemma4Adapter()
        llava_cfg = {
            "enabled": True,
            "mode": "judge",
            "n_samples": 2,
            "max_new_tokens": 512,
            "judge": {
                "enabled": False,
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            },
        }
        ctx = RecipeContext(
            cfg={
                "model": {"name_or_path": "google/gemma-4-e2b-it"},
                "runtime": {"show_progress": False},
                "evaluation": {
                    "enabled": True,
                    "n_samples": 5,
                    "max_seq_len": 2048,
                    "llava_bench": llava_cfg,
                },
            },
            adapter=adapter,
            model="target-model",
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        def fake_evaluate_and_print_llava_bench_judge(**kwargs):
            calls.append(kwargs)
            return {"count": 2}

        with patch.object(
            gemma4_mod,
            "evaluate_and_print_llava_bench_judge",
            fake_evaluate_and_print_llava_bench_judge,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.evaluate(ctx)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["model"], "target-model")
        self.assertIs(call["processor"], ctx.processor)
        self.assertEqual(call["device"], "cpu")
        self.assertIs(call["llava_cfg"], llava_cfg)
        self.assertEqual(call["model_cfg"], {"name_or_path": "google/gemma-4-e2b-it"})
        self.assertEqual(call["runtime_cfg"], {"show_progress": False})
        self.assertEqual(call["default_n_samples"], 5)
        self.assertEqual(call["default_max_seq_len"], 2048)

    def test_evaluate_dispatches_legacy_llava_bench_boolean_path(self):
        """Boolean llava_bench=true should keep the legacy COCO-style path."""
        calls = []
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "evaluation": {
                    "enabled": True,
                    "n_samples": 3,
                    "max_seq_len": 2048,
                    "llava_bench": True,
                }
            },
            adapter=adapter,
            model=object(),
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        def fake_evaluate_llava_bench(**kwargs):
            calls.append(("legacy", kwargs))
            return {"CIDEr": 0.5, "total_count": 3, "skipped_count": 0}

        def fake_print_coco_score_results(title, results):
            calls.append(("print", {"title": title, "results": results}))

        with patch.object(
            gemma4_mod, "evaluate_llava_bench", fake_evaluate_llava_bench
        ), patch.object(
            gemma4_mod, "print_coco_score_results", fake_print_coco_score_results
        ):
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                adapter.evaluate(ctx)

        self.assertIn("legacy", buffer.getvalue())
        self.assertEqual(calls[0][0], "legacy")
        self.assertEqual(calls[0][1]["n_samples"], 3)
        self.assertEqual(calls[0][1]["max_seq_len"], 2048)
        self.assertEqual(calls[1][0], "print")
        self.assertIn("Llava Bench", calls[1][1]["title"])

    def test_evaluate_dispatches_nested_legacy_mode(self):
        """Nested llava_bench mode=legacy should call the COCO-style helper."""
        calls = []
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "evaluation": {
                    "enabled": True,
                    "n_samples": 10,
                    "max_seq_len": 2048,
                    "llava_bench": {
                        "enabled": True,
                        "mode": "legacy",
                        "n_samples": 4,
                        "max_seq_len": 1024,
                    },
                }
            },
            adapter=adapter,
            model=object(),
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        def fake_evaluate_llava_bench(**kwargs):
            calls.append(("legacy", kwargs))
            return {"CIDEr": 0.5, "total_count": 4, "skipped_count": 0}

        with patch.object(
            gemma4_mod, "evaluate_llava_bench", fake_evaluate_llava_bench
        ), patch.object(
            gemma4_mod, "print_coco_score_results", lambda *args, **kwargs: None
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.evaluate(ctx)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["n_samples"], 4)
        self.assertEqual(calls[0][1]["max_seq_len"], 1024)

    def test_evaluate_rejects_unknown_llava_bench_mode(self):
        """Unknown llava_bench modes should fail clearly."""
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "evaluation": {
                    "enabled": True,
                    "llava_bench": {"enabled": True, "mode": "unknown"},
                }
            },
            adapter=adapter,
            model=object(),
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        with self.assertRaises(ValueError):
            adapter.evaluate(ctx)

    def test_evaluate_noop_when_disabled(self):
        """evaluation.enabled=false should return without error."""
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "evaluation": {
                    "enabled": False,
                    "llava_bench": {"enabled": True, "mode": "judge"},
                }
            },
            adapter=adapter,
            model=object(),
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        # Should not raise
        adapter.evaluate(ctx)


if __name__ == "__main__":
    unittest.main()

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

import tico.quantization.recipes.adapters.qwen3_vl as qwen_mod
import tico.quantization.recipes.data.vlm as vlm_data

import torch
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.dtypes import DType


class TinyModule(torch.nn.Module):
    """Tiny module exposing one linear submodule for hook registration."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        """Run a single linear layer."""
        return self.linear(x)


def _fake_qwen_model():
    """Return a fake Qwen3-VL model config."""
    return SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(depth=3, deepstack_visual_indexes=[0, 1]),
            text_config=SimpleNamespace(num_hidden_layers=4),
        )
    )


class TestQwen3VLAdapter(unittest.TestCase):
    def test_build_ptq_config_uses_architecture_counts_and_model_args(self):
        """Qwen3VLAdapter should infer architecture counts before building PTQ config."""
        captured = {}

        def fake_build_qwen3_vl_ptq_config(**kwargs):
            captured.update(kwargs)
            return {"ptq": "qwen"}

        ctx = RecipeContext(
            cfg={
                "model_args": {
                    "vision": {
                        "grid_thw": [1, 8, 8],
                        "visual_start_idx": 0,
                        "spatial_merge_size": 2,
                    }
                }
            },
            adapter=Qwen3VLAdapter(),
            model=_fake_qwen_model(),
        )

        with patch.object(
            qwen_mod, "build_qwen3_vl_ptq_config", fake_build_qwen3_vl_ptq_config
        ):
            result = Qwen3VLAdapter().build_ptq_config(
                ctx,
                {
                    "activation": "int16",
                    "linear_weight": 4,
                    "vision_patch_embed_weight": 8,
                    "embedding_weight": 8,
                    "lm_head_weight": 4,
                    "norm": "int16",
                    "norm_weight": "int16",
                    "quantize_vision": True,
                    "quantize_text": False,
                    "quantize_lm_head": True,
                    "strict_wrap": False,
                },
            )

        self.assertEqual(result, {"ptq": "qwen"})
        self.assertEqual(captured["num_vision_blocks"], 3)
        self.assertEqual(captured["num_text_layers"], 4)
        self.assertEqual(captured["num_deepstack_mergers"], 2)
        self.assertEqual(captured["model_args"]["vision"]["grid_thw"], (1, 8, 8))
        self.assertEqual(captured["linear_weight"].dtype, DType.uint(4))
        self.assertEqual(captured["vision_patch_embed_weight"].dtype, DType.uint(8))
        self.assertEqual(captured["lm_head_weight"].dtype, DType.uint(4))
        self.assertFalse(captured["strict_wrap"])

    def test_build_calibration_inputs_routes_mixed_dataset_config(self):
        """Qwen3VLAdapter should route mixed calibration datasets to the data helper."""
        captured = {}
        datasets = [
            {"dataset": "vqav2", "split": "testdev", "n_samples": 3},
            {"dataset": "wikitext2", "split": "train", "n_samples": 5},
        ]
        ctx = RecipeContext(
            cfg={
                "runtime": {"seed": 7},
                "calibration": {"datasets": datasets, "seq_len": 128},
            },
            adapter=Qwen3VLAdapter(),
            model=_fake_qwen_model(),
        )
        ctx.processor = object()

        def fake_build_vlm_calibration_inputs(**kwargs):
            captured.update(kwargs)
            return [{"input_ids": torch.ones(1, 2)}]

        with patch.object(
            qwen_mod,
            "build_vlm_calibration_inputs",
            fake_build_vlm_calibration_inputs,
        ):
            result = Qwen3VLAdapter().build_calibration_inputs(ctx)

        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0]["input_ids"], torch.ones(1, 2)))
        self.assertEqual(captured["datasets"], datasets)
        self.assertEqual(captured["max_seq_len"], 128)
        self.assertEqual(captured["seed"], 7)

    def test_vlm_data_helper_parses_mixed_dataset_string(self):
        """The VLM data helper should parse old CLI-style mixed dataset specs."""
        captured = {}

        def fake_get_mixed_calib_inputs(**kwargs):
            captured.update(kwargs)
            return ["mixed"]

        with patch.object(
            vlm_data, "get_mixed_calib_inputs", fake_get_mixed_calib_inputs
        ):
            result = vlm_data.build_vlm_calibration_inputs(
                processor=object(),
                dataset="vqav2:testdev:3,wikitext2:train:5",
                n_samples=1,
                max_seq_len=128,
                seed=11,
            )

        self.assertEqual(result, ["mixed"])
        self.assertEqual(
            captured["dataset_config"],
            {
                "vqav2": {"split": "testdev", "n_samples": 3},
                "wikitext2": {"split": "train", "n_samples": 5},
            },
        )
        self.assertEqual(captured["max_seq_len"], 128)
        self.assertEqual(captured["seed"], 11)

    def test_apply_smoothquant_maps_component_selection_to_excluded_appliers(self):
        """SmoothQuant component selection should translate to excluded applier names."""
        adapter = Qwen3VLAdapter()
        model = TinyModule()
        ctx = RecipeContext(
            cfg={},
            adapter=adapter,
            model=model,
            calibration_inputs=[{"x": torch.ones(1, 2)}],
        )
        captured = {}

        def fake_apply_smoothing(
            model, activation_max, alpha, custom_alpha_map=None, exclude_appliers=None
        ):
            captured["alpha"] = alpha
            captured["exclude_appliers"] = exclude_appliers
            captured["custom_alpha_map"] = custom_alpha_map

        with patch.object(
            adapter, "forward_calibration", lambda *args, **kwargs: None
        ), patch(
            "tico.quantization.algorithm.smoothquant.smooth_quant.apply_smoothing",
            fake_apply_smoothing,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.apply_smoothquant(ctx, {"alpha": 0.25, "components": "vision"})

        self.assertEqual(captured["alpha"], 0.25)
        self.assertEqual(
            captured["exclude_appliers"], ["_apply_if_qwen3vl_text_decoder"]
        )

    def test_evaluate_dispatches_coco_and_llava_bench(self):
        """Qwen3VLAdapter should dispatch COCO-style image caption benchmarks."""
        calls = []
        adapter = Qwen3VLAdapter()
        ctx = RecipeContext(
            cfg={
                "evaluation": {
                    "enabled": True,
                    "coco": True,
                    "llava_bench": True,
                    "n_samples": 2,
                    "max_seq_len": 128,
                }
            },
            adapter=adapter,
            model=object(),
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        def fake_evaluate_coco(**kwargs):
            calls.append(("coco", kwargs))
            return {"CIDEr": 1.0, "total_count": 2, "skipped_count": 0}

        def fake_evaluate_llava_bench(**kwargs):
            calls.append(("llava_bench", kwargs))
            return {"CIDEr": 2.0, "total_count": 2, "skipped_count": 1}

        def fake_print_coco_score_results(title, results):
            calls.append(("print", {"title": title, "results": results}))

        with patch.object(qwen_mod, "evaluate_coco", fake_evaluate_coco), patch.object(
            qwen_mod, "evaluate_llava_bench", fake_evaluate_llava_bench
        ), patch.object(
            qwen_mod, "print_coco_score_results", fake_print_coco_score_results
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.evaluate(ctx)

        self.assertEqual(calls[0][0], "coco")
        self.assertEqual(calls[0][1]["n_samples"], 2)
        self.assertEqual(calls[0][1]["max_seq_len"], 128)
        self.assertEqual(calls[1][0], "print")
        self.assertIn("COCO", calls[1][1]["title"])
        self.assertEqual(calls[2][0], "llava_bench")
        self.assertEqual(calls[2][1]["device"], "cpu")
        self.assertEqual(calls[3][0], "print")
        self.assertIn("Llava Bench", calls[3][1]["title"])

    def test_evaluate_dispatches_mmmu_and_ppl(self):
        """Qwen3VLAdapter should dispatch optional MMMU and PPL evaluation blocks."""
        calls = []
        adapter = Qwen3VLAdapter()
        ctx = RecipeContext(
            cfg={
                "runtime": {"show_progress": False},
                "calibration": {"seq_len": 128},
                "evaluation": {
                    "enabled": True,
                    "max_seq_len": 128,
                    "mmmu": {
                        "enabled": True,
                        "subjects": ["Accounting"],
                        "n_samples": 1,
                    },
                    "ppl": {
                        "enabled": True,
                        "dataset": "wikitext2",
                        "split": "test",
                        "stride": 32,
                    },
                },
            },
            adapter=adapter,
            model=object(),
            processor=SimpleNamespace(tokenizer=object()),
        )
        ctx.device = torch.device("cpu")

        def fake_evaluate_vlm_text_ppl(**kwargs):
            calls.append(("ppl", kwargs))
            return 12.5

        with patch.object(
            qwen_mod,
            "evaluate_and_print_mmmu",
            lambda **kwargs: calls.append(("mmmu", kwargs)),
        ), patch.object(
            qwen_mod,
            "evaluate_vlm_text_ppl",
            fake_evaluate_vlm_text_ppl,
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                adapter.evaluate(ctx)

        self.assertEqual(calls[0][0], "mmmu")
        self.assertEqual(calls[0][1]["dataset"], "MMMU/MMMU")
        self.assertEqual(calls[0][1]["subjects"], ["Accounting"])
        self.assertEqual(calls[1][0], "ppl")
        self.assertEqual(calls[1][1]["stride"], 32)


if __name__ == "__main__":
    unittest.main()

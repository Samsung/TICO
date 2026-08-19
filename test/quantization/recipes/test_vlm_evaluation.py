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
from unittest.mock import MagicMock, patch

import tico.quantization.evaluation.vlm_eval_utils as vlm_eval_utils
import tico.quantization.recipes.evaluation.vlm as vlm

import torch
from tico.quantization.evaluation.vlm_eval_utils import (
    _build_text_calib_inputs,
    _coerce_int_attr,
    _compute_image_max_pixels_for_budget,
    _extract_golds,
    _get_required_coco_eval_modules,
    _processor_vision_factor,
    build_messages,
    build_text_only_inputs,
    DATASETS,
    exact_match,
    get_item_alpaca,
    get_item_coco,
    get_item_llava_bench_in_the_wild,
    get_item_mmmu_calib,
    get_item_textvqa,
    get_item_vqav2,
    get_item_wikitext2,
    move_inputs_to_device,
    normalize_answer,
)


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


class TestNormalizeAnswer(unittest.TestCase):
    """Tests for normalize_answer."""

    def test_lowercases_and_strips(self):
        self.assertEqual(normalize_answer("  Hello World  "), "hello world")

    def test_removes_articles(self):
        self.assertEqual(normalize_answer("a cat an dog the bird"), "cat dog bird")

    def test_replaces_separators_with_spaces(self):
        self.assertEqual(normalize_answer("foo-bar/baz"), "foo bar baz")

    def test_removes_punctuation(self):
        self.assertEqual(normalize_answer("hello, world!"), "hello world")

    def test_collapses_whitespace(self):
        # "a" is removed as an article, so only "b" remains
        self.assertEqual(normalize_answer("  cat   dog  "), "cat dog")

    def test_empty_string(self):
        self.assertEqual(normalize_answer(""), "")


class TestExactMatch(unittest.TestCase):
    """Tests for exact_match."""

    def test_match_found(self):
        ok, gold = exact_match("cat", ["dog", "cat", "bird"])
        self.assertTrue(ok)
        self.assertEqual(gold, "cat")

    def test_match_after_normalization(self):
        ok, gold = exact_match("The Cat", ["a dog", "the cat"])
        self.assertTrue(ok)
        self.assertEqual(gold, "the cat")

    def test_no_match(self):
        ok, gold = exact_match("fish", ["dog", "cat"])
        self.assertFalse(ok)
        self.assertIsNone(gold)

    def test_empty_golds(self):
        ok, gold = exact_match("cat", [])
        self.assertFalse(ok)
        self.assertIsNone(gold)


class TestExtractGolds(unittest.TestCase):
    """Tests for _extract_golds."""

    def test_none(self):
        self.assertEqual(_extract_golds(None), [])

    def test_dict_with_answer_key(self):
        result = _extract_golds({"answer": [1, 2, 3]})
        self.assertEqual(result, ["1", "2", "3"])

    def test_list_of_dicts_with_answer(self):
        result = _extract_golds([{"answer": "yes"}, {"answer": "no"}])
        self.assertEqual(result, ["yes", "no"])

    def test_list_of_plain_values(self):
        result = _extract_golds(["a", "b", 42])
        self.assertEqual(result, ["a", "b", "42"])

    def test_single_scalar(self):
        result = _extract_golds(42)
        self.assertEqual(result, ["42"])

    def test_empty_list(self):
        self.assertEqual(_extract_golds([]), [])


class TestDatasetAdapters(unittest.TestCase):
    """Tests for dataset adapter functions."""

    def test_get_item_vqav2(self):
        ex = {"image": "img", "question": "What?", "answers": ["cat", "dog"]}
        item = get_item_vqav2(ex)
        self.assertEqual(item["image"], "img")
        self.assertEqual(item["question"], "What?")
        self.assertEqual(item["golds"], ["cat", "dog"])

    def test_get_item_vqav2_missing_question(self):
        ex = {"image": "img", "answers": ["cat"]}
        item = get_item_vqav2(ex)
        self.assertEqual(item["question"], "")
        self.assertEqual(item["golds"], ["cat"])

    def test_get_item_textvqa(self):
        ex = {"image": "img", "question": "Q?", "answers": ["a"]}
        item = get_item_textvqa(ex)
        self.assertEqual(item["image"], "img")
        self.assertEqual(item["question"], "Q?")
        self.assertEqual(item["golds"], ["a"])

    def test_get_item_coco(self):
        ex = {
            "image": "img",
            "question": "caption",
            "id": 1,
            "question_id": 10,
            "file_name": "f.jpg",
            "answer": ["cap1", "cap2"],
        }
        item = get_item_coco(ex)
        self.assertEqual(item["image"], "img")
        self.assertEqual(item["question"], "caption")
        self.assertEqual(item["id"], 1)
        self.assertEqual(item["image_id"], 10)
        self.assertEqual(item["file_name"], "f.jpg")
        self.assertEqual(item["golds"], ["cap1", "cap2"])

    def test_get_item_llava_bench_in_the_wild(self):
        ex = {
            "image": "img",
            "question": "Q?",
            "question_id": 42,
            "image_id": "img_001",
            "gpt_answer": "answer",
        }
        item = get_item_llava_bench_in_the_wild(ex)
        self.assertEqual(item["image"], "img")
        self.assertEqual(item["question"], "Q?")
        self.assertEqual(item["id"], 42)
        self.assertEqual(item["image_id"], 42)
        self.assertEqual(item["file_name"], "img_001")
        self.assertEqual(item["golds"], ["answer"])

    def test_get_item_wikitext2(self):
        ex = {"text": "hello world"}
        item = get_item_wikitext2(ex)
        self.assertEqual(item, {"text": "hello world"})

    def test_get_item_wikitext2_missing_text(self):
        item = get_item_wikitext2({})
        self.assertEqual(item, {"text": ""})

    def test_get_item_alpaca_with_input(self):
        ex = {"instruction": "Do X", "input": "data", "output": "result"}
        item = get_item_alpaca(ex)
        self.assertEqual(item, {"text": "Do X\ndata"})

    def test_get_item_alpaca_without_input(self):
        ex = {"instruction": "Do X", "input": "", "output": "result"}
        item = get_item_alpaca(ex)
        self.assertEqual(item, {"text": "Do X"})

    def test_get_item_alpaca_missing_input_key(self):
        ex = {"instruction": "Do X"}
        item = get_item_alpaca(ex)
        self.assertEqual(item, {"text": "Do X"})


class TestGetItemMmmuCalib(unittest.TestCase):
    """Tests for get_item_mmmu_calib."""

    def test_single_image_sample(self):
        ex = {
            "image_1": "img",
            "question": "Q?",
            "answer": "A",
        }
        item = get_item_mmmu_calib(ex)
        self.assertEqual(item["image"], "img")
        self.assertEqual(item["question"], "Q?")
        self.assertEqual(item["golds"], ["A"])

    def test_multi_image_sample_skipped(self):
        ex = {
            "image_1": "img1",
            "image_2": "img2",
            "question": "Q?",
            "answer": "A",
        }
        item = get_item_mmmu_calib(ex)
        self.assertIsNone(item["image"])
        self.assertEqual(item["question"], "")
        self.assertEqual(item["golds"], [])

    def test_fallback_to_image_key(self):
        ex = {
            "image": "fallback_img",
            "question": "Q?",
            "answer": "A",
        }
        item = get_item_mmmu_calib(ex)
        self.assertEqual(item["image"], "fallback_img")

    def test_no_answer(self):
        ex = {
            "image_1": "img",
            "question": "Q?",
            "answer": "",
        }
        item = get_item_mmmu_calib(ex)
        self.assertEqual(item["golds"], [])

    def test_missing_answer_key(self):
        ex = {
            "image_1": "img",
            "question": "Q?",
        }
        item = get_item_mmmu_calib(ex)
        self.assertEqual(item["golds"], [])


class TestDatasetsRegistry(unittest.TestCase):
    """Tests for the DATASETS registry structure."""

    def test_mmmu_pro_vision_registered(self):
        self.assertIn("mmmu_pro_vision", DATASETS)
        meta = DATASETS["mmmu_pro_vision"]
        self.assertEqual(meta["default_split"], "test")
        self.assertEqual(meta["candidates"], ["MMMU/MMMU_Pro"])
        self.assertEqual(meta["config"], "vision")
        self.assertFalse(meta["is_text_only"])
        self.assertIs(meta["adapter"], get_item_mmmu_calib)

    def test_all_datasets_have_required_keys(self):
        required_keys = {"default_split", "adapter", "candidates"}
        for name, meta in DATASETS.items():
            for key in required_keys:
                self.assertIn(key, meta, f"Dataset '{name}' missing key '{key}'")

    def test_text_only_datasets_marked(self):
        for name in ("wikitext2", "alpaca"):
            self.assertTrue(
                DATASETS[name].get("is_text_only", False),
                f"Dataset '{name}' should be text-only",
            )


class TestBuildMessages(unittest.TestCase):
    """Tests for build_messages."""

    def test_returns_user_message_with_image_and_text(self):
        messages = build_messages("What is this?")
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 1)
        msg = messages[0]
        self.assertEqual(msg["role"], "user")
        content = msg["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0], {"type": "image"})
        self.assertEqual(content[1]["type"], "text")
        self.assertIn("What is this?", content[1]["text"])
        self.assertIn("Return ONLY the final answer", content[1]["text"])


class TestCoerceIntAttr(unittest.TestCase):
    """Tests for _coerce_int_attr (new function from git diff)."""

    def test_none_returns_default(self):
        self.assertEqual(_coerce_int_attr(None, 42), 42)

    def test_scalar_int(self):
        self.assertEqual(_coerce_int_attr(16, 42), 16)

    def test_scalar_str(self):
        self.assertEqual(_coerce_int_attr("8", 42), 8)

    def test_single_element_list(self):
        self.assertEqual(_coerce_int_attr([16], 42), 16)

    def test_single_element_tuple(self):
        self.assertEqual(_coerce_int_attr((16,), 42), 16)

    def test_empty_list_returns_default(self):
        self.assertEqual(_coerce_int_attr([], 42), 42)

    def test_empty_tuple_returns_default(self):
        self.assertEqual(_coerce_int_attr((), 42), 42)


class TestProcessorVisionFactor(unittest.TestCase):
    """Tests for _processor_vision_factor (new function from git diff)."""

    def test_default_values(self):
        """When processor has no image_processor, defaults to 16*2=32."""
        processor = MagicMock()
        processor.image_processor = None
        factor = _processor_vision_factor(processor)
        self.assertEqual(factor, 32)

    def test_custom_patch_and_merge(self):
        processor = MagicMock()
        processor.image_processor.patch_size = 14
        processor.image_processor.merge_size = 2
        factor = _processor_vision_factor(processor)
        self.assertEqual(factor, 28)

    def test_list_attrs(self):
        processor = MagicMock()
        processor.image_processor.patch_size = [16]
        processor.image_processor.merge_size = [2]
        factor = _processor_vision_factor(processor)
        self.assertEqual(factor, 32)

    def test_none_attrs(self):
        processor = MagicMock()
        processor.image_processor.patch_size = None
        processor.image_processor.merge_size = None
        factor = _processor_vision_factor(processor)
        self.assertEqual(factor, 32)

    def test_minimum_clamped_to_one(self):
        processor = MagicMock()
        processor.image_processor.patch_size = 0
        processor.image_processor.merge_size = 0
        factor = _processor_vision_factor(processor)
        self.assertEqual(factor, 1)


class TestComputeImageMaxPixelsForBudget(unittest.TestCase):
    """Tests for _compute_image_max_pixels_for_budget (new function from git diff)."""

    def test_basic_budget(self):
        """With vision_factor=32 and max_seq_len=1024, visual_budget=768."""
        processor = MagicMock()
        processor.image_processor = None  # defaults: patch=16, merge=2 -> factor=32

        max_pixels, min_pixels = _compute_image_max_pixels_for_budget(
            max_seq_len=1024, processor=processor
        )
        # visual_budget = 1024 - 256 = 768
        # max_pixels = 768 * 32 * 32 = 786432
        expected_max = 768 * 32 * 32
        self.assertEqual(max_pixels, expected_max)
        # min_pixels = min(256*28*28, max_pixels) = min(200704, 786432) = 200704
        self.assertEqual(min_pixels, 200704)

    def test_tight_budget_min_pixels_capped(self):
        """When budget is very small, min_pixels should be capped to max_pixels."""
        processor = MagicMock()
        processor.image_processor = None

        max_pixels, min_pixels = _compute_image_max_pixels_for_budget(
            max_seq_len=300, processor=processor
        )
        # visual_budget = 300 - 256 = 44
        # max_pixels = 44 * 32 * 32 = 45056
        # min_pixels = min(200704, 45056) = 45056
        # max_pixels = max(45056, 45056) = 45056
        self.assertEqual(max_pixels, 45056)
        self.assertEqual(min_pixels, 45056)

    def test_raises_on_zero_budget(self):
        processor = MagicMock()
        processor.image_processor = None
        with self.assertRaises(ValueError):
            _compute_image_max_pixels_for_budget(
                max_seq_len=256, processor=processor  # visual_budget = 0
            )

    def test_raises_on_negative_budget(self):
        processor = MagicMock()
        processor.image_processor = None
        with self.assertRaises(ValueError):
            _compute_image_max_pixels_for_budget(max_seq_len=100, processor=processor)

    def test_custom_text_token_margin(self):
        processor = MagicMock()
        processor.image_processor = None

        max_pixels, min_pixels = _compute_image_max_pixels_for_budget(
            max_seq_len=1024, processor=processor, text_token_margin=512
        )
        # visual_budget = 1024 - 512 = 512
        # max_pixels = 512 * 32 * 32 = 524288
        self.assertEqual(max_pixels, 524288)


class TestMoveInputsToDevice(unittest.TestCase):
    """Tests for move_inputs_to_device."""

    def test_moves_tensors(self):
        t1 = torch.zeros(3)
        t2 = torch.ones(2)
        inputs = {"a": t1, "b": t2, "c": "not_a_tensor"}
        result = move_inputs_to_device(inputs, "cpu")
        # Tensors should still be tensors (moved to cpu)
        self.assertTrue(torch.is_tensor(result["a"]))
        self.assertTrue(torch.is_tensor(result["b"]))
        # Non-tensor preserved
        self.assertEqual(result["c"], "not_a_tensor")

    def test_preserves_non_tensor_values(self):
        inputs = {"text": "hello", "num": 42, "none": None}
        result = move_inputs_to_device(inputs, "cpu")
        self.assertEqual(result["text"], "hello")
        self.assertEqual(result["num"], 42)
        self.assertIsNone(result["none"])

    def test_returns_same_container(self):
        inputs = {"a": torch.zeros(1)}
        result = move_inputs_to_device(inputs, "cpu")
        self.assertIs(result, inputs)


class TestBuildTextOnlyInputs(unittest.TestCase):
    """Tests for build_text_only_inputs."""

    def test_without_max_seq_len(self):
        processor = MagicMock()
        processor.return_value = {"input_ids": "ok"}
        result = build_text_only_inputs(processor, "hello")
        processor.assert_called_once_with(text="hello", return_tensors="pt")
        self.assertEqual(result, {"input_ids": "ok"})

    def test_with_max_seq_len(self):
        processor = MagicMock()
        processor.return_value = {"input_ids": "ok"}
        build_text_only_inputs(processor, "hello", max_seq_len=128)
        processor.assert_called_once_with(
            text="hello",
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

    def test_max_seq_len_zero_no_truncation(self):
        processor = MagicMock()
        processor.return_value = {}
        build_text_only_inputs(processor, "hello", max_seq_len=0)
        processor.assert_called_once_with(text="hello", return_tensors="pt")


class TestBuildTextCalibInputs(unittest.TestCase):
    """Tests for _build_text_calib_inputs."""

    def test_returns_n_samples(self):
        processor = MagicMock()
        # Simulate a tokenizer that returns input_ids with shape [1, 100]
        processor.tokenizer.return_value = MagicMock(input_ids=torch.zeros(1, 100))
        result = _build_text_calib_inputs(
            processor=processor,
            text="some text",
            n_samples=5,
            max_seq_len=32,
            seed=42,
        )
        self.assertEqual(len(result), 5)
        for sample in result:
            self.assertIn("input_ids", sample)
            self.assertTrue(torch.is_tensor(sample["input_ids"]))

    def test_short_text_uses_all_tokens(self):
        """When text is shorter than max_seq_len, all tokens are used."""
        processor = MagicMock()
        processor.tokenizer.return_value = MagicMock(input_ids=torch.zeros(1, 10))
        result = _build_text_calib_inputs(
            processor=processor,
            text="short",
            n_samples=3,
            max_seq_len=128,
            seed=42,
        )
        self.assertEqual(len(result), 3)
        for sample in result:
            # Should use all 10 tokens
            self.assertEqual(sample["input_ids"].shape[1], 10)

    def test_reproducible_with_seed(self):
        processor = MagicMock()
        processor.tokenizer.return_value = MagicMock(input_ids=torch.zeros(1, 200))
        result1 = _build_text_calib_inputs(
            processor=processor, text="text", n_samples=3, max_seq_len=32, seed=42
        )
        result2 = _build_text_calib_inputs(
            processor=processor, text="text", n_samples=3, max_seq_len=32, seed=42
        )
        for r1, r2 in zip(result1, result2):
            self.assertTrue(torch.equal(r1["input_ids"], r2["input_ids"]))


class TestGetRequiredCocoEvalModules(unittest.TestCase):
    """Tests for _get_required_coco_eval_modules."""

    def test_always_includes_pycocotools(self):
        modules = _get_required_coco_eval_modules([])
        self.assertIn("pycocotools.coco", modules)

    def test_cider_metric(self):
        modules = _get_required_coco_eval_modules(["CIDEr"])
        self.assertIn("pycocotools.coco", modules)
        self.assertIn("pycocoevalcap.cider.cider", modules)

    def test_bleu_metrics(self):
        modules = _get_required_coco_eval_modules(["Bleu_1", "Bleu_4"])
        self.assertIn("pycocoevalcap.bleu.bleu", modules)

    def test_meteor_metric(self):
        modules = _get_required_coco_eval_modules(["METEOR"])
        self.assertIn("pycocoevalcap.meteor.meteor", modules)

    def test_rouge_metric(self):
        modules = _get_required_coco_eval_modules(["ROUGE_L"])
        self.assertIn("pycocoevalcap.rouge.rouge", modules)

    def test_deduplication(self):
        """Duplicate metrics should not produce duplicate modules."""
        modules = _get_required_coco_eval_modules(["CIDEr", "CIDEr", "Bleu_1"])
        self.assertEqual(len(modules), len(set(modules)))

    def test_combined_metrics(self):
        modules = _get_required_coco_eval_modules(
            ["CIDEr", "Bleu_1", "Bleu_2", "METEOR", "ROUGE_L"]
        )
        self.assertIn("pycocotools.coco", modules)
        self.assertIn("pycocoevalcap.cider.cider", modules)
        self.assertIn("pycocoevalcap.bleu.bleu", modules)
        self.assertIn("pycocoevalcap.meteor.meteor", modules)
        self.assertIn("pycocoevalcap.rouge.rouge", modules)


if __name__ == "__main__":
    unittest.main()

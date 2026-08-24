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
    _supports_qwen_style_pixel_budget,
    build_messages,
    build_text_only_inputs,
    build_vlm_inputs,
    DATASETS,
    exact_match,
    get_item_alpaca,
    get_item_coco,
    get_item_llava_bench_in_the_wild,
    get_item_mmlu,
    get_item_mmmu_calib,
    get_item_textvqa,
    get_item_videomme_text,
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

    def test_get_item_mmlu_with_answer(self):
        """Test get_item_mmlu with a standard 4-choice sample and correct answer."""
        ex = {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "answer": 2,
            "subject": "geography",
        }
        result = get_item_mmlu(ex)

        self.assertIn("text", result)
        text = result["text"]

        # Question is the first line
        self.assertTrue(text.startswith("What is the capital of France?"))

        # All four choices are labeled A-D
        self.assertIn("A. London", text)
        self.assertIn("B. Berlin", text)
        self.assertIn("C. Paris", text)
        self.assertIn("D. Madrid", text)

        # Correct answer (index 2 -> C) is appended
        self.assertIn("Answer: C", text)

    def test_get_item_mmlu_no_answer(self):
        """Test get_item_mmlu when answer is None (no answer line)."""
        ex = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5"],
            "answer": None,
        }
        result = get_item_mmlu(ex)
        text = result["text"]

        self.assertIn("A. 3", text)
        self.assertIn("B. 4", text)
        self.assertIn("C. 5", text)
        self.assertNotIn("Answer:", text)

    def test_get_item_mmlu_out_of_range_answer(self):
        """Test get_item_mmlu when answer index is out of range."""
        ex = {
            "question": "Question?",
            "choices": ["a", "b"],
            "answer": 5,  # out of range
        }
        result = get_item_mmlu(ex)
        text = result["text"]

        self.assertIn("A. a", text)
        self.assertIn("B. b", text)
        # Answer line should not be appended for out-of-range index
        self.assertNotIn("Answer:", text)

    def test_get_item_mmlu_empty_choices(self):
        """Test get_item_mmlu with empty choices list."""
        ex = {
            "question": "Question with no choices?",
            "choices": [],
            "answer": 0,
        }
        result = get_item_mmlu(ex)
        text = result["text"]

        # Only the question should be present
        self.assertEqual(text, "Question with no choices?")

    def test_get_item_mmlu_missing_fields(self):
        """Test get_item_mmlu with an empty dict (all fields missing)."""
        result = get_item_mmlu({})

        self.assertIn("text", result)
        self.assertEqual(result["text"], "")

    def test_get_item_videomme_text_with_answer(self):
        """Test get_item_videomme_text with a standard sample and answer text."""
        ex = {
            "videoID": "abc123",
            "question": "What happens at 0:10?",
            "options": ["A cat jumps", "A dog runs", "A bird flies", "Nothing"],
            "answer": "A cat jumps",
            "domain": "daily",
            "duration": "short",
            "task_type": "perception",
        }
        result = get_item_videomme_text(ex)

        self.assertIn("text", result)
        text = result["text"]

        # Question is the first line
        self.assertTrue(text.startswith("What happens at 0:10?"))

        # All four options are labeled A-D
        self.assertIn("A. A cat jumps", text)
        self.assertIn("B. A dog runs", text)
        self.assertIn("C. A bird flies", text)
        self.assertIn("D. Nothing", text)

        # Correct answer text is appended
        self.assertIn("Answer: A cat jumps", text)

    def test_get_item_videomme_text_no_answer(self):
        """Test get_item_videomme_text when answer is empty string."""
        ex = {
            "question": "What is shown?",
            "options": ["opt1", "opt2"],
            "answer": "",
        }
        result = get_item_videomme_text(ex)
        text = result["text"]

        self.assertIn("A. opt1", text)
        self.assertIn("B. opt2", text)
        self.assertNotIn("Answer:", text)

    def test_get_item_videomme_text_empty_options(self):
        """Test get_item_videomme_text with empty options list."""
        ex = {
            "question": "Question with no options?",
            "options": [],
            "answer": "something",
        }
        result = get_item_videomme_text(ex)
        text = result["text"]

        # Only the question and answer should be present
        self.assertTrue(text.startswith("Question with no options?"))
        self.assertIn("Answer: something", text)

    def test_get_item_videomme_text_missing_fields(self):
        """Test get_item_videomme_text with an empty dict (all fields missing)."""
        result = get_item_videomme_text({})

        self.assertIn("text", result)
        self.assertEqual(result["text"], "")


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

    def test_mmmu_pro_uses_registered_default_split(self):
        dataset = MagicMock()
        dataset.take.return_value = dataset

        with patch.object(
            vlm_eval_utils,
            "load_dataset",
            return_value=dataset,
        ) as load_dataset_mock:
            loaded, adapter = vlm_eval_utils.get_dataset(
                "mmmu_pro_vision",
                n=1,
            )

        self.assertIs(loaded, dataset)
        self.assertIs(adapter, get_item_mmmu_calib)
        load_dataset_mock.assert_called_once_with(
            path="MMMU/MMMU_Pro",
            name="vision",
            split="test",
            streaming=True,
        )

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

    def test_mmlu_registered(self):
        """Test that 'mmlu' is registered with correct fields."""
        self.assertIn("mmlu", DATASETS)
        meta = DATASETS["mmlu"]

        self.assertEqual(meta["default_split"], "test")
        self.assertIs(meta["adapter"], get_item_mmlu)
        self.assertEqual(meta["candidates"], ["cais/mmlu"])
        self.assertEqual(meta["config"], "all")
        self.assertTrue(meta["is_text_only"])

    def test_videomme_text_registered(self):
        """Test that 'videomme_text' is registered with correct fields."""
        self.assertIn("videomme_text", DATASETS)
        meta = DATASETS["videomme_text"]

        self.assertEqual(meta["default_split"], "test")
        self.assertIs(meta["adapter"], get_item_videomme_text)
        self.assertEqual(meta["candidates"], ["lmms-lab/Video-MME"])
        self.assertTrue(meta["is_text_only"])


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
    """Tests for prompt-aware Qwen image budgeting."""

    @staticmethod
    def _processor(
        *,
        prompt_ids,
        min_pixels=65536,
        max_pixels=16777216,
    ):
        processor = MagicMock()
        processor.image_token_id = 99
        processor.tokenizer.return_value = {"input_ids": prompt_ids}
        processor.image_processor = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size={
                "shortest_edge": min_pixels,
                "longest_edge": max_pixels,
            },
            valid_kwargs=SimpleNamespace(
                __annotations__={"min_pixels": int, "max_pixels": int}
            ),
        )
        return processor

    def test_uses_actual_prompt_tokens_and_processor_minimum(self):
        processor = self._processor(prompt_ids=[1, 99, 2, 3])
        max_pixels, min_pixels = _compute_image_max_pixels_for_budget(
            max_seq_len=1024,
            processor=processor,
            prompt="prompt",
        )

        self.assertEqual(max_pixels, (1024 - 3) * 32 * 32)
        self.assertEqual(min_pixels, 65536)
        processor.tokenizer.assert_called_once_with("prompt")

    def test_caps_at_processor_maximum(self):
        processor = self._processor(
            prompt_ids=[1, 99, 2],
            max_pixels=123456,
        )
        max_pixels, _ = _compute_image_max_pixels_for_budget(
            max_seq_len=1024,
            processor=processor,
            prompt="prompt",
        )

        self.assertEqual(max_pixels, 123456)

    def test_tight_budget_caps_processor_minimum(self):
        processor = self._processor(prompt_ids=[1, 99, 2, 3])
        max_pixels, min_pixels = _compute_image_max_pixels_for_budget(
            max_seq_len=10,
            processor=processor,
            prompt="prompt",
        )

        self.assertEqual(max_pixels, 7 * 32 * 32)
        self.assertEqual(min_pixels, max_pixels)

    def test_raises_when_prompt_consumes_context_budget(self):
        processor = self._processor(prompt_ids=[1, 99, 2, 3])
        with self.assertRaisesRegex(ValueError, "visual_budget=0"):
            _compute_image_max_pixels_for_budget(
                max_seq_len=3,
                processor=processor,
                prompt="prompt",
            )


class TestBuildVlmInputs(unittest.TestCase):
    """Tests for processor-specific multimodal budgeting."""

    @staticmethod
    def _qwen_processor(*, output_length=100):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "prompt"
        processor.image_token_id = 99
        processor.tokenizer.return_value = {"input_ids": [1, 99, 2, 3]}
        processor.image_processor = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size={
                "shortest_edge": 65536,
                "longest_edge": 16777216,
            },
            valid_kwargs=SimpleNamespace(
                __annotations__={"min_pixels": int, "max_pixels": int}
            ),
        )
        processor.return_value = {
            "input_ids": torch.zeros((1, output_length), dtype=torch.long)
        }
        return processor

    @staticmethod
    def _gemma_processor(*, output_length=100):
        processor = MagicMock()
        processor.apply_chat_template.return_value = "prompt"
        processor.image_processor = SimpleNamespace(
            patch_size=16,
            pooling_kernel_size=3,
            max_soft_tokens=280,
            valid_kwargs=SimpleNamespace(__annotations__={"max_soft_tokens": int}),
        )
        processor.return_value = {
            "input_ids": torch.zeros((1, output_length), dtype=torch.long)
        }
        return processor

    def test_qwen_receives_prompt_aware_pixel_budget(self):
        processor = self._qwen_processor()
        result = build_vlm_inputs(
            processor=processor,
            image="image",
            question="question",
            max_seq_len=128,
        )

        self.assertIs(result, processor.return_value)
        self.assertTrue(_supports_qwen_style_pixel_budget(processor))
        processor.assert_called_once_with(
            text="prompt",
            images="image",
            return_tensors="pt",
            max_pixels=(128 - 3) * 32 * 32,
            min_pixels=65536,
        )

    def test_gemma_does_not_receive_qwen_pixel_kwargs(self):
        processor = self._gemma_processor()
        build_vlm_inputs(
            processor=processor,
            image="image",
            question="question",
            max_seq_len=128,
        )

        self.assertFalse(_supports_qwen_style_pixel_budget(processor))
        processor.assert_called_once_with(
            text="prompt",
            images="image",
            return_tensors="pt",
        )

    def test_raises_when_processed_sequence_exceeds_budget(self):
        processor = self._gemma_processor(output_length=129)
        with self.assertRaisesRegex(
            ValueError,
            "sequence_length=129, max_seq_len=128",
        ):
            build_vlm_inputs(
                processor=processor,
                image="image",
                question="question",
                max_seq_len=128,
            )


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

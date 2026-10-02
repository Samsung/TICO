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
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

from tico.quantization.evaluation import (
    vlm_eval_utils as vlm,
    vlm_generation_utils as generation,
)


class FakeTokenizer:
    def __call__(self, prompt):
        return {"input_ids": [1, 2, 99, 3]}  # Three non-image tokens.

    def decode(self, ids, skip_special_tokens):
        return "Answer: C"


class FakeProcessor:
    """Synthetic Qwen-style processor; no downloads, image library or GPU."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_token_id = 99
        self.image_processor = SimpleNamespace(
            patch_size=16,
            merge_size=2,
            valid_kwargs=type(
                "Kwargs",
                (),
                {
                    "__annotations__": {"min_pixels": int, "max_pixels": int},
                },
            ),
        )
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        return "rendered image prompt"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        visual_tokens = kwargs.get("max_pixels", 4 * 1024) // 1024
        count = visual_tokens + 3
        return {
            "input_ids": torch.arange(count).reshape(1, -1),
            "attention_mask": torch.ones(1, count, dtype=torch.long),
            "pixel_values": torch.tensor([[float(visual_tokens), 1.0]]),
            "image_grid_thw": torch.tensor([[1, 2, 2 * visual_tokens]]),
        }


class FakeModel:
    def __init__(self, *, ids=(8, 9, 0), eos=(0,), structured=False):
        self.generation_config = SimpleNamespace(eos_token_id=list(eos))
        self.ids = ids
        self.structured = structured
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = torch.cat((kwargs["input_ids"], torch.tensor([self.ids])), dim=1)
        return SimpleNamespace(sequences=output) if self.structured else output


class TestVlmGenerationBudget(unittest.TestCase):
    def test_legacy_auto_budget(self):
        self.assertEqual(generation.resolve_generation_input_budget(2048, 256), 1792)
        self.assertEqual(generation.resolve_generation_input_budget(2048, 512), 1536)
        self.assertIsNone(generation.resolve_generation_input_budget(None, 16))

    def test_fixed_input_budget(self):
        for output in (50, 256, 512):
            self.assertEqual(
                generation.resolve_generation_input_budget(2048, output, 1536), 1536
            )
        self.assertEqual(
            generation.resolve_generation_input_budget(None, 512, 1536), 1536
        )

    def test_invalid_combinations_are_not_silently_clamped(self):
        for args in (
            (2048, 512, 1800),
            (2048, 2048),
            (0, 16),
            (2048, 0),
            (2048, True),
            (2048, 3.5),
            (2048, "256"),
            (2048, 16, -1),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                generation.resolve_generation_input_budget(*args)  # type: ignore[arg-type]


class TestVlmGenerationDiagnostics(unittest.TestCase):
    def run_image(self, *, output=256, cap=None, diagnostics=None, model=None):
        processor = FakeProcessor()
        model = model or FakeModel()
        result = vlm.generate_image_only_answer(
            model=model,
            processor=processor,
            image=object(),
            question="Answer directly.",
            device="cpu",
            max_seq_len=2048,
            max_new_tokens=output,
            input_max_seq_len=cap,
            diagnostics=diagnostics,
        )
        return result, processor, model

    def test_fixed_budget_sweep_has_identical_tensor_inputs(self):
        left: dict[str, Any] = {}
        right: dict[str, Any] = {}
        _, p256, m256 = self.run_image(output=256, cap=1536, diagnostics=left)
        _, p512, m512 = self.run_image(output=512, cap=1536, diagnostics=right)
        self.assertEqual(left["input_tokens"], 1536)
        self.assertEqual(left["tensor_inputs_sha256"], right["tensor_inputs_sha256"])
        self.assertEqual(left["image_grid_thw"], right["image_grid_thw"])
        self.assertEqual(p256.calls[0]["max_pixels"], p512.calls[0]["max_pixels"])
        self.assertEqual(m256.calls[0]["max_new_tokens"], 256)
        self.assertEqual(m512.calls[0]["max_new_tokens"], 512)
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            self.assertTrue(torch.equal(m256.calls[0][key], m512.calls[0][key]))

    def test_auto_budget_sweep_can_change_image_inputs(self):
        left: dict[str, Any] = {}
        right: dict[str, Any] = {}
        self.run_image(output=256, diagnostics=left)
        self.run_image(output=512, diagnostics=right)
        self.assertEqual(left["input_tokens"], 1792)
        self.assertEqual(right["input_tokens"], 1536)
        self.assertNotEqual(left["tensor_inputs_sha256"], right["tensor_inputs_sha256"])

    def test_invalid_cap_fails_before_processor_or_generate(self):
        processor, model = FakeProcessor(), FakeModel()
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            vlm.generate_image_only_answer(
                model=model,
                processor=processor,
                image=object(),
                device="cpu",
                max_seq_len=2048,
                max_new_tokens=512,
                input_max_seq_len=1800,
            )
        self.assertEqual(processor.calls, [])
        self.assertEqual(model.calls, [])

    def test_default_path_does_not_fingerprint(self):
        with patch.object(
            generation.hashlib, "sha256", side_effect=AssertionError("unexpected hash")
        ):
            result, _, model = self.run_image()
        self.assertEqual(result, "Answer: C")
        self.assertFalse(model.calls[0]["do_sample"])
        self.assertNotIn("diagnostics", model.calls[0])
        self.assertNotIn("input_max_seq_len", model.calls[0])

    def test_structured_generate_output(self):
        diagnostics: dict[str, Any] = {}
        result, _, _ = self.run_image(
            diagnostics=diagnostics, model=FakeModel(structured=True)
        )
        self.assertEqual(result, "Answer: C")
        self.assertEqual(diagnostics["stop_reason"], "eos")

    def test_general_vqa_generator_keeps_instruction_and_return_type(self):
        processor, model = FakeProcessor(), FakeModel()
        diagnostics: dict[str, Any] = {}
        result = vlm.generate_answer(
            model,
            processor,
            object(),
            "What is shown?",
            "cpu",
            16,
            0.0,
            128,
            diagnostics=diagnostics,
        )
        self.assertEqual(result, "Answer: C")
        text = processor.messages[0]["content"][1]["text"]
        self.assertIn("Return ONLY the final answer with no extra words.", text)
        self.assertEqual(diagnostics["input_max_seq_len"], 112)

    def test_cot_instruction_is_not_overridden_by_vqa_instruction(self):
        _, processor, _ = self.run_image()
        text = processor.messages[0]["content"][1]["text"]
        self.assertEqual(text, "Answer directly.")
        self.assertNotIn("Return ONLY", text)

    def test_non_qwen_processor_retains_its_settings_and_validates_length(self):
        processor = FakeProcessor()
        processor.image_processor = SimpleNamespace()
        vlm.generate_image_only_answer(
            model=FakeModel(),
            processor=processor,
            image=object(),
            device="cpu",
            max_seq_len=32,
            max_new_tokens=16,
        )
        self.assertNotIn("max_pixels", processor.calls[0])
        with self.assertRaisesRegex(ValueError, "Processed sequence exceeds"):
            vlm.generate_image_only_answer(
                model=FakeModel(),
                processor=processor,
                image=object(),
                device="cpu",
                max_seq_len=17,
                max_new_tokens=16,
            )


class TestVlmFingerprintAndStopping(unittest.TestCase):
    def test_hash_is_order_independent_and_detects_tensor_changes(self):
        a = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.ones(2, dtype=torch.bfloat16),
        }
        b = dict(reversed(list(a.items())))
        first: dict[str, Any] = {}
        second: dict[str, Any] = {}
        generation.record_generation_inputs(first, a, input_max_seq_len=16)
        generation.record_generation_inputs(second, b, input_max_seq_len=16)
        self.assertEqual(first["tensor_inputs_sha256"], second["tensor_inputs_sha256"])
        b["pixel_values"] = torch.zeros(2, dtype=torch.bfloat16)
        generation.record_generation_inputs(second, b, input_max_seq_len=16)
        self.assertNotEqual(
            first["tensor_inputs_sha256"], second["tensor_inputs_sha256"]
        )

    def test_fingerprint_supports_empty_scalar_and_noncontiguous_tensors(self):
        diagnostics: dict[str, Any] = {"old": "removed"}
        inputs = {
            "input_ids": torch.tensor([[1]]),
            "empty": torch.empty(0),
            "scalar": torch.tensor(2),
            "strided": torch.ones(3, 4).t(),
            "text": "not hashed",
        }
        generation.record_generation_inputs(diagnostics, inputs, input_max_seq_len=16)
        self.assertNotIn("old", diagnostics)
        self.assertEqual(len(diagnostics["input_tensors"]), 4)
        self.assertEqual(diagnostics["input_tensors"]["scalar"]["shape"], [])

    def stop(self, ids, eos=(0,), budget=3, generation_config=True):
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[0]))
        if generation_config:
            model.generation_config = SimpleNamespace(eos_token_id=eos)
        diagnostics: dict[str, Any] = {}
        generation.record_generation_output(
            diagnostics, torch.tensor(ids), model=model, max_new_tokens=budget
        )
        return diagnostics

    def test_eos_at_limit_is_not_marked_length_limited(self):
        result = self.stop([8, 9, 0])
        self.assertEqual(result["generated_tokens"], 3)
        self.assertTrue(result["reached_max_new_tokens"])
        self.assertFalse(result["length_limited"])
        self.assertEqual(result["stop_reason"], "eos")

    def test_padding_after_eos_is_excluded(self):
        result = self.stop([8, 0, 0, 0])
        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(result["generated_token_ids"], [8, 0])

    def test_multiple_eos_ids(self):
        self.assertEqual(self.stop([8, 12], eos=[0, 12])["stop_reason"], "eos")

    def test_length_limit_without_eos(self):
        result = self.stop([8, 9, 10])
        self.assertTrue(result["length_limited"])
        self.assertEqual(result["stop_reason"], "length")

    def test_unknown_eos_is_not_claimed_as_length_termination(self):
        result = self.stop([8, 9, 10], eos=None)
        self.assertIsNone(result["length_limited"])
        self.assertEqual(result["stop_reason"], "unknown")

    def test_no_generation_config_can_use_model_config(self):
        self.assertEqual(
            self.stop([8, 0], generation_config=False)["stop_reason"], "eos"
        )

    def test_early_non_eos_stop_is_other(self):
        self.assertEqual(self.stop([8], budget=3)["stop_reason"], "other")


class TestMmmuGenerationIntegration(unittest.TestCase):
    def test_evaluator_records_real_generation_diagnostics(self):
        import json
        import tempfile
        from pathlib import Path

        from tico.quantization.evaluation import mmmu_eval_utils as mmmu

        sample = {
            "id": "vision-1",
            "image": object(),
            "options": ["one", "two", "three", "four"],
            "answer": "C",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            with patch.object(mmmu, "load_data", return_value=[sample]):
                result = mmmu.evaluate_mmmu(
                    FakeModel(),
                    FakeProcessor(),
                    "MMMU/MMMU_Pro",
                    subjects=["vision"],
                    device="cpu",
                    n_shots=0,
                    max_seq_len=2048,
                    input_max_seq_len=1536,
                    output_jsonl=path,
                    verbose=False,
                )
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(result, {"vision": (1, 1, 0)})
        record = next(r for r in records if r["record_type"] == "sample")
        self.assertEqual(record["generation"]["input_tokens"], 1536)
        self.assertEqual(record["generation"]["generated_token_ids"], [8, 9, 0])
        self.assertEqual(record["generation"]["stop_reason"], "eos")
        self.assertEqual(len(record["generation"]["tensor_inputs_sha256"]), 64)
        self.assertEqual(record["predicted"], "C")
        self.assertEqual(record["parse_method"], "explicit")

    def test_non_vision_uses_existing_prompt_and_16_token_default(self):
        from tico.quantization.evaluation import mmmu_eval_utils as mmmu

        sample = {
            "id": "text-1",
            "image": object(),
            "question": "Which is correct?",
            "options": ["one", "two", "three", "four"],
            "answer": "C",
        }
        with patch.object(mmmu, "load_data", return_value=[sample]), patch.object(
            mmmu, "generate_answer", return_value="C"
        ) as generate:
            result = mmmu.evaluate_mmmu(
                object(),
                object(),
                "MMMU/MMMU",
                subjects=["Accounting"],
                n_shots=0,
                max_seq_len=2048,
                verbose=False,
            )
        self.assertEqual(result, {"Accounting": (1, 1, 0)})
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 16)
        self.assertIn("Which is correct?", generate.call_args.kwargs["question"])
        self.assertNotIn("diagnostics", generate.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()

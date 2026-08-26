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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tico.quantization.evaluation.vlm_eval_utils as vlm_eval_utils
import tico.quantization.recipes.data.llm as llm_data

import torch
from tico.quantization.recipes.data.dataset_usage import (
    CALIBRATION_ROLE,
    DatasetUsageError,
    EVALUATION_ROLE,
)


class TestDatasetRoleLoading(unittest.TestCase):
    """Test role-aware defaults and early safety checks in the shared loader."""

    @staticmethod
    def _dataset():
        dataset = MagicMock()
        dataset.take.return_value = dataset
        return dataset

    def test_role_is_required(self):
        with self.assertRaises(TypeError):
            # Intentionally omit the mandatory `role` keyword argument.
            vlm_eval_utils.get_dataset(  # type: ignore[call-arg] # pylint: disable=missing-kwoa
                "vqav2", n=1
            )

    def test_vqav2_calibration_defaults_to_train(self):
        dataset = self._dataset()
        with patch.object(
            vlm_eval_utils,
            "load_dataset",
            return_value=dataset,
        ) as load_dataset_mock:
            loaded, _ = vlm_eval_utils.get_dataset(
                "vqav2",
                role=CALIBRATION_ROLE,
                n=2,
            )

        self.assertIs(loaded, dataset)
        load_dataset_mock.assert_called_once_with(
            path="HuggingFaceM4/VQAv2",
            split="train",
            streaming=True,
        )
        dataset.take.assert_called_once_with(2)

    def test_vqav2_evaluation_defaults_to_validation(self):
        dataset = self._dataset()
        with patch.object(
            vlm_eval_utils,
            "load_dataset",
            return_value=dataset,
        ) as load_dataset_mock:
            vlm_eval_utils.get_dataset(
                "vqav2",
                role=EVALUATION_ROLE,
                n=1,
            )

        load_dataset_mock.assert_called_once_with(
            path="HuggingFaceM4/VQAv2",
            split="validation",
            streaming=True,
        )

    def test_mmlu_calibration_uses_auxiliary_train(self):
        dataset = self._dataset()
        with patch.object(
            vlm_eval_utils,
            "load_dataset",
            return_value=dataset,
        ) as load_dataset_mock:
            _, adapter = vlm_eval_utils.get_dataset(
                "mmlu",
                role=CALIBRATION_ROLE,
                n=1,
            )

        self.assertIs(adapter, vlm_eval_utils.get_item_mmlu_calib)
        load_dataset_mock.assert_called_once_with(
            path="cais/mmlu",
            name="all",
            split="auxiliary_train",
            streaming=True,
        )

    def test_mmlu_renderer_excludes_gold_answer(self):
        item = vlm_eval_utils.get_item_mmlu_calib(
            {
                "question": "Which value is even?",
                "choices": ["1", "2", "3", "5"],
                "answer": 1,
            }
        )

        self.assertEqual(
            item,
            {"text": "Which value is even?\nA. 1\nB. 2\nC. 3\nD. 5"},
        )
        self.assertNotIn("Answer:", item["text"])

    def test_evaluation_only_source_fails_before_download(self):
        with patch.object(vlm_eval_utils, "load_dataset") as load_dataset_mock:
            with self.assertRaises(DatasetUsageError):
                vlm_eval_utils.get_dataset(
                    "mmmu_pro_vision",
                    role=CALIBRATION_ROLE,
                    n=1,
                )

        load_dataset_mock.assert_not_called()


class TestLlmCalibrationRoleLoading(unittest.TestCase):
    """Test semantic split validation in the direct LLM calibration builder."""

    @staticmethod
    def _tokenizer(token_count: int = 16):
        """Return a minimal tokenizer with deterministic token IDs."""
        tokenizer = MagicMock()
        tokenizer.bos_token_id = None
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.arange(token_count, dtype=torch.long).unsqueeze(0)
        )
        return tokenizer

    def test_wikitext_train_calibration_loads_canonical_source(self):
        dataset = {"text": ["calibration corpus"]}
        tokenizer = self._tokenizer()

        with patch.object(
            llm_data,
            "load_dataset",
            return_value=dataset,
        ) as load_dataset_mock:
            samples = llm_data.build_wikitext_calibration_inputs(
                tokenizer=tokenizer,
                cache_dir=None,
                n_samples=1,
                seq_len=4,
                seed=7,
                device="cpu",
                dataset_name="wikitext2",
                split="train",
            )

        load_dataset_mock.assert_called_once_with(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            split="train",
            cache_dir=None,
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(tuple(samples[0].shape), (1, 4))

    def test_wikitext_test_calibration_fails_before_download(self):
        with patch.object(llm_data, "load_dataset") as load_dataset_mock:
            with self.assertRaisesRegex(
                DatasetUsageError,
                "test.*not calibration-safe",
            ):
                llm_data.build_wikitext_calibration_inputs(
                    tokenizer=self._tokenizer(),
                    cache_dir=None,
                    n_samples=1,
                    seq_len=4,
                    seed=7,
                    device="cpu",
                    dataset_name="Salesforce/wikitext",
                    split="test",
                )

        load_dataset_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()

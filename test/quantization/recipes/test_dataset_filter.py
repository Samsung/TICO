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

"""Unit tests for dataset_filter (per-class calibration sample filtering)."""

from typing import Any, Dict

from unittest.mock import MagicMock, patch

from tico.quantization.evaluation.vlm_eval_utils import (
    CalibFilterConfig,
    dataset_filter,
    get_mixed_calib_inputs,
)


def _make_example(image_classes):
    return {
        "image": None,
        "question": "What is this?",
        "answers": ["cat"],
        "image_classes": image_classes,
    }


class TestDatasetFilter:
    def test_basic_quota(self):
        """Each class should get at most n_per_class samples."""
        examples = [
            _make_example(["cat", "dog"]),
            _make_example(["cat"]),
            _make_example(["cat"]),
            _make_example(["dog"]),
            _make_example(["bird"]),
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=2))
        assert len(result) == 4

    def test_no_image_classes(self):
        """Samples without image_classes should be kept as-is."""
        examples: list[dict] = [
            {"image_classes": []},
            {"image_classes": []},
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=5))
        assert len(result) == 2

    def test_n_per_class_one(self):
        """With n_per_class=1, each class can only appear once."""
        examples = [
            _make_example(["cat"]),
            _make_example(["cat"]),
            _make_example(["dog"]),
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=1))
        assert len(result) == 2

    def test_multi_class_sample(self):
        """A sample with multiple classes counts for all of them."""
        examples = [
            _make_example(["cat", "dog", "bird"]),
            _make_example(["cat"]),
            _make_example(["dog"]),
            _make_example(["bird"]),
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=1))
        assert len(result) == 1

    def test_empty_input(self):
        result = dataset_filter([], CalibFilterConfig(n_per_class=5))
        assert result == []

    def test_large_n_per_class(self):
        """n_per_class larger than available samples keeps everything."""
        examples = [
            _make_example(["cat"]),
            _make_example(["cat"]),
            _make_example(["dog"]),
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=100))
        assert len(result) == 3


class TestDistinctImages:
    """Tests for the distinct_images deduplication feature."""

    def test_distinct_images_dedup(self):
        """When distinct_images=True, same image_id appears only once."""
        examples = [
            {**_make_example(["cat"]), "image_id": "img_001", "question_id": 1},
            {**_make_example(["cat"]), "image_id": "img_001", "question_id": 2},
            {**_make_example(["cat"]), "image_id": "img_002", "question_id": 3},
        ]
        result = dataset_filter(
            examples, CalibFilterConfig(n_per_class=10, distinct_images=True)
        )
        assert len(result) == 2
        assert result[0]["question_id"] == 1
        assert result[1]["question_id"] == 3

    def test_distinct_images_false_allows_duplicates(self):
        """When distinct_images=False, same image_id can appear multiple times."""
        examples = [
            {**_make_example(["cat"]), "image_id": "img_001", "question_id": 1},
            {**_make_example(["cat"]), "image_id": "img_001", "question_id": 2},
        ]
        result = dataset_filter(
            examples, CalibFilterConfig(n_per_class=10, distinct_images=False)
        )
        assert len(result) == 2

    def test_distinct_images_default_true(self):
        """distinct_images defaults to True."""
        examples = [
            {**_make_example(["cat"]), "image_id": "img_001", "question_id": 1},
            {**_make_example(["cat"]), "image_id": "img_001", "question_id": 2},
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=10))
        assert len(result) == 1

    def test_distinct_images_no_image_id(self):
        """Samples without image_id are not deduplicated."""
        examples = [
            _make_example(["cat"]),
            _make_example(["cat"]),
        ]
        result = dataset_filter(
            examples, CalibFilterConfig(n_per_class=10, distinct_images=True)
        )
        assert len(result) == 2


class TestMixedModeClassFiltering:
    """Tests that get_mixed_calib_inputs routes textvqa through class filtering
    when a per-dataset ``filter`` block is present."""

    def test_textvqa_uses_class_filter_when_filter_block_set(self):
        """When a filter block with n_per_class > 0 is set, textvqa should use
        get_calib_inputs with a CalibFilterConfig."""
        dataset_config: Dict[str, Dict[str, Any]] = {
            "vqav2": {"n_samples": 10},
            "textvqa": {
                "n_samples": 50,
                "filter": {
                    "field": "image_classes",
                    "n_per_class": 5,
                },
            },
            "wikitext2": {"n_samples": 128},
        }

        processor = MagicMock()

        with patch(
            "tico.quantization.evaluation.vlm_eval_utils.get_calib_inputs"
        ) as mock_calib, patch(
            "tico.quantization.evaluation.vlm_eval_utils.get_dataset"
        ) as mock_get_dataset, patch(
            "tico.quantization.evaluation.vlm_eval_utils._build_text_calib_inputs"
        ) as mock_text:
            # textvqa class-filtering returns dummy inputs
            mock_calib.return_value = [{"input_ids": "textvqa_filtered"}]

            # vqav2 streaming returns 2 samples with images
            vqav2_ds = MagicMock()
            vqav2_adapter = MagicMock(
                return_value={"image": MagicMock(), "question": "q", "golds": []}
            )
            mock_get_dataset.return_value = (vqav2_ds, vqav2_adapter)
            vqav2_ds.__iter__ = MagicMock(
                return_value=iter([{"image": MagicMock()}, {"image": MagicMock()}])
            )

            # wikitext2 text inputs
            mock_text.return_value = [{"input_ids": "wikitext"}]

            with patch(
                "tico.quantization.evaluation.vlm_eval_utils.build_vlm_inputs"
            ) as mock_build:
                mock_build.return_value = {"input_ids": "vqav2_input"}

                result = get_mixed_calib_inputs(
                    processor=processor,
                    dataset_config=dataset_config,
                    max_seq_len=2048,
                )

            # textvqa should have been routed through get_calib_inputs with filter_config
            mock_calib.assert_called_once()
            call_kwargs = mock_calib.call_args.kwargs
            assert call_kwargs["dataset"] == "textvqa"
            fc = call_kwargs["filter_config"]
            assert isinstance(fc, CalibFilterConfig)
            assert fc.n_per_class == 5
            assert fc.filter_field == "image_classes"

            # Result should contain textvqa filtered + vqav2 + wikitext inputs
            assert len(result) >= 1

    def test_textvqa_not_filtered_when_no_filter_block(self):
        """When no filter block is present, textvqa uses the default streaming path."""
        dataset_config: Dict[str, Dict[str, Any]] = {
            "textvqa": {"n_samples": 5},
        }

        processor = MagicMock()

        with patch(
            "tico.quantization.evaluation.vlm_eval_utils.get_calib_inputs"
        ) as mock_calib, patch(
            "tico.quantization.evaluation.vlm_eval_utils.get_dataset"
        ) as mock_get_dataset, patch(
            "tico.quantization.evaluation.vlm_eval_utils.build_vlm_inputs"
        ) as mock_build:
            ds = MagicMock()
            adapter = MagicMock(
                return_value={"image": MagicMock(), "question": "q", "golds": []}
            )
            mock_get_dataset.return_value = (ds, adapter)
            ds.__iter__ = MagicMock(return_value=iter([{"image": MagicMock()}] * 5))
            mock_build.return_value = {"input_ids": "input"}

            get_mixed_calib_inputs(
                processor=processor,
                dataset_config=dataset_config,
                max_seq_len=2048,
            )

            # get_calib_inputs (class filtering path) should NOT have been called
            mock_calib.assert_not_called()

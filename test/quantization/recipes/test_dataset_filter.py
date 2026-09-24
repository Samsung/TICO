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

import tico.quantization.recipes.data.vlm as vlm_data
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


def _class_occurrences(examples):
    """Count in how many of the *selected* examples each class appears."""
    occurrences: Dict[str, int] = {}
    for ex in examples:
        for cls in ex.get("image_classes", []):
            occurrences[cls] = occurrences.get(cls, 0) + 1
    return occurrences


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

    def test_missing_filter_field_raises(self):
        """A misspelled filter_field raises ValueError naming dataset and field.

        The filtered path loads the full dataset (``n=-1``) and ignores
        ``n_samples``; silently returning everything on a typo like
        ``image_class`` instead of ``image_classes`` would turn a small
        calibration run into full-dataset preprocessing.
        """
        examples = [
            _make_example(["cat"]),
            _make_example(["dog"]),
        ]
        raised = None
        try:
            dataset_filter(
                examples,
                CalibFilterConfig(n_per_class=1, filter_field="image_class"),
                dataset_name="textvqa",
            )
        except ValueError as e:
            raised = e
        assert raised is not None, "expected ValueError for unknown filter field"
        msg = str(raised)
        assert "image_class" in msg
        assert "textvqa" in msg

    def test_inactive_config_returns_unchanged(self):
        """Filtering is disabled for n_per_class <= 0 or a missing config."""
        examples = [_make_example(["cat"])]
        assert dataset_filter(examples, CalibFilterConfig(n_per_class=0)) is examples
        assert dataset_filter(examples, None) is examples

    def test_strict_multilabel_cap(self):
        """A multi-label sample is kept only when ALL its classes are under quota.

        With ``n_per_class=1`` and ``[cat]`` followed by ``[cat, dog]``, the
        second sample must be skipped because ``cat`` is already at quota;
        otherwise ``cat`` would appear in two selected samples while the
        internal counter reports one.
        """
        examples = [
            _make_example(["cat"]),
            _make_example(["cat", "dog"]),
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=1))
        occurrences = _class_occurrences(result)
        assert len(result) == 1
        assert occurrences == {"cat": 1}
        assert all(n <= 1 for n in occurrences.values())

    def test_class_occurrences_never_exceed_quota(self):
        """Class occurrences in the selected set never exceed n_per_class."""
        examples = [
            _make_example(["cat", "dog"]),
            _make_example(["cat", "dog"]),
            _make_example(["cat"]),
            _make_example(["dog"]),
            _make_example(["bird"]),
            _make_example(["cat", "bird"]),
        ]
        result = dataset_filter(examples, CalibFilterConfig(n_per_class=2))
        occurrences = _class_occurrences(result)
        assert occurrences == {"cat": 2, "dog": 2, "bird": 1}
        assert all(n <= 2 for n in occurrences.values())
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


class TestFilterConfigNormalization:
    """_normalize_filter_config must preserve every supported filter key."""

    @staticmethod
    def _normalize_verbose(verbose_value):
        config = vlm_data.normalize_mixed_dataset_config(
            {
                "textvqa": {
                    "n_samples": 8,
                    "filter": {"n_per_class": 2, "verbose": verbose_value},
                }
            },
            default_n_samples=8,
        )
        return config["textvqa"]["filter"]["verbose"]

    def test_verbose_true_preserved(self):
        assert self._normalize_verbose(True) is True

    def test_verbose_false_preserved(self):
        """A YAML ``verbose: false`` must not be discarded by normalization."""
        assert self._normalize_verbose(False) is False

    def test_verbose_defaults_true(self):
        """Without an explicit key, verbose falls back to True."""
        config = vlm_data.normalize_mixed_dataset_config(
            {"textvqa": {"n_samples": 8, "filter": {"n_per_class": 2}}},
            default_n_samples=8,
        )
        assert config["textvqa"]["filter"]["verbose"] is True

    def test_negative_n_per_class_raises(self):
        """A negative n_per_class is a configuration error, not 'disabled'."""
        raised = None
        try:
            vlm_data.normalize_mixed_dataset_config(
                {"textvqa": {"n_samples": 8, "filter": {"n_per_class": -1}}},
                default_n_samples=8,
            )
        except ValueError as e:
            raised = e
        assert raised is not None, "expected ValueError for negative n_per_class"
        msg = str(raised)
        assert "n_per_class" in msg
        assert ">= 0" in msg

    def test_n_per_class_none_and_zero_allowed(self):
        """null and 0 mean the filter is disabled; they are not errors."""
        for value in (None, 0):
            config = vlm_data.normalize_mixed_dataset_config(
                {"textvqa": {"n_samples": 8, "filter": {"n_per_class": value}}},
                default_n_samples=8,
            )
            assert config["textvqa"]["filter"]["n_per_class"] == 0


class TestMixedModeClassFiltering:
    """Tests that get_mixed_calib_inputs routes textvqa through class filtering
    when a per-dataset ``filter`` block is present."""

    def test_textvqa_filter_verbose_flows_to_calib_config(self):
        """A normalized ``verbose: false`` must reach the CalibFilterConfig.

        Regression test: _normalize_filter_config used to drop the ``verbose``
        key, so ``filter_dict.get("verbose", True)`` in get_mixed_calib_inputs
        always fell back to True regardless of the configured value.
        """
        dataset_config = vlm_data.normalize_mixed_dataset_config(
            {
                "textvqa": {
                    "n_samples": 50,
                    "filter": {"n_per_class": 5, "verbose": False},
                }
            },
            default_n_samples=8,
        )

        processor = MagicMock()

        with patch(
            "tico.quantization.evaluation.vlm_eval_utils.get_calib_inputs"
        ) as mock_calib:
            mock_calib.return_value = [{"input_ids": "textvqa_filtered"}]

            get_mixed_calib_inputs(
                processor=processor,
                dataset_config=dataset_config,
                max_seq_len=2048,
            )

        mock_calib.assert_called_once()
        fc = mock_calib.call_args.kwargs["filter_config"]
        assert isinstance(fc, CalibFilterConfig)
        assert fc.verbose is False

    def test_filtered_branch_forwards_dataset_policy_flags(self):
        """The filtered branch must forward dataset-usage policy flags.

        ``allow_benchmark_overlap`` / ``allow_unregistered_dataset`` must reach
        get_calib_inputs so that adding a ``filter`` block changes only sample
        selection, not dataset-usage validation.
        """
        dataset_config: Dict[str, Dict[str, Any]] = {
            "textvqa": {
                "n_samples": 50,
                "filter": {"n_per_class": 5},
            },
        }

        processor = MagicMock()

        def _call_and_get_kwargs(**flag_kwargs):
            with patch(
                "tico.quantization.evaluation.vlm_eval_utils.get_calib_inputs"
            ) as mock_calib:
                mock_calib.return_value = [{"input_ids": "textvqa_filtered"}]
                get_mixed_calib_inputs(
                    processor=processor,
                    dataset_config=dataset_config,
                    max_seq_len=2048,
                    **flag_kwargs,
                )
            mock_calib.assert_called_once()
            return mock_calib.call_args.kwargs

        kwargs = _call_and_get_kwargs(
            allow_benchmark_overlap=True, allow_unregistered_dataset=True
        )
        assert kwargs["allow_benchmark_overlap"] is True
        assert kwargs["allow_unregistered_dataset"] is True

        # Defaults must be forwarded explicitly as False, not omitted.
        kwargs = _call_and_get_kwargs()
        assert kwargs["allow_benchmark_overlap"] is False
        assert kwargs["allow_unregistered_dataset"] is False

    def test_inactive_filter_falls_through_to_normal_path(self):
        """A non-positive n_per_class must not enter the filtered branch.

        The filter is disabled, so the dataset uses the normal loading path
        with the configured n_samples (not get_calib_inputs' default of 28).
        """
        for n_per_class in (-1, 0):
            dataset_config: Dict[str, Dict[str, Any]] = {
                "textvqa": {
                    "n_samples": 7,
                    "filter": {"n_per_class": n_per_class},
                },
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
                ds.__iter__ = MagicMock(return_value=iter([{"image": MagicMock()}]))
                mock_build.return_value = {"input_ids": "input"}

                get_mixed_calib_inputs(
                    processor=processor,
                    dataset_config=dataset_config,
                    max_seq_len=2048,
                )

            mock_calib.assert_not_called()
            assert mock_get_dataset.call_args.kwargs["n"] == 7

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
            assert call_kwargs["n_samples"] == 50
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

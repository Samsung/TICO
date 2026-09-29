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

from collections.abc import Mapping, Sequence
from typing import Any

from tico.quantization.evaluation.vlm_eval_utils import (
    get_calib_inputs,
    get_mixed_calib_inputs,
)
from tico.quantization.recipes.data.dataset_config import (
    DatasetConfig,
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CALIBRATION_N_SAMPLES,
    DEFAULT_CALIBRATION_SEED,
    normalize_mixed_dataset_config,
)


def build_vlm_calibration_inputs(
    *,
    processor: Any,
    dataset: str | None = None,
    datasets: str | Mapping[str, Any] | Sequence[Any] | None = None,
    n_samples: int = DEFAULT_CALIBRATION_N_SAMPLES,
    split: str | None = None,
    max_seq_len: int | None = None,
    seed: int = DEFAULT_CALIBRATION_SEED,
    allow_benchmark_overlap: bool = False,
    allow_unregistered_dataset: bool = False,
) -> list[dict]:
    """
    Build VLM calibration inputs from either one dataset or a mixed dataset set.

    Per-dataset filtering
    ---------------------
    When a dataset entry in ``datasets`` contains a ``filter`` block with
    ``n_per_class > 0``, that dataset is loaded in non-streaming mode and
    filtered to select up to ``n_per_class`` samples per class (as determined
    by ``filter.field``, default ``image_classes``), instead of taking the
    first ``n_samples``.  Only selected rows are materialized (decoded).

    Example YAML::

        calibration:
          datasets:
            textvqa:
              n_samples: 50
              filter:
                field: image_classes
                n_per_class: 5
            wikitext2:
              n_samples: 128

    Args:
        processor: Hugging Face processor used to build model inputs.
        dataset: Single dataset key or a comma-separated mixed dataset spec.
        datasets: Explicit mixed dataset configuration. When set, this takes
            precedence over ``dataset``.
        n_samples: Default sample count used by single-dataset mode and by
            mixed dataset entries that omit their own ``n_samples``.
        split: Optional split for single-dataset mode. If omitted, the
            dataset registry default is used.
        max_seq_len: Optional maximum text sequence length.
        seed: Random seed for text-only mixed calibration sampling.
        allow_benchmark_overlap: Permit explicitly transductive calibration data.
        allow_unregistered_dataset: Permit calibration with a source that has no
            registered safety policy.

    Returns:
        A list of processor output dictionaries for calibration.
    """
    if datasets is not None:
        dataset_config = normalize_mixed_dataset_config(datasets, n_samples)
        return get_mixed_calib_inputs(
            processor=processor,
            dataset_config=dataset_config,
            max_seq_len=max_seq_len or 2048,
            seed=seed,
            allow_benchmark_overlap=allow_benchmark_overlap,
            allow_unregistered_dataset=allow_unregistered_dataset,
        )

    if dataset is not None and "," in dataset:
        dataset_config = normalize_mixed_dataset_config(dataset, n_samples)
        return get_mixed_calib_inputs(
            processor=processor,
            dataset_config=dataset_config,
            max_seq_len=max_seq_len or 2048,
            seed=seed,
            allow_benchmark_overlap=allow_benchmark_overlap,
            allow_unregistered_dataset=allow_unregistered_dataset,
        )

    return get_calib_inputs(
        dataset or DEFAULT_CALIBRATION_DATASET,
        processor,
        n_samples=n_samples,
        split=split,
        max_seq_len=max_seq_len,
        allow_benchmark_overlap=allow_benchmark_overlap,
        allow_unregistered_dataset=allow_unregistered_dataset,
    )

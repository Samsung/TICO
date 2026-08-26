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

DatasetConfig = dict[str, dict[str, Any]]


def _coerce_positive_int(value: Any, *, context: str) -> int:
    """Convert a configuration value to a positive integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer, got {value!r}.") from exc

    if parsed <= 0:
        raise ValueError(f"{context} must be positive, got {parsed}.")
    return parsed


def _parse_dataset_spec(
    spec: str, default_n_samples: int
) -> tuple[str, dict[str, Any]]:
    """
    Parse one mixed calibration dataset specification string.

    Supported forms are:
      - ``dataset``
      - ``dataset:n_samples``
      - ``dataset:split:n_samples``
    """
    parts = [part.strip() for part in spec.split(":")]
    if not parts or not parts[0]:
        raise ValueError(f"Invalid calibration dataset spec: {spec!r}.")

    dataset = parts[0]
    config: dict[str, Any] = {}

    if len(parts) == 1:
        config["n_samples"] = default_n_samples
    elif len(parts) == 2:
        config["n_samples"] = _coerce_positive_int(
            parts[1], context=f"{dataset}.n_samples"
        )
    elif len(parts) == 3:
        split = parts[1]
        if not split:
            raise ValueError(f"{dataset}.split must not be empty.")
        config["split"] = split
        config["n_samples"] = _coerce_positive_int(
            parts[2], context=f"{dataset}.n_samples"
        )
    else:
        raise ValueError(
            f"Invalid calibration dataset spec {spec!r}. Expected 'dataset', "
            "'dataset:n_samples', or 'dataset:split:n_samples'."
        )

    return dataset, config


def _normalize_filter_config(filter_cfg: Any, *, context: str) -> dict[str, Any]:
    """Normalize a per-dataset ``filter`` block.

    Expected keys (all optional except ``n_per_class``):
      - ``n_per_class`` (int, required for the filter to activate)
      - ``field`` (str, default ``"image_classes"``)
      - ``classes`` (list[str] | None)
      - ``max_classes`` (int | None)
      - ``distinct_images`` (bool, default True)
      - ``verbose`` (bool, default True)
    """
    if not isinstance(filter_cfg, Mapping):
        raise TypeError(
            f"{context}.filter must be a mapping, got {type(filter_cfg).__name__}."
        )

    normalized: dict[str, Any] = {}
    n_per_class = filter_cfg.get("n_per_class", 0)
    if n_per_class is not None:
        n_per_class = int(n_per_class)
    normalized["n_per_class"] = n_per_class or 0

    field = filter_cfg.get("field", "image_classes")
    normalized["field"] = str(field)

    classes = filter_cfg.get("classes")
    if classes is not None:
        if not isinstance(classes, (list, tuple)):
            raise TypeError(
                f"{context}.filter.classes must be a list or null, "
                f"got {type(classes).__name__}."
            )
        normalized["classes"] = [str(c) for c in classes]
    else:
        normalized["classes"] = None

    max_classes = filter_cfg.get("max_classes")
    if max_classes is not None:
        max_classes = int(max_classes)
    normalized["max_classes"] = max_classes

    normalized["distinct_images"] = bool(filter_cfg.get("distinct_images", True))

    return normalized


def _normalize_mapping_dataset_config(
    datasets: Mapping[str, Any],
    default_n_samples: int,
) -> DatasetConfig:
    """Normalize a mapping-style mixed dataset configuration."""
    normalized: DatasetConfig = {}

    for dataset, config in datasets.items():
        if not dataset:
            raise ValueError("Calibration dataset name must not be empty.")

        if isinstance(config, Mapping):
            entry: dict[str, Any] = {
                "n_samples": _coerce_positive_int(
                    config.get("n_samples", default_n_samples),
                    context=f"{dataset}.n_samples",
                )
            }
            split = config.get("split")
            if split is not None:
                entry["split"] = str(split)

            # Pass through per-dataset filter block
            filter_cfg = config.get("filter")
            if filter_cfg is not None:
                entry["filter"] = _normalize_filter_config(
                    filter_cfg, context=f"{dataset}"
                )
        else:
            entry = {
                "n_samples": _coerce_positive_int(
                    config, context=f"{dataset}.n_samples"
                )
            }

        normalized[str(dataset)] = entry

    return normalized


def _normalize_sequence_dataset_config(
    datasets: Sequence[Any],
    default_n_samples: int,
) -> DatasetConfig:
    """Normalize a sequence-style mixed dataset configuration."""
    normalized: DatasetConfig = {}

    for index, item in enumerate(datasets):
        if isinstance(item, str):
            dataset, config = _parse_dataset_spec(item, default_n_samples)
        elif isinstance(item, Mapping):
            dataset = item.get("dataset", item.get("name"))
            if not dataset:
                raise ValueError(
                    f"calibration.datasets[{index}] must define 'dataset' or 'name'."
                )
            config = {
                "n_samples": _coerce_positive_int(
                    item.get("n_samples", default_n_samples),
                    context=f"calibration.datasets[{index}].n_samples",
                )
            }
            split = item.get("split")
            if split is not None:
                config["split"] = str(split)

            # Pass through per-dataset filter block
            filter_cfg = item.get("filter")
            if filter_cfg is not None:
                config["filter"] = _normalize_filter_config(
                    filter_cfg, context=f"calibration.datasets[{index}]"
                )
        else:
            raise TypeError(
                "Each calibration dataset entry must be a string or mapping, "
                f"got {type(item).__name__}."
            )

        normalized[str(dataset)] = config

    return normalized


def normalize_mixed_dataset_config(
    datasets: str | Mapping[str, Any] | Sequence[Any],
    default_n_samples: int,
) -> DatasetConfig:
    """Normalize mixed calibration dataset settings for VLM recipes."""
    if isinstance(datasets, str):
        entries = [entry.strip() for entry in datasets.split(",") if entry.strip()]
        normalized = _normalize_sequence_dataset_config(entries, default_n_samples)
    elif isinstance(datasets, Mapping):
        normalized = _normalize_mapping_dataset_config(datasets, default_n_samples)
    else:
        normalized = _normalize_sequence_dataset_config(datasets, default_n_samples)

    if not normalized:
        raise ValueError("At least one calibration dataset must be configured.")
    return normalized


def build_vlm_calibration_inputs(
    *,
    processor: Any,
    dataset: str | None = None,
    datasets: str | Mapping[str, Any] | Sequence[Any] | None = None,
    n_samples: int = 128,
    split: str | None = None,
    max_seq_len: int | None = None,
    seed: int = 42,
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
    first ``n_samples``.

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
        dataset or "vqav2",
        processor,
        n_samples=n_samples,
        split=split,
        max_seq_len=max_seq_len,
        allow_benchmark_overlap=allow_benchmark_overlap,
        allow_unregistered_dataset=allow_unregistered_dataset,
    )

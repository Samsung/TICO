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

"""
Pure calibration-dataset configuration normalization.

Shared by the calibration loader (``recipes.data.vlm``) and by cache-identity
code that must mirror the loader defaults (``recipes.stages.gptq``).

This module is intentionally free of heavy dependencies: do NOT import torch,
Hugging Face ``datasets``/``transformers``, or ``tico.quantization.evaluation``
here, so that configuration handling stays importable in minimal environments
(e.g. algorithm-only CI jobs).
"""

from collections.abc import Mapping, Sequence
from typing import Any


DatasetConfig = dict[str, dict[str, Any]]

# Single source of calibration defaults.
DEFAULT_CALIBRATION_DATASET = "vqav2"
DEFAULT_CALIBRATION_N_SAMPLES = 128
DEFAULT_CALIBRATION_SEED = 42


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
      - ``n_per_class`` (int, >= 0; required for the filter to activate)
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
    if normalized["n_per_class"] < 0:
        raise ValueError(
            f"{context}.filter.n_per_class must be >= 0, "
            f"got {normalized['n_per_class']}."
        )

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
    normalized["verbose"] = bool(filter_cfg.get("verbose", True))

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

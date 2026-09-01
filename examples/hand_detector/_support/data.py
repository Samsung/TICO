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

"""Input loading helpers for hand-detector calibration and evaluation."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch


_NUMERIC_SUFFIX = re.compile(r"(\d+)$")


def normalize_input_array(array: np.ndarray) -> torch.Tensor:
    """Convert one supported NumPy image layout into NHWC float32 format."""
    value = np.asarray(array)
    if value.ndim == 3:
        value = value[None, ...]
    if value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(
            f"Expected a rank-3 image or single-batch rank-4 array, got {value.shape}."
        )
    if value.shape[1] == 3 and value.shape[3] != 3:
        value = np.transpose(value, (0, 2, 3, 1))
    if value.shape[3] != 3:
        raise ValueError(
            f"Expected an NHWC or NCHW image with 3 channels, got {value.shape}."
        )
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / 255.0
    else:
        value = value.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(value))


def load_npy_inputs(
    directory: Path,
    limit: int | None = None,
    *,
    offset: int = 0,
    pattern: str = "palm*.npy",
) -> list[torch.Tensor]:
    """Load a naturally sorted slice of representative input arrays."""
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")
    paths = sorted(directory.glob(pattern), key=_natural_path_key)
    paths = paths[offset : None if limit is None else offset + limit]
    if not paths:
        raise FileNotFoundError(
            f"No input arrays matched {pattern!r} under {directory} at offset {offset}."
        )
    return [normalize_input_array(np.load(path)) for path in paths]


def list_npy_inputs(
    directory: Path,
    *,
    pattern: str = "palm*.npy",
) -> list[Path]:
    """Return naturally sorted input paths without loading their contents."""
    return sorted(directory.glob(pattern), key=_natural_path_key)


def make_synthetic_inputs(count: int, seed: int) -> list[torch.Tensor]:
    """Create deterministic NHWC [0, 1] inputs for smoke tests only."""
    if count <= 0:
        raise ValueError("count must be positive.")
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(1, 192, 192, 3, generator=generator) for _ in range(count)]


def _natural_path_key(path: Path) -> tuple[str, int, str]:
    stem = path.stem
    match = _NUMERIC_SUFFIX.search(stem)
    if match is None:
        return stem, -1, path.name
    return stem[: match.start()], int(match.group(1)), path.name

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

import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch


Qwen3VLVisionGridInput = torch.Tensor | Sequence[int]


@dataclass(frozen=True)
class Qwen3VLVisionProfile:
    """Describe one fixed Qwen3-VL vision deployment profile.

    The profile owns only the processor-visible temporal, height, and width
    grid. Model architecture properties such as ``spatial_merge_size`` remain
    owned by the wrapped vision model and are validated when an export adapter
    is materialized.
    """

    temporal: int
    height: int
    width: int

    def __post_init__(self) -> None:
        """Validate that every profile dimension is a positive integer."""
        values = (self.temporal, self.height, self.width)
        if any(type(value) is not int for value in values):
            raise TypeError(
                "Qwen3-VL vision profile dimensions must be integers, " f"got {values}."
            )
        if any(value <= 0 for value in values):
            raise ValueError(
                "Qwen3-VL vision profile dimensions must be positive, " f"got {values}."
            )

    @classmethod
    def from_grid_thw(
        cls,
        value: "Qwen3VLVisionProfile | Qwen3VLVisionGridInput",
    ) -> "Qwen3VLVisionProfile":
        """Normalize a profile object or concrete ``(T, H, W)`` value."""
        if isinstance(value, cls):
            return value

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.dim() == 2 and tuple(tensor.shape) == (1, 3):
                tensor = tensor[0]
            if tensor.dim() != 1 or tensor.numel() != 3:
                raise ValueError(
                    "Qwen3-VL vision grid tensor must have shape `(3,)` or "
                    f"`(1, 3)`, got {tuple(value.shape)}."
                )
            if (
                tensor.dtype == torch.bool
                or tensor.is_floating_point()
                or tensor.is_complex()
            ):
                raise TypeError(
                    "Qwen3-VL vision grid tensor must use an integer dtype, "
                    f"got {tensor.dtype}."
                )
            raw_values = tensor.tolist()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 3:
                raise ValueError(
                    "Qwen3-VL vision grid must contain exactly three values, "
                    f"got {value}."
                )
            raw_values = list(value)
        else:
            raise TypeError(
                "Qwen3-VL vision profile must be a profile, integer tensor, "
                f"list, or tuple, got {type(value).__name__}."
            )

        normalized: list[int] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                raise TypeError(
                    "Qwen3-VL vision profile dimensions must be integers, "
                    f"got {raw_values}."
                )
            try:
                normalized.append(operator.index(raw_value))
            except TypeError as exc:
                raise TypeError(
                    "Qwen3-VL vision profile dimensions must be integers, "
                    f"got {raw_values}."
                ) from exc

        return cls(*normalized)

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        """Return the profile as a ``(temporal, height, width)`` tuple."""
        return self.temporal, self.height, self.width

    @property
    def num_patch_tokens(self) -> int:
        """Return the number of vision tokens before spatial merging."""
        return self.temporal * self.height * self.width

    @property
    def attention_split_sizes(self) -> tuple[int, ...]:
        """Return one static attention chunk size per temporal frame."""
        return (self.height * self.width,) * self.temporal

    @property
    def key(self) -> str:
        """Return the stable key used for profile lookup and artifact naming."""
        return f"t{self.temporal}_h{self.height}_w{self.width}"

    def stage_stem(self, stage: str) -> str:
        """Return a profile-qualified artifact stem for one static stage."""
        stage = stage.strip()
        if not stage:
            raise ValueError("stage must not be empty.")
        return f"{stage}_{self.key}"

    def stage_filename(self, stage: str, artifact_tag: str) -> str:
        """Return a profile-qualified Circle filename for one static stage."""
        if not artifact_tag:
            raise ValueError("artifact_tag must not be empty.")
        return f"{self.stage_stem(stage)}.{artifact_tag}.circle"

    @property
    def artifact_stem(self) -> str:
        """Return the profile-specific vision-prefill artifact stem."""
        return self.stage_stem("vision_prefill")

    def circle_filename(self, artifact_tag: str) -> str:
        """Return the profile-specific vision-prefill Circle filename."""
        return self.stage_filename("vision_prefill", artifact_tag)

    def validate_spatial_merge_size(self, spatial_merge_size: int) -> None:
        """Validate that the spatial grid is compatible with the model merger."""
        if type(spatial_merge_size) is not int:
            raise TypeError(
                "spatial_merge_size must be an integer, "
                f"got {type(spatial_merge_size).__name__}."
            )
        if spatial_merge_size <= 0:
            raise ValueError(
                "spatial_merge_size must be positive, " f"got {spatial_merge_size}."
            )
        if self.height % spatial_merge_size or self.width % spatial_merge_size:
            raise ValueError(
                "Qwen3-VL vision profile height and width must be divisible by "
                "spatial_merge_size: "
                f"grid_thw={self.grid_thw}, "
                f"spatial_merge_size={spatial_merge_size}."
            )

    def num_visual_tokens(self, spatial_merge_size: int) -> int:
        """Return the number of visual tokens after spatial merging."""
        self.validate_spatial_merge_size(spatial_merge_size)
        return (
            self.temporal
            * (self.height // spatial_merge_size)
            * (self.width // spatial_merge_size)
        )

    def to_tensor(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the profile as a batch-one ``torch.long`` tensor."""
        return torch.tensor([self.grid_thw], dtype=torch.long, device=device)

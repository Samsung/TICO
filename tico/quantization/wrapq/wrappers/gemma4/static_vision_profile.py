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

"""Canonical static vision profiles for Gemma4 export and runtime."""

from __future__ import annotations

import hashlib
import json

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


DEFAULT_GEMMA4_STATIC_VISION_PROFILE = "e2b_66x36_264"


@dataclass(frozen=True)
class Gemma4StaticVisionProfile:
    """Describe one fixed Gemma4 image-to-text deployment profile.

    The profile combines two coordinate spaces that must stay consistent:

    * Vision geometry: a fixed row-major patch grid followed by padding patches.
    * Text fusion: a fixed contiguous span that receives projected visual tokens.

    ``image_position_ids`` is the processor-facing name for the patch coordinate
    tensor. The Gemma4 vision model receives the same tensor under the argument
    name ``pixel_position_ids``.
    """

    name: str
    visual_start_idx: int
    num_visual_tokens: int
    max_soft_tokens: int
    patch_grid_height: int
    patch_grid_width: int
    patch_size: int
    pooling_kernel_size: int
    batch_size: int = 1
    tokenizer_add_bos_token: bool = False

    @property
    def num_valid_patches(self) -> int:
        """Return the number of non-padding image patches."""
        return self.patch_grid_height * self.patch_grid_width

    @property
    def num_patches(self) -> int:
        """Return the fixed padded patch capacity."""
        return self.max_soft_tokens * self.pooling_kernel_size**2

    @property
    def num_padding_patches(self) -> int:
        """Return the number of padding patches in the fixed input tensor."""
        return self.num_patches - self.num_valid_patches

    @property
    def soft_grid_height(self) -> int:
        """Return the pooled visual-token grid height."""
        return self.patch_grid_height // self.pooling_kernel_size

    @property
    def soft_grid_width(self) -> int:
        """Return the pooled visual-token grid width."""
        return self.patch_grid_width // self.pooling_kernel_size

    @property
    def image_height(self) -> int:
        """Return the canonical processed image height in pixels."""
        return self.patch_grid_height * self.patch_size

    @property
    def image_width(self) -> int:
        """Return the canonical processed image width in pixels."""
        return self.patch_grid_width * self.patch_size

    @property
    def patch_vector_size(self) -> int:
        """Return the flattened RGB patch width consumed by the vision tower."""
        return 3 * self.patch_size * self.patch_size

    @property
    def visual_end_idx(self) -> int:
        """Return the exclusive end of the text fusion span."""
        return self.visual_start_idx + self.num_visual_tokens

    def validate(
        self,
        *,
        max_seq_len: int | None = None,
        vision_config: Any | None = None,
    ) -> None:
        """Validate the internal geometry and optional model constraints."""
        if not self.name:
            raise ValueError("Gemma4 static vision profile name must be non-empty.")
        if self.batch_size != 1:
            raise ValueError(
                "Gemma4 static vision profiles currently require batch_size=1, "
                f"got {self.batch_size}."
            )
        for field_name in (
            "num_visual_tokens",
            "max_soft_tokens",
            "patch_grid_height",
            "patch_grid_width",
            "patch_size",
            "pooling_kernel_size",
        ):
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(
                    f"Gemma4 static vision profile {field_name} must be positive, "
                    f"got {value}."
                )
        if self.visual_start_idx < 0:
            raise ValueError(
                "Gemma4 static vision profile visual_start_idx must be "
                f"non-negative, got {self.visual_start_idx}."
            )
        if self.patch_grid_height % self.pooling_kernel_size:
            raise ValueError(
                "patch_grid_height must be divisible by pooling_kernel_size: "
                f"height={self.patch_grid_height}, "
                f"kernel={self.pooling_kernel_size}."
            )
        if self.patch_grid_width % self.pooling_kernel_size:
            raise ValueError(
                "patch_grid_width must be divisible by pooling_kernel_size: "
                f"width={self.patch_grid_width}, "
                f"kernel={self.pooling_kernel_size}."
            )

        if self.num_visual_tokens > self.max_soft_tokens:
            raise ValueError(
                "num_visual_tokens cannot exceed max_soft_tokens: "
                f"{self.num_visual_tokens} > {self.max_soft_tokens}."
            )
        derived_visual_tokens = self.soft_grid_height * self.soft_grid_width
        if derived_visual_tokens != self.num_visual_tokens:
            raise ValueError(
                "Gemma4 static vision profile visual-token count does not match "
                "the pooled patch grid: "
                f"configured={self.num_visual_tokens}, "
                f"derived={derived_visual_tokens}, "
                f"patch_grid=({self.patch_grid_height}, {self.patch_grid_width}), "
                f"pooling_kernel_size={self.pooling_kernel_size}."
            )
        if self.num_valid_patches > self.num_patches:
            raise ValueError(
                "The valid patch grid exceeds the padded patch capacity: "
                f"valid={self.num_valid_patches}, capacity={self.num_patches}."
            )
        if max_seq_len is not None and self.visual_end_idx > int(max_seq_len):
            raise ValueError(
                "The visual-token span exceeds max_seq_len: "
                f"span=[{self.visual_start_idx}, {self.visual_end_idx}), "
                f"max_seq_len={int(max_seq_len)}."
            )

        if vision_config is not None:
            config_patch_size = int(getattr(vision_config, "patch_size"))
            config_pooling_kernel_size = int(
                getattr(vision_config, "pooling_kernel_size")
            )
            config_max_soft_tokens = int(
                getattr(
                    vision_config,
                    "default_output_length",
                    self.max_soft_tokens,
                )
            )
            if config_patch_size != self.patch_size:
                raise ValueError(
                    "Static profile patch_size does not match the vision config: "
                    f"profile={self.patch_size}, config={config_patch_size}."
                )
            if config_pooling_kernel_size != self.pooling_kernel_size:
                raise ValueError(
                    "Static profile pooling_kernel_size does not match the vision "
                    f"config: profile={self.pooling_kernel_size}, "
                    f"config={config_pooling_kernel_size}."
                )
            if config_max_soft_tokens != self.max_soft_tokens:
                raise ValueError(
                    "Static profile max_soft_tokens does not match the vision "
                    f"config default_output_length: profile={self.max_soft_tokens}, "
                    f"config={config_max_soft_tokens}."
                )

    def validate_processor(self, processor: Any) -> None:
        """Validate tokenizer and image-processor settings used by the profile."""
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "add_bos_token"):
            actual_add_bos = bool(tokenizer.add_bos_token)
            if actual_add_bos != self.tokenizer_add_bos_token:
                raise ValueError(
                    "Tokenizer add_bos_token does not match the static profile: "
                    f"profile={self.tokenizer_add_bos_token}, "
                    f"processor={actual_add_bos}."
                )

        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return
        expected_fields = {
            "patch_size": self.patch_size,
            "pooling_kernel_size": self.pooling_kernel_size,
            "max_soft_tokens": self.max_soft_tokens,
        }
        for field_name, expected in expected_fields.items():
            if not hasattr(image_processor, field_name):
                continue
            actual_value = getattr(image_processor, field_name)
            if actual_value is None:
                continue
            actual = int(actual_value)
            if actual != expected:
                raise ValueError(
                    f"Image processor {field_name} does not match the static "
                    f"profile: profile={expected}, processor={actual}."
                )

        image_seq_length = getattr(processor, "image_seq_length", None)
        if (
            image_seq_length is not None
            and int(image_seq_length) != self.max_soft_tokens
        ):
            raise ValueError(
                "Processor image_seq_length does not match max_soft_tokens: "
                f"profile={self.max_soft_tokens}, "
                f"processor={int(image_seq_length)}."
            )

    def build_image_position_ids(
        self,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.LongTensor:
        """Build the canonical row-major patch coordinates and padding suffix."""
        self.validate()
        coords = torch.arange(
            self.num_valid_patches,
            dtype=torch.long,
            device=device,
        )
        valid_positions = torch.stack(
            (
                coords % self.patch_grid_width,
                coords // self.patch_grid_width,
            ),
            dim=-1,
        )
        padding_positions = torch.full(
            (self.num_padding_patches, 2),
            -1,
            dtype=torch.long,
            device=device,
        )
        return torch.cat((valid_positions, padding_positions), dim=0).unsqueeze(0)

    def position_ids_sha256(self) -> str:
        """Return a stable SHA-256 digest for the canonical position tensor."""
        position_ids = self.build_image_position_ids().contiguous()
        return hashlib.sha256(position_ids.numpy().tobytes()).hexdigest()

    def validate_image_position_ids(
        self,
        image_position_ids: torch.Tensor,
        *,
        tensor_name: str = "image_position_ids",
    ) -> None:
        """Require a processor position tensor to match this profile exactly."""
        if not isinstance(image_position_ids, torch.Tensor):
            raise TypeError(f"{tensor_name} must be a torch.Tensor.")
        actual = (
            image_position_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        )
        expected = self.build_image_position_ids()
        if torch.equal(actual, expected):
            return

        actual_digest = hashlib.sha256(actual.numpy().tobytes()).hexdigest()
        actual_valid = (
            int((actual[..., 0] >= 0).sum().item())
            if actual.dim() == 3 and actual.shape[-1] == 2
            else None
        )
        raise ValueError(
            f"{tensor_name} does not match Gemma4 static vision profile "
            f"{self.name!r}: expected_shape={tuple(expected.shape)}, "
            f"actual_shape={tuple(actual.shape)}, "
            f"expected_valid_patches={self.num_valid_patches}, "
            f"actual_valid_patches={actual_valid}, "
            f"expected_sha256={self.position_ids_sha256()}, "
            f"actual_sha256={actual_digest}."
        )

    def validate_text_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        image_token_id: int,
    ) -> None:
        """Validate the contiguous image-token span in processed text IDs."""
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError(
                    "Gemma4 static text fusion currently requires batch_size=1, "
                    f"got input_ids shape {tuple(input_ids.shape)}."
                )
            token_row = input_ids[0]
        elif input_ids.dim() == 1:
            token_row = input_ids
        else:
            raise ValueError(
                "input_ids must have shape (S,) or (1, S), got "
                f"{tuple(input_ids.shape)}."
            )

        positions = torch.nonzero(
            token_row == int(image_token_id),
            as_tuple=True,
        )[0]
        expected = torch.arange(
            self.visual_start_idx,
            self.visual_end_idx,
            device=positions.device,
        )
        if not torch.equal(positions, expected):
            raise ValueError(
                "Processed input_ids do not match the Gemma4 static text fusion "
                f"profile {self.name!r}: expected image-token span "
                f"[{self.visual_start_idx}, {self.visual_end_idx}), "
                f"actual_positions={positions.detach().cpu().tolist()}."
            )

    def validate_processor_outputs(
        self,
        outputs: Mapping[str, Any],
        *,
        image_token_id: int,
    ) -> None:
        """Validate one processor output against vision and text profile fields."""
        input_ids = outputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Processor output must contain tensor input_ids.")
        self.validate_text_input_ids(input_ids, image_token_id=image_token_id)

        image_position_ids = outputs.get("image_position_ids")
        if not isinstance(image_position_ids, torch.Tensor):
            raise ValueError("Processor output must contain tensor image_position_ids.")
        self.validate_image_position_ids(image_position_ids)

        pixel_values = outputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("Processor output must contain tensor pixel_values.")
        expected_pixel_shape = (
            self.batch_size,
            self.num_patches,
            self.patch_vector_size,
        )
        if tuple(pixel_values.shape) != expected_pixel_shape:
            raise ValueError(
                "pixel_values does not match the Gemma4 static vision profile: "
                f"expected_shape={expected_pixel_shape}, "
                f"actual_shape={tuple(pixel_values.shape)}."
            )

        soft_token_counts = outputs.get("num_soft_tokens_per_image")
        if soft_token_counts is not None:
            counts = torch.as_tensor(soft_token_counts).reshape(-1)
            if counts.numel() != self.batch_size or not torch.all(
                counts == self.num_visual_tokens
            ):
                raise ValueError(
                    "num_soft_tokens_per_image does not match the static "
                    f"profile: expected={self.num_visual_tokens}, "
                    f"actual={counts.tolist()}."
                )

    def validate_wrapped_model(self, wrapped_model: Any) -> None:
        """Validate profile fields copied into a wrapped Gemma4 model."""
        wrapped_start_idx = int(getattr(wrapped_model, "visual_start_idx"))
        wrapped_visual_tokens = int(getattr(wrapped_model, "num_visual_tokens"))
        if wrapped_start_idx != self.visual_start_idx:
            raise ValueError(
                "Wrapped Gemma4 visual_start_idx does not match the static "
                f"profile: wrapped={wrapped_start_idx}, "
                f"profile={self.visual_start_idx}."
            )
        if wrapped_visual_tokens != self.num_visual_tokens:
            raise ValueError(
                "Wrapped Gemma4 num_visual_tokens does not match the static "
                f"profile: wrapped={wrapped_visual_tokens}, "
                f"profile={self.num_visual_tokens}."
            )

    def to_vision_model_args(self) -> dict[str, Any]:
        """Return the normalized model_args.vision representation."""
        return {
            "profile": self.name,
            "visual_start_idx": self.visual_start_idx,
            "num_visual_tokens": self.num_visual_tokens,
            "max_soft_tokens": self.max_soft_tokens,
            "patch_grid_height": self.patch_grid_height,
            "patch_grid_width": self.patch_grid_width,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable deployment manifest."""
        manifest = asdict(self)
        manifest.update(
            {
                "num_valid_patches": self.num_valid_patches,
                "num_patches": self.num_patches,
                "num_padding_patches": self.num_padding_patches,
                "soft_grid_height": self.soft_grid_height,
                "soft_grid_width": self.soft_grid_width,
                "image_height": self.image_height,
                "image_width": self.image_width,
                "patch_vector_size": self.patch_vector_size,
                "visual_end_idx": self.visual_end_idx,
                "image_position_ids_shape": [self.batch_size, self.num_patches, 2],
                "image_position_ids_sha256": self.position_ids_sha256(),
            }
        )
        return manifest

    def save_manifest(self, path: str | Path) -> None:
        """Write the canonical profile manifest as formatted JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


_PROFILE_REGISTRY: dict[str, Gemma4StaticVisionProfile] = {
    "e2b_57x42_266": Gemma4StaticVisionProfile(
        name="e2b_57x42_266",
        visual_start_idx=1,
        num_visual_tokens=266,
        max_soft_tokens=280,
        patch_grid_height=42,
        patch_grid_width=57,
        patch_size=16,
        pooling_kernel_size=3,
    ),
    DEFAULT_GEMMA4_STATIC_VISION_PROFILE: Gemma4StaticVisionProfile(
        name=DEFAULT_GEMMA4_STATIC_VISION_PROFILE,
        visual_start_idx=1,
        num_visual_tokens=264,
        max_soft_tokens=280,
        patch_grid_height=36,
        patch_grid_width=66,
        patch_size=16,
        pooling_kernel_size=3,
    ),
}


def get_gemma4_static_vision_profile(name: str) -> Gemma4StaticVisionProfile:
    """Return a registered Gemma4 static vision profile by name."""
    try:
        profile = _PROFILE_REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(
            "Unknown Gemma4 static vision profile "
            f"{name!r}. Available profiles: {sorted(_PROFILE_REGISTRY)}."
        ) from exc
    profile.validate()
    return profile


def canonicalize_gemma4_static_vision_model_args(
    model_args: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expand a named profile into explicit ``model_args.vision`` fields."""
    normalized = deepcopy(dict(model_args or {}))
    vision = normalized.setdefault("vision", {})
    if not isinstance(vision, dict):
        raise TypeError("model_args.vision must be a mapping.")

    profile_name = vision.get("profile")
    if profile_name is None:
        return normalized

    profile = get_gemma4_static_vision_profile(str(profile_name))
    canonical = profile.to_vision_model_args()
    for key, expected in canonical.items():
        if key in vision and vision[key] != expected:
            raise ValueError(
                "model_args.vision conflicts with named Gemma4 static profile "
                f"{profile.name!r}: field={key!r}, "
                f"configured={vision[key]!r}, expected={expected!r}."
            )
    vision.update(canonical)
    return normalized


def build_gemma4_static_vision_profile(
    model_args: Mapping[str, Any],
    *,
    vision_config: Any,
    max_seq_len: int,
) -> Gemma4StaticVisionProfile:
    """Build and validate a named or inline Gemma4 static vision profile."""
    normalized = canonicalize_gemma4_static_vision_model_args(model_args)
    vision = normalized.get("vision", {})
    if not isinstance(vision, Mapping):
        raise TypeError("model_args.vision must be a mapping.")

    required = (
        "visual_start_idx",
        "num_visual_tokens",
        "max_soft_tokens",
        "patch_grid_height",
        "patch_grid_width",
    )
    missing = [key for key in required if key not in vision]
    if missing:
        raise ValueError(
            "Gemma4 static vision export requires model_args.vision fields: "
            + ", ".join(missing)
        )

    profile_name = vision.get("profile")
    if profile_name is not None:
        profile = get_gemma4_static_vision_profile(str(profile_name))
    else:
        profile = Gemma4StaticVisionProfile(
            name="custom",
            visual_start_idx=int(vision["visual_start_idx"]),
            num_visual_tokens=int(vision["num_visual_tokens"]),
            max_soft_tokens=int(vision["max_soft_tokens"]),
            patch_grid_height=int(vision["patch_grid_height"]),
            patch_grid_width=int(vision["patch_grid_width"]),
            patch_size=int(getattr(vision_config, "patch_size")),
            pooling_kernel_size=int(getattr(vision_config, "pooling_kernel_size")),
        )
    profile.validate(max_seq_len=max_seq_len, vision_config=vision_config)

    expected_height = profile.image_height
    expected_width = profile.image_width
    configured_height = vision.get("image_height")
    configured_width = vision.get("image_width")
    if configured_height is not None and int(configured_height) != expected_height:
        raise ValueError(
            "model_args.vision.image_height does not match the static patch grid: "
            f"configured={int(configured_height)}, expected={expected_height}."
        )
    if configured_width is not None and int(configured_width) != expected_width:
        raise ValueError(
            "model_args.vision.image_width does not match the static patch grid: "
            f"configured={int(configured_width)}, expected={expected_width}."
        )
    return profile

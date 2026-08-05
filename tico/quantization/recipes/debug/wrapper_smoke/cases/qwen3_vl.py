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

"""Smoke cases for Qwen3-VL wrapper checks."""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import (
    clone_module,
    first_tensor,
    smoke_section,
)


_QWEN3_VL_SIZE_PROFILE_TINY = "tiny"
_QWEN3_VL_SIZE_PROFILE_4B_DIMS = "qwen3_vl_4b_dims"
_QWEN3_VL_SIZE_PROFILE_4B_STATIC_RUNTIME = "qwen3_vl_4b_static_runtime"
_QWEN3_VL_SIZE_PROFILES = frozenset(
    {
        _QWEN3_VL_SIZE_PROFILE_TINY,
        _QWEN3_VL_SIZE_PROFILE_4B_DIMS,
        _QWEN3_VL_SIZE_PROFILE_4B_STATIC_RUNTIME,
    }
)
_QWEN3_VL_4B_WIDTH_PROFILES = frozenset(
    {
        _QWEN3_VL_SIZE_PROFILE_4B_DIMS,
        _QWEN3_VL_SIZE_PROFILE_4B_STATIC_RUNTIME,
    }
)

_QWEN3_VL_4B_STATIC_MAX_SEQ = 2_048
_QWEN3_VL_4B_STATIC_GRID_THW = (1, 54, 72)
_QWEN3_VL_4B_STATIC_VISUAL_CAPACITY = 1_000
_QWEN3_VL_4B_STATIC_NON_VISUAL_TOKENS = 14
_QWEN3_VL_4B_STATIC_VISUAL_START_IDX = 4
_QWEN3_VL_4B_SPATIAL_MERGE_SIZE = 2

_QWEN3_VL_4B_SUPPORTED_CASES = frozenset(
    {
        "qwen3_vl_text_attention_prefill",
        "qwen3_vl_text_attention_decode",
        "qwen3_vl_text_mlp",
        "qwen3_vl_text_decoder_layer_prefill",
        "qwen3_vl_text_decoder_layer_decode",
        "qwen3_vl_vision_attention",
        "qwen3_vl_vision_mlp",
        "qwen3_vl_vision_block",
        "qwen3_vl_vision_patch_embed",
        "qwen3_vl_vision_patch_merger",
        "qwen3_vl_vision_model",
    }
)


@dataclass(frozen=True)
class Qwen3VLStaticRuntimeShape:
    """Fixed Qwen3-VL-4B text and image shape contract used by TICO."""

    max_seq: int = _QWEN3_VL_4B_STATIC_MAX_SEQ
    grid_thw: tuple[int, int, int] = _QWEN3_VL_4B_STATIC_GRID_THW
    visual_capacity: int = _QWEN3_VL_4B_STATIC_VISUAL_CAPACITY
    non_visual_tokens: int = _QWEN3_VL_4B_STATIC_NON_VISUAL_TOKENS
    visual_start_idx: int = _QWEN3_VL_4B_STATIC_VISUAL_START_IDX
    spatial_merge_size: int = _QWEN3_VL_4B_SPATIAL_MERGE_SIZE

    def __post_init__(self) -> None:
        if self.max_seq < 2:
            raise ValueError(
                f"Qwen3-VL static max_seq must be at least 2, got {self.max_seq}."
            )
        if len(self.grid_thw) != 3 or any(value <= 0 for value in self.grid_thw):
            raise ValueError(
                "Qwen3-VL static grid_thw must contain three positive integers, "
                f"got {self.grid_thw}."
            )
        if self.spatial_merge_size <= 0:
            raise ValueError("Qwen3-VL spatial_merge_size must be positive.")
        if self.visual_capacity <= 0:
            raise ValueError("Qwen3-VL static visual_capacity must be positive.")
        if self.visual_capacity > self.max_seq:
            raise ValueError(
                "Qwen3-VL static visual_capacity must not exceed max_seq, got "
                f"{self.visual_capacity} > {self.max_seq}."
            )
        if self.non_visual_tokens < 0:
            raise ValueError("Qwen3-VL static non_visual_tokens must be non-negative.")

        _, grid_h, grid_w = self.grid_thw
        if grid_h % self.spatial_merge_size != 0:
            raise ValueError(
                "Qwen3-VL static grid height must be divisible by "
                f"spatial_merge_size={self.spatial_merge_size}."
            )
        if grid_w % self.spatial_merge_size != 0:
            raise ValueError(
                "Qwen3-VL static grid width must be divisible by "
                f"spatial_merge_size={self.spatial_merge_size}."
            )
        if self.visual_capacity < self.num_visual_tokens:
            raise ValueError(
                "Qwen3-VL static visual_capacity must cover all merged visual "
                f"tokens, got {self.visual_capacity} < {self.num_visual_tokens}."
            )
        if self.valid_seq_len > self.max_seq:
            raise ValueError(
                "Qwen3-VL static non-visual and visual tokens do not fit in "
                f"max_seq: {self.valid_seq_len} > {self.max_seq}."
            )
        if self.visual_start_idx < 0:
            raise ValueError("Qwen3-VL static visual_start_idx must be non-negative.")
        if self.visual_start_idx > self.non_visual_tokens:
            raise ValueError(
                "Qwen3-VL static visual_start_idx must fit within the non-visual "
                "token budget."
            )
        if self.visual_start_idx + self.num_visual_tokens > self.valid_seq_len:
            raise ValueError(
                "Qwen3-VL static visual segment does not fit in the valid sequence."
            )

    @property
    def num_patch_tokens(self) -> int:
        """Return the number of vision tokens before spatial merging."""
        grid_t, grid_h, grid_w = self.grid_thw
        return grid_t * grid_h * grid_w

    @property
    def num_visual_tokens(self) -> int:
        """Return the number of merged visual tokens inserted into text."""
        grid_t, grid_h, grid_w = self.grid_thw
        merge = self.spatial_merge_size
        return grid_t * (grid_h // merge) * (grid_w // merge)

    @property
    def valid_seq_len(self) -> int:
        """Return the unpadded logical sequence length."""
        return self.non_visual_tokens + self.num_visual_tokens

    @property
    def visual_arena_start(self) -> int:
        """Return the first physical slot reserved for visual tokens."""
        return self.max_seq - self.visual_capacity


def _has_qwen3_vl() -> CaseAvailability:
    """Return availability for Hugging Face Qwen3-VL modules."""
    try:
        from tico.quantization.wrapq.utils.version import has_transformers_for

        if not has_transformers_for("qwen3-vl"):
            return CaseAvailability(
                False, "required transformers Qwen3-VL modules are unavailable"
            )
        return CaseAvailability(True)
    except Exception as exc:
        return CaseAvailability(False, f"failed to check Qwen3-VL availability: {exc}")


def _set_eager_attention(cfg: Any) -> Any:
    """Set eager attention on configs that expose a configurable implementation."""
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _qwen3_vl_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the Qwen3-VL-specific wrapper-smoke configuration mapping."""
    section = smoke_section(cfg)
    qwen_cfg = section.get("qwen3_vl", {})
    if not isinstance(qwen_cfg, Mapping):
        raise ValueError("debug.wrapper_smoke.qwen3_vl must be a mapping.")
    return qwen_cfg


def _qwen3_vl_size_profile(cfg: Mapping[str, Any]) -> str:
    """Return and validate the requested Qwen3-VL smoke size profile."""
    qwen_cfg = _qwen3_vl_options(cfg)
    profile = (
        str(qwen_cfg.get("size_profile", _QWEN3_VL_SIZE_PROFILE_TINY)).strip().lower()
    )
    if profile not in _QWEN3_VL_SIZE_PROFILES:
        choices = ", ".join(sorted(_QWEN3_VL_SIZE_PROFILES))
        raise ValueError(
            f"Unsupported Qwen3-VL wrapper-smoke size profile '{profile}'. "
            f"Expected one of: {choices}."
        )
    return profile


def _parse_grid_thw(value: Any) -> tuple[int, int, int]:
    """Parse one static ``(T, H, W)`` image-grid value."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            "debug.wrapper_smoke.qwen3_vl.static_runtime.grid_thw must be "
            "a three-element list or tuple."
        )
    return int(value[0]), int(value[1]), int(value[2])


def _qwen3_vl_static_runtime_shape(
    cfg: Mapping[str, Any],
) -> Qwen3VLStaticRuntimeShape:
    """Parse and validate the Qwen3-VL-4B static-runtime options."""
    qwen_cfg = _qwen3_vl_options(cfg)
    static_cfg = qwen_cfg.get("static_runtime", {})
    if not isinstance(static_cfg, Mapping):
        raise ValueError(
            "debug.wrapper_smoke.qwen3_vl.static_runtime must be a mapping."
        )

    return Qwen3VLStaticRuntimeShape(
        max_seq=int(static_cfg.get("max_seq", _QWEN3_VL_4B_STATIC_MAX_SEQ)),
        grid_thw=_parse_grid_thw(
            static_cfg.get("grid_thw", _QWEN3_VL_4B_STATIC_GRID_THW)
        ),
        visual_capacity=int(
            static_cfg.get("visual_capacity", _QWEN3_VL_4B_STATIC_VISUAL_CAPACITY)
        ),
        non_visual_tokens=int(
            static_cfg.get("non_visual_tokens", _QWEN3_VL_4B_STATIC_NON_VISUAL_TOKENS)
        ),
        visual_start_idx=int(
            static_cfg.get("visual_start_idx", _QWEN3_VL_4B_STATIC_VISUAL_START_IDX)
        ),
    )


def _build_text_config(*, size_profile: str, max_seq: int) -> Any:
    """Create a tiny or Qwen3-VL-4B-width text configuration.

    Real-width profiles retain one synthetic decoder layer. The context capacity
    follows the smoke/runtime input rather than the checkpoint's 262,144-token
    limit so wrapper-owned static masks remain bounded.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig

    if size_profile == _QWEN3_VL_SIZE_PROFILE_TINY:
        params: dict[str, Any] = {
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "max_position_embeddings": max_seq,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
            },
        }
    elif size_profile in _QWEN3_VL_4B_WIDTH_PROFILES:
        params = {
            "vocab_size": 151_936,
            "hidden_size": 2_560,
            "intermediate_size": 9_728,
            "num_hidden_layers": 1,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "max_position_embeddings": max_seq,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000.0,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            },
        }
    else:
        raise AssertionError(f"Unhandled Qwen3-VL size profile: {size_profile}")

    text_cfg = Qwen3VLTextConfig(
        **params,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
    )
    return _set_eager_attention(text_cfg)


def _build_vision_config(*, size_profile: str, **overrides: Any) -> Any:
    """Create a tiny or Qwen3-VL-4B-width bounded vision config."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    if size_profile == _QWEN3_VL_SIZE_PROFILE_TINY:
        params: dict[str, Any] = {
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1],
        }
    elif size_profile in _QWEN3_VL_4B_WIDTH_PROFILES:
        params = {
            "hidden_size": 1_024,
            "intermediate_size": 4_096,
            "num_heads": 16,
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "in_channels": 3,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 2_560,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2_304,
            "deepstack_visual_indexes": [0],
        }
    else:
        raise AssertionError(f"Unhandled Qwen3-VL size profile: {size_profile}")

    params.update(overrides)
    return _set_eager_attention(Qwen3VLVisionConfig(**params))


def _make_bounded_vision_model(
    *, size_profile: str, **overrides: Any
) -> torch.nn.Module:
    """Build a vision-only Qwen3-VL model without allocating text embeddings.

    The tiny profile preserves the original two-layer smoke configuration. The
    real-width profiles use one vision layer because only block zero is consumed
    by the bounded attention/MLP/block cases.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    model_overrides = dict(overrides)
    if size_profile in _QWEN3_VL_4B_WIDTH_PROFILES:
        model_overrides.setdefault("depth", 1)
        model_overrides.setdefault("deepstack_visual_indexes", [0])

    vision_cfg = _build_vision_config(
        size_profile=size_profile,
        **model_overrides,
    )
    return Qwen3VLVisionModel(vision_cfg).eval()


def _make_tiny_qwen3vl_config() -> Any:
    """Create a tiny Qwen3-VL config that is large enough for image-token tests."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1],
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "num_hidden_layers": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 1024,
            "vocab_size": 1000,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=998,
        video_token_id=999,
    )


def _make_tiny_qwen3vl_model() -> torch.nn.Module:
    """Build the existing tiny multimodal model without downloading weights."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    return Qwen3VLModel(_make_tiny_qwen3vl_config()).eval()


def _rope(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic rotary position embeddings."""
    emb = torch.randn(seq_len, head_dim)
    return emb.cos(), emb.sin()


def _text_rope(
    batch_size: int, seq_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic text RoPE embeddings."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _get_position_embeddings(visual_model: torch.nn.Module, grid_thw: torch.Tensor):
    """Return Qwen3-VL vision RoPE embeddings for a synthetic image grid."""
    pos_embeds = visual_model.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb = visual_model.rot_pos_emb(grid_thw)
    seq_len, _ = pos_embeds.size()
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return emb.cos(), emb.sin()


def _get_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Return cumulative sequence lengths for one synthetic Qwen3-VL image grid."""
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
    return torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)


def _make_ptq_config(
    cfg: Any, thw: Tuple[int, int, int], visual_start_idx: int = 0
) -> Any:
    """Create the PTQ config used by synthetic Qwen3-VL model examples."""
    from tico.quantization.config.ptq import PTQConfig

    return PTQConfig(
        model_args={
            "vision": {
                "grid_thw": thw,
                "visual_start_idx": visual_start_idx,
                "spatial_merge_size": cfg.vision_config.spatial_merge_size,
            }
        }
    )


def _create_patchified_pixel_values(
    vision_cfg: Any,
    thw: Tuple[int, int, int],
) -> torch.Tensor:
    """Create batch-one flattened patches for one synthetic visual item."""
    num_patches = thw[0] * thw[1] * thw[2]
    patch_dim = (
        vision_cfg.in_channels
        * vision_cfg.temporal_patch_size
        * vision_cfg.patch_size
        * vision_cfg.patch_size
    )
    return torch.randn(1, num_patches, patch_dim)


def _compute_3d_position_ids(
    input_ids: torch.Tensor,
    thw: Tuple[int, int, int],
    spatial_merge_size: int,
    image_token_id: int,
) -> torch.Tensor:
    """Compute multimodal 3D RoPE position IDs for a single visual segment."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    position_ids = torch.ones(
        3, batch_size, seq_len, dtype=input_ids.dtype, device=device
    )
    for i in range(batch_size):
        image_mask = input_ids[i] == image_token_id
        image_positions = torch.nonzero(image_mask, as_tuple=True)[0]
        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        if len(image_positions) > 0:
            start_pos = image_positions[0].item()
            text_len = start_pos - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )
            llm_grid_t = 1
            llm_grid_h = thw[1] // spatial_merge_size
            llm_grid_w = thw[2] // spatial_merge_size
            t_index = (
                torch.arange(llm_grid_t, device=device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)
            num_visual_tokens = (thw[1] // spatial_merge_size) * (
                thw[2] // spatial_merge_size
            )
            st = start_pos + num_visual_tokens
        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
            )
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, :] = llm_positions
    return position_ids


def _create_image_input(
    cfg: Any,
    seq_len: int,
    thw: Tuple[int, int, int],
    *,
    visual_start_idx: int = 0,
    include_generation_fields: bool = False,
) -> dict[str, Any]:
    """Create one synthetic Qwen3-VL image prompt without a processor."""
    spatial_merge_size = cfg.vision_config.spatial_merge_size
    num_visual_tokens = (thw[1] // spatial_merge_size) * (thw[2] // spatial_merge_size)
    if visual_start_idx + num_visual_tokens > seq_len:
        raise ValueError("visual tokens do not fit into the synthetic sequence")
    input_ids = torch.randint(
        0, cfg.text_config.vocab_size - 2, (1, seq_len), dtype=torch.long
    )
    input_ids[
        0, visual_start_idx : visual_start_idx + num_visual_tokens
    ] = cfg.image_token_id
    pixel_values = _create_patchified_pixel_values(cfg.vision_config, thw)
    image_grid_thw = torch.tensor([thw])
    position_ids = _compute_3d_position_ids(
        input_ids, thw, spatial_merge_size, cfg.image_token_id
    )
    example: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": None,
        "position_ids": position_ids,
        "past_key_values": None,
        "inputs_embeds": None,
        "pixel_values": pixel_values,
        "pixel_values_videos": None,
        "image_grid_thw": image_grid_thw,
        "video_grid_thw": None,
        "cache_position": None,
    }
    if include_generation_fields:
        example["labels"] = None
        example["logits_to_keep"] = 0
    return example


def _causal_mask(seq_len: int, fill_value: float = -120.0) -> torch.Tensor:
    mask = torch.full((1, 1, seq_len, seq_len), fill_value)
    return torch.triu(mask, diagonal=1)


class QwenBaseCase(WrapperSmokeCase):
    """Base class for Qwen3-VL wrapper smoke cases with size profiles."""

    tags: tuple[str, ...] = ("qwen3_vl",)

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Qwen3-VL modules."""
        return _has_qwen3_vl()

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Reject unsupported real-width cases before model construction."""
        self._validated_size_profile(cfg)

    def _validated_size_profile(self, cfg: Mapping[str, Any]) -> str:
        """Validate the case/profile pair and cache the static shape."""
        profile = _qwen3_vl_size_profile(cfg)
        if (
            profile in _QWEN3_VL_4B_WIDTH_PROFILES
            and self.name not in _QWEN3_VL_4B_SUPPORTED_CASES
        ):
            supported = ", ".join(sorted(_QWEN3_VL_4B_SUPPORTED_CASES))
            raise ValueError(
                f"Case '{self.name}' does not support Qwen3-VL size profile "
                f"'{profile}'. The profile is limited to bounded module-level "
                f"and one-layer vision cases: {supported}."
            )

        self._active_size_profile = profile
        self._active_static_runtime_shape = (
            _qwen3_vl_static_runtime_shape(cfg)
            if profile == _QWEN3_VL_SIZE_PROFILE_4B_STATIC_RUNTIME
            else None
        )
        return profile

    def _static_runtime_shape(self) -> Qwen3VLStaticRuntimeShape | None:
        """Return the active static-runtime shape after validation."""
        return getattr(self, "_active_static_runtime_shape", None)

    def _is_wide_profile(self, cfg: Mapping[str, Any]) -> bool:
        """Return whether the selected profile uses Qwen3-VL-4B widths."""
        return self._validated_size_profile(cfg) in _QWEN3_VL_4B_WIDTH_PROFILES

    def _text_seq_len(self, default: int) -> int:
        """Return the text prefill/decode capacity for the active profile."""
        shape = self._static_runtime_shape()
        return shape.max_seq if shape is not None else int(default)

    def _vision_grid_tuple(self, default: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the fixed image grid for the active profile."""
        shape = self._static_runtime_shape()
        return shape.grid_thw if shape is not None else default

    def _vision_patch_seq_len(self, default: int) -> int:
        """Return the number of pre-merge vision patch tokens."""
        shape = self._static_runtime_shape()
        return shape.num_patch_tokens if shape is not None else int(default)

    def _batch_size(self, default: int) -> int:
        """Use batch one for real-width profiles and preserve tiny behavior."""
        profile = getattr(self, "_active_size_profile", _QWEN3_VL_SIZE_PROFILE_TINY)
        return 1 if profile in _QWEN3_VL_4B_WIDTH_PROFILES else int(default)

    def _calibration_sample_count(self, cfg: Mapping[str, Any], *, default: int) -> int:
        """Avoid retaining multiple full static-runtime samples in memory."""
        profile = self._validated_size_profile(cfg)
        return 1 if profile == _QWEN3_VL_SIZE_PROFILE_4B_STATIC_RUNTIME else default

    def _make_text_config(
        self, cfg: Mapping[str, Any], *, tiny_max_seq: int = 128
    ) -> Any:
        """Create a profile-aware one-layer Qwen3-VL text config."""
        profile = self._validated_size_profile(cfg)
        shape = self._static_runtime_shape()
        max_seq = shape.max_seq if shape is not None else int(tiny_max_seq)
        return _build_text_config(size_profile=profile, max_seq=max_seq)

    def _make_vision_config(self, cfg: Mapping[str, Any], **overrides: Any) -> Any:
        """Create a profile-aware bounded Qwen3-VL vision config."""
        return _build_vision_config(
            size_profile=self._validated_size_profile(cfg),
            **overrides,
        )

    def _make_vision_model(
        self, cfg: Mapping[str, Any], **overrides: Any
    ) -> torch.nn.Module:
        """Create a profile-aware vision-only Qwen3-VL model."""
        return _make_bounded_vision_model(
            size_profile=self._validated_size_profile(cfg),
            **overrides,
        )

    def prepare_model(
        self, model: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Prepare real-width modules in place to limit peak host memory."""
        from tico.quantization import prepare

        inplace = self.inplace_prepare or self._is_wide_profile(cfg)
        return prepare(model, self.ptq_config(cfg), inplace=inplace)

    def convert_model(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Convert real-width modules in place to limit peak host memory."""
        from tico.quantization import convert

        inplace = self.inplace_convert or self._is_wide_profile(cfg)
        return convert(prepared, inplace=inplace)

    def export_filename(self, cfg: Mapping[str, Any]) -> str:
        """Include non-default profiles in generated Circle filenames."""
        profile = self._validated_size_profile(cfg)
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            return super().export_filename(cfg)
        return f"{self.name}.{profile}.q.circle"


class QwenTextAttentionBaseCase(QwenBaseCase):
    """Base class for Qwen3-VL text attention smoke cases."""

    tags: tuple[str, ...] = ("qwen3_vl", "text", "attention")
    max_mean_abs_diff = 2.0
    seq_len = 8
    export_mode = "prefill"

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text attention module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg)
        self.seq_len = self._text_seq_len(8)
        module = Qwen3VLTextAttention(self.text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped text attention module in the case-specific mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module(self.export_mode).eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )


class QwenTextAttentionPrefillCase(QwenTextAttentionBaseCase):
    """Smoke case for the Qwen3-VL text attention prefill wrapper path."""

    name = "qwen3_vl_text_attention_prefill"
    description = "Quantize one tiny Qwen3-VL text attention module in prefill mode."
    tags = ("qwen3_vl", "text", "attention", "prefill")
    export_mode = "prefill"

    def _sample(self) -> ForwardInput:
        """Create one synthetic prefill text attention input."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        rope = _text_rope(1, self.seq_len, self.text_cfg.head_dim)
        return ForwardInput((hidden, rope))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create prefill text attention calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the prefill text attention evaluation sample."""
        return self._sample()

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original text attention prefill signature with an explicit mask."""
        hidden, rope = sample.args
        mask = _causal_mask(hidden.shape[1])
        return reference(hidden, position_embeddings=rope, attention_mask=mask)[0]

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional prefill export inputs for the text attention adapter."""
        return ForwardInput(eval_sample.args)


class QwenTextAttentionDecodeCase(QwenTextAttentionBaseCase):
    """Smoke case for the Qwen3-VL text attention decode wrapper path."""

    name = "qwen3_vl_text_attention_decode"
    description = (
        "Quantize one tiny Qwen3-VL text attention module in static decode mode."
    )
    tags = ("qwen3_vl", "text", "attention", "decode")
    compare_reference_source = "prepared"
    export_mode = "decode"

    def _sample(self) -> ForwardInput:
        """Create one synthetic static decode text attention input."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        rope = _text_rope(1, 1, self.text_cfg.head_dim)
        mask = torch.zeros(1, 1, self.seq_len)
        past_k = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            self.seq_len - 1,
            self.text_cfg.head_dim,
        )
        past_v = torch.randn_like(past_k)
        return ForwardInput(
            (hidden, rope),
            {
                "attention_mask": mask,
                "past_key_values": (past_k, past_v),
                "use_cache": True,
            },
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create static decode text attention calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the static decode text attention evaluation sample."""
        return self._sample()

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional static decode inputs expected by the export adapter."""
        hidden, pos = eval_sample.args
        mask = eval_sample.kwargs["attention_mask"]
        past = eval_sample.kwargs["past_key_values"]
        return ForwardInput((hidden, pos, mask, past))


class QwenTextMLPCase(QwenBaseCase):
    """Smoke case for qwen/quantize_text_mlp.py."""

    name = "qwen3_vl_text_mlp"
    description = "Quantize one tiny Qwen3-VL text MLP module."
    tags = ("qwen3_vl", "text", "mlp")
    max_mean_abs_diff = 2.0

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text MLP module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg)
        self.seq_len = self._text_seq_len(8)
        self.batch_size = self._batch_size(2)
        module = Qwen3VLTextMLP(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one profile-aware text MLP input."""
        return ForwardInput(
            (
                torch.randn(
                    self.batch_size,
                    self.seq_len,
                    self.text_cfg.hidden_size,
                ),
            )
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text MLP calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text MLP evaluation sample."""
        return self._sample()


class QwenTextDecoderLayerBaseCase(QwenBaseCase):
    """Base class for Qwen3-VL text decoder-layer smoke cases."""

    tags: tuple[str, ...] = ("qwen3_vl", "text", "decoder_layer")
    max_mean_abs_diff = 3.0
    inplace_convert = True
    seq_len = 8
    export_mode = "prefill"

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text decoder layer and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLTextDecoderLayer,
        )

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg)
        self.seq_len = self._text_seq_len(8)
        module = Qwen3VLTextDecoderLayer(self.text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped text decoder layer in the case-specific mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module(self.export_mode).eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )


class QwenTextDecoderLayerPrefillCase(QwenTextDecoderLayerBaseCase):
    """Smoke case for the Qwen3-VL text decoder-layer prefill wrapper path."""

    name = "qwen3_vl_text_decoder_layer_prefill"
    description = "Quantize one tiny Qwen3-VL text decoder layer in prefill mode."
    tags = ("qwen3_vl", "text", "decoder_layer", "prefill")
    export_mode = "prefill"

    def _sample(self) -> ForwardInput:
        """Create one synthetic text decoder-layer prefill input."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        pos = _text_rope(1, self.seq_len, self.text_cfg.head_dim)
        mask = _causal_mask(self.seq_len)
        position_ids = torch.arange(self.seq_len).unsqueeze(0)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": pos,
                "attention_mask": mask,
                "position_ids": position_ids,
            },
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text decoder-layer prefill calibration samples."""
        count = self._calibration_sample_count(cfg, default=5)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text decoder-layer prefill evaluation sample."""
        return self._sample()

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional prefill inputs expected by the export adapter."""
        hidden = eval_sample.kwargs["hidden_states"]
        mask = eval_sample.kwargs["attention_mask"]
        pos = eval_sample.kwargs["position_embeddings"]
        return ForwardInput((hidden, mask, pos))


class QwenTextDecoderLayerDecodeCase(QwenTextDecoderLayerBaseCase):
    """Smoke case for the Qwen3-VL text decoder-layer decode wrapper path."""

    name = "qwen3_vl_text_decoder_layer_decode"
    description = "Quantize one tiny Qwen3-VL text decoder layer in static decode mode."
    tags = ("qwen3_vl", "text", "decoder_layer", "decode")
    compare_reference_source = "prepared"
    export_mode = "decode"

    def after_prepare(self, prepared: torch.nn.Module, cfg: Mapping[str, Any]) -> None:
        """Force tuple return so hidden states and cache tensors are available."""
        wrapped = getattr(prepared, "wrapped", prepared)
        if hasattr(wrapped, "return_type"):
            wrapped.return_type = "tuple"

    def _sample(self) -> ForwardInput:
        """Create one synthetic static decode text decoder-layer input."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        pos = _text_rope(1, 1, self.text_cfg.head_dim)
        mask = torch.zeros(1, 1, 1, self.seq_len)
        position_ids = torch.full((1, 1), self.seq_len - 1, dtype=torch.long)
        past_k = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            self.seq_len - 1,
            self.text_cfg.head_dim,
        )
        past_v = torch.randn_like(past_k)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": pos,
                "attention_mask": mask,
                "position_ids": position_ids,
                "past_key_values": (past_k, past_v),
                "use_cache": True,
            },
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text decoder-layer decode calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text decoder-layer decode evaluation sample."""
        return self._sample()

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional static decode inputs expected by the export adapter."""
        hidden = eval_sample.kwargs["hidden_states"]
        mask = eval_sample.kwargs["attention_mask"]
        pos = eval_sample.kwargs["position_embeddings"]
        past = eval_sample.kwargs["past_key_values"]
        return ForwardInput((hidden, mask, pos, past))


class QwenTextModelCase(QwenBaseCase):
    """Smoke case for qwen/quantize_text_model.py."""

    name = "qwen3_vl_text_model"
    description = "Quantize a tiny Qwen3-VL text model."
    tags = ("qwen3_vl", "text", "model")
    max_mean_abs_diff = 5.0
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny text model and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg)
        module = Qwen3VLTextModel(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic text-model input."""
        ids = torch.randint(0, self.text_cfg.vocab_size, (1, 8))
        return ForwardInput((), {"input_ids": ids, "use_cache": False})

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create text-model calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the text-model evaluation sample."""
        return self._sample()

    def output_tensor(self, output: Any) -> torch.Tensor:
        """Select last_hidden_state from model outputs."""
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return first_tensor(output)


class QwenVisionMLPCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_mlp.py."""

    name = "qwen3_vl_vision_mlp"
    description = "Quantize one tiny Qwen3-VL vision MLP module."
    tags = ("qwen3_vl", "vision", "mlp")
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 1.0

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision MLP module and reference copy."""
        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            visual = _make_tiny_qwen3vl_model().visual
        else:
            visual = self._make_vision_model(cfg)
        self.vision_cfg = visual.config
        self.hidden_size = self.vision_cfg.hidden_size
        self.seq_len = self._vision_patch_seq_len(16)
        module = visual.blocks[0].mlp.eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one profile-aware vision MLP input."""
        return ForwardInput((torch.randn(self.seq_len, self.hidden_size),))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision MLP calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision MLP evaluation sample."""
        return self._sample()


class QwenVisionAttentionCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_attention.py."""

    name = "qwen3_vl_vision_attention"
    description = "Quantize one tiny Qwen3-VL vision attention module."
    tags = ("qwen3_vl", "vision", "attention")
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 1.5

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision attention module and reference copy."""
        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            visual = _make_tiny_qwen3vl_model().visual
        else:
            visual = self._make_vision_model(cfg)
        self.hidden_size = visual.config.hidden_size
        self.grid_tuple = self._vision_grid_tuple((1, 8, 8))
        self.grid_thw = torch.tensor([self.grid_tuple], dtype=torch.long)
        self.cu_seqlens = _get_cu_seqlens(self.grid_thw)
        self.position_embeddings = _get_position_embeddings(visual, self.grid_thw)
        self.seq_len = int(self.cu_seqlens[-1].item())
        module = visual.blocks[0].attn.eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic vision attention input."""
        hidden = torch.randn(self.seq_len, self.hidden_size)
        return ForwardInput((hidden, self.cu_seqlens, None, self.position_embeddings))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision attention calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision attention evaluation sample."""
        return self._sample()


class QwenVisionBlockCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_block.py."""

    name = "qwen3_vl_vision_block"
    description = "Quantize one tiny Qwen3-VL vision block."
    tags = ("qwen3_vl", "vision", "block")
    max_mean_abs_diff = 3.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a profile-aware vision block and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            self.vision_cfg = self._make_vision_config(cfg)
            self.seq_len = 8
            self.cu_seqlens = torch.tensor([0, self.seq_len])
            self.position_embeddings = _rope(
                self.seq_len,
                self.vision_cfg.hidden_size // self.vision_cfg.num_heads,
            )
            module = Qwen3VLVisionBlock(self.vision_cfg).eval()
        else:
            visual = self._make_vision_model(cfg)
            self.vision_cfg = visual.config
            self.grid_tuple = self._vision_grid_tuple((1, 2, 4))
            self.grid_thw = torch.tensor([self.grid_tuple], dtype=torch.long)
            self.cu_seqlens = _get_cu_seqlens(self.grid_thw)
            self.position_embeddings = _get_position_embeddings(visual, self.grid_thw)
            self.seq_len = int(self.cu_seqlens[-1].item())
            module = visual.blocks[0].eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one profile-aware vision block input."""
        hidden = torch.randn(self.seq_len, self.vision_cfg.hidden_size)
        return ForwardInput(
            (hidden, self.cu_seqlens),
            {"position_embeddings": self.position_embeddings},
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision block calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision block evaluation sample."""
        return self._sample()


class QwenVisionPatchEmbedCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_patch_embed.py."""

    name = "qwen3_vl_vision_patch_embed"
    description = "Quantize one tiny Qwen3-VL vision patch-embed module."
    tags = ("qwen3_vl", "vision", "patch_embed")
    max_mean_abs_diff = 3.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision patch-embed module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        overrides: dict[str, Any] = {"in_channels": 3}
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            overrides.update(
                hidden_size=32,
                temporal_patch_size=2,
                patch_size=16,
            )
        self.vision_cfg = self._make_vision_config(cfg, **overrides)
        self.grid_tuple = self._vision_grid_tuple((1, 2, 2))
        module = Qwen3VLVisionPatchEmbed(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one profile-aware patch-embed input."""
        return ForwardInput(
            (_create_patchified_pixel_values(self.vision_cfg, self.grid_tuple),)
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create patch-embed calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the patch-embed evaluation sample."""
        return self._sample()


class QwenVisionPatchMergerCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_patch_merger.py."""

    name = "qwen3_vl_vision_patch_merger"
    description = "Quantize one tiny Qwen3-VL vision patch-merger module."
    tags = ("qwen3_vl", "vision", "patch_merger")
    max_mean_abs_diff = 3.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision patch-merger module and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        overrides: dict[str, Any] = {}
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            overrides.update(
                hidden_size=32,
                spatial_merge_size=2,
                out_hidden_size=64,
            )
        self.vision_cfg = self._make_vision_config(cfg, **overrides)
        self.seq_len = self._vision_patch_seq_len(8)
        module = Qwen3VLVisionPatchMerger(
            self.vision_cfg, use_postshuffle_norm=False
        ).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one profile-aware patch-merger input."""
        return ForwardInput((torch.randn(self.seq_len, self.vision_cfg.hidden_size),))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create patch-merger calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the patch-merger evaluation sample."""
        return self._sample()


class QwenVisionModelCase(QwenBaseCase):
    """Smoke case for qwen/quantize_vision_model.py."""

    name = "qwen3_vl_vision_model"
    description = "Quantize a tiny Qwen3-VL vision model."
    tags = ("qwen3_vl", "vision", "model")
    max_mean_abs_diff = 5.0
    inplace_prepare = True
    inplace_convert = True

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the vision PTQ config with static grid metadata."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"vision": {"grid_thw": self.grid_tuple}})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny vision model and reference copy."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        overrides: dict[str, Any] = {
            "depth": 1,
            "deepstack_visual_indexes": [0],
        }
        if profile == _QWEN3_VL_SIZE_PROFILE_TINY:
            overrides["num_position_embeddings"] = 64
        self.vision_cfg = self._make_vision_config(cfg, **overrides)
        self.grid_tuple = self._vision_grid_tuple((1, 2, 2))
        self.grid_thw = torch.tensor([self.grid_tuple])
        module = Qwen3VLVisionModel(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic vision-model input."""
        pixel_values = _create_patchified_pixel_values(self.vision_cfg, self.grid_tuple)
        return ForwardInput((pixel_values, self.grid_thw))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create vision-model calibration samples."""
        count = self._calibration_sample_count(cfg, default=2)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the vision-model evaluation sample."""
        return self._sample()

    def output_tensor(self, output: Any) -> torch.Tensor:
        """Select a stable vision-model output tensor."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return first_tensor(output)


class QwenModelCase(QwenBaseCase):
    """Smoke case for qwen/quantize_model.py."""

    name = "qwen3_vl_model"
    description = "Quantize a tiny multimodal Qwen3-VL model."
    tags: tuple[str, ...] = ("qwen3_vl", "model")
    max_mean_abs_diff = 10.0
    inplace_prepare = True
    inplace_convert = True
    include_generation_fields = False

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the multimodal PTQ config with static vision metadata."""
        return _make_ptq_config(self.qwen_cfg, self.thw)

    def _model_class(self) -> type[torch.nn.Module]:
        """Return the Hugging Face model class used by this case."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        return Qwen3VLModel

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny multimodal Qwen3-VL model and reference copy."""
        torch.manual_seed(123)
        self._validated_size_profile(cfg)
        self.qwen_cfg = _make_tiny_qwen3vl_config()
        self.thw = (1, 8, 8)
        model_cls = self._model_class()
        module = model_cls(self.qwen_cfg).eval()
        return module, clone_module(module)

    def _sample(self, *, for_eval: bool = False) -> ForwardInput:
        """Create one synthetic multimodal model input."""
        sample = _create_image_input(
            self.qwen_cfg,
            seq_len=50,
            thw=self.thw,
            include_generation_fields=self.include_generation_fields,
        )
        if for_eval:
            sample = dict(sample)
            sample["position_ids"] = None
            sample["return_dict"] = False
        return ForwardInput((), sample)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create multimodal model calibration samples."""
        return [self._sample() for _ in range(2)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the multimodal model evaluation sample."""
        return self._sample(for_eval=True)


class QwenForConditionalGenerationCase(QwenModelCase):
    """Smoke case for qwen/quantize_for_conditional_generation.py."""

    name = "qwen3_vl_for_conditional_generation"
    description = "Quantize a tiny Qwen3-VL for-conditional-generation model."
    tags = ("qwen3_vl", "model", "generation")
    include_generation_fields = True

    def _model_class(self) -> type[torch.nn.Module]:
        """Return the generation model class used by this case."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        return Qwen3VLForConditionalGeneration


QWEN3_VL_CASES: tuple[WrapperSmokeCase, ...] = (
    QwenTextAttentionPrefillCase(),
    QwenTextAttentionDecodeCase(),
    QwenTextMLPCase(),
    QwenTextDecoderLayerPrefillCase(),
    QwenTextDecoderLayerDecodeCase(),
    QwenTextModelCase(),
    QwenVisionAttentionCase(),
    QwenVisionMLPCase(),
    QwenVisionBlockCase(),
    QwenVisionPatchEmbedCase(),
    QwenVisionPatchMergerCase(),
    QwenVisionModelCase(),
    QwenModelCase(),
    QwenForConditionalGenerationCase(),
)

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

"""Smoke cases for Gemma4 wrapper checks."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import (
    clone_module,
    smoke_section,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    DEFAULT_GEMMA4_STATIC_VISION_PROFILE,
    Gemma4StaticVisionProfile,
    get_gemma4_static_vision_profile,
)


_GEMMA4_FULL_ROPE_PARAMETERS: dict[str, Any] = {
    "rope_type": "proportional",
    "partial_rotary_factor": 0.25,
    "rope_theta": 1_000_000.0,
}
_GEMMA4_SLIDING_ROPE_PARAMETERS: dict[str, Any] = {
    "rope_type": "default",
    "rope_theta": 10_000.0,
}

_GEMMA4_SIZE_PROFILE_TINY = "tiny"
_GEMMA4_SIZE_PROFILE_E2B_DIMS = "e2b_dims"
_GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME = "e2b_static_runtime"
_GEMMA4_SIZE_PROFILES = frozenset(
    {
        _GEMMA4_SIZE_PROFILE_TINY,
        _GEMMA4_SIZE_PROFILE_E2B_DIMS,
        _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME,
    }
)
_GEMMA4_E2B_WIDTH_PROFILES = frozenset(
    {
        _GEMMA4_SIZE_PROFILE_E2B_DIMS,
        _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME,
    }
)

_GEMMA4_E2B_PLE_DIM = 256
_GEMMA4_E2B_STATIC_MAX_SEQ = 2_048
_GEMMA4_E2B_STATIC_PROFILE = get_gemma4_static_vision_profile(
    DEFAULT_GEMMA4_STATIC_VISION_PROFILE
)
_GEMMA4_E2B_STATIC_NUM_VISUAL_TOKENS = _GEMMA4_E2B_STATIC_PROFILE.num_visual_tokens
_GEMMA4_E2B_STATIC_MAX_SOFT_TOKENS = _GEMMA4_E2B_STATIC_PROFILE.max_soft_tokens
_GEMMA4_E2B_STATIC_PATCH_GRID_HEIGHT = _GEMMA4_E2B_STATIC_PROFILE.patch_grid_height
_GEMMA4_E2B_STATIC_PATCH_GRID_WIDTH = _GEMMA4_E2B_STATIC_PROFILE.patch_grid_width
_GEMMA4_E2B_VISION_POOLING_KERNEL_SIZE = _GEMMA4_E2B_STATIC_PROFILE.pooling_kernel_size
_GEMMA4_SUPPORTED_MAX_SOFT_TOKENS = frozenset({70, 140, 280, 560, 1_120})

# The E2B-width profiles intentionally cover bounded module-level cases and
# one-layer vision composites. Text/full multimodal models and vocabulary-sized
# embeddings remain tiny-only because they defeat the smoke workflow.
_GEMMA4_E2B_DIMS_SUPPORTED_CASES = frozenset(
    {
        "gemma4_text_mlp",
        "gemma4_text_attention_prefill",
        "gemma4_text_attention_decode",
        "gemma4_text_attention_sliding_prefill",
        "gemma4_text_attention_sliding_decode",
        "gemma4_text_attention_k_eq_v_prefill",
        "gemma4_text_attention_shared_kv_prefill",
        "gemma4_text_attention_shared_kv_decode",
        "gemma4_text_decoder_layer_prefill",
        "gemma4_text_decoder_layer_sliding_prefill",
        "gemma4_text_decoder_layer_decode",
        "gemma4_text_decoder_layer_shared_kv",
        "gemma4_vision_attention",
        "gemma4_vision_encoder_layer",
        "gemma4_vision_encoder",
        "gemma4_vision_patch_embedder",
        "gemma4_vision_pooler",
        "gemma4_vision_model",
        "gemma4_multimodal_embedder",
    }
)

# The static-runtime profile follows the real E2B execution contract rather
# than exercising synthetic feature variants. In particular, E2B itself uses
# attention_k_eq_v=False, so the K=V-only branch remains available in e2b_dims
# but is intentionally excluded here.
_GEMMA4_E2B_STATIC_RUNTIME_SUPPORTED_CASES = _GEMMA4_E2B_DIMS_SUPPORTED_CASES - {
    "gemma4_text_attention_k_eq_v_prefill"
}


@dataclass(frozen=True)
class Gemma4StaticRuntimeShape:
    """Fixed input-shape contract used by E2B static-runtime smoke exports."""

    max_seq: int = _GEMMA4_E2B_STATIC_MAX_SEQ
    num_visual_tokens: int = _GEMMA4_E2B_STATIC_NUM_VISUAL_TOKENS
    max_soft_tokens: int = _GEMMA4_E2B_STATIC_MAX_SOFT_TOKENS
    patch_grid_height: int = _GEMMA4_E2B_STATIC_PATCH_GRID_HEIGHT
    patch_grid_width: int = _GEMMA4_E2B_STATIC_PATCH_GRID_WIDTH
    pooling_kernel_size: int = _GEMMA4_E2B_VISION_POOLING_KERNEL_SIZE

    def __post_init__(self) -> None:
        if self.max_seq < 2:
            raise ValueError(
                f"Gemma4 static max_seq must be at least 2, got {self.max_seq}."
            )
        if self.max_soft_tokens not in _GEMMA4_SUPPORTED_MAX_SOFT_TOKENS:
            supported = ", ".join(
                str(value) for value in sorted(_GEMMA4_SUPPORTED_MAX_SOFT_TOKENS)
            )
            raise ValueError(
                "Gemma4 static max_soft_tokens must match a processor-supported "
                f"budget ({supported}), got {self.max_soft_tokens}."
            )

        profile = Gemma4StaticVisionProfile(
            name="wrapper_smoke",
            visual_start_idx=_GEMMA4_E2B_STATIC_PROFILE.visual_start_idx,
            num_visual_tokens=self.num_visual_tokens,
            max_soft_tokens=self.max_soft_tokens,
            patch_grid_height=self.patch_grid_height,
            patch_grid_width=self.patch_grid_width,
            patch_size=_GEMMA4_E2B_STATIC_PROFILE.patch_size,
            pooling_kernel_size=self.pooling_kernel_size,
        )
        profile.validate(max_seq_len=self.max_seq)

    @property
    def visual_grid_height(self) -> int:
        """Return the valid post-pooling visual-grid height."""
        return self.patch_grid_height // self.pooling_kernel_size

    @property
    def visual_grid_width(self) -> int:
        """Return the valid post-pooling visual-grid width."""
        return self.patch_grid_width // self.pooling_kernel_size

    @property
    def num_valid_patches(self) -> int:
        """Return the number of non-padding patches in the fixed input."""
        return self.patch_grid_height * self.patch_grid_width

    @property
    def num_patches(self) -> int:
        """Return the processor-compatible padded patch-slot count."""
        return self.max_soft_tokens * self.pooling_kernel_size**2

    @property
    def num_padding_patches(self) -> int:
        """Return the number of padded patch slots."""
        return self.num_patches - self.num_valid_patches

    @property
    def num_padding_soft_tokens(self) -> int:
        """Return the number of pooler output slots removed as padding."""
        return self.max_soft_tokens - self.num_visual_tokens


def _has_gemma4() -> CaseAvailability:
    """Return availability for Hugging Face Gemma4 modules."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )

        return CaseAvailability(True)
    except Exception as exc:
        return CaseAvailability(False, f"Gemma4 modules are unavailable: {exc}")


def _set_eager_attention(cfg: Any) -> Any:
    """Set eager attention on configs that expose a configurable implementation."""
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _rope_parameters_for_layer_types(
    layer_types: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Return RoPE parameters whose keys exactly match the requested layer types.

    Hugging Face validates Gemma4 RoPE parameters as a nested layer-type mapping
    only when every top-level RoPE key is present in ``config.layer_types``. Tiny
    smoke configs often use a subset of real Gemma4 layer types, so the default
    Gemma4 RoPE dict can trigger warnings when it contains unused keys.
    """
    rope_parameters: dict[str, dict[str, Any]] = {}
    if "sliding_attention" in layer_types:
        rope_parameters["sliding_attention"] = dict(_GEMMA4_SLIDING_ROPE_PARAMETERS)
    if "full_attention" in layer_types:
        rope_parameters["full_attention"] = dict(_GEMMA4_FULL_ROPE_PARAMETERS)
    return rope_parameters


def _gemma4_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the Gemma4-specific wrapper-smoke configuration mapping."""
    section = smoke_section(cfg)
    gemma4_cfg = section.get("gemma4", {})
    if not isinstance(gemma4_cfg, Mapping):
        raise ValueError("debug.wrapper_smoke.gemma4 must be a mapping.")
    return gemma4_cfg


def _gemma4_size_profile(cfg: Mapping[str, Any]) -> str:
    """Return and validate the requested Gemma4 smoke size profile."""
    gemma4_cfg = _gemma4_options(cfg)
    profile = (
        str(gemma4_cfg.get("size_profile", _GEMMA4_SIZE_PROFILE_TINY)).strip().lower()
    )
    if profile not in _GEMMA4_SIZE_PROFILES:
        choices = ", ".join(sorted(_GEMMA4_SIZE_PROFILES))
        raise ValueError(
            f"Unsupported Gemma4 wrapper-smoke size profile '{profile}'. "
            f"Expected one of: {choices}."
        )
    return profile


def _gemma4_static_runtime_shape(
    cfg: Mapping[str, Any],
) -> Gemma4StaticRuntimeShape:
    """Parse and validate the E2B static-runtime input-shape options."""
    gemma4_cfg = _gemma4_options(cfg)
    static_cfg = gemma4_cfg.get("static_runtime", {})
    if not isinstance(static_cfg, Mapping):
        raise ValueError("debug.wrapper_smoke.gemma4.static_runtime must be a mapping.")

    profile = get_gemma4_static_vision_profile(
        str(static_cfg.get("profile", DEFAULT_GEMMA4_STATIC_VISION_PROFILE))
    )
    return Gemma4StaticRuntimeShape(
        max_seq=int(static_cfg.get("max_seq", _GEMMA4_E2B_STATIC_MAX_SEQ)),
        num_visual_tokens=int(
            static_cfg.get("num_visual_tokens", profile.num_visual_tokens)
        ),
        max_soft_tokens=int(static_cfg.get("max_soft_tokens", profile.max_soft_tokens)),
        patch_grid_height=int(
            static_cfg.get("patch_grid_height", profile.patch_grid_height)
        ),
        patch_grid_width=int(
            static_cfg.get("patch_grid_width", profile.patch_grid_width)
        ),
        pooling_kernel_size=int(
            static_cfg.get("pooling_kernel_size", profile.pooling_kernel_size)
        ),
    )


def _build_text_config(
    *,
    size_profile: str,
    layer_types: tuple[str, ...] = ("full_attention",),
    attention_k_eq_v: bool = False,
    num_kv_shared_layers: int = 0,
    hidden_size_per_layer_input: int = 0,
) -> Any:
    """Create a tiny or E2B-width Gemma4 text config.

    The helper intentionally provides ``layer_types`` and ``rope_parameters`` as
    a matched pair. This prevents Hugging Face from treating nested Gemma4 RoPE
    parameters as one global default-RoPE config, which otherwise emits
    ``Unrecognized keys in rope_parameters`` warnings in one-layer smoke cases.

    Both E2B profiles copy the original channel and projection dimensions while
    preserving the short synthetic topology selected by each smoke case. PLE is
    disabled through ``hidden_size_per_layer_input=0`` unless a caller explicitly
    requests it.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    if size_profile == _GEMMA4_SIZE_PROFILE_TINY:
        dimension_kwargs: dict[str, Any] = {
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "head_dim": 32,
            "global_head_dim": 32,
            "max_position_embeddings": 128,
        }
    elif size_profile in _GEMMA4_E2B_WIDTH_PROFILES:
        dimension_kwargs = {
            "vocab_size": 262_144,
            "hidden_size": 1_536,
            "intermediate_size": 6_144,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            # E2B falls back to num_key_value_heads for ordinary full attention.
            # Keep this explicit so the synthetic K=V case also has a valid
            # global-KV projection shape.
            "num_global_key_value_heads": 1,
            "head_dim": 256,
            "global_head_dim": 512,
            "hidden_activation": "gelu_pytorch_tanh",
            "max_position_embeddings": 131_072,
            "rms_norm_eps": 1e-6,
            "sliding_window": 512,
            "final_logit_softcapping": 30.0,
            "use_double_wide_mlp": True,
            "vocab_size_per_layer_input": 262_144,
        }
    else:
        raise AssertionError(f"Unhandled Gemma4 size profile: {size_profile}")

    text_cfg = Gemma4TextConfig(
        **dimension_kwargs,
        num_hidden_layers=len(layer_types),
        layer_types=list(layer_types),
        rope_parameters=_rope_parameters_for_layer_types(layer_types),
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        enable_moe_block=False,
        attention_k_eq_v=attention_k_eq_v,
        num_kv_shared_layers=num_kv_shared_layers,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
    )
    return _set_eager_attention(text_cfg)


def _build_vision_config(*, size_profile: str) -> Any:
    """Create a tiny or E2B-width Gemma4 vision config.

    The E2B-width variant keeps a single encoder layer. It reproduces original
    projection, patch, and position-table dimensions without constructing the
    complete 16-layer vision tower.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    if size_profile == _GEMMA4_SIZE_PROFILE_TINY:
        vision_cfg = Gemma4VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attention_dropout=0.0,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            use_clipped_linears=False,
            rope_parameters={"rope_type": "default", "rope_theta": 100.0},
            standardize=True,
        )
    elif size_profile in _GEMMA4_E2B_WIDTH_PROFILES:
        vision_cfg = Gemma4VisionConfig(
            hidden_size=768,
            intermediate_size=3_072,
            num_hidden_layers=1,
            num_attention_heads=12,
            num_key_value_heads=12,
            head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            attention_bias=False,
            attention_dropout=0.0,
            max_position_embeddings=131_072,
            rms_norm_eps=1e-6,
            pooling_kernel_size=3,
            patch_size=16,
            position_embedding_size=10_240,
            use_clipped_linears=True,
            rope_parameters={"rope_type": "default", "rope_theta": 100.0},
            standardize=False,
        )
    else:
        raise AssertionError(f"Unhandled Gemma4 size profile: {size_profile}")

    return _set_eager_attention(vision_cfg)


def _build_vision_patch_embedder_config(*, size_profile: str) -> Any:
    """Create a patch-embedder config for the requested size profile."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    if size_profile == _GEMMA4_SIZE_PROFILE_TINY:
        return Gemma4VisionConfig(
            hidden_size=32,
            patch_size=4,
            position_embedding_size=8,
        )
    return _build_vision_config(size_profile=size_profile)


def _text_rope(
    batch_size: int, seq_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Gemma4 text RoPE embeddings."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _attention_mask(seq_len: int, kv_len: int | None = None) -> torch.Tensor:
    """Create an additive attention mask for synthetic Gemma4 attention tests."""
    kv_len = seq_len if kv_len is None else kv_len
    return torch.zeros(1, 1, seq_len, kv_len)


def _causal_mask(seq_len: int, fill_value: float = -120.0) -> torch.Tensor:
    """Create an additive causal mask with a large negative upper triangle."""
    mask = torch.zeros(1, 1, seq_len, seq_len)
    blocked = torch.full_like(mask, float(fill_value))
    return torch.triu(blocked, diagonal=1)


def _clone_value(value: Any) -> Any:
    """Clone tensors nested inside a small smoke-test value."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _clone_forward_input(sample: ForwardInput) -> ForwardInput:
    """Clone a smoke input so reference and quantized runs do not share mutable state."""
    return ForwardInput(
        tuple(_clone_value(arg) for arg in sample.args),
        {key: _clone_value(value) for key, value in sample.kwargs.items()},
    )


def _sliding_window_causal_mask(
    seq_len: int,
    sliding_window: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    fill_value: float = -120.0,
) -> torch.Tensor:
    """Create a fixed-shape additive causal sliding-window mask.

    A query at position ``q`` can attend to keys in the inclusive interval
    ``[max(0, q - sliding_window + 1), q]``. Future keys and keys older than
    the configured window receive ``fill_value``.

    Parameters
    ----------
    seq_len:
        Static query and key/value sequence length.
    sliding_window:
        Number of visible tokens including the current query token.
    batch_size:
        Static batch size represented by the returned mask.
    dtype:
        Floating-point dtype of the additive mask.
    device:
        Device on which to create the mask.
    fill_value:
        Additive value assigned to blocked positions.

    Returns
    -------
    torch.Tensor
        A tensor with shape ``(batch_size, 1, seq_len, seq_len)``.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}.")

    query_positions = torch.arange(seq_len, device=device).view(seq_len, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, seq_len)

    future_positions = key_positions > query_positions
    positions_before_window = key_positions < query_positions - sliding_window + 1
    blocked_positions = future_positions | positions_before_window

    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    mask.masked_fill_(blocked_positions, float(fill_value))
    return (
        mask.view(1, 1, seq_len, seq_len)
        .expand(batch_size, 1, seq_len, seq_len)
        .contiguous()
    )


def _static_pixel_position_ids(
    shape: Gemma4StaticRuntimeShape,
    *,
    batch_size: int = 1,
) -> torch.Tensor:
    """Create the padded 2-D patch layout used by the E2B image processor.

    The default contract contains a valid ``36 x 66`` patch grid (2,376
    patches) followed by 144 ``(-1, -1)`` padding slots. Pooling by ``3 x 3``
    produces 280 fixed output slots, of which 16 are padding and 264 are valid
    visual tokens.
    """
    coords = torch.arange(shape.num_valid_patches)
    valid = torch.stack(
        (coords % shape.patch_grid_width, coords // shape.patch_grid_width),
        dim=-1,
    )
    padding = torch.full(
        (shape.num_padding_patches, 2),
        -1,
        dtype=valid.dtype,
    )
    position_ids = torch.cat((valid, padding), dim=0)
    return position_ids.unsqueeze(0).expand(batch_size, -1, -1).long()


def _padding_positions_from_ids(pixel_position_ids: torch.Tensor) -> torch.Tensor:
    """Return the Boolean padding mask encoded by ``(-1, -1)`` positions."""
    return (pixel_position_ids == -1).all(dim=-1)


def _vision_additive_padding_mask(
    padding_positions: torch.Tensor,
    *,
    fill_value: float = -120.0,
) -> torch.Tensor:
    """Create a bidirectional additive mask that blocks padded key positions."""
    batch_size, seq_len = padding_positions.shape
    mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    return mask.masked_fill(
        padding_positions[:, None, None, :],
        float(fill_value),
    )


class Gemma4BaseCase(WrapperSmokeCase):
    """Base class for Gemma4 E2B wrapper smoke cases."""

    tags: tuple[str, ...] = ("gemma4", "e2b")
    # Set by concrete cases (class attribute or during build()).
    seq_len: int
    text_cfg: Any

    def availability(self) -> CaseAvailability:
        """Return whether Gemma4 modules can be imported."""
        return _has_gemma4()

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Reject unsupported size profiles before constructing a module."""
        self._validated_size_profile(cfg)

    def _validated_size_profile(self, cfg: Mapping[str, Any]) -> str:
        """Validate that this case supports the requested Gemma4 profile."""
        profile = _gemma4_size_profile(cfg)
        supported_cases = None
        if profile == _GEMMA4_SIZE_PROFILE_E2B_DIMS:
            supported_cases = _GEMMA4_E2B_DIMS_SUPPORTED_CASES
        elif profile == _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME:
            supported_cases = _GEMMA4_E2B_STATIC_RUNTIME_SUPPORTED_CASES

        if supported_cases is not None and self.name not in supported_cases:
            supported = ", ".join(sorted(supported_cases))
            raise ValueError(
                f"Case '{self.name}' does not support Gemma4 size profile "
                f"'{profile}'. The profile is limited to bounded module-level "
                f"cases: {supported}."
            )

        self._active_size_profile = profile
        self._active_static_runtime_shape = (
            _gemma4_static_runtime_shape(cfg)
            if profile == _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME
            else None
        )
        return profile

    def _static_runtime_shape(self) -> Gemma4StaticRuntimeShape | None:
        """Return the active static-runtime shape after profile validation."""
        return getattr(self, "_active_static_runtime_shape", None)

    def _text_prefill_seq_len(self, default: int) -> int:
        """Return the text prefill length for the active profile."""
        shape = self._static_runtime_shape()
        return shape.max_seq if shape is not None else int(default)

    def _decode_max_seq(self, default: int) -> int:
        """Return the fixed decode cache capacity for the active profile."""
        shape = self._static_runtime_shape()
        return shape.max_seq if shape is not None else int(default)

    def _vision_patch_seq_len(self, default: int) -> int:
        """Return the padded vision patch-slot count for the active profile."""
        shape = self._static_runtime_shape()
        return shape.num_patches if shape is not None else int(default)

    def _visual_token_seq_len(self, default: int) -> int:
        """Return the number of valid post-pooling visual tokens."""
        shape = self._static_runtime_shape()
        return shape.num_visual_tokens if shape is not None else int(default)

    def _vision_pool_output_length(self, default: int) -> int:
        """Return the fixed pooler output-slot count before padding removal."""
        shape = self._static_runtime_shape()
        return shape.max_soft_tokens if shape is not None else int(default)

    def _calibration_sample_count(self, cfg: Mapping[str, Any], *, default: int) -> int:
        """Avoid retaining multiple large static-runtime samples in memory."""
        profile = self._validated_size_profile(cfg)
        return 1 if profile == _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME else default

    def _case_pixel_position_ids(self, batch_size: int = 1) -> torch.Tensor:
        """Create tiny or static-runtime pixel positions for this case."""
        shape = self._static_runtime_shape()
        if shape is not None:
            return _static_pixel_position_ids(shape, batch_size=batch_size)
        return _pixel_position_ids(batch_size, self.seq_len)

    def _case_vision_position_ids(self, batch_size: int = 1) -> torch.Tensor:
        """Create positions while preserving the legacy tiny encoder layout."""
        shape = self._static_runtime_shape()
        if shape is not None:
            return _static_pixel_position_ids(shape, batch_size=batch_size)
        return _vision_position_ids(batch_size, self.seq_len)

    def _case_padding_positions(self, batch_size: int = 1) -> torch.Tensor:
        """Return the padding mask corresponding to this case's patch layout."""
        return _padding_positions_from_ids(
            self._case_pixel_position_ids(batch_size=batch_size)
        )

    def _case_vision_attention_mask(self, batch_size: int = 1) -> torch.Tensor:
        """Return the tiny all-visible or static padded-key attention mask."""
        shape = self._static_runtime_shape()
        if shape is None:
            return torch.zeros(batch_size, 1, self.seq_len, self.seq_len)
        return _vision_additive_padding_mask(
            self._case_padding_positions(batch_size=batch_size)
        )

    def _case_vision_keep_mask(self, batch_size: int = 1) -> torch.Tensor:
        """Return the Boolean valid-patch mask expected by the vision encoder."""
        return torch.logical_not(self._case_padding_positions(batch_size=batch_size))

    def _case_pixel_values(
        self,
        *,
        batch_size: int,
        patch_dim: int,
    ) -> torch.Tensor:
        """Create image-like patch values and zero any static padding slots."""
        if self._static_runtime_shape() is None:
            return torch.randn(batch_size, self.seq_len, patch_dim)
        pixel_values = torch.rand(batch_size, self.seq_len, patch_dim)
        return pixel_values.masked_fill(
            self._case_padding_positions(batch_size=batch_size).unsqueeze(-1),
            0.0,
        )

    def _case_hidden_states(
        self,
        *,
        batch_size: int,
        hidden_size: int,
    ) -> torch.Tensor:
        """Create hidden states and zero static padded patch slots."""
        hidden = torch.randn(batch_size, self.seq_len, hidden_size)
        if self._static_runtime_shape() is None:
            return hidden
        return hidden.masked_fill(
            self._case_padding_positions(batch_size=batch_size).unsqueeze(-1),
            0.0,
        )

    def _per_layer_input(self, seq_len: int) -> torch.Tensor | None:
        """Create the external PLE input required by static decoder graphs."""
        dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        if dim <= 0:
            return None
        return torch.randn(1, seq_len, dim)

    def _make_text_config(
        self,
        cfg: Mapping[str, Any],
        *,
        layer_types: tuple[str, ...] = ("full_attention",),
        attention_k_eq_v: bool = False,
        num_kv_shared_layers: int = 0,
        hidden_size_per_layer_input: int = 0,
    ) -> Any:
        """Create a text config after validating this case's profile support."""
        return _build_text_config(
            size_profile=self._validated_size_profile(cfg),
            layer_types=layer_types,
            attention_k_eq_v=attention_k_eq_v,
            num_kv_shared_layers=num_kv_shared_layers,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
        )

    def _make_vision_config(self, cfg: Mapping[str, Any]) -> Any:
        """Create a vision config after validating this case's profile support."""
        return _build_vision_config(
            size_profile=self._validated_size_profile(cfg),
        )

    def _make_vision_patch_embedder_config(self, cfg: Mapping[str, Any]) -> Any:
        """Create a patch-embedder config after profile validation."""
        return _build_vision_patch_embedder_config(
            size_profile=self._validated_size_profile(cfg),
        )

    def prepare_model(
        self, model: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Prepare E2B-width modules in place to avoid an extra large copy."""
        from tico.quantization import prepare

        profile = self._validated_size_profile(cfg)
        inplace = self.inplace_prepare or profile in _GEMMA4_E2B_WIDTH_PROFILES
        return prepare(model, self.ptq_config(cfg), inplace=inplace)

    def convert_model(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Convert E2B-width modules in place to limit peak host memory."""
        from tico.quantization import convert

        profile = self._validated_size_profile(cfg)
        inplace = self.inplace_convert or profile in _GEMMA4_E2B_WIDTH_PROFILES
        return convert(prepared, inplace=inplace)

    def export_filename(self, cfg: Mapping[str, Any]) -> str:
        """Include the non-default size profile in the Circle filename."""
        profile = self._validated_size_profile(cfg)
        if profile == _GEMMA4_SIZE_PROFILE_TINY:
            return super().export_filename(cfg)
        return f"{self.name}.{profile}.q.circle"


class Gemma4TextMLPCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 text MLP."""

    name = "gemma4_text_mlp"
    description = "Quantize one tiny dense Gemma4 text MLP module."
    tags = ("gemma4", "e2b", "text", "mlp")

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text MLP and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg, layer_types=("full_attention",))
        self.seq_len = self._text_prefill_seq_len(default=8)
        module = Gemma4TextMLP(self.text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create calibration samples."""
        return [
            ForwardInput((torch.randn(1, self.seq_len, self.text_cfg.hidden_size),))
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create an evaluation sample."""
        return ForwardInput((torch.randn(1, self.seq_len, self.text_cfg.hidden_size),))


class Gemma4TextAttentionBaseCase(Gemma4BaseCase):
    """Base class for mode-specific Gemma4 text attention smoke cases."""

    tags: tuple[str, ...] = ("gemma4", "e2b", "text", "attention")
    max_mean_abs_diff = 2.0
    seq_len = 8
    max_seq = 8
    layer_idx = 0
    layer_types: tuple[str, ...] = ("full_attention",)
    attention_k_eq_v = False
    num_kv_shared_layers = 0
    export_mode = "prefill"
    smoke_sliding_window: int | None = None

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a mode-specific Gemma4 text attention module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(
            cfg,
            layer_types=self.layer_types,
            attention_k_eq_v=self.attention_k_eq_v,
            num_kv_shared_layers=self.num_kv_shared_layers,
        )
        if (
            self.smoke_sliding_window is not None
            and self._static_runtime_shape() is None
        ):
            self.text_cfg.sliding_window = self.smoke_sliding_window

        if self.export_mode == "decode":
            self.max_seq = self._decode_max_seq(default=type(self).max_seq)
            self.seq_len = 1
        else:
            self.seq_len = self._text_prefill_seq_len(default=type(self).seq_len)

        module = Gemma4TextAttention(self.text_cfg, layer_idx=self.layer_idx).eval()
        self.attention_head_dim = int(module.head_dim)
        self.is_sliding_attention = bool(module.is_sliding)
        self.sliding_window = int(module.sliding_window or 0)
        return module, clone_module(module)

    def _shared_key_value(self, kv_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a synthetic full K/V tuple for a shared-KV consumer layer."""
        key_states = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            kv_len,
            self.attention_head_dim,
        )
        return key_states, torch.randn_like(key_states)

    def _sample(self) -> ForwardInput:
        """Create one mode-specific synthetic Gemma4 text attention sample."""
        raise NotImplementedError

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 attention wrapper without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the selected reference module without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        return reference(*cloned.args, **dict(cloned.kwargs))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create mode-specific Gemma4 text attention calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the mode-specific Gemma4 text attention evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Return the static attention adapter selected by this smoke case."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if not hasattr(wrapped, "as_export_module"):
            return quantized
        return wrapped.as_export_module(
            self.export_mode,
            return_kv=not bool(getattr(wrapped, "is_kv_shared_layer", False)),
        ).eval()


class Gemma4TextAttentionPrefillBaseCase(Gemma4TextAttentionBaseCase):
    """Base class for Gemma4 text attention prefill smoke cases."""

    export_mode = "prefill"

    def _case_text_attention_mask(self) -> torch.Tensor:
        """Return the full or sliding prefill mask for the selected layer."""
        if self.is_sliding_attention:
            return _sliding_window_causal_mask(
                self.seq_len,
                self.sliding_window,
                fill_value=-120.0,
            )
        if self._static_runtime_shape() is not None:
            return _causal_mask(self.seq_len, fill_value=-120.0)
        return _attention_mask(self.seq_len)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 text attention prefill sample."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    1,
                    self.seq_len,
                    self.attention_head_dim,
                ),
                "attention_mask": self._case_text_attention_mask(),
                "shared_kv_states": {},
            },
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional inputs expected by the prefill export adapter."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        export_kwargs = {}
        shared_key_value = kwargs.get("shared_key_value")
        if shared_key_value is not None:
            export_kwargs["shared_key_value"] = shared_key_value
        return ForwardInput(
            (
                kwargs["hidden_states"],
                kwargs["attention_mask"],
                kwargs["position_embeddings"],
            ),
            export_kwargs,
        )


class Gemma4TextAttentionDecodeBaseCase(Gemma4TextAttentionBaseCase):
    """Base class for Gemma4 text attention single-token decode smoke cases."""

    export_mode = "decode"
    compare_reference_source = "prepared"
    seq_len = 1
    max_seq = 8

    def _case_text_attention_mask(self) -> torch.Tensor:
        """Return a fixed-capacity full or sliding single-token decode mask."""
        if self.is_sliding_attention:
            full_mask = _sliding_window_causal_mask(
                self.max_seq,
                self.sliding_window,
                fill_value=-120.0,
            )
            return full_mask[..., -1:, :]
        return _attention_mask(1, self.max_seq)

    def _sample(self) -> ForwardInput:
        """Create one synthetic non-shared Gemma4 attention decode sample."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        past_len = self.max_seq - 1
        past_key = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            past_len,
            self.attention_head_dim,
        )
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    1,
                    1,
                    self.attention_head_dim,
                ),
                "attention_mask": self._case_text_attention_mask(),
                "past_key_value": (past_key, torch.randn_like(past_key)),
                "use_cache": True,
            },
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional inputs expected by the decode export adapter."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        export_kwargs = {}
        past_key_value = kwargs.get("past_key_value")
        shared_key_value = kwargs.get("shared_key_value")
        if past_key_value is not None:
            export_kwargs["past_key_value"] = past_key_value
        if shared_key_value is not None:
            export_kwargs["shared_key_value"] = shared_key_value
        return ForwardInput(
            (
                kwargs["hidden_states"],
                kwargs["attention_mask"],
                kwargs["position_embeddings"],
            ),
            export_kwargs,
        )


class Gemma4TextAttentionPrefillCase(Gemma4TextAttentionPrefillBaseCase):
    """Smoke case for Gemma4 full attention in prefill mode."""

    name = "gemma4_text_attention_prefill"
    description = "Quantize one Gemma4 full-attention module in prefill mode."
    tags = ("gemma4", "e2b", "text", "attention", "prefill")
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 1


class Gemma4TextAttentionDecodeCase(Gemma4TextAttentionDecodeBaseCase):
    """Smoke case for Gemma4 full attention in decode mode."""

    name = "gemma4_text_attention_decode"
    description = "Quantize one Gemma4 full-attention module in decode mode."
    tags = ("gemma4", "e2b", "text", "attention", "decode")
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 1


class Gemma4TextSlidingAttentionPrefillCase(Gemma4TextAttentionPrefillBaseCase):
    """Smoke case for Gemma4 sliding attention in prefill mode."""

    name = "gemma4_text_attention_sliding_prefill"
    description = "Quantize one Gemma4 sliding-attention module in prefill mode."
    tags = ("gemma4", "e2b", "text", "attention", "prefill", "sliding")
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 0
    smoke_sliding_window = 4


class Gemma4TextSlidingAttentionDecodeCase(Gemma4TextAttentionDecodeBaseCase):
    """Smoke case for Gemma4 sliding attention in decode mode."""

    name = "gemma4_text_attention_sliding_decode"
    description = "Quantize one Gemma4 sliding-attention module in decode mode."
    tags = ("gemma4", "e2b", "text", "attention", "decode", "sliding")
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 0
    smoke_sliding_window = 4


class Gemma4TextAttentionKEqVPrefillCase(Gemma4TextAttentionPrefillBaseCase):
    """Smoke case for Gemma4 K-equals-V attention in prefill mode."""

    name = "gemma4_text_attention_k_eq_v_prefill"
    description = (
        "Quantize one Gemma4 full-attention module with attention_k_eq_v=True "
        "in prefill mode."
    )
    tags = ("gemma4", "e2b", "text", "attention", "prefill", "k_eq_v")
    layer_types = ("full_attention",)
    layer_idx = 0
    attention_k_eq_v = True


class Gemma4TextAttentionSharedKVPrefillCase(Gemma4TextAttentionPrefillBaseCase):
    """Smoke case for a Gemma4 shared-KV consumer in prefill mode."""

    name = "gemma4_text_attention_shared_kv_prefill"
    description = "Quantize one Gemma4 shared-KV attention consumer in prefill mode."
    tags = ("gemma4", "e2b", "text", "attention", "prefill", "shared_kv")
    layer_types = ("full_attention", "full_attention")
    layer_idx = 1
    num_kv_shared_layers = 1

    def _sample(self) -> ForwardInput:
        """Create one synthetic shared-KV prefill sample."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        shared_key_value = self._shared_key_value(self.seq_len)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    1,
                    self.seq_len,
                    self.attention_head_dim,
                ),
                "attention_mask": self._case_text_attention_mask(),
                "shared_kv_states": {"full_attention": shared_key_value},
                "shared_key_value": shared_key_value,
            },
        )


class Gemma4TextAttentionSharedKVDecodeCase(Gemma4TextAttentionDecodeBaseCase):
    """Smoke case for a Gemma4 shared-KV consumer in decode mode."""

    name = "gemma4_text_attention_shared_kv_decode"
    description = "Quantize one Gemma4 shared-KV attention consumer in decode mode."
    tags = ("gemma4", "e2b", "text", "attention", "decode", "shared_kv")
    layer_types = ("full_attention", "full_attention")
    layer_idx = 1
    num_kv_shared_layers = 1

    def _sample(self) -> ForwardInput:
        """Create one synthetic shared-KV single-token decode sample."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        shared_key_value = self._shared_key_value(self.max_seq)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    1,
                    1,
                    self.attention_head_dim,
                ),
                "attention_mask": self._case_text_attention_mask(),
                "shared_kv_states": {"full_attention": shared_key_value},
                "shared_key_value": shared_key_value,
            },
        )


class Gemma4TextDecoderLayerBaseCase(Gemma4BaseCase):
    """Base class for tiny Gemma4 text decoder-layer smoke cases."""

    tags: tuple[str, ...] = ("gemma4", "e2b", "text", "decoder_layer")
    max_mean_abs_diff = 2.5
    max_seq: int
    seq_len = 8
    layer_idx = 0
    layer_types: tuple[str, ...] = ("full_attention",)
    attention_k_eq_v = False
    num_kv_shared_layers = 0
    export_mode = "prefill"
    return_kv_on_export = True
    compare_reference_source = "reference"

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the NPU-export PTQ config used by Gemma4 decoder-layer smoke checks."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "npu_export"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny dense Gemma4 text decoder layer and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        hidden_size_per_layer_input = (
            _GEMMA4_E2B_PLE_DIM
            if profile == _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME
            else 0
        )
        self.text_cfg = self._make_text_config(
            cfg,
            layer_types=self.layer_types,
            attention_k_eq_v=self.attention_k_eq_v,
            num_kv_shared_layers=self.num_kv_shared_layers,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
        )
        self.seq_len = (
            1
            if self.export_mode == "decode"
            else self._text_prefill_seq_len(default=type(self).seq_len)
        )
        if self.export_mode == "decode":
            self.max_seq = self._decode_max_seq(default=type(self).max_seq)
        module = Gemma4TextDecoderLayer(self.text_cfg, layer_idx=self.layer_idx).eval()
        self.attention_head_dim = int(module.self_attn.head_dim)
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic prefill decoder-layer sample."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "position_embeddings": _text_rope(1, self.seq_len, self.attention_head_dim),
            "attention_mask": _causal_mask(
                self.seq_len,
                fill_value=float(self.ptq_config({}).attention_mask_fill_value),
            ),
            "shared_kv_states": {},
        }
        per_layer_input = self._per_layer_input(self.seq_len)
        if per_layer_input is not None:
            kwargs["per_layer_input"] = per_layer_input
        return ForwardInput((), kwargs)

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 decoder layer without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 decoder layer without wrapper-only kwargs."""
        cloned = _clone_forward_input(sample)
        kwargs = dict(cloned.kwargs)
        kwargs.pop("shared_key_value", None)
        output = reference(*cloned.args, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create decoder-layer calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the decoder-layer evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped decoder layer in the configured static mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module(
                self.export_mode, return_kv=self.return_kv_on_export
            ).eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the decoder-layer adapter."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        hidden = kwargs["hidden_states"]
        mask = kwargs["attention_mask"]
        rope = kwargs["position_embeddings"]
        shared_key_value = kwargs.get("shared_key_value")
        per_layer_input = kwargs.get("per_layer_input")
        export_kwargs = {}
        if per_layer_input is not None:
            export_kwargs["per_layer_input"] = per_layer_input
        if shared_key_value is not None:
            export_kwargs["shared_key_value"] = shared_key_value
        return ForwardInput((hidden, mask, rope), export_kwargs)


class Gemma4TextDecoderLayerPrefillCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for one tiny Gemma4 text decoder layer in prefill mode."""

    name = "gemma4_text_decoder_layer_prefill"
    description = "Quantize one tiny dense Gemma4 text decoder layer in prefill mode."
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 1
    export_mode = "prefill"


class Gemma4TextDecoderLayerSlidingPrefillCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for one Gemma4 sliding-attention decoder layer.

    The case creates a two-layer text configuration and selects layer zero.
    This keeps layer zero as sliding attention while satisfying Gemma4's
    requirement that the final decoder layer use full attention.

    The sliding window is intentionally smaller than the sequence length so
    the input covers both future-token masking and left-side window masking.
    """

    name = "gemma4_text_decoder_layer_sliding_prefill"
    description = (
        "Quantize one tiny Gemma4 sliding-attention decoder layer with "
        "a causal sliding-window mask."
    )
    tags = (
        "gemma4",
        "e2b",
        "text",
        "decoder_layer",
        "prefill",
        "sliding",
    )

    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 0
    export_mode = "prefill"

    seq_len = 8
    sliding_window = 4
    mask_fill_value = -120.0

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build an NPU-export PTQ config matching the sample mask fill value."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(
            model_args={"profile": "npu_export"},
            attention_mask_fill_value=self.mask_fill_value,
        )

    def build(
        self,
        cfg: Mapping[str, Any],
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a sliding-attention decoder layer and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        torch.manual_seed(123)
        profile = self._validated_size_profile(cfg)
        hidden_size_per_layer_input = (
            _GEMMA4_E2B_PLE_DIM
            if profile == _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME
            else 0
        )
        self.text_cfg = self._make_text_config(
            cfg,
            layer_types=self.layer_types,
            hidden_size_per_layer_input=hidden_size_per_layer_input,
        )
        self.seq_len = self._text_prefill_seq_len(default=type(self).seq_len)
        self.sliding_window = (
            int(self.text_cfg.sliding_window)
            if profile == _GEMMA4_SIZE_PROFILE_E2B_STATIC_RUNTIME
            else type(self).sliding_window
        )
        self.text_cfg.sliding_window = self.sliding_window

        module = Gemma4TextDecoderLayer(
            self.text_cfg,
            layer_idx=self.layer_idx,
        ).eval()
        self.attention_head_dim = int(module.self_attn.head_dim)

        if not module.self_attn.is_sliding:
            raise RuntimeError(
                "The smoke case did not build a sliding-attention layer."
            )
        if module.self_attn.sliding_window != self.sliding_window:
            raise RuntimeError(
                "The decoder layer does not use the requested sliding window: "
                f"expected {self.sliding_window}, "
                f"got {module.self_attn.sliding_window}."
            )

        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one fixed-shape sliding-window prefill sample."""
        batch_size = 1
        hidden = torch.randn(
            batch_size,
            self.seq_len,
            self.text_cfg.hidden_size,
        )
        attention_mask = _sliding_window_causal_mask(
            self.seq_len,
            self.sliding_window,
            batch_size=batch_size,
            dtype=hidden.dtype,
            device=hidden.device,
            fill_value=self.mask_fill_value,
        )

        kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "position_embeddings": _text_rope(
                batch_size,
                self.seq_len,
                self.attention_head_dim,
            ),
            "attention_mask": attention_mask,
            "shared_kv_states": {},
        }
        per_layer_input = self._per_layer_input(self.seq_len)
        if per_layer_input is not None:
            kwargs["per_layer_input"] = per_layer_input
        return ForwardInput((), kwargs)


class Gemma4TextDecoderLayerDecodeCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for one tiny Gemma4 text decoder layer in decode mode."""

    name = "gemma4_text_decoder_layer_decode"
    description = "Quantize one tiny dense Gemma4 text decoder layer in decode mode."
    tags = ("gemma4", "e2b", "text", "decoder_layer", "decode")
    compare_reference_source = "prepared"
    seq_len = 1
    max_seq = 8
    export_mode = "decode"

    def _sample(self) -> ForwardInput:
        """Create one synthetic single-token decoder-layer decode sample."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        past_len = self.max_seq - 1
        past = (
            torch.randn(
                1,
                self.text_cfg.num_key_value_heads,
                past_len,
                self.attention_head_dim,
            ),
            torch.randn(
                1,
                self.text_cfg.num_key_value_heads,
                past_len,
                self.attention_head_dim,
            ),
        )
        kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "position_embeddings": _text_rope(1, 1, self.attention_head_dim),
            "attention_mask": _attention_mask(1, self.max_seq),
            "past_key_value": past,
            "use_cache": True,
        }
        per_layer_input = self._per_layer_input(1)
        if per_layer_input is not None:
            kwargs["per_layer_input"] = per_layer_input
        return ForwardInput((), kwargs)

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static decode inputs expected by the decoder-layer adapter."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        return ForwardInput(
            (
                kwargs["hidden_states"],
                kwargs["attention_mask"],
                kwargs["position_embeddings"],
            ),
            {
                "past_key_value": kwargs["past_key_value"],
                **(
                    {"per_layer_input": kwargs["per_layer_input"]}
                    if "per_layer_input" in kwargs
                    else {}
                ),
            },
        )


class Gemma4TextDecoderLayerSharedKVCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for a Gemma4 shared-KV consumer decoder layer."""

    name = "gemma4_text_decoder_layer_shared_kv"
    description = "Quantize one tiny Gemma4 decoder layer that consumes shared K/V."
    tags = ("gemma4", "e2b", "text", "decoder_layer", "shared_kv")
    layer_types = ("full_attention", "full_attention")
    layer_idx = 1
    num_kv_shared_layers = 1
    export_mode = "prefill"

    def _sample(self) -> ForwardInput:
        """Create one synthetic shared-KV decoder-layer sample."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        key_states = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            self.seq_len,
            self.attention_head_dim,
        )
        value_states = torch.randn_like(key_states)
        shared_key_value = (key_states, value_states)
        kwargs: dict[str, Any] = {
            "hidden_states": hidden,
            "position_embeddings": _text_rope(1, self.seq_len, self.attention_head_dim),
            "attention_mask": (
                _causal_mask(self.seq_len, fill_value=-120.0)
                if self._static_runtime_shape() is not None
                else _attention_mask(self.seq_len)
            ),
            "shared_kv_states": {"full_attention": shared_key_value},
            "shared_key_value": shared_key_value,
        }
        per_layer_input = self._per_layer_input(self.seq_len)
        if per_layer_input is not None:
            kwargs["per_layer_input"] = per_layer_input
        return ForwardInput((), kwargs)


class Gemma4TextModelCase(Gemma4BaseCase):
    """Smoke case for one tiny dense Gemma4 text model."""

    name = "gemma4_text_model"
    description = (
        "Quantize one tiny dense Gemma4 text model with full and sliding attention."
    )
    tags = ("gemma4", "e2b", "text", "model")
    max_mean_abs_diff = 3.0
    seq_len = 8
    supports_circle_export = False
    circle_export_unsupported_reason = (
        "This case validates PTQ numerical parity only. "
        "Full Gemma4TextModel Circle export requires a dedicated static adapter."
    )

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4 text-model smoke checks."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text model and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(
            cfg,
            layer_types=("sliding_attention", "full_attention"),
        )
        module = Gemma4TextModel(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 text-model input."""
        input_ids = torch.randint(0, self.text_cfg.vocab_size, (1, self.seq_len))
        attention_mask = torch.ones_like(input_ids)
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 text-model calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 text-model evaluation sample."""
        return self._sample()


def _vision_rope(
    batch_size: int,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Gemma4 vision RoPE embeddings."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _vision_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = 4
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _pixel_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence.

    The pooler requires ``pixel_position_ids`` with shape ``(B, S, 2)`` where
    the last dimension encodes ``(x, y)`` patch coordinates.  We build a
    simple square grid layout that is compatible with the ``output_length``
    used in pooler tests: ``seq_len = output_length * k^2`` where ``k`` is
    the pooling factor.
    """
    side = int(seq_len**0.5)
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _padding_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create an all-False padding mask (no padding)."""
    return torch.zeros(batch_size, seq_len, dtype=torch.bool)


class Gemma4VisionAttentionCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision attention module."""

    name = "gemma4_vision_attention"
    description = "Quantize one tiny Gemma4 vision attention module."
    tags = ("gemma4", "e2b", "vision", "attention")
    max_mean_abs_diff = 2.0
    seq_len = 8

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision attention module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionAttention

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_config(cfg)
        self.seq_len = self._vision_patch_seq_len(default=type(self).seq_len)
        module = Gemma4VisionAttention(self.vision_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision attention input."""
        batch_size = 1
        hidden = self._case_hidden_states(
            batch_size=batch_size,
            hidden_size=self.vision_cfg.hidden_size,
        )
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _vision_rope(
                    batch_size,
                    self.seq_len,
                    self.vision_cfg.head_dim,
                ),
                "attention_mask": self._case_vision_attention_mask(batch_size),
                "position_ids": self._case_vision_position_ids(batch_size),
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision attention calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision attention evaluation sample."""
        return self._sample()

    def export_input(
        self,
        eval_sample: ForwardInput,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create static attention inputs without compatibility-only position ids."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        return ForwardInput(
            (
                kwargs["hidden_states"],
                kwargs["position_embeddings"],
                kwargs["attention_mask"],
            ),
            {},
        )


class Gemma4VisionEncoderLayerCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision encoder layer."""

    name = "gemma4_vision_encoder_layer"
    description = "Quantize one tiny Gemma4 vision encoder layer."
    tags = ("gemma4", "e2b", "vision", "encoder_layer")
    max_mean_abs_diff = 2.5
    seq_len = 8

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision encoder layer and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoderLayer

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_config(cfg)
        self.seq_len = self._vision_patch_seq_len(default=type(self).seq_len)
        module = Gemma4VisionEncoderLayer(self.vision_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision encoder-layer input."""
        batch_size = 1
        hidden = self._case_hidden_states(
            batch_size=batch_size,
            hidden_size=self.vision_cfg.hidden_size,
        )
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _vision_rope(
                    batch_size,
                    self.seq_len,
                    self.vision_cfg.head_dim,
                ),
                "attention_mask": self._case_vision_attention_mask(batch_size),
                "position_ids": self._case_vision_position_ids(batch_size),
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision encoder-layer calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=8))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision encoder-layer evaluation sample."""
        return self._sample()

    def export_module(
        self,
        quantized: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> torch.nn.Module:
        """Return the static vision encoder-layer prefill adapter."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module("prefill").eval()
        return quantized

    def export_input(
        self,
        eval_sample: ForwardInput,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create static encoder-layer inputs without position ids."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        return ForwardInput(
            (
                kwargs["hidden_states"],
                kwargs["attention_mask"],
                kwargs["position_embeddings"],
            ),
            {},
        )


class Gemma4TextScaledWordEmbeddingCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 text scaled word embedding module."""

    name = "gemma4_text_scaled_word_embedding"
    description = "Quantize one tiny Gemma4 text scaled word embedding module."
    tags = ("gemma4", "e2b", "text", "embedding")
    max_mean_abs_diff = 1.0
    vocab_size = 256
    embedding_dim = 64
    seq_len = 16
    embed_scale = 0.125

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text scaled word embedding module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextScaledWordEmbedding,
        )

        torch.manual_seed(123)
        self._validated_size_profile(cfg)
        module = Gemma4TextScaledWordEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
            embed_scale=self.embed_scale,
        ).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 text scaled word embedding input."""
        batch_size = 1
        input_ids = torch.randint(
            0, self.vocab_size, (batch_size, self.seq_len), dtype=torch.long
        )
        return ForwardInput((input_ids,))

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 text scaled word embedding calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=8))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 text scaled word embedding evaluation sample."""
        return self._sample()


class Gemma4VisionPatchEmbedderCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision patch embedder module."""

    name = "gemma4_vision_patch_embedder"
    description = "Quantize one tiny Gemma4 vision patch embedder module."
    tags = ("gemma4", "e2b", "vision", "patch_embedder")
    max_mean_abs_diff = 2.0
    hidden_size = 32
    patch_size = 4
    position_embedding_size = 8
    batch_size = 1
    num_patches = 16

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision patch embedder module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_patch_embedder_config(cfg)
        self.hidden_size = int(self.vision_cfg.hidden_size)
        self.patch_size = int(self.vision_cfg.patch_size)
        self.position_embedding_size = int(self.vision_cfg.position_embedding_size)
        self.num_patches = self._vision_patch_seq_len(default=type(self).num_patches)
        self.seq_len = self.num_patches
        module = Gemma4VisionPatchEmbedder(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision patch embedder input."""
        patch_dim = 3 * self.patch_size**2
        if self._static_runtime_shape() is None:
            pixel_values = torch.randn(self.batch_size, self.num_patches, patch_dim)
            pixel_position_ids = _pixel_position_ids(
                self.batch_size,
                self.num_patches,
            )
            padding_positions = torch.zeros(
                self.batch_size, self.num_patches, dtype=torch.bool
            )
        else:
            pixel_values = self._case_pixel_values(
                batch_size=self.batch_size,
                patch_dim=patch_dim,
            )
            pixel_position_ids = self._case_pixel_position_ids(self.batch_size)
            padding_positions = self._case_padding_positions(self.batch_size)
        return ForwardInput((pixel_values, pixel_position_ids, padding_positions))

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision patch embedder calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision patch embedder evaluation sample."""
        return self._sample()

    def export_module(
        self,
        quantized: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> torch.nn.Module:
        """Bake the fixed patch-coordinate profile into an export adapter."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if not hasattr(wrapped, "as_export_module"):
            return quantized
        return wrapped.as_export_module(
            mode="prefill",
            pixel_position_ids=self._case_pixel_position_ids(self.batch_size),
            padding_positions=self._case_padding_positions(self.batch_size),
        ).eval()

    def export_input(
        self,
        eval_sample: ForwardInput,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Keep only pixel values in the static patch-embedder ABI."""
        cloned = _clone_forward_input(eval_sample)
        return ForwardInput((cloned.args[0],), {})


class Gemma4VisionPoolerCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision pooler module."""

    name = "gemma4_vision_pooler"
    description = "Quantize one tiny Gemma4 vision pooler module."
    tags = ("gemma4", "e2b", "vision", "pooler")
    max_mean_abs_diff = 2.0
    # seq_len=16 and output_length=4 so that k=2 (16 / 4 = 4, sqrt(4) = 2).
    seq_len = 16
    output_length = 4

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision pooler module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_config(cfg)
        self.seq_len = self._vision_patch_seq_len(default=type(self).seq_len)
        self.output_length = self._vision_pool_output_length(
            default=type(self).output_length
        )
        module = Gemma4VisionPooler(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision pooler input."""
        batch_size = 1
        return ForwardInput(
            (),
            {
                "hidden_states": self._case_hidden_states(
                    batch_size=batch_size,
                    hidden_size=self.vision_cfg.hidden_size,
                ),
                "pixel_position_ids": self._case_pixel_position_ids(batch_size),
                "padding_positions": self._case_padding_positions(batch_size),
                "output_length": self.output_length,
            },
        )

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 vision pooler without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = module(*cloned.args, **dict(cloned.kwargs))
        # Return only the pooled features for comparison.
        return output[0] if isinstance(output, tuple) else output

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 vision pooler without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        return output[0] if isinstance(output, tuple) else output

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision pooler calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision pooler evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped pooler in prefill mode with fixed output_length.

        Passes ``pixel_position_ids`` so the export adapter precomputes the
        padding-baked pooling weight matrix and valid-prefix length at
        construction time, replacing dynamic mask operations with a static
        ``matmul``.
        """
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            pixel_pos_ids = self._case_pixel_position_ids(1)
            return wrapped.as_export_module(
                mode="prefill",
                output_length=self.output_length,
                pixel_position_ids=pixel_pos_ids,
            ).eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Keep only hidden states in the static pooler ABI.

        Position IDs, padding, output length, and output validity are all
        construction-time profile data.
        """
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        hidden = kwargs["hidden_states"]
        return ForwardInput((hidden,), {})


class Gemma4VisionModelCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision model."""

    name = "gemma4_vision_model"
    description = (
        "Quantize one tiny Gemma4 vision model (patch_embedder + encoder + pooler)."
    )
    tags = ("gemma4", "e2b", "vision", "model")
    max_mean_abs_diff = 3.0
    # seq_len=36 and output_length=4 so that k=2 (36 / 3^2 = 4, sqrt(4) = 2).
    seq_len = 36

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision model and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_config(cfg)
        self.seq_len = self._vision_patch_seq_len(default=type(self).seq_len)
        module = Gemma4VisionModel(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision model input.

        The HF Gemma4VisionModel expects pre-flattened patches:
            pixel_values: (B, num_patches, 3*patch_size^2)
            pixel_position_ids: (B, num_patches, 2)
        """
        batch_size = 1
        patch_size = self.vision_cfg.patch_size
        patch_dim = 3 * patch_size**2
        pixel_values = self._case_pixel_values(
            batch_size=batch_size,
            patch_dim=patch_dim,
        )
        pixel_position_ids = self._case_pixel_position_ids(batch_size)
        return ForwardInput(
            (),
            {
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
                "return_dict": True,
            },
        )

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 vision model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = module(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 vision model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision model calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision model evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export one vision model specialized for a fixed position profile.

        Construction-time position IDs materialize patch embeddings, encoder
        templates, and pooler geometry. The returned runtime module accepts only
        pixel values.
        """
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            pixel_pos_ids = self._case_pixel_position_ids(1)
            return wrapped.as_export_module(
                mode="prefill",
                pixel_position_ids=pixel_pos_ids,
            ).eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Keep only pixel values in the static vision-model ABI."""
        cloned = _clone_forward_input(eval_sample)
        pixel_values = dict(cloned.kwargs)["pixel_values"]
        return ForwardInput((pixel_values,), {})


class Gemma4MultimodalEmbedderCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 multimodal embedder module."""

    name = "gemma4_multimodal_embedder"
    description = (
        "Quantize one tiny Gemma4 multimodal embedder (RMSNorm + Linear projection)."
    )
    tags = ("gemma4", "e2b", "multimodal", "embedder")
    max_mean_abs_diff = 2.0
    seq_len = 16

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 multimodal embedder and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_config(cfg)
        self.text_cfg = self._make_text_config(cfg)
        self.seq_len = self._visual_token_seq_len(default=type(self).seq_len)
        module = Gemma4MultimodalEmbedder(self.vision_cfg, self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 multimodal embedder input."""
        batch_size = 1
        inputs_embeds = torch.randn(
            batch_size, self.seq_len, self.vision_cfg.hidden_size
        )
        return ForwardInput((inputs_embeds,))

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 multimodal embedder calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 multimodal embedder evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped multimodal embedder in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the multimodal embedder adapter.

        The export adapter's forward() takes inputs_embeds.
        """
        cloned = _clone_forward_input(eval_sample)
        inputs_embeds = cloned.args[0]
        return ForwardInput((inputs_embeds,), {})


class Gemma4VisionEncoderCase(Gemma4BaseCase):
    """Smoke case for a tiny Gemma4 vision encoder."""

    name = "gemma4_vision_encoder"
    description = "Quantize a tiny Gemma4 vision encoder."
    tags = ("gemma4", "e2b", "vision", "encoder")
    max_mean_abs_diff = 3.0
    seq_len = 8

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision encoder and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoder

        torch.manual_seed(123)
        self.vision_cfg = self._make_vision_config(cfg)
        self.seq_len = self._vision_patch_seq_len(default=type(self).seq_len)
        module = Gemma4VisionEncoder(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision encoder input."""
        batch_size = 1
        hidden = self._case_hidden_states(
            batch_size=batch_size,
            hidden_size=self.vision_cfg.hidden_size,
        )
        return ForwardInput(
            (),
            {
                "inputs_embeds": hidden,
                "attention_mask": self._case_vision_keep_mask(batch_size),
                "pixel_position_ids": self._case_vision_position_ids(batch_size),
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision encoder calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=8))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision encoder evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped vision encoder in prefill mode.

        Passes ``pixel_position_ids`` so the export adapter precomputes
        position embeddings and the bidirectional attention mask as
        registered buffers, replacing dynamic RoPE and mask computation
        with static gather from precomputed tables.
        """
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            pixel_pos_ids = self._case_vision_position_ids(1)
            return wrapped.as_export_module(
                mode="prefill", pixel_position_ids=pixel_pos_ids
            ).eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the vision encoder adapter.

        The export adapter's ``forward_export`` only takes ``inputs_embeds``.
        Position embeddings and attention mask are precomputed buffers.
        """
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        hidden = kwargs["inputs_embeds"]
        return ForwardInput((hidden,), {})


class Gemma4ModelCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4Model (image-text)."""

    name = "gemma4_model"
    description = (
        "Quantize one tiny Gemma4Model (vision tower + language model + fusion)."
    )
    tags = ("gemma4", "e2b", "model", "image-text")
    max_mean_abs_diff = 5.0
    seq_len = 16
    num_visual_tokens = 4

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4Model smoke checks."""
        from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=int(self.vision_cfg.num_hidden_layers),
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": self.num_visual_tokens,
                }
            },
        )

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4Model and reference copy."""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg, layer_types=("full_attention",))
        self.vision_cfg = self._make_vision_config(cfg)

        config = Gemma4Config(
            text_config=self.text_cfg,
            vision_config=self.vision_cfg,
            audio_config=None,
            image_token_id=10,
            video_token_id=11,
            audio_token_id=12,
        )
        module = Gemma4Model(config).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4Model text-only input.

        Token IDs are kept in [0, 9] to avoid colliding with the image
        placeholder token ID (10).
        """
        input_ids = torch.randint(0, 10, (1, self.seq_len))
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4Model calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4Model evaluation sample."""
        return self._sample()

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the Gemma4Model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = module(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4Model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped Gemma4Model in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the Gemma4Model export adapter.

        The export adapter's forward_export() takes precomputed inputs:
        - inputs_embeds: (1, S, H)
        - per_layer_inputs: (1, S, L, P) or None
        - attention_masks: dict[layer_type -> mask]
        - position_embeddings: dict[layer_type -> (cos, sin)]
        """
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(1, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(1, self.seq_len, num_layers, ple_dim)

        attention_masks: dict[str, torch.Tensor] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(1, 1, self.seq_len, self.seq_len)
            cos = torch.ones(1, self.seq_len, head_dim)
            sin = torch.zeros(1, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        return ForwardInput(
            (inputs_embeds, per_layer_inputs, attention_masks, position_embeddings),
            {},
        )


class Gemma4ForConditionalGenerationCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4ForConditionalGeneration."""

    name = "gemma4_for_conditional_generation"
    description = (
        "Quantize one tiny Gemma4ForConditionalGeneration "
        "(vision + text decoder + lm_head + softcapping)."
    )
    tags = ("gemma4", "e2b", "model", "conditional_generation", "image-text")
    max_mean_abs_diff = 5.0
    seq_len = 16
    num_visual_tokens = 4

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4ForConditionalGeneration smoke checks."""
        from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=int(self.vision_cfg.num_hidden_layers),
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": self.num_visual_tokens,
                }
            },
        )

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4ForConditionalGeneration and reference copy."""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg, layer_types=("full_attention",))
        # Enable logit softcapping to exercise that code path.
        self.text_cfg.final_logit_softcapping = 30.0
        self.vision_cfg = self._make_vision_config(cfg)

        config = Gemma4Config(
            text_config=self.text_cfg,
            vision_config=self.vision_cfg,
            audio_config=None,
            image_token_id=10,
            video_token_id=11,
            audio_token_id=12,
        )
        module = Gemma4ForConditionalGeneration(config).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4ForConditionalGeneration text-only input.

        Token IDs are kept in [0, 9] to avoid colliding with the image
        placeholder token ID (10).
        """
        input_ids = torch.randint(0, 10, (1, self.seq_len))
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4ForConditionalGeneration calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4ForConditionalGeneration evaluation sample."""
        return self._sample()

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the Gemma4ForConditionalGeneration without sharing mutable sample state.

        The wrapper returns logits directly (not a Gemma4CausalLMOutputWithPast).
        """
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4ForConditionalGeneration without sharing mutable state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        return output.logits if hasattr(output, "logits") else output

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped Gemma4ForConditionalGeneration in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the export adapter.

        The export adapter's forward_export() takes precomputed inputs:
        - inputs_embeds: (1, S, H)
        - per_layer_inputs: (1, S, L, P) or None
        - attention_masks: dict[layer_type -> mask]
        - position_embeddings: dict[layer_type -> (cos, sin)]
        """
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(1, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(1, self.seq_len, num_layers, ple_dim)

        attention_masks: dict[str, torch.Tensor] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(1, 1, self.seq_len, self.seq_len)
            cos = torch.ones(1, self.seq_len, head_dim)
            sin = torch.zeros(1, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        return ForwardInput(
            (inputs_embeds, per_layer_inputs, attention_masks, position_embeddings),
            {},
        )


class Gemma4ForCausalLMCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4ForCausalLM (text-only)."""

    name = "gemma4_for_causal_lm"
    description = (
        "Quantize one tiny Gemma4ForCausalLM " "(text decoder + lm_head + softcapping)."
    )
    tags = ("gemma4", "e2b", "model", "causal_lm", "text")
    max_mean_abs_diff = 5.0
    seq_len = 16

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4ForCausalLM smoke checks."""
        from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=0,
        )

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4ForCausalLM and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        torch.manual_seed(123)
        self.text_cfg = self._make_text_config(cfg, layer_types=("full_attention",))
        # Enable logit softcapping to exercise that code path.
        self.text_cfg.final_logit_softcapping = 30.0

        module = Gemma4ForCausalLM(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4ForCausalLM text-only input."""
        input_ids = torch.randint(0, self.text_cfg.vocab_size, (1, self.seq_len))
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4ForCausalLM calibration samples."""
        return [
            self._sample()
            for _ in range(self._calibration_sample_count(cfg, default=3))
        ]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4ForCausalLM evaluation sample."""
        return self._sample()

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the Gemma4ForCausalLM without sharing mutable sample state.

        The wrapper returns logits directly (not a Gemma4CausalLMOutputWithPast).
        """
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4ForCausalLM without sharing mutable state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        return output.logits if hasattr(output, "logits") else output

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped Gemma4ForCausalLM in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the export adapter.

        The export adapter's forward_export() takes precomputed inputs:
        - inputs_embeds: (1, S, H)
        - per_layer_inputs: (1, S, L, P) or None
        - attention_masks: dict[layer_type -> mask]
        - position_embeddings: dict[layer_type -> (cos, sin)]
        """
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(1, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(1, self.seq_len, num_layers, ple_dim)

        attention_masks: dict[str, torch.Tensor] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(1, 1, self.seq_len, self.seq_len)
            cos = torch.ones(1, self.seq_len, head_dim)
            sin = torch.zeros(1, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        return ForwardInput(
            (inputs_embeds, per_layer_inputs, attention_masks, position_embeddings),
            {},
        )


GEMMA4_CASES = (
    Gemma4TextMLPCase(),
    Gemma4TextAttentionPrefillCase(),
    Gemma4TextAttentionDecodeCase(),
    Gemma4TextSlidingAttentionPrefillCase(),
    Gemma4TextSlidingAttentionDecodeCase(),
    Gemma4TextAttentionKEqVPrefillCase(),
    Gemma4TextAttentionSharedKVPrefillCase(),
    Gemma4TextAttentionSharedKVDecodeCase(),
    Gemma4TextDecoderLayerPrefillCase(),
    Gemma4TextDecoderLayerSlidingPrefillCase(),
    Gemma4TextDecoderLayerDecodeCase(),
    Gemma4TextDecoderLayerSharedKVCase(),
    Gemma4TextScaledWordEmbeddingCase(),
    Gemma4VisionPatchEmbedderCase(),
    Gemma4TextModelCase(),
    Gemma4VisionAttentionCase(),
    Gemma4VisionEncoderLayerCase(),
    Gemma4VisionPoolerCase(),
    Gemma4VisionModelCase(),
    Gemma4MultimodalEmbedderCase(),
    Gemma4VisionEncoderCase(),
    Gemma4ModelCase(),
    Gemma4ForConditionalGenerationCase(),
    Gemma4ForCausalLMCase(),
)

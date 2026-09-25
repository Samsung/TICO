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

from dataclasses import dataclass

import torch

from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.utils import torch_dtype_from_name


@dataclass
class Qwen3VLGPTQConfig(GPTQConfig):
    """
    Configuration for GPTQ on Qwen3-VL.

    This config extends the generic GPTQ configuration with Qwen3-VL specific
    switches so that the quantizer can process the model in stage order:

        1) vision patch embed
        2) vision blocks
        3) vision merger / deepstack mergers
        4) text decoder layers
        5) lm_head (optional)

    The main purpose of this configuration is to support layerwise/stagewise
    GPTQ for Qwen3-VL.
    """

    # ------------------------------------------------------------------
    # Model identity
    # ------------------------------------------------------------------
    model_type: str = "qwen3_vl"

    # ------------------------------------------------------------------
    # Stage-level enable/disable switches
    # ------------------------------------------------------------------
    quantize_vision: bool = True
    quantize_text: bool = True
    quantize_lm_head: bool = False

    # ------------------------------------------------------------------
    # Vision-side stage switches
    # ------------------------------------------------------------------
    quantize_vision_patch_embed: bool = True
    quantize_vision_blocks: bool = True
    quantize_vision_merger: bool = True
    quantize_vision_deepstack_mergers: bool = True

    # ------------------------------------------------------------------
    # Text-side stage switches
    # ------------------------------------------------------------------
    quantize_text_layers: bool = True

    # ------------------------------------------------------------------
    # Cache behavior
    # ------------------------------------------------------------------
    move_cache_to_cpu: bool = False
    cache_dtype: torch.dtype | None = None

    # ------------------------------------------------------------------
    # Optional attribute paths for architecture lookup
    # These defaults follow the current Qwen3-VL HF structure.
    # ------------------------------------------------------------------
    visual_attr: str = "model.visual"
    visual_blocks_attr: str = "model.visual.blocks"
    visual_patch_embed_attr: str = "model.visual.patch_embed.proj"
    visual_merger_attr: str = "model.visual.merger"
    visual_deepstack_mergers_attr: str = "model.visual.deepstack_merger_list"

    language_model_attr: str = "model.language_model"
    text_layers_attr: str = "model.language_model.layers"
    lm_head_attr: str = "lm_head"

    # ------------------------------------------------------------------
    # Hessian dtype
    # ------------------------------------------------------------------
    # Dtype used for Hessian (H) and dXXT accumulation.
    # Defaults to FP32 for speed and lower memory. Set to torch.float64
    # for higher-precision accumulation.
    hessian_dtype: torch.dtype = torch.float32

    # Dtype of the input Gram matrices (inp @ inp.T) and the GPTQv2 dXXT
    # cross-term (dX @ inp.T). Hessian storage and factorization follow
    # hessian_dtype. Defaults to torch.float32, it is faster and uses
    # less memory.
    inp_dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------
    # GPTQv2 options
    # ------------------------------------------------------------------
    # GPTQv2: uses FP inference for collecting inputs during quantization
    gptq_v2: bool = False

    # GPTQv2: Path to a DIRECTORY for the FP inputs cache.
    # If set and <dir>/manifest.json exists and passes validation, FP inputs
    # are loaded from per-stage shards on demand instead of running the
    # original model. If set and no valid manifest exists, FP inputs are
    # collected during quantization: each stage is deduplicated (shared
    # inputs such as q/k/v and gate/up are stored once), written as an
    # atomic per-stage shard, and the manifest that publishes the cache is
    # written atomically only after the whole conversion succeeds, so a
    # failed run never publishes a partial cache.
    # If None, FP inputs are collected on-the-fly (default behavior).
    fp_inputs_cache_path: str | None = None

    # GPTQv2: Optional calibration dataset spec (dataset names with sample
    # counts, e.g. "textvqa:50,wikitext2:128") recorded in the FP inputs
    # cache manifest fingerprint. On a warm run the spec is compared against
    # the cached one so that a cache built for different calibration data is
    # rejected instead of silently reused. If None, the calibration component
    # of the fingerprint is not verified. The recipe pipeline stamps this
    # automatically from the ``calibration`` section of the YAML config.
    calibration_dataset_spec: str | None = None

    # GPTQv2: scaling factor for the asymmetric correction (P matrix)
    # `alpha` is the correction strength for GPTQv2's input-error compensation.
    # It scales the `P` matrix that adjusts weight updates to account for upstream quantization error in the activations.
    # A value of `0` disables the correction (standard GPTQ), while values around `0.25` provide the best empirical results.
    gptq_v2_alpha: float = 0.25

    # Use running average for Hessian accumulation.
    # When False, uses summation.
    normalize_H: bool = True

    def __post_init__(self) -> None:
        """Convert string dtype options (from YAML) to torch.dtype."""
        if isinstance(self.hessian_dtype, str):
            self.hessian_dtype = torch_dtype_from_name(self.hessian_dtype)
        if isinstance(self.inp_dtype, str):
            self.inp_dtype = torch_dtype_from_name(self.inp_dtype)

    @property
    def name(self) -> str:
        return "qwen3_vl_gptq"

    def validate(self) -> None:
        """
        Validate Qwen3-VL specific GPTQ settings.

        Raises:
            ValueError: If a numeric or logical option is invalid.
            TypeError: If a field has an unexpected type.
        """
        super().validate()

        for dtype_field in ("hessian_dtype", "inp_dtype"):
            dtype_value = getattr(self, dtype_field)
            if not isinstance(dtype_value, torch.dtype):
                raise TypeError(
                    f"{dtype_field} must be a torch.dtype. got {type(dtype_value)}"
                )
            if dtype_value not in (torch.float32, torch.float64):
                raise ValueError(
                    f"{dtype_field} must be torch.float32 or torch.float64. "
                    f"got {dtype_value}"
                )

        if self.model_type != "qwen3_vl":
            raise ValueError(f"model_type must be 'qwen3_vl'. got {self.model_type!r}")

        if not isinstance(self.quantize_lm_head, bool):
            raise TypeError(
                f"quantize_lm_head must be bool. got {type(self.quantize_lm_head)}"
            )

        if not (self.quantize_vision or self.quantize_text or self.quantize_lm_head):
            raise ValueError(
                "At least one of quantize_vision, quantize_text, or "
                "quantize_lm_head must be True."
            )

        if not self.quantize_vision:
            if self.quantize_vision_patch_embed:
                raise ValueError(
                    "quantize_vision_patch_embed=True requires quantize_vision=True."
                )
            if self.quantize_vision_blocks:
                raise ValueError(
                    "quantize_vision_blocks=True requires quantize_vision=True."
                )
            if self.quantize_vision_merger:
                raise ValueError(
                    "quantize_vision_merger=True requires quantize_vision=True."
                )
            if self.quantize_vision_deepstack_mergers:
                raise ValueError(
                    "quantize_vision_deepstack_mergers=True requires "
                    "quantize_vision=True."
                )

        if not self.quantize_text and self.quantize_text_layers:
            raise ValueError("quantize_text_layers=True requires quantize_text=True.")

        if self.cache_dtype is not None and not isinstance(
            self.cache_dtype, torch.dtype
        ):
            raise TypeError(
                f"cache_dtype must be torch.dtype or None. got {type(self.cache_dtype)}"
            )

        if self.calibration_dataset_spec is not None and not isinstance(
            self.calibration_dataset_spec, str
        ):
            raise TypeError(
                "calibration_dataset_spec must be str or None. "
                f"got {type(self.calibration_dataset_spec)}"
            )

        attr_fields = {
            "visual_attr": self.visual_attr,
            "visual_blocks_attr": self.visual_blocks_attr,
            "visual_patch_embed_attr": self.visual_patch_embed_attr,
            "visual_merger_attr": self.visual_merger_attr,
            "visual_deepstack_mergers_attr": self.visual_deepstack_mergers_attr,
            "language_model_attr": self.language_model_attr,
            "text_layers_attr": self.text_layers_attr,
            "lm_head_attr": self.lm_head_attr,
        }

        for field_name, field_value in attr_fields.items():
            if not isinstance(field_value, str) or not field_value:
                raise ValueError(
                    f"{field_name} must be a non-empty string. got {field_value!r}"
                )

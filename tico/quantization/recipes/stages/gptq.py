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

from pathlib import Path
from typing import Any, Mapping

import torch

from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.base import Stage
from tico.quantization.recipes.utils import filter_dataclass_kwargs, stage_payload


class GPTQStage(Stage):
    name = "gptq"

    _SENSITIVITY_MODES = {"compute", "load", "save", "cache"}

    @staticmethod
    def _is_smse_mode(payload: Mapping[str, Any]) -> bool:
        """Return True when the GPTQ stage should use sensitivity-aware MSE."""
        return payload.get("mse") in {"smse", "smse_for_gptq"}

    @classmethod
    def _sensitivity_mode_and_path(
        cls,
        payload: Mapping[str, Any],
    ) -> tuple[str, Path | None]:
        """Resolve the sensitivity cache mode and path from stage configuration."""
        raw_cfg = payload.get("sensitivity")
        if raw_cfg is None:
            return "compute", None

        if not isinstance(raw_cfg, Mapping):
            raise TypeError(
                "GPTQ sensitivity config must be a mapping with `mode` and `path`."
            )

        mode = str(raw_cfg.get("mode", "compute")).lower()
        if mode not in cls._SENSITIVITY_MODES:
            supported = ", ".join(sorted(cls._SENSITIVITY_MODES))
            raise ValueError(
                f"Unsupported GPTQ sensitivity mode {mode!r}. "
                f"Supported modes: {supported}."
            )

        raw_path = raw_cfg.get("path")
        if mode == "compute":
            if raw_path is not None:
                raise ValueError(
                    "GPTQ sensitivity mode 'compute' does not use "
                    "`sensitivity.path`. Use mode 'save', 'load', or 'cache' when "
                    "a path is needed."
                )
            return mode, None

        if raw_path is None or str(raw_path).strip() == "":
            raise ValueError(
                f"GPTQ sensitivity mode {mode!r} requires `sensitivity.path`."
            )

        return mode, Path(raw_path)

    @staticmethod
    def _load_sensitivity(path: Path) -> dict[str, torch.Tensor]:
        """Load sensitivity tensors from disk."""
        if not path.exists():
            raise FileNotFoundError(f"GPTQ sensitivity file does not exist: {path}")

        print(f"Loading GPTQ sensitivity information from {path.resolve()}")
        sensitivity = torch.load(path, map_location="cpu")
        if not isinstance(sensitivity, dict):
            raise TypeError(
                f"GPTQ sensitivity file must contain a dict. got {type(sensitivity)}"
            )
        return sensitivity

    @staticmethod
    def _save_sensitivity(path: Path, sensitivity: dict[str, torch.Tensor]) -> None:
        """Save sensitivity tensors to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving GPTQ sensitivity information to {path.resolve()}")
        torch.save(sensitivity, path)

    @staticmethod
    def _compute_sensitivity(ctx: RecipeContext) -> dict[str, torch.Tensor]:
        """Compute GPTQ sensitivity tensors from calibration inputs."""
        print("Computing GPTQ sensitivity information …")
        calibrator = SensitivityCalibrator(ctx.require_model(), ctx.calibration_inputs)
        sensitivity = calibrator.compute_sensitivity_info()
        if not isinstance(sensitivity, dict):
            raise TypeError(
                "SensitivityCalibrator.compute_sensitivity_info() must return a dict. "
                f"got {type(sensitivity)}"
            )
        return sensitivity

    @classmethod
    def _resolve_sensitivity(
        cls,
        ctx: RecipeContext,
        payload: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Load, compute, save, or cache sensitivity tensors for GPTQ SMSE."""
        mode, path = cls._sensitivity_mode_and_path(payload)

        if mode == "compute":
            return cls._compute_sensitivity(ctx)

        assert path is not None

        if mode == "load":
            return cls._load_sensitivity(path)

        if mode == "save":
            sensitivity = cls._compute_sensitivity(ctx)
            cls._save_sensitivity(path, sensitivity)
            ctx.artifacts["gptq_sensitivity_path"] = str(path)
            return sensitivity

        if mode == "cache":
            if path.exists():
                return cls._load_sensitivity(path)

            sensitivity = cls._compute_sensitivity(ctx)
            cls._save_sensitivity(path, sensitivity)
            ctx.artifacts["gptq_sensitivity_path"] = str(path)
            return sensitivity

        raise AssertionError(f"Unhandled sensitivity mode: {mode}")

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        payload = stage_payload(stage_cfg)

        if self._is_smse_mode(payload):
            payload["sensitivity"] = self._resolve_sensitivity(ctx, payload)

        # Generic GPTQ now has LLaMA-specific safety switches such as
        # quantize_lm_head=False by default and use_orig_model_inference.
        config_cls = (
            Qwen3VLGPTQConfig if ctx.adapter.family == "qwen3_vl" else GPTQConfig
        )
        gptq_config = config_cls(**filter_dataclass_kwargs(config_cls, payload))

        print(f"Applying {gptq_config.name} …")
        q_model = prepare(ctx.require_model(), gptq_config, inplace=True)

        ctx.adapter.forward_calibration(
            ctx,
            q_model,
            ctx.calibration_inputs,
            desc="GPTQ calibration",
        )

        ctx.model = convert(q_model, inplace=True)
        return ctx

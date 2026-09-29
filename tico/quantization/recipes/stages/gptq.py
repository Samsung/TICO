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

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.base import BaseConfig
from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig
from tico.quantization.config.gptq import GPTQConfig, UniversalGPTQConfig
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.dataset_config import (
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CALIBRATION_N_SAMPLES,
    DEFAULT_CALIBRATION_SEED,
    normalize_mixed_dataset_config,
)
from tico.quantization.recipes.stages.base import Stage
from tico.quantization.recipes.utils import filter_dataclass_kwargs, stage_payload


class GPTQStage(Stage):
    name = "gptq"
    requires_calibration_inputs = True

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

    @staticmethod
    def _calibration_dataset_spec(
        calibration_cfg: Mapping[str, Any],
        runtime_cfg: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Build a complete, order-preserving provenance spec of the calibration
        datasets.

        Recorded in the FP-inputs cache manifest fingerprint so a warm cache
        is only reused for identical calibration data.  The spec is built from
        the same normalized form the loader produces
        (``normalize_mixed_dataset_config``), so every accepted ``datasets``
        form (string, mapping, sequence) is covered, entry order is
        significant, and split/filter settings are included.  Mirrors
        the defaults of ``build_vlm_calibration_inputs`` (imported from
        ``recipes.data.dataset_config`` so both sides stay in sync).

        The sampling seed is taken from the ``runtime`` section, exactly as
        the adapters and the runner do (``runtime.seed``, int-coerced);
        ``calibration.seed`` has never reached the loader, so it is ignored
        here as well (with a warning).  ``seq_len`` and
        ``allow_benchmark_overlap`` are included because they change the
        calibration inputs; the top-level ``split`` is included only for the
        single-dataset path, mirroring the adapters, which ignore it when
        ``datasets`` is set.  ``allow_unregistered_dataset`` is a guard flag
        with no effect on the data and is deliberately excluded.
        """
        if "seed" in calibration_cfg:
            print(
                "[GPTQ] WARNING: calibration.seed does not reach the "
                "calibration loader; set runtime.seed instead. It is ignored "
                "by the FP-inputs cache fingerprint."
            )
        default_n = calibration_cfg.get("n_samples", DEFAULT_CALIBRATION_N_SAMPLES)
        datasets = calibration_cfg.get("datasets")
        single_dataset = datasets is None
        if single_dataset:
            datasets = calibration_cfg.get("dataset") or DEFAULT_CALIBRATION_DATASET
        normalized = normalize_mixed_dataset_config(datasets, default_n)  # type: ignore[arg-type]
        provenance = {
            # The loader's sampling seed lives in the runtime section
            # (adapters and runner use runtime.seed, int-coerced).
            "seed": int((runtime_cfg or {}).get("seed", DEFAULT_CALIBRATION_SEED)),
            "seq_len": calibration_cfg.get("seq_len"),
            "allow_benchmark_overlap": bool(
                calibration_cfg.get("allow_benchmark_overlap", False)
            ),
            "datasets": [
                {
                    "dataset": name,
                    **{k: v for k, v in entry.items() if v is not None},
                }
                for name, entry in normalized.items()
            ],
        }
        # Top-level split reaches the loader only on the single-dataset path.
        if single_dataset:
            provenance["split"] = calibration_cfg.get("split")
        return json.dumps(provenance, sort_keys=True, default=str)

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        payload = stage_payload(stage_cfg)

        if self._is_smse_mode(payload):
            payload["sensitivity"] = self._resolve_sensitivity(ctx, payload)

        # Select the GPTQ variant. `variant: universal` selects the
        # model-agnostic frontier-based quantizer; the default variant uses
        # the model-family-specific quantizer.
        variant = str(payload.pop("variant", "default")).strip().lower()
        if variant not in {"default", "universal"}:
            raise ValueError(
                f"Unsupported GPTQ variant {variant!r}. "
                "Supported variants: default, universal."
            )
        # Stamp the calibration dataset spec for configs that declare it
        # (e.g. Qwen3VLGPTQConfig): the FP-inputs cache fingerprint compares
        # it on warm runs so a cache built for different calibration data is
        # rejected instead of silently reused. filter_dataclass_kwargs drops
        # the key for config classes without the field.
        if "calibration_dataset_spec" not in payload:
            calibration_cfg = ctx.cfg.get("calibration", {})
            if isinstance(calibration_cfg, Mapping) and calibration_cfg:
                runtime_cfg = ctx.cfg.get("runtime", {})
                payload["calibration_dataset_spec"] = self._calibration_dataset_spec(
                    calibration_cfg,
                    runtime_cfg if isinstance(runtime_cfg, Mapping) else None,
                )

        # Map model family to the appropriate GPTQ config class.
        # Families with a dedicated multimodal GPTQ config (vision + text
        # stagewise quantization) get their own class; everything else falls
        # back to the generic GPTQConfig (single decoder stack).
        _FAMILY_CONFIG_MAP: dict[str, type[BaseConfig]] = {
            "qwen3_vl": Qwen3VLGPTQConfig,
            "gemma4": Gemma4GPTQConfig,
        }
        if variant == "universal":
            config_cls: type[BaseConfig] = UniversalGPTQConfig
        else:
            config_cls = _FAMILY_CONFIG_MAP.get(ctx.adapter.family, GPTQConfig)
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

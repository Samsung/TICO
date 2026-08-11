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

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

import torch

from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.evaluation.selection import (
    get_selected_evaluation_targets,
)


class ModelAdapter(ABC):
    """Model-family-specific hooks for a common quantization recipe runner."""

    family: str
    evaluation_targets: frozenset[str] = frozenset()
    evaluation_target_requirements: Mapping[str, str] = {}

    def validate_evaluation_config(self, cfg: Mapping[str, Any]) -> None:
        """Validate selected top-level evaluation targets before model execution."""
        eval_cfg = cfg.get("evaluation")
        if eval_cfg is None:
            return
        if not isinstance(eval_cfg, Mapping):
            raise TypeError("evaluation must be a mapping.")
        if not eval_cfg.get("enabled", False):
            return

        selected_targets = get_selected_evaluation_targets(eval_cfg)
        if selected_targets is None:
            return

        unsupported_targets = [
            target
            for target in selected_targets
            if target not in self.evaluation_targets
        ]
        if unsupported_targets:
            raise ValueError(
                "Unsupported evaluation target(s) for model family "
                f"{self.family!r}: {unsupported_targets}. Supported targets: "
                f"{sorted(self.evaluation_targets)}."
            )

        for target_name, config_key in self.evaluation_target_requirements.items():
            if target_name not in selected_targets:
                continue
            if eval_cfg.get(config_key):
                continue
            raise ValueError(
                f"Evaluation target {target_name!r} requires non-empty "
                f"evaluation.{config_key}."
            )

    @abstractmethod
    def load_model(self, ctx: RecipeContext) -> RecipeContext:
        raise NotImplementedError

    @abstractmethod
    def build_calibration_inputs(self, ctx: RecipeContext) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_calibration(
        self,
        ctx: RecipeContext,
        model: torch.nn.Module,
        calibration_inputs: Sequence[Any],
        *,
        desc: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, ctx: RecipeContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def export(self, ctx: RecipeContext) -> None:
        raise NotImplementedError

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


class ModelAdapter(ABC):
    """Model-family-specific hooks for a common quantization recipe runner."""

    family: str

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

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class RecipeContext:
    """Mutable state shared by recipe stages."""

    cfg: dict[str, Any]
    adapter: Any
    model: Any = None
    original_model: Any = None
    tokenizer: Any = None
    processor: Any = None
    calibration_inputs: list[Any] = field(default_factory=list)
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    output_dir: Path | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    def require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("RecipeContext.model is not initialized.")
        return self.model

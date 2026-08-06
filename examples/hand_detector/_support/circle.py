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

"""Circle-side layout optimization helpers for the hand detector."""

from __future__ import annotations

from pathlib import Path

from tico.circle.document import CircleDocument
from tico.circle.passes import (
    CirclePassContext,
    CirclePassManager,
    CirclePassManagerResult,
    CirclePassStrategy,
    EliminateTransposeBoundedLayoutRegionPass,
    RemoveRedundantLayoutOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.utils.model import CircleModel


def optimize_layout_transitions(
    circle_model: CircleModel,
) -> tuple[CircleModel, CirclePassManagerResult]:
    """Optimize layout round trips in a serialized Circle model."""
    document = CircleDocument.from_bytes(circle_model.circle_binary)
    pipeline = CirclePassManager(
        [
            EliminateTransposeBoundedLayoutRegionPass(),
            RemoveRedundantLayoutOpsPass(),
            DeadCodeEliminationPass(),
            CompactIndicesPass(),
        ],
        strategy=CirclePassStrategy.RESTART,
    )
    result = pipeline.run(document, CirclePassContext())
    document.verify(raise_on_error=True)
    return CircleModel(document.to_bytes()), result


def save_layout_optimized_circle(
    circle_model: CircleModel,
    output_path: str | Path,
) -> tuple[Path, CirclePassManagerResult]:
    """Optimize one Circle model and save the resulting binary."""
    optimized, result = optimize_layout_transitions(circle_model)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(output)
    return output, result

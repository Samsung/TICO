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

from __future__ import annotations

from tico.circle.document import CircleDocument
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.rewrite import compact_model


class CompactIndicesPass(CirclePass):
    """Remove unused tensors, buffers, and operator codes and compact indices."""

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Compact every model-global and subgraph-local index space."""

        stats = compact_model(document.model)
        changes = (
            stats.removed_tensors
            + stats.removed_buffers
            + stats.removed_operator_codes
            + stats.remapped_references
        )
        diagnostic = (
            "Removed "
            f"{stats.removed_tensors} tensors, "
            f"{stats.removed_buffers} buffers, and "
            f"{stats.removed_operator_codes} operator codes; "
            f"remapped {stats.remapped_references} references."
        )
        context.logger.debug(diagnostic)
        return CirclePassResult(
            modified=stats.modified,
            changes=changes,
            diagnostics=(diagnostic,),
        )

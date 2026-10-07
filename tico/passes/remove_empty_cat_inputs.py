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

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx

import torch
from torch.export import ExportedProgram

from tico.passes import ops
from tico.passes.remove_unused_placeholder import remove_unused_constant_placeholders
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import (
    trace_const_diff_on_pass,
    trace_graph_diff_on_pass,
)
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import CatArgs


def _is_legacy_empty_tensor(node: "torch.fx.Node") -> bool:
    """
    Return whether ``node`` is a rank-1 tensor with a static size of zero.

    ``torch.cat`` skips such tensors even when the other inputs have a different
    rank (a legacy behavior kept for backward compatibility).
    """
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return False
    if val.dim() != 1:
        return False
    size = val.shape[0]
    # A symbolic size is a SymInt, not an int, and is never treated as empty.
    if not isinstance(size, int) or size != 0:
        return False
    return True


def _promoted_dtype(nodes: List["torch.fx.Node"]) -> torch.dtype:
    """Return the dtype ``torch.cat`` produces for the given tensor inputs."""
    dtypes = [node.meta["val"].dtype for node in nodes]
    result = dtypes[0]
    for dtype in dtypes[1:]:
        result = torch.promote_types(result, dtype)
    return result


@trace_graph_diff_on_pass
@trace_const_diff_on_pass
class RemoveEmptyCatInputs(PassBase):
    """
    Remove rank-1 empty inputs from ``aten.cat`` and drop the ``cat`` when a single
    input remains.

    PyTorch lets ``torch.cat`` accept a 1-D tensor of shape ``[0]`` next to inputs of
    any other rank and simply ignores it. Hugging Face ``DynamicCache`` relies on
    this: the first ``update()`` concatenates ``torch.tensor([])`` with the new key
    and value states. Circle ``CONCATENATION`` requires every input to have the same
    rank, so the exported operator is rejected by Circle consumers.

    Preconditions
        - The node is ``aten.cat``.
        - At least one input has a tensor ``meta["val"]`` of rank 1 with a static
          size of zero.
        - At least one input does not satisfy the previous bullet.
        - The dtype promoted from the remaining inputs equals the ``cat`` output
          dtype, so the removed inputs did not take part in type promotion.

    Transformation
        - Matching inputs are removed from the ``cat`` input list.
        - When exactly one input remains, all uses of the ``cat`` are redirected to
          that input; its shape and dtype equal the ``cat`` output by construction.
        - Lifted constant placeholders that only fed removed inputs are deleted from
          the graph, the ``ExportedProgram`` constants, and the graph signature.
          User input placeholders are never removed.

    Postconditions
        No ``aten.cat`` keeps a rank-1 zero-size input of the output dtype.

    Semantic assumptions
        A rank-1 zero-size tensor contributes no elements along any concatenation
        dimension, so removing it preserves the value, shape, and dtype of the
        result as long as it does not change type promotion. Empty inputs that
        widen the result dtype, inputs with symbolic sizes, and zero-size inputs
        with the same rank as the output are left untouched.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        placeholder_candidates: List["torch.fx.Node"] = []
        for cat in graph.nodes:
            if not is_target_node(cat, ops.aten.cat):
                continue

            cat_val = cat.meta.get("val")
            if not isinstance(cat_val, torch.Tensor):
                continue

            args = CatArgs(*cat.args, **cat.kwargs)  # type: ignore[arg-type]
            inputs = args.tensors
            dim = args.dim

            kept = [t for t in inputs if not _is_legacy_empty_tensor(t)]
            if len(kept) == len(inputs) or len(kept) == 0:
                continue
            if any(not isinstance(t.meta.get("val"), torch.Tensor) for t in kept):
                continue
            # Dropping an input must not change the promoted result dtype.
            if _promoted_dtype(kept) != cat_val.dtype:
                continue

            removed = [t for t in inputs if t not in kept]
            for node in removed:
                if node.op == "placeholder":
                    placeholder_candidates.append(node)
                placeholder_candidates.extend(
                    n for n in node.all_input_nodes if n.op == "placeholder"
                )

            if len(kept) == 1:
                remaining = kept[0]
                assert extract_shape(remaining) == extract_shape(cat), (
                    f"{cat.name}: shape mismatch after removing empty inputs "
                    f"({extract_shape(remaining)} vs {extract_shape(cat)})"
                )
                remaining_val = remaining.meta.get("val")
                assert (
                    isinstance(remaining_val, torch.Tensor)
                    and remaining_val.dtype == cat_val.dtype
                )
                cat.replace_all_uses_with(remaining, propagate_meta=False)
                logger.debug(
                    f"{cat.name} is replaced with {remaining.name}; "
                    f"empty inputs removed: {[n.name for n in removed]}"
                )
            else:
                cat.args = (kept, dim)
                cat.kwargs = {}
                logger.debug(
                    f"Empty inputs removed from {cat.name}: {[n.name for n in removed]}"
                )

            modified = True

        if not modified:
            return PassResult(False)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        removed_placeholders = remove_unused_constant_placeholders(
            exported_program, candidates=placeholder_candidates
        )
        if removed_placeholders:
            logger.debug(
                f"Unused constant placeholders are removed: {removed_placeholders}"
            )

        return PassResult(modified)

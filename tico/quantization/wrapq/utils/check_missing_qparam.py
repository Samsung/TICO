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

from typing import Optional, Set

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY
from tico.utils import logging


def _is_tensor_like_node(node: torch.fx.Node) -> bool:
    """
    Return True if the node appears to produce a tensor or a container of tensors.

    We rely on FX/export metadata because the graph may contain non-tensor nodes
    such as shape/index bookkeeping ops that should not be checked for qparam.
    """
    val = node.meta.get("val", None)

    if isinstance(val, torch.Tensor):
        return True

    if isinstance(val, (tuple, list)) and val:
        return any(isinstance(x, torch.Tensor) for x in val)

    if "tensor_meta" in node.meta:
        return True

    return False


def check_missing_qparam(
    exported_program: ExportedProgram,
    *,
    strict: bool = False,
    ignore_targets: Optional[Set[object]] = None,
) -> None:
    """
    Inspect the final graph once after all quantization-related passes complete.

    This checker warns or raises if tensor-producing call_function nodes still do
    not have QPARAM metadata. It is intentionally not implemented as a pass
    because PassManager may restart and rerun passes multiple times, while this
    check should run exactly once on the final graph.

    Args:
        exported_program:
            The exported program to inspect.
        strict:
            If True, raise RuntimeError when any missing-qparam node is found.
            If False, only emit warnings.
        ignore_targets:
            A set of call targets to exclude from the check.
    """
    logger = logging.getLogger(__name__)

    ignore_targets = ignore_targets or set()
    graph = exported_program.graph_module.graph

    missing: list[torch.fx.Node] = []
    quantized_nodes = 0
    fp_nodes = 0

    for node in graph.nodes:
        if node.op != "call_function":
            continue

        if node.target in ignore_targets:
            continue

        if not _is_tensor_like_node(node):
            continue

        if QPARAM_KEY in node.meta:
            quantized_nodes += 1
        else:
            fp_nodes += 1
            missing.append(node)

    # Summary statistics
    logger.debug(f"[QuantCheck] quantized nodes : {quantized_nodes}")
    logger.debug(f"[QuantCheck] fp nodes        : {fp_nodes}")

    if not missing:
        return

    # Short message for user-facing output
    print(
        f"[QuantCheck] WARNING: {len(missing)} nodes without qparam detected "
        "(see logs)."
    )

    # Detailed logs for debugging
    for node in missing:
        target_name = getattr(node.target, "__name__", str(node.target))

        logger.debug(
            f"[QuantCheck] Missing qparam:\n"
            f"  name   : {node.name}\n"
            f"  target : {target_name}\n"
            f"  users  : {len(node.users)}\n"
            f"  trace  : {node.meta.get('stack_trace')}",
        )

    if strict:
        raise RuntimeError(
            f"[QuantCheck] {len(missing)} nodes without qparam detected."
        )

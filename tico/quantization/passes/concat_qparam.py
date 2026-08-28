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

"""Concat-specific quantization-parameter propagation rules."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from tico.utils.validate_args_kwargs import CatArgs


_MIN_WIDTH_CONCAT_RANK = 3
_TICO_CONCAT_TYPE_SUFFIXES = (
    "tico.ops.concat.Concat",
    "tico.quantization.wrapq.wrappers.ops.quant_concat.QuantConcat",
)


def is_width_direction_concat(
    node: torch.fx.Node,
    args: CatArgs,
) -> bool:
    """Return whether one TICO Concat joins channels-last width.

    The backend permits inputs of a width-direction Concat to keep distinct
    per-tensor scale and zero point values. Qparam propagation must therefore
    not copy the Concat output domain to its inputs, or an input domain to the
    Concat output.

    The TICO module-origin check prevents an unrelated rank-three operation,
    such as sequence concatenation implemented with raw ``torch.cat``, from
    inheriting the hand-detector backend exception. Rank-two tensors are also
    excluded because dimension zero commonly represents a batch-like axis.
    """
    if not _originates_from_tico_concat(node) or not args.tensors:
        return False

    ranks: set[int] = set()
    for tensor in args.tensors:
        value = tensor.meta.get("val")
        shape = getattr(value, "shape", None)
        if shape is None:
            return False
        ranks.add(len(shape))

    if len(ranks) != 1:
        return False
    rank = ranks.pop()
    if rank < _MIN_WIDTH_CONCAT_RANK:
        return False

    dim = int(args.dim)
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        return False
    return dim == rank - 2


def _originates_from_tico_concat(node: torch.fx.Node) -> bool:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, Mapping):
        return False
    for entry in stack.values():
        if not isinstance(entry, tuple) or len(entry) < 2:
            continue
        type_name = str(entry[1])
        if type_name.endswith(_TICO_CONCAT_TYPE_SUFFIXES):
            return True
    return False

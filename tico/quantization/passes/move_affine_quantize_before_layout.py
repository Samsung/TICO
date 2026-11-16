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

"""Commute an existing fixed affine Q/DQ with a private static layout chain."""

import math

import torch
from torch.export import ExportedProgram

from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
)
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val

_LAYOUTS = frozenset({torch.ops.aten.reshape.default, torch.ops.aten.permute.default})


def _arguments(node: torch.fx.Node) -> tuple:
    """Read schema defaults without changing the original argument representation."""
    schema = getattr(node.target, "_schema")
    return tuple(
        (
            node.args[index]
            if index < len(node.args)
            else node.kwargs.get(arg.name, arg.default_value)
        )
        for index, arg in enumerate(schema.arguments)
    )


def _set_input(node: torch.fx.Node, value: torch.fx.Node) -> None:
    if node.args:
        node.args = (value, *node.args[1:])
    else:
        schema = getattr(node.target, "_schema")
        node.kwargs = {**node.kwargs, schema.arguments[0].name: value}


def _static_float(node: torch.fx.Node) -> bool:
    value = node.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and value.layout == torch.strided
        and all(type(dim) is int and dim > 0 for dim in value.shape)
    )


def _protected(node: torch.fx.Node) -> bool:
    return bool(node.meta.get(EXPLICIT_EMBEDDING) or node.meta.get(FLOAT_EMBEDDING))


def _mx_source(node: torch.fx.Node) -> bool:
    if node.target != torch.ops.circle_custom.dequantize_mx.default:
        return False
    dq_args = _arguments(node)
    q = dq_args[0]
    if not isinstance(q, torch.fx.Node) or q.target != (
        torch.ops.circle_custom.quantize_mx.default
    ):
        return False
    q_args = _arguments(q)
    producer = q_args[0]
    return (
        isinstance(producer, torch.fx.Node)
        and q_args[1] == "int8"
        and type(q_args[2]) is int
        and type(dq_args[2]) is int
        and q_args[1:] == dq_args[1:]
        and -node.meta["val"].ndim <= q_args[2] < node.meta["val"].ndim
        and q_args[3:] == ("max", "nearest")
        and _static_float(producer)
        and not any(_protected(n) for n in (node, q, producer))
    )


def _layout_source(node: torch.fx.Node) -> torch.fx.Node | None:
    if node.op != "call_function" or node.target not in _LAYOUTS:
        return None
    if not _static_float(node) or _protected(node) or QPARAM_KEY in node.meta:
        return None
    args = _arguments(node)
    source, dims = args
    if (
        not isinstance(source, torch.fx.Node)
        or not _static_float(source)
        or not isinstance(dims, (tuple, list))
        or any(type(dim) is not int for dim in dims)
    ):
        return None
    if node.target == torch.ops.aten.reshape.default:
        if any(dim == 0 or dim < -1 for dim in dims) or dims.count(-1) > 1:
            return None
        if math.prod(source.meta["val"].shape) != math.prod(node.meta["val"].shape):
            return None
    else:
        rank = source.meta["val"].ndim
        if len(dims) != rank or any(not -rank <= dim < rank for dim in dims):
            return None
        if len({dim % rank for dim in dims}) != rank:
            return None
    return source


@trace_graph_diff_on_pass
class MoveAffineQuantizeBeforeLayout(PassBase):
    """Move MXINT8 -> layouts -> INT16 Q/DQ to MXINT8 -> INT16 Q/DQ -> layouts.

    Run after layout normalization and embedding protection, before FoldQuantOps.
    Match only scalar, fixed per-tensor affine default overloads with identical
    scale, zero point, storage dtype and range, and implicit float32 DQ output. Each static
    reshape/permute result must have exactly the next layout or Q as its sole user
    (including graph outputs). The MX source and final DQ may have multiple users.
    Protected or already annotated boundaries are left alone.

    Reuse both affine nodes and every argument: their rounding/clipping and the
    original MX quantizer remain unchanged. Only the selected source connection
    changes. Q_A(L(u)) == L(Q_A(u)) needs no MX block-preservation test on L,
    since folding will execute L in the existing per-tensor INT16 domain.
    """

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph = exported_program.graph
        modified = False
        for dq in list(graph.nodes):
            if dq.op != "call_function" or dq.target != (
                torch.ops.quantized_decomposed.dequantize_per_tensor.default
            ):
                continue
            dq_args = _arguments(dq)
            q = dq_args[0]
            if not isinstance(q, torch.fx.Node) or q.target != (
                torch.ops.quantized_decomposed.quantize_per_tensor.default
            ):
                continue
            q_args = _arguments(q)
            scale, zp, qmin, qmax, dtype = q_args[1:]
            if (
                type(scale) not in (float, int)
                or not math.isfinite(scale)
                or scale <= 0
                or any(type(value) is not int for value in (zp, qmin, qmax))
                or dtype != torch.int16
                or (qmin, qmax) not in ((-32768, 32767), (-32767, 32767))
                or not qmin <= zp <= qmax
                or q_args[1:] != dq_args[1:6]
                # Folding currently accepts only the implicit float32 DQ ABI.
                or "out_dtype" in dq.kwargs
                or set(q.users) != {dq}
                or any(_protected(n) or QPARAM_KEY in n.meta for n in (q, dq))
                or not _static_float(dq)
            ):
                continue

            chain = []
            current = q_args[0]
            user = q
            while isinstance(current, torch.fx.Node):
                source = _layout_source(current)
                if source is None or set(current.users) != {user}:
                    break
                chain.append(current)
                user, current = current, source
            if not chain or not isinstance(current, torch.fx.Node):
                continue
            if not _static_float(current) or not _mx_source(current):
                continue

            # Functional export normally has no mutations here. Also protect
            # direct invocations: a write through another alias of the shared
            # source between a view and Q would change the snapshot being read.
            first, last = chain[-1], chain[0]
            intervening = first.next
            while intervening is not q and not intervening.is_impure():
                intervening = intervening.next
            if intervening is not q:
                continue

            # All checks precede mutation. Rewire only this private layout path;
            # the MX source's other consumers and graph-output uses stay intact.
            dq.replace_all_uses_with(last)
            _set_input(q, current)
            _set_input(first, dq)
            first.prepend(q)
            first.prepend(dq)
            for node in (q, dq, *reversed(chain)):
                node.meta.pop("tensor_meta", None)
                set_new_meta_val(node)
            modified = True

        if modified:
            graph.lint()
            exported_program.graph_module.recompile()
        return PassResult(modified)

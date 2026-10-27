# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import copy
import math

import torch
from torch.export import ExportedProgram

from tico.quantization.passes.concat_qparam import is_width_direction_concat
from tico.quantization.passes.mx_qparam import remap_mx_qparam
from tico.quantization.passes.remove_weight_dequant_op import infer_dtype
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.mx.dtypes import is_mx_dtype
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import ArgTypeError
from tico.utils.validate_args_kwargs import (
    CatArgs,
    PermuteArgs,
    QuantizePerTensorArgs,
    ReshapeArgs,
)


def _retarget_affine_per_tensor_quantize(
    src: torch.fx.Node, dst: torch.fx.Node
) -> None:
    """Retarget a private Quantize boundary without changing its source domain."""
    if len(dst.users) != 1 or src not in dst.users:
        return

    qparam = src.meta.get(QPARAM_KEY)
    if not isinstance(qparam, QuantParam):
        return
    qparams = [qparam]
    if QPARAM_KEY in dst.meta:
        existing = dst.meta[QPARAM_KEY]
        if not isinstance(existing, QuantParam):
            return
        qparams.append(existing)

    # Resolve schema keyword names (input/zero_point) before using the existing
    # validator, whose corresponding fields are named tensor/zero_p.
    schema = getattr(dst.target, "_schema")
    values = [
        dst.args[index] if index < len(dst.args) else dst.kwargs.get(argument.name)
        for index, argument in enumerate(schema.arguments)
    ]
    try:
        args = QuantizePerTensorArgs(*values)  # type: ignore[arg-type]
        dtype = infer_dtype(args.quant_min, args.quant_max, args.dtype)
    except (ArgTypeError, TypeError, ValueError, RuntimeError):
        return
    if (
        not math.isfinite(args.scale)
        or args.scale <= 0
        or not args.quant_min <= args.zero_p <= args.quant_max
    ):
        return

    for candidate in qparams:
        if (
            is_mx_dtype(candidate.dtype)
            or candidate.dtype != dtype
            or candidate.quantized_dimension is not None
            or not isinstance(candidate.scale, list)
            or len(candidate.scale) != 1
            or not isinstance(candidate.zero_point, list)
            or len(candidate.zero_point) != 1
            or not isinstance(candidate.scale[0], (float, int))
            or not math.isfinite(candidate.scale[0])
            or candidate.scale[0] <= 0
            or not isinstance(candidate.zero_point[0], int)
            or not args.quant_min <= candidate.zero_point[0] <= args.quant_max
        ):
            return

    # Stage every replacement before mutating the node. Preserve argument
    # placement, storage dtype, range, and all unrelated kwargs and metadata.
    assert qparam.scale is not None and qparam.zero_point is not None
    scale, zero_point = float(qparam.scale[0]), qparam.zero_point[0]
    new_qparam = copy.deepcopy(qparam)
    new_args = list(dst.args)
    new_kwargs = dict(dst.kwargs)
    for index, name, value in (
        (1, "scale", scale),
        (2, "zero_point", zero_point),
    ):
        if index < len(new_args):
            new_args[index] = value
        else:
            new_kwargs[name] = value
    dst.args = tuple(new_args)
    dst.kwargs = new_kwargs
    dst.meta[QPARAM_KEY] = new_qparam
    logging.getLogger(__name__).debug(
        f"{dst.name} is retargeted to {src.name}'s affine quantparam: "
        f"scale={scale}, zero_point={zero_point}."
    )


@trace_graph_diff_on_pass
class PropagateQParamBackward(PassBase):
    """
    This pass propagates quantization parameters backward.

    BEFORE)

    node -> reshape (with meta[QPARAM_KEY])

    AFTER)

    node (with meta[QPARAM_KEY]) -> reshape (with meta[QPARAM_KEY])
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

        def _propagate_qparam_if_possible(src: torch.fx.Node, dst: torch.fx.Node):
            if QPARAM_KEY not in src.meta:
                return

            # Explicit quantizers are domain boundaries, including unsupported
            # overloads. Only a private affine per-tensor Quantize can retarget.
            schema = getattr(dst.target, "_schema", None)
            if getattr(schema, "name", None) in {
                "circle_custom::quantize_mx",
                "quantized_decomposed::quantize_per_tensor",
                "quantized_decomposed::quantize_per_channel",
            }:
                if dst.target == (
                    torch.ops.quantized_decomposed.quantize_per_tensor.default
                ):
                    _retarget_affine_per_tensor_quantize(src, dst)
                return

            src_qparam = src.meta[QPARAM_KEY]
            if is_mx_dtype(src_qparam.dtype):
                # A producer's established domain belongs to all its users. In
                # particular, never rewrite an explicit Quantize destination
                # using the domain of a later view or requantization boundary.
                if QPARAM_KEY in dst.meta or len(dst.users) != 1:
                    return
                mapped = remap_mx_qparam(src_qparam, src, backward=True)
                assert mapped is not None
                dst.meta[QPARAM_KEY] = mapped
                logger.debug(f"{src.name}'s MX quantparam is remapped to {dst.name}.")
                return

            if (
                QPARAM_KEY in dst.meta
                and src_qparam.dtype != dst.meta[QPARAM_KEY].dtype
            ):
                return

            dst.meta[QPARAM_KEY] = copy.deepcopy(src_qparam)

            logger.debug(f"{src.name}'s quantparam is propagated to {dst.name}.")

        # Do reverse-order traversal for backward propagation
        for node in reversed(graph.nodes):
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.cat.default:
                concat_args = CatArgs(*node.args, **node.kwargs)
                concat_inputs = concat_args.tensors

                node_qparam = node.meta.get(QPARAM_KEY)
                if is_width_direction_concat(node, concat_args) and (
                    node_qparam is None or not is_mx_dtype(node_qparam.dtype)
                ):
                    continue

                for concat_input in concat_inputs:
                    # An established producer domain also belongs to its other
                    # consumers and to graph input/output ABIs. Concat converts
                    # such inputs on its own edges instead of relabeling them.
                    # Explicit Quantize boundaries retain their private-retarget
                    # contract; unannotated inputs can still inherit the domain.
                    schema = getattr(concat_input.target, "_schema", None)
                    if QPARAM_KEY in concat_input.meta and getattr(
                        schema, "name", None
                    ) not in {
                        "circle_custom::quantize_mx",
                        "quantized_decomposed::quantize_per_tensor",
                        "quantized_decomposed::quantize_per_channel",
                    }:
                        continue
                    _propagate_qparam_if_possible(node, concat_input)
            elif node.target == torch.ops.aten.reshape.default:
                args = ReshapeArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(node, args.input)
            elif node.target == torch.ops.aten.permute.default:
                permute_args = PermuteArgs(*node.args, **node.kwargs)
                _propagate_qparam_if_possible(node, permute_args.input)
            # TODO Support more ops.

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        # Run only once.
        return PassResult(False)

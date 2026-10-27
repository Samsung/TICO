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
import operator

from collections import defaultdict
from typing import Any

import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.passes.decompose_fake_quantize_tensor_qparams import get_quant_type
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_mx_dtype, quant_min_max, set_new_meta_val
from tico.utils.validate_args_kwargs import (
    AddTensorArgs,
    BmmArgs,
    CatArgs,
    CircleRMSNormArgs,
    LinearArgs,
    MulTensorArgs,
    PermuteArgs,
    ReluArgs,
    ReshapeArgs,
    RMSNormArgs,
    SigmoidArgs,
    SplitWithSizesArgs,
)


def qparam_dtype(node: torch.fx.Node) -> str:
    assert QPARAM_KEY in node.meta
    return node.meta[QPARAM_KEY].dtype


def convert_qparam(qparam: QuantParam, target_dtype: str) -> QuantParam:
    """Convert a QuantParam from one dtype to another.

    Supports conversions between int16 and unsigned types (uint8, uint4).
    The conversion preserves the representable value range.
    """
    if qparam.dtype == target_dtype:
        return copy.deepcopy(qparam)

    assert qparam.scale is not None and len(qparam.scale) == 1
    assert qparam.zero_point is not None and len(qparam.zero_point) == 1

    src_dtype = qparam.dtype
    src_scale = qparam.scale[0]
    src_zerop = qparam.zero_point[0]
    src_min, src_max = quant_min_max(src_dtype)

    # Compute the real value range from the source qparam
    real_max = src_scale * (src_max - src_zerop)
    real_min = src_scale * (src_min - src_zerop)

    dst_min, dst_max = quant_min_max(target_dtype)
    n_levels = dst_max - dst_min + 1  # 256 for uint8, 16 for uint4, 65536 for int16

    if target_dtype == "int16":
        # Converting to int16: symmetric, zero_point = 0
        abs_max = max(abs(real_max), abs(real_min))
        dst_scale = abs_max / 32767
        dst_zerop = 0
    else:
        # Converting to unsigned: use full range [0, n_levels-1]
        dst_scale = (real_max - real_min) / (n_levels - 1)
        dst_zerop = round(-real_min / dst_scale)

    new_qparam = QuantParam()
    new_qparam.scale = [dst_scale]
    new_qparam.zero_point = [dst_zerop]
    new_qparam.dtype = target_dtype
    return new_qparam


def _insert_quantize_op_before(node, inp, qparam: QuantParam | None = None):
    graph = node.graph
    if qparam is None:
        qparam = node.meta[QPARAM_KEY]
    assert qparam.scale is not None
    assert qparam.zero_point is not None
    scale = qparam.scale[0]
    zerop = qparam.zero_point[0]
    min_, max_ = quant_min_max(qparam.dtype)
    # Sub-byte dtypes (uint4/int4) have no serializable torch dtype; resolve
    # the storage dtype from the (quant_min, quant_max) convention instead.
    dtype = get_quant_type(min_, max_)
    with graph.inserting_before(node):
        q_args = (inp, scale, zerop, min_, max_, dtype)
        quantize = create_node(
            graph,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=q_args,
            origin=node,
        )
        quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)
        set_new_meta_val(quantize)

    node.replace_input_with(inp, quantize)

    return quantize


def _insert_quantize_op_after(node):
    graph = node.graph
    qparam: QuantParam = node.meta[QPARAM_KEY]
    assert qparam.scale is not None
    assert qparam.zero_point is not None
    scale = qparam.scale[0]
    zerop = qparam.zero_point[0]
    min_, max_ = quant_min_max(qparam.dtype)
    # Sub-byte dtypes (uint4/int4) have no serializable torch dtype; resolve
    # the storage dtype from the (quant_min, quant_max) convention instead.
    dtype = get_quant_type(min_, max_)
    with graph.inserting_after(node):
        q_args = (node, scale, zerop, min_, max_, dtype)
        quantize = create_node(
            graph,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)

    return quantize


def _insert_mx_quantize_op_after(node, qparam: QuantParam):
    graph = node.graph
    if qparam is None:
        qparam = node.meta[QPARAM_KEY]
    assert qparam.quantized_dimension is not None
    assert qparam.dtype is not None

    with graph.inserting_after(node):
        q_args = (node, qparam.dtype, qparam.quantized_dimension)
        quantize = create_node(
            graph,
            torch.ops.circle_custom.quantize_mx_decomposed.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)

    return quantize


def _linear_handler(node, logger):
    lin_args = LinearArgs(*node.args, **node.kwargs)
    inp = lin_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])

        # Run the linear on the input's grid and convert the output back to
        # the node's original grid with the inserted quantize.
        # NOTE Running on a narrow grid can degrade accuracy. It is
        # important to mitigate this accuracy drop in backend.
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in ("int16", "uint8", "uint4"):
        quantize = _insert_quantize_op_after(node)

        node.meta[QPARAM_KEY] = copy.deepcopy(
            inp.meta[QPARAM_KEY]
        )  # convert_qparam(node.meta[QPARAM_KEY], "uint8")
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _add_handler(node, logger):
    add_args = AddTensorArgs(*node.args, **node.kwargs)
    x = add_args.input
    y = add_args.other

    if not isinstance(x, torch.fx.Node):
        return
    if not isinstance(y, torch.fx.Node):
        return

    if QPARAM_KEY not in x.meta:
        return
    if QPARAM_KEY not in y.meta:
        return
    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(x) == qparam_dtype(node) and (qparam_dtype(y) == qparam_dtype(node) or (is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(node)) and is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(x)))):
        return

    # if qparam_dtype(x) != qparam_dtype(y):
    #    return

    if (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        and not is_mx_dtype(qparam_dtype(y))
        and qparam_dtype(node) in ("int16", "uint8", "uint4")
    ):
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(x))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (is_mx_dtype(qparam_dtype(x)) or is_mx_dtype(qparam_dtype(y))) and qparam_dtype(
        node
    ) in ("int16", "uint8", "uint4"):
        mx_node = x
        if qparam_dtype(y) != qparam_dtype(x):
            if qparam_dtype(x) in ("int16", "uint8", "uint4"):
                quantize = _insert_mx_quantize_op_after(x, y.meta[QPARAM_KEY])
                mx_node = y
                logger.debug(
                    f"_insert_mx_quantize_op_after.default is inserted after {x.name}."
                )
            if qparam_dtype(y) in ("int16", "uint8", "uint4"):
                quantize = _insert_mx_quantize_op_after(y, x.meta[QPARAM_KEY])
                logger.debug(
                    f"_insert_mx_quantize_op_after.default is inserted after {y.name}."
                )
        quantize = _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(mx_node.meta[QPARAM_KEY])
        logger.debug(
            f"_insert_quantize_op_after.default is inserted after {node.name}."
        )
    elif (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        or qparam_dtype(y) in ("int16", "uint8", "uint4")
    ) and is_mx_dtype(qparam_dtype(node)):
        if qparam_dtype(y) in ("int16", "uint8", "uint4"):
            quantize = _insert_mx_quantize_op_after(y, node.meta[QPARAM_KEY])
            logger.debug(
                f"_insert_mx_quantize_op_after.default is inserted after {y.name}."
            )
        if qparam_dtype(x) in ("int16", "uint8", "uint4"):
            quantize = _insert_mx_quantize_op_after(x, node.meta[QPARAM_KEY])
            logger.debug(
                f"_insert_mx_quantize_op_after.default is inserted after {x.name}."
            )
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: x={qparam_dtype(x)}, y={qparam_dtype(y)}, "
            f"out={qparam_dtype(node)} (node {node.name})"
        )


def _mul_handler(node, logger):
    mul_args = MulTensorArgs(*node.args, **node.kwargs)
    x = mul_args.input
    y = mul_args.other

    if not isinstance(x, torch.fx.Node):
        return
    if not isinstance(y, torch.fx.Node):
        return

    if QPARAM_KEY not in x.meta:
        return
    if QPARAM_KEY not in y.meta:
        return
    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(x) == qparam_dtype(node) and (qparam_dtype(y) == qparam_dtype(node) or (is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(node)) and is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(x)))):
        return
        
    if (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        and not is_mx_dtype(qparam_dtype(y))
        and qparam_dtype(node) in ("int16", "uint8", "uint4")
    ):
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(x))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (is_mx_dtype(qparam_dtype(x)) or is_mx_dtype(qparam_dtype(y))) and qparam_dtype(
        node
    ) in ("int16", "uint8", "uint4"):
        mx_node = x
        if qparam_dtype(y) != qparam_dtype(x):
            if qparam_dtype(x) in ("int16", "uint8", "uint4"):
                quantize = _insert_mx_quantize_op_after(x, y.meta[QPARAM_KEY])
                mx_node = y
                logger.debug(
                    f"_insert_mx_quantize_op_after.default is inserted after {x.name}."
                )
            if qparam_dtype(y) in ("int16", "uint8", "uint4"):
                quantize = _insert_mx_quantize_op_after(y, x.meta[QPARAM_KEY])
                logger.debug(
                    f"_insert_mx_quantize_op_after.default is inserted after {y.name}."
                )

        quantize = _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(mx_node.meta[QPARAM_KEY])
        logger.debug(
            f"_insert_quantize_op_after.default is inserted after {node.name}."
        )
    elif (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        or qparam_dtype(y) in ("int16", "uint8", "uint4")
    ) and is_mx_dtype(qparam_dtype(node)):
        if qparam_dtype(y) in ("int16", "uint8", "uint4"):
            quantize = _insert_mx_quantize_op_after(y, node.meta[QPARAM_KEY])
            logger.debug(
                f"_insert_mx_quantize_op_after.default is inserted after {y.name}."
            )
        if qparam_dtype(x) in ("int16", "uint8", "uint4"):
            quantize = _insert_mx_quantize_op_after(x, node.meta[QPARAM_KEY])
            logger.debug(
                f"_insert_mx_quantize_op_after.default is inserted after {x.name}."
            )
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: x={qparam_dtype(x)}, y={qparam_dtype(y)}, "
            f"out={qparam_dtype(node)} (node {node.name})"
        )


def _cat_handler(node, logger):
    cat_args = CatArgs(*node.args, **node.kwargs)
    tensors = cat_args.tensors

    if any(QPARAM_KEY not in x.meta for x in tensors):
        return

    if QPARAM_KEY not in node.meta:
        return

    assert len(tensors) > 0
    in_dtype = qparam_dtype(tensors[0])
    if in_dtype == qparam_dtype(node):
        return

    if any(qparam_dtype(x) != in_dtype for x in tensors):
        logger.warning(
            f"{node.name}: cat inputs have mixed dtypes "
            f"{[qparam_dtype(x) for x in tensors]} with output "
            f"{qparam_dtype(node)}; skipping dtype alignment."
        )
        return

    if in_dtype in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run cat on the inputs' grid; the inserted quantize converts the
        # output back to the node's original grid.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], in_dtype)
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif is_mx_dtype(in_dtype) and qparam_dtype(node) in ("int16", "uint8", "uint4"):
        for inp in tensors:
            quantize = _insert_quantize_op_before(node, inp)

            quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
            logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {in_dtype} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _bmm_handler(node, logger):
    bmm_args = BmmArgs(*node.args, **node.kwargs)
    x = bmm_args.input
    y = bmm_args.mat2

    if QPARAM_KEY not in x.meta:
        return
    if QPARAM_KEY not in y.meta:
        return
    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(x) == qparam_dtype(node) and qparam_dtype(y) == qparam_dtype(node):
        return

    # If x and node agree but y differs with a non-MX dtype, there is
    # nothing to fix (e.g. Q@K^T where Q and output are int16, K is uint8).
    # MX dtypes on y still need to be handled below.
    if qparam_dtype(x) == qparam_dtype(node) and is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(node)):
        return

    if (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        and not is_mx_dtype(qparam_dtype(y))
        and qparam_dtype(node) in ("int16", "uint8", "uint4")
    ):
        # Non-MX dtype mismatch: run the bmm on the input's grid and insert a
        # quantize that converts the output back to the node's original grid.
        # Covers int16<->uint8, int16<->uint4 and uint8<->uint4.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(x))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (is_mx_dtype(qparam_dtype(x)) or is_mx_dtype(qparam_dtype(y))) and qparam_dtype(
        node
    ) in ("int16", "uint8", "uint4"):
        mx_node = x
        if qparam_dtype(y) != qparam_dtype(x):
            if qparam_dtype(x) in ("int16", "uint8", "uint4"):
                quantize = _insert_mx_quantize_op_after(x, y.meta[QPARAM_KEY])
                mx_node = y
                logger.debug(
                    f"_insert_mx_quantize_op_after.default is inserted after {x.name}."
                )
            if qparam_dtype(y) in ("int16", "uint8", "uint4"):
                quantize = _insert_mx_quantize_op_after(y, x.meta[QPARAM_KEY])
                logger.debug(
                    f"_insert_mx_quantize_op_after.default is inserted after {y.name}."
                )
        quantize = _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(mx_node.meta[QPARAM_KEY])
        logger.debug(
            f"_insert_quantize_op_after.default is inserted after {node.name}."
        )
    elif (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        or qparam_dtype(y) in ("int16", "uint8", "uint4")
    ) and is_mx_dtype(qparam_dtype(node)):
        # Align plain-quantized operands (int16/uint8/uint4) onto the output's
        # MX grid.
        if qparam_dtype(y) in ("int16", "uint8", "uint4"):
            quantize = _insert_mx_quantize_op_after(y, node.meta[QPARAM_KEY])
            logger.debug(
                f"_insert_mx_quantize_op_after.default is inserted after {y.name}."
            )
        if qparam_dtype(x) in ("int16", "uint8", "uint4"):
            quantize = _insert_mx_quantize_op_after(x, node.meta[QPARAM_KEY])
            logger.debug(
                f"_insert_mx_quantize_op_after.default is inserted after {x.name}."
            )
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: x={qparam_dtype(x)}, y={qparam_dtype(y)}, "
            f"out={qparam_dtype(node)} (node {node.name})"
        )


def _permute_handler(node, logger):
    per_args = PermuteArgs(*node.args, **node.kwargs)
    inp = per_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and qparam_dtype(node) in ("uint8", "uint4"):
        # A new Quantize Op (s16 to u8/u4) is inserted before (not after)
        # permute Op to reduce tensor size ealier
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    elif qparam_dtype(inp) in ("uint8", "uint4", "int16") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run permute on the input's grid; the inserted quantize converts
        # the output back to the node's original grid.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Quantize the MX input onto the output's plain grid before permute.
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _reshape_handler(node, logger):
    reshape_args = ReshapeArgs(*node.args, **node.kwargs)
    inp = reshape_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and qparam_dtype(node) in ("uint8", "uint4"):
        # A new Quantize Op (s16 to u8/u4) is inserted before (not after)
        # reshape Op to reduce tensor size ealier
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    elif qparam_dtype(inp) in ("uint8", "uint4", "int16") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run reshape on the input's grid; the inserted quantize converts
        # the output back to the node's original grid.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        quantize = _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(f"quantize_per_tensor.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in ("int16", "uint8", "uint4"):
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _split_handler(node, logger):
    reshape_args = SplitWithSizesArgs(*node.args, **node.kwargs)
    inp = reshape_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(f"quantize_per_tensor.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in ("int16", "uint8", "uint4"):
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    elif qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # split is dtype agnostic: quantize the input onto the output's grid
        # so that no extra conversion is needed afterwards.
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _sigmoid_handler(node, logger):
    sigmoid_args = SigmoidArgs(*node.args, **node.kwargs)
    inp = sigmoid_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(f"quantize_per_tensor.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) == "int16":
        # no way to calibrate for "int16"
        assert False  # please consider changing quantization parameters

        _insert_quantize_op_before(node, inp)

        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run sigmoid on the input's grid; the inserted quantize converts
        # the output back to the node's original grid.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _rmsnorm_handler(node, logger):
    rms_args = RMSNormArgs(*node.args, **node.kwargs)
    inp = rms_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(f"quantize_per_tensor.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) == "int16":
        # no way to calibrate for "int16"
        assert False  # please consider changing quantization parameters
        #   #TODO scale of rmsnorm is (0..1) for every input (we need recalibration here)
        _insert_quantize_op_before(node, inp)

        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run rmsnorm on the input's grid; the inserted quantize converts
        # the output back to the node's original grid.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _circle_rmsnorm_handler(node, logger):
    rms_args = CircleRMSNormArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
    inp = rms_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(f"quantize_per_tensor.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) == "int16":
        inp_args = getattr(inp, "all_input_nodes", None)
        if inp_args is not None and len(inp_args) == 1:
            inp_inp = inp_args[0]
            if QPARAM_KEY not in inp.meta:
                return
            if qparam_dtype(inp_inp) == "int16":
                # TODO copy qparam from single ancestor,
                # so that all ops between ancestor and
                # node does not modify scale (Quantization/Layout/...)
                _insert_quantize_op_before(node, inp, inp_inp.meta[QPARAM_KEY])
                logger.debug(
                    f"quantize_per_tensor.default is inserted after {node.name}."
                )
            else:
                assert False
        else:
            assert False
            # no way to calibrate for "int16"

            # TODO scale of rmsnorm is (0..1) for every input (we need recalibration here)

    elif qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run the rmsnorm on the input's grid and convert the output back to
        # the node's original grid with the inserted quantize.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _get_item_handler(node, logger):
    inp = node.args[0]
    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(
            f"_insert_mx_quantize_op_after.default is inserted after {inp.name}."
        )
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in ("int16", "uint8", "uint4"):
        _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(inp.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # getitem is dtype agnostic: run it on the input's grid and convert
        # the output back with the inserted quantize.
        quantize = _insert_quantize_op_after(node)
        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = copy.deepcopy(inp.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


def _relu_handler(node, logger):
    relu_args = ReluArgs(*node.args, **node.kwargs)
    inp = relu_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) in ("int16", "uint8", "uint4") and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run relu on the input's grid; the inserted quantize converts the
        # output back to the node's original grid.
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], qparam_dtype(inp))
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Quantize the MX input onto the output's plain grid before relu.
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype: From {qparam_dtype(inp)} to {qparam_dtype(node)} "
            f"(node {node.name})"
        )


_op_handler: defaultdict[Any, Any | None] = defaultdict(lambda: None)
_op_handler[torch.ops.aten.linear.default] = _linear_handler
_op_handler[torch.ops.aten.add.Tensor] = _add_handler
_op_handler[torch.ops.aten.mul.Tensor] = _mul_handler
_op_handler[torch.ops.aten.cat.default] = _cat_handler
_op_handler[torch.ops.aten.bmm.default] = _bmm_handler
_op_handler[torch.ops.aten.permute.default] = _permute_handler
_op_handler[torch.ops.aten.reshape.default] = _reshape_handler
_op_handler[torch.ops.aten.relu.default] = _relu_handler
_op_handler[torch.ops.aten.split_with_sizes.default] = _split_handler
_op_handler[torch.ops.aten.sigmoid.default] = _sigmoid_handler
_op_handler[torch.ops.aten.rms_norm.default] = _rmsnorm_handler
_op_handler[operator.getitem] = _get_item_handler


@trace_graph_diff_on_pass
class InsertQuantizeOnDtypeMismatch(PassBase):
    """
    Insert quantize Op in the operators where circle's type inference is violated.
    Example. FullyConnected
    [BEFORE]
      Op (uint8) -  aten.linear.default (int16)
    [AFTER]
      Op (uint8) -  aten.linear.default (uint8) - quantized_decomposed.quantize_per_tensor.default (int16)
    Why is this pass necessary?
    - For some operators, circle's type inference pass overwrites the input's dtype to
    the output's dtype. For the above example, fully-connected layer (aten.linear.default)'s
    output dtype (int16) is updated to the input dtype (uint8), which breaks the semantics.
    This problem can occur in the tools (ex: circle2circle) that automatically apply type inference.
    - To resolve the issue, we insert quantize operators not to violate circle's type inference logic.
    - NOTE For some cases, Quantize Op is inserted before the operators.

    Let's assume Reshape Op's input is int16 and output is uint8. There are two possible places to insert
    Quantize Op.

    1. Insert Quantize before Reshape.

    ```
    Predecessor (int16)-> Quantize (uint8) -> Reshape (uint8) -> ...
    ```

    2. Insert Quantize after Reshape.

    ```
    Predecessor (int16)-> Reshape (int16) -> Quantize (uint8) -> ...
    ```

    Comparing 1) and 2), the difference is that Reshape operation is conducted in uint8 or int16.
    We go with 1), which does Reshape in uint8, for faster execution. Note that Reshape Op does not
    change the value, so its dytpe does not affect accuracy.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        # hack to remove dependecy on initialiazation order
        _op_handler[torch.ops.circle_custom.rms_norm.default] = _circle_rmsnorm_handler

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph
        for _ in range(5):  # TODO (wihtout additional passes?)
            for node in graph.nodes:
                if node.op != "call_function":
                    continue

                handler = _op_handler[node.target]
                if handler is not None:
                    handler(node, logger)

            graph.eliminate_dead_code()
            graph.lint()
            graph_module.recompile()

        # Run only once.
        return PassResult(False)


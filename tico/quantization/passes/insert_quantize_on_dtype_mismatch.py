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

from tico.quantization.passes.preserve_embedding_quantization import FLOAT_EMBEDDING
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.mx.dtypes import elem_format_from_mx_dtype, is_mx_dtype
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import quant_min_max
from tico.utils.validate_args_kwargs import (
    AddTensorArgs,
    BmmArgs,
    CatArgs,
    LinearArgs,
    MulTensorArgs,
    PermuteArgs,
    ReluArgs,
    ReshapeArgs,
    RMSNormArgs,
    SigmoidArgs,
    SplitWithSizesArgs,
    SubTensorArgs,
)


def qparam_dtype(node: torch.fx.Node) -> str:
    assert QPARAM_KEY in node.meta
    return node.meta[QPARAM_KEY].dtype


# Bit width of the supported affine storage dtypes.
_DTYPE_WIDTH = {"uint4": 4, "uint8": 8, "int16": 16}


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


# Convert i16 qparam to u8 qparam
# scale and zero_point are inferred from i16 qparam
def _i16_to_u8(qparam: QuantParam) -> QuantParam:
    # Assume per-tensor quantization
    assert qparam.scale is not None and len(qparam.scale) == 1
    assert qparam.dtype == "int16"

    s16_scale = qparam.scale[0]
    max_ = s16_scale * 32767  # numeric_limits<int16>
    min_ = -max_

    u8_scale = (max_ - min_) / 255
    u8_zerop = round(-min_ / u8_scale)

    new_qparam = QuantParam()
    new_qparam.scale = [u8_scale]
    new_qparam.zero_point = [u8_zerop]
    new_qparam.dtype = "uint8"

    return new_qparam


# Convert u8 qparam to i16 qparam
# scale is inferred from u8 qparam
def _u8_to_i16(qparam: QuantParam) -> QuantParam:
    # Assume per-tensor quantization
    assert qparam.scale is not None and len(qparam.scale) == 1
    assert qparam.zero_point is not None and len(qparam.zero_point) == 1
    assert qparam.dtype == "uint8"

    u8_scale = qparam.scale[0]
    u8_zerop = qparam.zero_point[0]
    max_ = u8_scale * (255 - u8_zerop)
    min_ = u8_scale * (-u8_zerop)

    abs_max = max(abs(max_), abs(min_))
    s16_scale = abs_max / 32767
    s16_zerop = 0

    new_qparam = QuantParam()
    new_qparam.scale = [s16_scale]
    new_qparam.zero_point = [s16_zerop]
    new_qparam.dtype = "int16"

    return new_qparam


def _insert_quantize_op_before(node, inp, qparam: QuantParam | None = None):
    graph = node.graph
    if qparam is None:
        qparam = node.meta[QPARAM_KEY]
    assert qparam.scale is not None and len(qparam.scale) == 1
    assert qparam.zero_point is not None and len(qparam.zero_point) == 1
    assert qparam.quantized_dimension is None
    scale = qparam.scale[0]
    zerop = qparam.zero_point[0]
    min_, max_ = quant_min_max(qparam.dtype)
    # torch has no uint4 storage type; keep uint8 storage like the quant op.
    dtype = torch.uint8 if qparam.dtype == "uint4" else getattr(torch, qparam.dtype)

    with graph.inserting_before(node):
        q_args = (inp, scale, zerop, min_, max_, dtype)
        quantize = create_node(
            graph,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=q_args,
            origin=node,
        )
        quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)
        # Folded quantized edges may already have integer value metadata, but
        # PyTorch's quantize_per_tensor meta kernel accepts only float32 input.
        # Circle requantization needs just the source shape and destination dtype.
        quantize.meta["val"] = torch.empty_like(inp.meta["val"], dtype=dtype)

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
    dtype = getattr(torch, qparam.dtype)
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
    # Mirror ``_insert_quantize_op_before``: the Quantize output carries the
    # destination storage dtype, whatever the producer's FX value dtype is.
    quantize.meta["val"] = torch.empty_like(node.meta["val"], dtype=dtype)

    return quantize


def _insert_mx_quantize_op_after(node, qparam: QuantParam):
    """Insert an MX quantize op converting ``node``'s affine output to the MX
    domain described by ``qparam`` (block element dtype + quantized dimension).

    An affine ``quantize_per_tensor`` cannot express a conversion into an MX
    domain, so MX-boundary legalization uses ``circle_custom::quantize_mx``.
    """
    graph = node.graph
    assert is_mx_dtype(qparam.dtype)
    assert qparam.quantized_dimension is not None
    elem_format = elem_format_from_mx_dtype(qparam.dtype)

    with graph.inserting_after(node):
        q_args = (node, elem_format, qparam.quantized_dimension)
        quantize = create_node(
            graph,
            torch.ops.circle_custom.quantize_mx.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)
    # The MX quantize fake kernel returns its input unchanged.
    quantize.meta["val"] = node.meta["val"]

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

    if qparam_dtype(inp) == "uint8" and qparam_dtype(node) == "int16":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])

        # Update node's qparam from i16 to u8
        # NOTE This would severely degrade accuracy. It is
        # important to mitigate this accuracy drop in backend.
        node.meta[QPARAM_KEY] = _i16_to_u8(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Run the linear in the MX domain and requantize the output back to
        # the annotated affine grid.
        quantize = _insert_quantize_op_after(node)

        node.meta[QPARAM_KEY] = copy.deepcopy(inp.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(inp) in ("int16", "uint8", "uint4") and is_mx_dtype(
        qparam_dtype(node)
    ):
        # Convert the affine activation to the annotated MX domain.
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])
        logger.debug(f"quantize_mx.default is inserted after {inp.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype mismatch at {node.name}: "
            f"in={qparam_dtype(inp)}, out={qparam_dtype(node)}"
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

    if qparam_dtype(x) == qparam_dtype(y) == qparam_dtype(node):
        return

    if not (
        is_mx_dtype(qparam_dtype(x))
        or is_mx_dtype(qparam_dtype(y))
        or is_mx_dtype(qparam_dtype(node))
    ):
        # Legacy affine behavior.
        if qparam_dtype(x) == qparam_dtype(node):
            return

        if qparam_dtype(x) != qparam_dtype(y):
            return

        if qparam_dtype(x) == "int16" and qparam_dtype(node) == "uint8":
            quantize = _insert_quantize_op_after(node)

            quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
            node.meta[QPARAM_KEY] = _u8_to_i16(node.meta[QPARAM_KEY])
            logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
            return

        raise NotYetSupportedError(
            f"Unsupported dtype mismatch at {node.name}: "
            f"x={qparam_dtype(x)}, y={qparam_dtype(y)}, out={qparam_dtype(node)}"
        )

    # Uniform MX family with the output on x's grid is directly supported.
    if (
        is_mx_dtype(qparam_dtype(x))
        and is_mx_dtype(qparam_dtype(y))
        and is_mx_dtype(qparam_dtype(node))
        and qparam_dtype(x) == qparam_dtype(node)
    ):
        return

    if (is_mx_dtype(qparam_dtype(x)) or is_mx_dtype(qparam_dtype(y))) and (
        qparam_dtype(node) in ("int16", "uint8", "uint4")
    ):
        # One input is MX and the annotated output is affine: bring the affine
        # input into the MX domain, run the add in MX and requantize the output
        # back to the affine grid.
        mx_node = x
        if qparam_dtype(y) != qparam_dtype(x):
            if qparam_dtype(x) in ("int16", "uint8", "uint4"):
                _insert_mx_quantize_op_after(x, y.meta[QPARAM_KEY])
                mx_node = y
                logger.debug(f"quantize_mx.default is inserted after {x.name}.")
            if qparam_dtype(y) in ("int16", "uint8", "uint4"):
                _insert_mx_quantize_op_after(y, x.meta[QPARAM_KEY])
                logger.debug(f"quantize_mx.default is inserted after {y.name}.")

        _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(mx_node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (not is_mx_dtype(qparam_dtype(x)) and not is_mx_dtype(qparam_dtype(y))) and (
        is_mx_dtype(qparam_dtype(node))
    ):
        # Affine inputs with an annotated MX output: convert both inputs.
        _insert_mx_quantize_op_after(y, node.meta[QPARAM_KEY])
        logger.debug(f"quantize_mx.default is inserted after {y.name}.")
        _insert_mx_quantize_op_after(x, node.meta[QPARAM_KEY])
        logger.debug(f"quantize_mx.default is inserted after {x.name}.")
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

    # The op is directly executable when the output stays on x's grid and y is
    # either on the same grid or at least in the same quantization family
    # (the backend allows cross-type MUL inside one family).
    if qparam_dtype(x) == qparam_dtype(node) and (
        qparam_dtype(y) == qparam_dtype(node)
        or (
            is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(node))
            and is_mx_dtype(qparam_dtype(y)) == is_mx_dtype(qparam_dtype(x))
        )
    ):
        return

    if qparam_dtype(x) == "int16" and qparam_dtype(node) == "uint8":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _u8_to_i16(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (is_mx_dtype(qparam_dtype(x)) or is_mx_dtype(qparam_dtype(y))) and (
        qparam_dtype(node) in ("int16", "uint8", "uint4")
    ):
        # One input is MX and the annotated output is affine: bring the affine
        # input into the MX domain, run the mul in MX and requantize the output
        # back to the affine grid.
        mx_node = x
        if qparam_dtype(y) != qparam_dtype(x):
            if qparam_dtype(x) in ("int16", "uint8", "uint4"):
                _insert_mx_quantize_op_after(x, y.meta[QPARAM_KEY])
                mx_node = y
                logger.debug(f"quantize_mx.default is inserted after {x.name}.")
            if qparam_dtype(y) in ("int16", "uint8", "uint4"):
                _insert_mx_quantize_op_after(y, x.meta[QPARAM_KEY])
                logger.debug(f"quantize_mx.default is inserted after {y.name}.")

        _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(mx_node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        or qparam_dtype(y) in ("int16", "uint8", "uint4")
    ) and is_mx_dtype(qparam_dtype(node)):
        # Affine input(s) with an annotated MX output: convert the affine
        # operand(s) to the MX domain.
        if qparam_dtype(y) in ("int16", "uint8", "uint4"):
            _insert_mx_quantize_op_after(y, node.meta[QPARAM_KEY])
            logger.debug(f"quantize_mx.default is inserted after {y.name}.")
        if qparam_dtype(x) in ("int16", "uint8", "uint4"):
            _insert_mx_quantize_op_after(x, node.meta[QPARAM_KEY])
            logger.debug(f"quantize_mx.default is inserted after {x.name}.")
    elif (
        qparam_dtype(x) == qparam_dtype(y)
        and qparam_dtype(x) in ("uint8", "int16")
        and qparam_dtype(node) in ("uint8", "int16")
    ):
        # Same-family mismatch in the opposite direction: compute in the
        # inputs' domain and requantize the output to the annotated domain.
        # Example: x=uint8, y=uint8, out=int16
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = copy.deepcopy(x.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype mismatch at {node.name}: "
            f"x={qparam_dtype(x)}, y={qparam_dtype(y)}, out={qparam_dtype(node)}"
        )


def _sub_handler(node, logger):
    sub_args = SubTensorArgs(*node.args, **node.kwargs)
    x = sub_args.input
    y = sub_args.other

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

    x_dtype = qparam_dtype(x)
    y_dtype = qparam_dtype(y)
    out_dtype = qparam_dtype(node)

    if not (
        x_dtype in _DTYPE_WIDTH
        and y_dtype in _DTYPE_WIDTH
        and out_dtype in _DTYPE_WIDTH
    ):
        raise NotYetSupportedError(
            f"Unsupported dtype: x={x_dtype}, y={y_dtype}, "
            f"out={out_dtype} (node {node.name})"
        )

    if _DTYPE_WIDTH[y_dtype] <= _DTYPE_WIDTH[x_dtype]:
        if x_dtype == out_dtype:
            return

        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], x_dtype)
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
        return

    if _DTYPE_WIDTH[out_dtype] >= _DTYPE_WIDTH[y_dtype]:
        op_dtype = out_dtype
    else:
        op_dtype = y_dtype

    if out_dtype != op_dtype:
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = convert_qparam(node.meta[QPARAM_KEY], op_dtype)
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")

    _insert_quantize_op_before(node, x, copy.deepcopy(node.meta[QPARAM_KEY]))
    logger.debug(
        f"quantize_per_tensor.default is inserted before {node.name} for {x.name}."
    )


def _same_domain(lhs: QuantParam, rhs: QuantParam) -> bool:
    """Return whether two qparams describe one integer grid.

    The grid is defined by the storage dtype, scale, zero point and the
    granularity axis. Observer ``min``/``max`` statistics are metadata and must
    not trigger a conversion. Scales compare exactly: an approximate match
    would silently skip a conversion the backend requires.
    """
    return (
        lhs.dtype == rhs.dtype
        and lhs.scale == rhs.scale
        and lhs.zero_point == rhs.zero_point
        and lhs.quantized_dimension == rhs.quantized_dimension
    )


def _cat_handler(node, logger):
    """Make every CONCATENATION input and output share one quantization domain.

    Circle CONCATENATION copies integer codes, so a backend expects identical
    qparams on all of its inputs and its output. Any domain change is made
    explicit with a per-tensor Quantize, regardless of the concat axis or the
    module that produced the concat:

    * identical input and output domains: nothing is inserted;
    * every input shares domain ``A`` and the output requests ``B``: the
      concat runs in ``A`` and one Quantize converts its output to ``B``;
    * distinct input domains: each input connection whose domain differs from
      the output domain gets a Quantize to that domain. Producers keep their
      own domain, including graph inputs, graph outputs and tensors shared
      with other consumers, and a repeated operand reuses one conversion.

    An input that is itself a requantization keeps that earlier rounding step;
    the new Quantize is chained after it instead of retargeting it.
    """
    cat_args = CatArgs(*node.args, **node.kwargs)
    tensors = cat_args.tensors

    # A CPU float layout may leave the embedding island at an integer concat.
    # Convert that edge rather than assigning an integer grid to the float value.
    if QPARAM_KEY in node.meta:
        for inp in dict.fromkeys(tensors):
            if inp.meta.get(FLOAT_EMBEDDING) and QPARAM_KEY not in inp.meta:
                _insert_quantize_op_before(node, inp)
        tensors = CatArgs(*node.args, **node.kwargs).tensors

    if any(QPARAM_KEY not in x.meta for x in tensors):
        return

    if QPARAM_KEY not in node.meta:
        return

    assert len(tensors) > 0
    destination: QuantParam = node.meta[QPARAM_KEY]
    sources = [x.meta[QPARAM_KEY] for x in tensors]
    if all(_same_domain(source, destination) for source in sources):
        return

    # MX legalization. An affine requantize cannot enter or leave the MX
    # domain, so MX boundaries are converted with the MX-aware helpers before
    # falling back to the affine logic (or the silent passthrough) below.
    if is_mx_dtype(destination.dtype) and all(
        source.dtype in ("int16", "uint8", "uint4") for source in sources
    ):
        # Affine inputs with an MX output: convert every input to the MX
        # domain. Distinct inputs keep their own conversion.
        for inp in dict.fromkeys(tensors):
            _insert_mx_quantize_op_after(inp, destination)
            logger.debug(f"quantize_mx.default is inserted after {inp.name}.")
        return

    if destination.dtype in ("int16", "uint8", "uint4") and any(
        is_mx_dtype(source.dtype) for source in sources
    ):
        # MX inputs with an affine output: requantize each MX input onto the
        # output grid, then reconcile the remaining affine domains below.
        for inp in dict.fromkeys(tensors):
            if is_mx_dtype(inp.meta[QPARAM_KEY].dtype):
                _insert_quantize_op_before(node, inp)
                logger.debug(
                    f"quantize_per_tensor.default is inserted before {node.name}."
                )
        tensors = CatArgs(*node.args, **node.kwargs).tensors
        sources = [x.meta[QPARAM_KEY] for x in tensors]
        if all(_same_domain(source, destination) for source in sources):
            return

    qparams = sources + [destination]
    # Preserve the existing treatment of incompletely annotated concats,
    # including MX domains that carry no affine scale.
    if any(
        not qparam.dtype or qparam.scale is None or qparam.zero_point is None
        for qparam in qparams
    ):
        return

    if any(
        qparam.dtype not in ("uint8", "int16")
        or qparam.quantized_dimension is not None
        or len(qparam.scale) != 1
        or len(qparam.zero_point) != 1
        for qparam in qparams
    ):
        raise NotYetSupportedError(
            f"Unsupported distinct-domain concat {node.name}: "
            f"inputs={[qparam_dtype(x) for x in tensors]}, "
            f"output={destination.dtype}. "
            "Only per-tensor affine uint8/int16 qparams are supported."
        )

    first = sources[0]
    if all(_same_domain(source, first) for source in sources[1:]):
        # Every input already shares one grid. Concatenate in that grid and
        # requantize the whole result once at the output boundary.
        quantize = _insert_quantize_op_after(node)
        quantize.meta[QPARAM_KEY] = copy.deepcopy(destination)
        node.meta[QPARAM_KEY] = copy.deepcopy(first)
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
        return

    # Normalize to the intended output grid. Convert only this consumer's
    # connections: producers may be graph inputs, graph outputs, or shared
    # with other operators. ``replace_input_with`` rewires every occurrence of
    # a repeated operand to the same Quantize.
    for inp in dict.fromkeys(tensors):
        if not _same_domain(inp.meta[QPARAM_KEY], destination):
            _insert_quantize_op_before(node, inp)
            logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")


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

    # The op is directly executable when the output stays on x's grid and the
    # second operand belongs to the same quantization family.
    if qparam_dtype(x) == qparam_dtype(node) and is_mx_dtype(
        qparam_dtype(y)
    ) == is_mx_dtype(qparam_dtype(node)):
        return

    if qparam_dtype(x) == "int16" and qparam_dtype(node) == "uint8":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _u8_to_i16(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(x) == "uint8" and qparam_dtype(node) == "int16":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _i16_to_u8(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (is_mx_dtype(qparam_dtype(x)) or is_mx_dtype(qparam_dtype(y))) and (
        qparam_dtype(node) in ("int16", "uint8", "uint4")
    ):
        # One input is MX and the annotated output is affine: bring the affine
        # input into the MX domain, run the bmm in MX and requantize the output
        # back to the affine grid.
        mx_node = x
        if qparam_dtype(y) != qparam_dtype(x):
            if qparam_dtype(x) in ("int16", "uint8", "uint4"):
                _insert_mx_quantize_op_after(x, y.meta[QPARAM_KEY])
                mx_node = y
                logger.debug(f"quantize_mx.default is inserted after {x.name}.")
            if qparam_dtype(y) in ("int16", "uint8", "uint4"):
                _insert_mx_quantize_op_after(y, x.meta[QPARAM_KEY])
                logger.debug(f"quantize_mx.default is inserted after {y.name}.")

        _insert_quantize_op_after(node)
        node.meta[QPARAM_KEY] = copy.deepcopy(mx_node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif (
        qparam_dtype(x) in ("int16", "uint8", "uint4")
        and qparam_dtype(y) in ("int16", "uint8", "uint4")
        and is_mx_dtype(qparam_dtype(node))
    ):
        # Affine inputs with an annotated MX output: convert both inputs.
        _insert_mx_quantize_op_after(y, node.meta[QPARAM_KEY])
        logger.debug(f"quantize_mx.default is inserted after {y.name}.")
        _insert_mx_quantize_op_after(x, node.meta[QPARAM_KEY])
        logger.debug(f"quantize_mx.default is inserted after {x.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype mismatch at {node.name}: "
            f"x={qparam_dtype(x)}, y={qparam_dtype(y)}, out={qparam_dtype(node)}"
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

    if qparam_dtype(inp) == "int16" and qparam_dtype(node) == "uint8":
        # A new Quantize Op (s16 to u8) is inserted before (not after)
        # permute Op to reduce tensor size ealier
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    elif qparam_dtype(inp) == "uint8" and qparam_dtype(node) == "int16":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _i16_to_u8(node.meta[QPARAM_KEY])
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
            f"Unsupported dtype mismatch at {node.name}: "
            f"in={qparam_dtype(inp)}, out={qparam_dtype(node)}"
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

    if qparam_dtype(inp) == "int16" and qparam_dtype(node) == "uint8":
        # A new Quantize Op (s16 to u8) is inserted before (not after)
        # reshape Op to reduce tensor size ealier
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    elif qparam_dtype(inp) == "uint8" and qparam_dtype(node) == "int16":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _i16_to_u8(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Quantize the MX input onto the output's plain grid before reshape.
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
    else:
        raise NotYetSupportedError(
            f"Unsupported dtype mismatch at {node.name}: "
            f"in={qparam_dtype(inp)}, out={qparam_dtype(node)}"
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

    if qparam_dtype(inp) == "int16" and qparam_dtype(node) == "uint8":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _u8_to_i16(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted after {node.name}.")
    elif qparam_dtype(inp) == "uint8" and qparam_dtype(node) == "int16":
        quantize = _insert_quantize_op_after(node)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        node.meta[QPARAM_KEY] = _i16_to_u8(node.meta[QPARAM_KEY])
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
            f"Unsupported dtype mismatch at {node.name}: "
            f"in={qparam_dtype(inp)}, out={qparam_dtype(node)}"
        )


def _split_handler(node, logger):
    split_args = SplitWithSizesArgs(*node.args, **node.kwargs)
    inp = split_args.input

    if QPARAM_KEY not in inp.meta:
        return

    if QPARAM_KEY not in node.meta:
        return

    if qparam_dtype(inp) == qparam_dtype(node):
        return

    if qparam_dtype(inp) == "int16" and is_mx_dtype(qparam_dtype(node)):
        _insert_mx_quantize_op_after(inp, node.meta[QPARAM_KEY])

        logger.debug(f"quantize_mx.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
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

        logger.debug(f"quantize_mx.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Quantize the MX input onto the output's plain grid before sigmoid.
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
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

        logger.debug(f"quantize_mx.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
        # Quantize the MX input onto the output's plain grid before rms_norm.
        quantize = _insert_quantize_op_before(node, inp)

        quantize.meta[QPARAM_KEY] = copy.deepcopy(node.meta[QPARAM_KEY])
        logger.debug(f"quantize_per_tensor.default is inserted before {node.name}.")
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

        logger.debug(f"quantize_mx.default is inserted after {inp.name}.")
    elif is_mx_dtype(qparam_dtype(inp)) and qparam_dtype(node) in (
        "int16",
        "uint8",
        "uint4",
    ):
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


_op_handler: defaultdict[Any, Any | None] = defaultdict(lambda: None)
_op_handler[torch.ops.aten.linear.default] = _linear_handler
_op_handler[torch.ops.aten.add.Tensor] = _add_handler
_op_handler[torch.ops.aten.sub.Tensor] = _sub_handler
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

    For ``aten.cat`` the same mechanism legalizes every quantization-domain
    mismatch, not only dtype changes: a Circle CONCATENATION must use one
    scale, zero point and dtype on all of its inputs and its output. See
    ``_cat_handler`` for the input-connection and output-boundary rules.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph: torch.fx.Graph = graph_module.graph

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

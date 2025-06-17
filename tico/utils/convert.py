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

import operator
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.export import export, ExportedProgram

from tico.config import CompileConfigBase, get_default_config
from tico.experimental.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.experimental.quantization.passes.insert_quantize_on_dtype_mismatch import (
    InsertQuantizeOnDtypeMismatch,
)
from tico.experimental.quantization.passes.propagate_qparam_backward import (
    PropagateQParamBackward,
)
from tico.experimental.quantization.passes.propagate_qparam_forward import (
    PropagateQParamForward,
)
from tico.experimental.quantization.passes.quantize_bias import QuantizeBias
from tico.experimental.quantization.passes.remove_weight_dequant_op import (
    RemoveWeightDequantOp,
)
from tico.passes.cast_aten_where_arg_type import CastATenWhereArgType
from tico.passes.cast_mixed_type_args import CastMixedTypeArgs
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.convert_conv1d_to_conv2d import ConvertConv1dToConv2d
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.passes.convert_repeat_to_expand_copy import ConvertRepeatToExpandCopy
from tico.passes.convert_to_relu6 import ConvertToReLU6
from tico.passes.decompose_addmm import DecomposeAddmm
from tico.passes.decompose_batch_norm import DecomposeBatchNorm
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
)
from tico.passes.decompose_group_norm import DecomposeGroupNorm
from tico.passes.decompose_grouped_conv2d import DecomposeGroupedConv2d
from tico.passes.decompose_slice_scatter import DecomposeSliceScatter
from tico.passes.extract_dtype_kwargs import ExtractDtypeKwargsPass
from tico.passes.fill_meta_val import FillMetaVal
from tico.passes.fuse_leading_unsqueeze_reshape import FuseLeadingUnsqueezeReshape
from tico.passes.fuse_redundant_reshape_to_mean import FuseRedundantReshapeToMean
from tico.passes.legalize_causal_mask_value import LegalizeCausalMaskValue
from tico.passes.legalize_predefined_layout_operators import (
    LegalizePreDefinedLayoutOperators,
)
from tico.passes.lower_pow2_to_mul import LowerPow2ToMul
from tico.passes.lower_to_resize_nearest_neighbor import LowerToResizeNearestNeighbor
from tico.passes.lower_to_slice import passes as LowerToSlicePasses
from tico.passes.merge_consecutive_cat import MergeConsecutiveCat
from tico.passes.remove_nop import RemoveNop
from tico.passes.remove_redundant_assert_nodes import RemoveRedundantAssertionNodes
from tico.passes.remove_redundant_expand import RemoveRedundantExpand
from tico.passes.remove_redundant_permute import passes as RemoveRedundantPermutePasses
from tico.passes.remove_redundant_reshape import passes as RemoveRedundantViewPasses
from tico.passes.remove_redundant_slice import RemoveRedundantSlice
from tico.passes.remove_redundant_to_copy import RemoveRedundantToCopy
from tico.passes.restore_linear import RestoreLinear
from tico.passes.segment_index_select import SegmentIndexSelectConst
from tico.serialize.circle_serializer import build_circle
from tico.serialize.operators.node_visitor import get_support_targets
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.model import CircleModel
from tico.utils.passes import PassManager
from tico.utils.trace_decorators import (
    trace_const_diff_on_func,
    trace_graph_diff_on_func,
)
from tico.utils.utils import has_quantization_ops, SuppressWarning


@trace_const_diff_on_func
@trace_graph_diff_on_func
def traced_run_decompositions(exported_program: ExportedProgram):
    """
    Let's preserve convolution operators.
    `run_decompositions()` converts all Conv-related Ops to generic `aten.convolution`.
    But, we should re-convert them to specific circle ops such as CircleConv2D, TransposeConv, etc.
    Therefore, we do not decompose Conv-related Ops and convert them directly to circle ops.
    """

    def run_decompositions_v25(ep: ExportedProgram):
        _preserve_ops = (
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv1d.padding,
            torch.ops.aten.instance_norm.default,
            torch.ops.aten._safe_softmax.default,
            torch.ops.aten.relu6.default,  # Do not decompose to hardtanh
            torch.ops.aten.linear.default,
        )
        ep = ep.run_decompositions(_preserve_ops=_preserve_ops)

        return ep

    def run_decompositions(ep: ExportedProgram):
        _decomp_table = torch.export.default_decompositions()  # type: ignore[attr-defined]
        _preserve_ops = (
            torch.ops.aten.conv2d.default,
            torch.ops.aten.conv2d.padding,
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv1d.padding,
            torch.ops.aten.instance_norm.default,
            torch.ops.aten._safe_softmax.default,
            torch.ops.aten.relu6.default,  # Do not decompose to hardtanh
            torch.ops.aten.prelu.default,
            torch.ops.aten.linear.default,
        )
        for op in _preserve_ops:
            if op in _decomp_table:
                del _decomp_table[op]

        ep = ep.run_decompositions(decomp_table=_decomp_table)
        return ep

    if torch.__version__.startswith("2.5"):
        return run_decompositions_v25(exported_program)
    elif (
        torch.__version__.startswith("2.6")
        or torch.__version__.startswith("2.7")
        or torch.__version__.startswith("2.8")
    ):
        return run_decompositions(exported_program)
    else:
        raise RuntimeError(f"Unsupported PyTorch version: {torch.__version__}")


def check_unsupported_target(exported_program: ExportedProgram):
    logger = logging.getLogger(__name__)

    supported_target = list(get_support_targets())
    # Ignore `getitem` since it is no-op for multiple outputs.
    supported_target.append(operator.getitem)
    unsupported = []
    for n in exported_program.graph.nodes:
        if n.op != "call_function":
            continue
        if not n.target in supported_target:
            unsupported.append(n)

    if unsupported:
        for node in unsupported:
            logger.error(
                f"NOT SUPPORTED OPERATOR\n\t(op) {node.target.__name__}\n\t(trace) {node.meta.get('stack_trace')}"
            )
        raise NotYetSupportedError("NOT SUPPORTED OPERATOR IN GRAPH MODULE")


def convert_exported_module_to_circle(
    exported_program: ExportedProgram,
    config: CompileConfigBase = get_default_config(),
) -> bytes:
    logger = logging.getLogger(__name__)
    logger.debug("Input ExportedProgram (must be core aten)")
    logger.debug(exported_program)

    # PRE-EDGE PASSES
    #
    # Here are the passes that run before to_edge() conversion.
    # Let's decompose nodes that are not Aten Canonical, which can't be converted to the edge IR.
    decompose_quantize_op = PassManager(
        passes=[
            DecomposeFakeQuantize(),
            DecomposeFakeQuantizeTensorQParams(),
        ]
    )
    decompose_quantize_op.run(exported_program)

    # This pass should be run before 'RestoreLinear' and after 'decompose_quantize_op'.
    # TODO run pass regardless of the orders.
    with SuppressWarning(UserWarning, ".*quantize_per_tensor"), SuppressWarning(
        UserWarning,
        ".*TF32 acceleration on top of oneDNN is available for Intel GPUs.*",
    ):
        # Warning details:
        #   ...site-packages/torch/_subclasses/functional_tensor.py:364
        #   UserWarning: At pre-dispatch tracing, we assume that any custom op marked with
        #     CompositeImplicitAutograd and have functional schema are safe to not decompose.
        exported_program = traced_run_decompositions(exported_program)

    # TODO Distinguish legalize and optimize
    circle_legalize = PassManager(
        passes=[
            FillMetaVal(),
            ExtractDtypeKwargsPass(),
            RemoveNop(),
            ConvertLayoutOpToReshape(),
            RestoreLinear(),
            ConvertToReLU6(),
            DecomposeAddmm(),
            DecomposeSliceScatter(),
            DecomposeGroupNorm(),
            DecomposeBatchNorm(),
            DecomposeGroupedConv2d(),
            CastATenWhereArgType(),
            ConvertRepeatToExpandCopy(),
            *RemoveRedundantPermutePasses(),
            RemoveRedundantAssertionNodes(),
            RemoveRedundantExpand(),
            RemoveRedundantSlice(),
            FuseRedundantReshapeToMean(),
            *RemoveRedundantViewPasses(),
            RemoveRedundantToCopy(),
            MergeConsecutiveCat(),
            CastMixedTypeArgs(preserve_ep_invariant=True),
            ConstPropPass(),
            SegmentIndexSelectConst(),
            LegalizeCausalMaskValue(enabled=config.get("legalize_causal_mask_value")),
            LowerToResizeNearestNeighbor(),
            LegalizePreDefinedLayoutOperators(),
            LowerPow2ToMul(),
            ConvertConv1dToConv2d(),
            *LowerToSlicePasses(),
            FuseLeadingUnsqueezeReshape(),
        ]
    )
    circle_legalize.run(exported_program)

    # After this stage, ExportedProgram invariant is broken, i.e.,
    # graph can have a constant torch.tensor not lifted to a placeholder
    circle_legalize = PassManager(
        passes=[
            FillMetaVal(),
            CastMixedTypeArgs(preserve_ep_invariant=False),
        ]
    )
    circle_legalize.run(exported_program)

    # TODO Give an option to enable quantiztion to user
    enable_quantization = has_quantization_ops(exported_program.graph)
    if enable_quantization:
        quantize_graph = PassManager(
            passes=[
                FoldQuantOps(),
                RemoveWeightDequantOp(),
                PropagateQParamForward(),
                PropagateQParamBackward(),
                QuantizeBias(),
                InsertQuantizeOnDtypeMismatch(),
            ]
        )
        quantize_graph.run(exported_program)

    check_unsupported_target(exported_program)
    circle_program = build_circle(exported_program)

    return circle_program


def convert(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    config: CompileConfigBase = get_default_config(),
) -> CircleModel:
    with torch.no_grad():
        exported_program = export(mod, args, kwargs, strict=strict)

    circle_binary = convert_exported_module_to_circle(exported_program, config=config)

    return CircleModel(circle_binary)


def convert_from_exported_program(
    exported_program: ExportedProgram,
    config: CompileConfigBase = get_default_config(),
) -> CircleModel:
    circle_binary = convert_exported_module_to_circle(exported_program, config=config)

    return CircleModel(circle_binary)


def convert_from_pt2(
    pt2_path: str | os.PathLike, config: CompileConfigBase = get_default_config()
) -> CircleModel:
    exported_program = torch.export.load(pt2_path)
    circle_binary = convert_exported_module_to_circle(exported_program, config=config)

    return CircleModel(circle_binary)

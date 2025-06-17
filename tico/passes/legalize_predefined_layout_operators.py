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

from types import NoneType
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_graph import extract_shape
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import (
    AvgPool2dArgs,
    Conv2DArgs,
    DequantizePerChannelArgs,
    DequantizePerTensorArgs,
    InstanceNormArgs,
    MaxPool2dWithIndicesArgs,
)


def get_permute_weight_input(conv_args: Conv2DArgs) -> torch.fx.Node:
    """
    Retrieves the weight input for the permute operation.

    This function extracts the weight tensor from the given convolution arguments.

    If the weight is in floating point format, it is returned directly.
    If the weight is quantized and followed by a Dequantize operation, the function
     returns the input of the Dequantize node (i.e., the original quantized weight)
    """
    weight = conv_args.weight

    dq_args: Optional[DequantizePerChannelArgs | DequantizePerTensorArgs] = None
    if weight.target == torch.ops.quantized_decomposed.dequantize_per_channel.default:
        dq_args = DequantizePerChannelArgs(*weight.args, *weight.kwargs)  # type: ignore[arg-type]
    elif weight.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default:
        dq_args = DequantizePerTensorArgs(*weight.args, *weight.kwargs)  # type: ignore[arg-type]

    return getattr(dq_args, "input", weight)


@trace_graph_diff_on_pass
class LegalizePreDefinedLayoutOperators(PassBase):
    """
    Pytorch basically assumes NCHW memory format. But, Circle assumes NHWC. Specifcally, some operators have kernels only for NHWC memory format.
    So, we need to permute the dimensions accordingly.

    NOTE. This pass DOES NOT CHANGE node.kwargs["memory_format"]. It changes memory formats by inserting `aten.permute` operators.

    [1] aten.conv2d with group = 1 (circle_custom.conv2d)

        [BEFORE PASS]
          Input[NCHW] ------------------- aten.conv2d[NCHW] ---- OUTPUT[NCHW]
          Weight[NCHW] - (aten.dequantize) ---/
          Bias --------- (aten.dequantize) --/

        [AFTER PASS]
          Input[NCHW] ---- aten.permute(NCHW_to_NHWC) ---------- circle_cumstom.conv2d[NHWC] ---- aten.permute(NHWC_to_NCHW) ---- OUTPUT[NCHW]
          Weight[NCHW] - (aten.dequantize) - aten.permute(NCHW_to_NHWC) ---/
          Bias --------- (aten.dequantize) -------------------------------/

    [2] aten.conv2d with group == Input[C] (circle_custom.depthwise_conv2d)

        NOTE: Weight layout is CNHW (IOHW)

        [BEFORE PASS]
          Input[NCHW] -------------- aten.conv2d[NCHW] ---- OUTPUT[NCHW]
          Weight[CNHW] - (aten.dequantize) --/
          Bias ----------(aten.dequantize) -/

        [AFTER PASS]
          Input[NCHW] ---- aten.permute(NCHW_to_NHWC) ---- circle_cumstom.depthwise_conv2d[NHWC] ---- aten.permute(NHWC_to_NCHW) ---- OUTPUT[NCHW]
          Weight[CNHW] - (aten.dequantize) - aten.permute(CNHW_to_NHWC) ---/
          Bias ----------(aten.dequantize) -------------------------------/
    """

    def __init__(self):
        super().__init__()

    def legalize_conv2d(self, exported_program, node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # conv2d            (Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
        # conv2d.padding    (Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, str padding="valid", SymInt[2] dilation=1, SymInt groups=1) -> Tensor
        args = Conv2DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        padding = args.padding
        groups = args.groups

        input_shape = extract_shape(input)
        if not (len(input_shape) == 4):
            raise NotYetSupportedError(
                f"Only support 4D input tensor: node's input shape: {input_shape}"
            )

        if not (groups == 1 or groups == input_shape[1]):
            raise NotYetSupportedError(
                f"Only support groups=1 or groups=input_channels: node's groups: {groups}, input channels: {input_shape[1]}"
            )

        NCHW_to_NHWC = [0, 2, 3, 1]
        # TODO Introduce a method that inserts permute op.
        # input permute
        with graph.inserting_after(input):
            input_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(input, NCHW_to_NHWC),
                origin=input,
            )
            node.update_arg(node.args.index(input), input_permute)

        # weight permute
        weight = get_permute_weight_input(args)
        with graph.inserting_after(weight):
            if groups == 1:
                # circle_custom.conv2d
                perm = [0, 2, 3, 1]  # OIHW_to_OHWI
            elif groups == input_shape[1]:
                # circle_custom.depthwise_conv2d
                perm = [1, 2, 3, 0]  # O1HW_to_1HWO
            else:
                assert groups == 1 or groups == input_shape[1]  # Cannot reach here

            weight_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(weight, perm),
                origin=weight,
            )
            if args.weight.target in [
                torch.ops.quantized_decomposed.dequantize_per_channel.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]:
                dq = args.weight
                dq.update_arg(dq.args.index(weight), weight_permute)
                # Need to update dq.meta["val"] in FillMetaVal pass.
                del dq.meta["val"]
            else:
                node.update_arg(node.args.index(weight), weight_permute)

        with graph.inserting_before(node):
            if groups == 1:
                if isinstance(padding, list):
                    legalized_op = torch.ops.circle_custom.conv2d
                elif isinstance(padding, str):
                    legalized_op = torch.ops.circle_custom.conv2d.padding
            elif groups == input_shape[1]:
                if isinstance(padding, list):
                    legalized_op = torch.ops.circle_custom.depthwise_conv2d
                elif isinstance(padding, str):
                    legalized_op = torch.ops.circle_custom.depthwise_conv2d.padding
            else:
                assert groups == 1 or groups == input_shape[1]  # Cannot reach here

            circle_op = create_node(
                graph, legalized_op, args=node.args, kwargs=node.kwargs, origin=node
            )
            # output permute
            NHWC_to_NCHW = [0, 3, 1, 2]
            conv_out_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(circle_op, NHWC_to_NCHW),
            )
        node.replace_all_uses_with(conv_out_permute, propagate_meta=True)

        logger.debug(f"{node.name} is replaced with {circle_op.name}")
        modified = True
        return modified

    def legalize_instance_norm(self, exported_program, node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
        args = InstanceNormArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input
        weight = args.weight
        bias = args.bias
        eps = args.eps

        running_mean = args.running_mean
        running_var = args.running_var
        use_input_stats = args.use_input_stats

        if not (use_input_stats == True):
            raise NotYetSupportedError("Only support use_input_stats is True.")
        if not isinstance(running_mean, NoneType):
            raise NotYetSupportedError("Only support running_mean=None")
        if not isinstance(running_var, NoneType):
            raise NotYetSupportedError("Only support running_var=None")

        if weight is None:
            # TODO Support weight=None
            raise NotYetSupportedError("Only support weight is not None.")
        if bias is None:
            # TODO Support bias=None
            raise NotYetSupportedError("Only support bias is not None.")

        with graph.inserting_after(input):
            # input permute
            NCHW_to_NHWC = [0, 2, 3, 1]
            input_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(input, NCHW_to_NHWC),
                origin=input,
            )
            node.update_arg(node.args.index(input), input_permute)
        with graph.inserting_before(node):
            # circle instnorm
            circle_instnorm = create_node(
                graph,
                torch.ops.circle_custom.instance_norm,
                args=node.args,
                kwargs=node.kwargs,
                origin=node,
            )
            # output permute
            NHWC_to_NCHW = [0, 3, 1, 2]
            instnorm_out_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(circle_instnorm, NHWC_to_NCHW),
            )
        node.replace_all_uses_with(instnorm_out_permute, propagate_meta=True)

        logger.debug(f"{node.name} is replaced with {circle_instnorm.name}")
        modified = True
        return modified

    def legalize_max_pool2d_with_indices(self, exported_program, node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
        args = MaxPool2dWithIndicesArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input_ = args.input
        kernel_size = args.kernel_size
        stride = args.stride
        padding = args.padding
        dilation = args.dilation
        ceil_mode = args.ceil_mode
        if ceil_mode:
            raise NotYetSupportedError("Only support non-ceil model.")
        if len(node.users.keys()) != 1:
            raise NotYetSupportedError(
                "Only support maxpool2d with 'return_indices=False'."
            )

        NCHW_to_NHWC = [0, 2, 3, 1]
        # TODO Introduce a method that inserts permute op.
        # input permute
        with graph.inserting_after(input_):
            input_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(input_, NCHW_to_NHWC),
                origin=input_,
            )
            node.update_arg(node.args.index(input_), input_permute)
        with graph.inserting_before(node):
            legalized_op = torch.ops.circle_custom.maxpool2d
            circle_maxpool2d = create_node(
                graph, legalized_op, args=node.args, kwargs=node.kwargs, origin=node
            )
            # output permute
            NHWC_to_NCHW = [0, 3, 1, 2]
            maxpool_out_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(circle_maxpool2d, NHWC_to_NCHW),
            )
        get_item, *_ = node.users.keys()
        get_item.replace_all_uses_with(maxpool_out_permute, propagate_meta=True)

        logger.debug(f"{node.name} is replaced with {circle_maxpool2d.name}")
        modified = True
        return modified

    def legalize_avg_pool2d(self, exported_program, node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False

        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> (Tensor)
        args = AvgPool2dArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input_ = args.input
        kernel_size = args.kernel_size
        stride = args.stride
        padding = args.padding
        ceil_mode = args.ceil_mode
        if ceil_mode:
            raise NotYetSupportedError("Only support non-ceil model.")
        divisor_override = args.divisor_override
        if divisor_override is not None:
            raise NotYetSupportedError(
                "For the case that the divisor_override is not None is not yet supported."
            )

        NCHW_to_NHWC = [0, 2, 3, 1]
        # TODO Introduce a method that inserts permute op.
        # input permute
        with graph.inserting_after(input_):
            input_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(input_, NCHW_to_NHWC),
                origin=input_,
            )
            node.update_arg(node.args.index(input_), input_permute)
        with graph.inserting_before(node):
            legalized_op = torch.ops.circle_custom.avgpool2d
            circle_avgpool2d = create_node(
                graph, legalized_op, args=node.args, kwargs=node.kwargs, origin=node
            )
            # output permute
            NHWC_to_NCHW = [0, 3, 1, 2]
            avgpool_out_permute = create_node(
                graph,
                torch.ops.aten.permute.default,
                args=(circle_avgpool2d, NHWC_to_NCHW),
            )
        node.replace_all_uses_with(avgpool_out_permute, propagate_meta=True)

        logger.debug(f"{node.name} is replaced with {circle_avgpool2d.name}")
        modified = True
        return modified

    def call(self, exported_program: ExportedProgram) -> PassResult:
        target_to_legalize_func = {
            torch.ops.aten.conv2d.default: self.legalize_conv2d,
            torch.ops.aten.conv2d.padding: self.legalize_conv2d,
            torch.ops.aten.max_pool2d_with_indices.default: self.legalize_max_pool2d_with_indices,
            torch.ops.aten.avg_pool2d.default: self.legalize_avg_pool2d,
            torch.ops.aten.instance_norm.default: self.legalize_instance_norm,
        }

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, list(target_to_legalize_func.keys())):
                continue

            modified |= target_to_legalize_func[node.target](exported_program, node)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)

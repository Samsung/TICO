# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
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

from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.errors import NotYetSupportedError
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import Conv3DArgs


@trace_graph_diff_on_pass
class ConvertConv3dToConv2d(PassBase):
    """
    This pass converts `torch.ops.aten.conv3d` to multiple `torch.ops.aten.conv2d` operations

    [before]                       input(dim=5)         weight(dim=5)
                                      │                   │
                                      │                   │
                                   conv3d<----------------+
                                      │
                                      │
                                   output(dim=5)

    [after]                        input(dim=5)                              weight(dim=5)
                                      │                                         │
                                      │                                 ┌───────┴───────┐
                                      │                                 │ weight slice  │
                                      │                                 │ (kT times)    │
                                      │                                 └───────┬───────┘
                                      │                                         │
                                      │                                 ┌───────┴───────┐
                                      │                                 │  squeeze dims │
                                      │                                 │ (remove dim=2)│
                                      │                                 └───────┬───────┘
                                      │                                         │
                                      │                                 ┌───────┴────────────┐
                                      │                                 │ weight_2d[0..kT-1] │
                                      │                                 │ [C_out,C_in,kH,kW] │
                                      │                                 └───────┬────────────┘
                                      │                                         │
                    ┌─────────────────┴──────────────────────────────┐          |
                    │  temporal padding (if needed)                  │          |
                    │  ┌────────────┐  ┌────────────┐  ┌───────────┐ │          |
                    │  │ zeros      │  │  input     │  │zeros      │ │          |
                    │  │ [N,C,p,H,W]│  │ [N,C,T,H,W]│  │[N,C,p,H,W]│ │          |
                    │  └────┬───────┘  └────┬───────┘  └────┬──────┘ │          |
                    │       └───────────┼───┴───────────────┘        │          |
                    │                   │                            │          |
                    │           ┌───────┴───────┐                    │          |
                    │           │     cat       │                    │          |
                    │           │ (dim=2)       │                    │          |
                    │           └───────┬───────┘                    │          |
                    │                   │                            │          |
                    │           ┌───────┴───────┐                    │          |
                    │           │  padded_input │                    │          |
                    │           │ [N,C,T+2p,H,W]│                    │          |
                    │           └───────┬───────┘                    │          |
                    └───────────────────┼────────────────────────────┘          |
                                        │                                       |
                    ┌───────────────────┴───────────────────────────────┐       |
                    │           Temporal Processing Loop                │       |
                    │  ┌────────────────────────────────────────────┐   │       |
                    │  │ For t_out = 0..T_out-1:                    │   │       |
                    │  │   For i = 0..kT-1:                         │   │       |
                    │  │     t_idx = t_out*stride[0] + i*dilation[0]│   │       |
                    │  │     ┌─────────────────────────┐            │   │       |
                    │  │     │ slice input[t_idx]      │            │   │       |
                    │  │     │ [N,C,H,W]               │            │   │       |
                    │  │     └─────────┬───────────────┘            │   │       |
                    │  │               │                            │   │       |
                    │  │     ┌─────────┴───────────────┐            │   │       |
                    │  │     │ squeeze dims            │            │   │       |
                    │  │     │ [N,C,H,W]               │            │   │       |
                    │  │     └─────────┬───────────────┘            │   │       |
                    │  │               │                            │   │       |
                    │  │     ┌─────────┴───────────────┐            │   │       |
                    │  │     │ conv2d(input,weight)    │            │   │───────┘
                    │  │     │ [N,C_out,H_out,W_out]   │            │   │
                    │  │     └─────────┬───────────────┘            │   │
                    │  │               │                            │   │
                    │  │     ┌─────────┴───────────────┐            │   │
                    │  │     │ where(valid_mask,       │            │   │
                    │  │     │      conv2d, zeros)     │            │   │
                    │  │     └─────────┬───────────────┘            │   │
                    │  │               │                            │   │
                    │  │     ┌─────────┴───────────────┐            │   │
                    │  │     │ accumulate (add)        │            │   │
                    │  │     └─────────┬───────────────┘            │   │
                    │  └───────────────┼────────────────────────────┘   │
                    │                  │                                │
                    │           ┌──────┴───────────┐                    │
                    │           │ add bias (if any)│                    │
                    │           └───────┬──────────┘                    │
                    │                   │                               │
                    │           ┌───────┴──────────┐                    │
                    │           │ unsqueeze (dim=2)│                    │
                    │           └───────┬──────────┘                    │
                    └───────────────────┼───────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────────┐
                    │           cat (dim=2)                     │
                    │           [N,C_out,T_out,H_out,W_out]     │
                    └───────────────────┬───────────────────────┘
                                        │
                                   output(dim=5)
    """

    def __init__(self):
        super().__init__()

    def _parse_3d_padding(self, padding, kernel_size):
        """
        Parse 3D padding parameter and return (temporal, H, W) tuple.

        Args:
            padding: Can be str ('same', 'valid'), int, list, or tuple
            kernel_size: 3D kernel size (kT, kH, kW)

        Returns:
            Tuple of 3 padding values: (temporal_padding, H_padding, W_padding)
        """
        if isinstance(padding, str):
            if padding == "same":
                # For 'same' padding, use kernel_size // 2
                if isinstance(kernel_size, int):
                    return kernel_size // 2, kernel_size // 2, kernel_size // 2
                else:
                    return kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2
            elif padding == "valid":
                return 0, 0, 0
            else:
                raise NotYetSupportedError(f"Unsupported padding string: {padding}")
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 1:
                return padding[0], padding[0], padding[0]
            elif len(padding) == 3:
                return padding[0], padding[1], padding[2]
            else:
                raise NotYetSupportedError(f"Unsupported padding format: {padding}")
        else:  # int
            return padding, padding, padding

    def convert(self, exported_program: ExportedProgram, node: torch.fx.Node) -> bool:
        logger = logging.getLogger(__name__)
        modified = False
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        # Extract conv3d arguments
        args = Conv3DArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        input = args.input
        weight = args.weight
        bias = args.bias
        stride = args.stride
        padding = args.padding
        dilation = args.dilation
        groups = args.groups

        input_shape = extract_shape(input)
        weight_shape = extract_shape(weight)

        if not (len(input_shape) == 5):
            raise NotYetSupportedError(
                f"Only support 5D input tensor: node's input shape: {input_shape}"
            )

        if not (len(weight_shape) == 5):
            raise NotYetSupportedError(
                f"Only support 5D weight tensor: node's weight shape: {weight_shape}"
            )

        N, C_in, T_in, H_in, W_in = input_shape
        C_out, C_in_weight, kT, kH, kW = weight_shape

        temporal_padding, h_padding, w_padding = self._parse_3d_padding(
            padding, (kT, kH, kW)
        )

        # Calculate output dimensions
        T_out = (T_in + 2 * temporal_padding - dilation[0] * (kT - 1) - 1) // stride[
            0
        ] + 1

        H_out = (H_in + 2 * h_padding - dilation[1] * (kH - 1) - 1) // stride[1] + 1
        W_out = (W_in + 2 * w_padding - dilation[2] * (kW - 1) - 1) // stride[2] + 1

        # Find the next node after conv3d
        next_node = node.next
        if next_node is None:
            # If no next node, find the output node
            for n in graph.nodes:
                if n.op == "output":
                    next_node = n
                    break

        if next_node is None:
            raise RuntimeError("Could not find insertion point for temporal outputs")

        # Create all nodes before the next node in one go
        with graph.inserting_before(next_node):
            # Step 1: Create weight_2d layers first (they depend only on weight)
            weight_2d_layers = []
            for t in range(kT):
                # Slice weight for temporal dimension t: [C_out, C_in, t, kH, kW] -> [C_out, C_in, kH, kW]
                weight_slice = create_node(
                    graph,
                    torch.ops.aten.slice.Tensor,
                    args=(weight, 2, t, t + 1, 1),
                    origin=weight,
                )

                # Remove temporal dimension: [C_out, C_in, 1, kH, kW] -> [C_out, C_in, kH, kW]
                weight_2d = create_node(
                    graph,
                    torch.ops.aten.squeeze.dims,
                    args=(weight_slice, [2]),
                    origin=weight_slice,
                )
                weight_2d_layers.append(weight_2d)

            # Step 2: Create padded input (if needed) using cat
            if temporal_padding > 0:
                # Create zero padding: [N, C, padding, H, W]
                zero_padding = create_node(
                    graph,
                    torch.ops.aten.zeros.default,
                    args=([N, C_in, temporal_padding, H_in, W_in],),
                    kwargs={
                        "dtype": input.meta.get("dtype", torch.float32),
                        "device": input.meta.get("device", "cpu"),
                    },
                    origin=input,
                )

                # Cat: [zero_padding, input, zero_padding] -> [N, C, T+2*padding, H, W]
                padded_input = create_node(
                    graph,
                    torch.ops.aten.cat.default,
                    args=([zero_padding, input, zero_padding], 2),
                    origin=input,
                )
                T_padded = T_in + 2 * temporal_padding
            else:
                padded_input = input
                T_padded = T_in

            # Step 3: Process each temporal output position
            temporal_outputs = []
            for t_out in range(T_out):
                # Calculate input time position
                t_in = t_out * stride[0]

                # Initialize accumulator for this temporal position
                acc = None

                for i, weight_2d in enumerate(weight_2d_layers):
                    # Calculate actual time index with dilation
                    t_idx = t_in + i * dilation[0]

                    # Create constant for time index
                    t_idx_const = create_node(
                        graph,
                        torch.ops.aten.scalar_tensor.default,
                        args=(t_idx,),
                        kwargs={"dtype": torch.int64},
                        origin=node,
                    )

                    # Create constant for T_padded
                    t_padded_const = create_node(
                        graph,
                        torch.ops.aten.scalar_tensor.default,
                        args=(T_padded,),
                        kwargs={"dtype": torch.int64},
                        origin=node,
                    )

                    # Check if t_idx < T_padded
                    valid_mask = create_node(
                        graph,
                        torch.ops.aten.lt.Tensor,
                        args=(t_idx_const, t_padded_const),
                        origin=node,
                    )

                    # Slice input at time t_idx: [N, C_in, T_padded, H_in, W_in] -> [N, C_in, H_in, W_in]
                    input_slice = create_node(
                        graph,
                        torch.ops.aten.slice.Tensor,
                        args=(padded_input, 2, t_idx, t_idx + 1, 1),
                        origin=padded_input,
                    )

                    # Remove temporal dimension: [N, C_in, 1, H_in, W_in] -> [N, C_in, H_in, W_in]
                    input_2d = create_node(
                        graph,
                        torch.ops.aten.squeeze.dims,
                        args=(input_slice, [2]),
                        origin=input_slice,
                    )

                    # Create conv2d operation with proper input
                    conv2d = create_node(
                        graph,
                        torch.ops.aten.conv2d.default,
                        args=(
                            input_2d,  # input is now available
                            weight_2d,
                            None,  # bias = False
                            [stride[1], stride[2]],
                            [h_padding, w_padding],
                            [dilation[1], dilation[2]],
                            groups,
                        ),
                        origin=node,
                    )

                    # Create zero tensor with calculated shape
                    # conv2d output shape: [N, C_out, H_out, W_out]
                    zero_tensor = create_node(
                        graph,
                        torch.ops.aten.zeros.default,
                        args=([N, C_out, H_out, W_out],),
                        kwargs={
                            "dtype": input.meta.get("dtype", torch.float32),
                            "device": input.meta.get("device", "cpu"),
                        },
                        origin=conv2d,
                    )

                    # Apply conditional execution
                    conv2d_masked = create_node(
                        graph,
                        torch.ops.aten.where.self,
                        args=(valid_mask, conv2d, zero_tensor),
                        origin=conv2d,
                    )

                    if acc is None:
                        # First temporal slice
                        acc = conv2d_masked
                    else:
                        # Add subsequent temporal slices
                        acc = create_node(
                            graph,
                            torch.ops.aten.add.Tensor,
                            args=(acc, conv2d_masked),
                            origin=acc,
                        )

                # Add bias if present
                if bias is not None:
                    bias_reshaped = create_node(
                        graph,
                        torch.ops.aten.reshape.default,
                        args=(bias, [1, C_out, 1, 1]),
                        origin=bias,
                    )
                    acc = create_node(
                        graph,
                        torch.ops.aten.add.Tensor,
                        args=(acc, bias_reshaped),
                        origin=acc,
                    )

                temporal_outputs.append(acc)

            # Step 4: Stack temporal outputs using cat instead of stack
            # First, unsqueeze each temporal output to add the time dimension
            unsqueezed_outputs = []
            for i, temp_output in enumerate(temporal_outputs):
                # Add time dimension: [N, C_out, H_out, W_out] -> [N, C_out, 1, H_out, W_out]
                unsqueezed = create_node(
                    graph,
                    torch.ops.aten.unsqueeze.default,
                    args=(temp_output, 2),
                    origin=temp_output,
                )
                unsqueezed_outputs.append(unsqueezed)

            # Cat along time dimension: [N, C_out, T_out, H_out, W_out]
            stacked_output = create_node(
                graph,
                torch.ops.aten.cat.default,
                args=(unsqueezed_outputs, 2),
                origin=node,
            )

        # Replace the original node
        node.replace_all_uses_with(stacked_output, propagate_meta=False)
        logger.debug(f"{node.name} is replaced with conv2d decomposition")
        modified = True

        return modified

    def call(self, exported_program: ExportedProgram) -> PassResult:
        target_conv_op = [torch.ops.aten.conv3d.default, torch.ops.aten.conv3d.padding]
        graph_module = exported_program.graph_module
        graph = graph_module.graph

        modified = False

        # Process all Conv3D nodes in forward pass order
        for node in graph.nodes:
            if not is_target_node(node, target_conv_op):
                continue
            modified |= self.convert(exported_program, node)

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)

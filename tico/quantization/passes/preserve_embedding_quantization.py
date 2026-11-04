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

"""Keep row-wise embedding domain conversions explicit during quant folding."""

import numpy as np
import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.validate_args_kwargs import parse_node_args, QuantizePerTensorArgs

# Export-only policy. CPU planning rediscovers the pattern from serialized tensors.
EXPLICIT_EMBEDDING = "_explicit_embedding_quantization_"
FLOAT_EMBEDDING = "_float_embedding_tensor_"


def _schema_name(node: torch.fx.Node) -> str:
    return getattr(getattr(node.target, "_schema", None), "name", "")


class PreserveEmbeddingQuantization(PassBase):
    """Retain DQ/Gather/Q and optional scaled embedding Q/DQ/Mul/Q boundaries.

    Match a constant UINT8 rank-two table dequantized per row (axis zero),
    followed by ATen embedding. Only the lookup's Q/DQ and elementwise MUL
    continuation form the CPU island. A terminal DQ is folded to its explicit Q,
    retaining the pre-existing integer output ABI. Other operators, including
    shared FC consumers of the weight, keep their existing quantization policy.
    """

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph = exported_program.graph
        user_inputs = exported_program.graph_signature.user_inputs
        starts = []
        for node in graph.nodes:
            if node.target != torch.ops.aten.embedding.default:
                continue
            dq = node.args[0]
            if not isinstance(dq, torch.fx.Node) or _schema_name(dq) != (
                "quantized_decomposed::dequantize_per_channel"
            ):
                continue
            weight = dq.args[0]
            if (
                not isinstance(weight, torch.fx.Node)
                or weight.op != "placeholder"
                or weight.name in user_inputs
                or weight.meta["val"].dtype != torch.uint8
                or weight.meta["val"].ndim != 2
                or dq.args[3] != 0
                or dq.args[4:6] != (0, 255)
            ):
                continue
            dq.meta[FLOAT_EMBEDDING] = True
            dq.meta[EXPLICIT_EMBEDDING] = True
            node.meta[FLOAT_EMBEDDING] = True
            node.meta[EXPLICIT_EMBEDDING] = True
            starts.append(node)

        allowed = {
            "quantized_decomposed::quantize_per_tensor",
            "quantized_decomposed::dequantize_per_tensor",
            "aten::mul",
        }
        pending = list(starts)
        seen = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            for user in node.users:
                if user.op == "call_function" and (
                    _schema_name(user) in allowed
                    or (
                        _schema_name(user) == "aten::clamp"
                        and user.meta.get("_embedding_range_clamp_")
                    )
                ):
                    user.meta[EXPLICIT_EMBEDDING] = True
                    pending.append(user)

        # Complete graph mutations first. A reduced-range Q inserts a float
        # clamp between its input DQ and itself; classification must see this
        # updated edge rather than depend on the order of membership-set visits.
        for node in graph.nodes:
            if node not in seen or _schema_name(node) != (
                "quantized_decomposed::quantize_per_tensor"
            ):
                continue
            args = parse_node_args(QuantizePerTensorArgs, node)
            # Circle QUANTIZE expresses storage-range saturation. A reduced
            # range needs a separate clamp, rather than silently widening it.
            supported_range = (
                args.dtype == torch.uint8
                and (args.quant_min, args.quant_max) == (0, 255)
            ) or (
                args.dtype == torch.int16
                and args.quant_min in (-32768, -32767)
                and args.quant_max == 32767
            )
            if not supported_range:
                raise RuntimeError(
                    "Unsupported explicit embedding Quantize storage range"
                )
            if args.quant_min == -32767 and not args.tensor.meta.get(
                "_embedding_range_clamp_"
            ):
                lower = float(
                    np.float32(args.quant_min - args.zero_p) * np.float32(args.scale)
                )
                with graph.inserting_before(node):
                    clamp = create_node(
                        graph,
                        torch.ops.aten.clamp.default,
                        args=(args.tensor, lower, None),
                        origin=node,
                    )
                clamp.meta["val"] = torch.empty_like(
                    args.tensor.meta["val"], dtype=torch.float32
                )
                clamp.meta[EXPLICIT_EMBEDDING] = True
                clamp.meta[FLOAT_EMBEDDING] = True
                clamp.meta["_embedding_range_clamp_"] = True
                clamp.meta.pop(QPARAM_KEY, None)
                seen.add(clamp)
                node.replace_input_with(args.tensor, clamp)
            node.meta[QPARAM_KEY] = QuantParam(
                scale=[args.scale],
                zero_point=[args.zero_p],
                dtype=str(args.dtype).removeprefix("torch."),
            )

        # Every generated clamp is now a member of the CPU island. Determine
        # internal versus terminal DQs only after those new consumers exist.
        for node in graph.nodes:
            if node not in seen:
                continue
            name = _schema_name(node)
            if name == "quantized_decomposed::quantize_per_tensor":
                continue
            elif name == "quantized_decomposed::dequantize_per_tensor":
                # Only DQs used inside the island are retained. Boundary DQs
                # expose their Quantize's integer tensor to downstream NPU ops.
                node.meta[FLOAT_EMBEDDING] = any(user in seen for user in node.users)
            else:
                node.meta[FLOAT_EMBEDDING] = True
            if name == "aten::mul":
                # The multiplier may have its own fake-quantized constant.
                # Preserve its DQ and annotate the quantized constant later.
                for operand in node.all_input_nodes:
                    if (
                        _schema_name(operand)
                        == "quantized_decomposed::dequantize_per_tensor"
                    ):
                        source = operand.args[0]
                        if operand not in seen and (
                            not isinstance(source, torch.fx.Node)
                            or source.op != "placeholder"
                            or source.name in user_inputs
                        ):
                            raise RuntimeError(
                                "Explicit embedding dynamic quantized MUL operand "
                                "is unsupported; static scale required."
                            )
                        operand.meta[EXPLICIT_EMBEDDING] = True
                        operand.meta[FLOAT_EMBEDDING] = True
        graph.lint()
        exported_program.graph_module.recompile()
        return PassResult(False)

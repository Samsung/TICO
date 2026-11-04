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

import copy

import numpy as np
import torch
from torch.export import ExportedProgram

from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.validate_args_kwargs import (
    DequantizePerTensorArgs,
    LinearArgs,
    parse_node_args,
    QuantizePerTensorArgs,
)

# Export-only policy. CPU planning rediscovers the pattern from serialized tensors.
EXPLICIT_EMBEDDING = "_explicit_embedding_quantization_"
FLOAT_EMBEDDING = "_float_embedding_tensor_"


def _schema_name(node: torch.fx.Node) -> str:
    return getattr(getattr(node.target, "_schema", None), "name", "")


_FLOAT_LAYOUTS = {"aten::reshape", "aten::permute", "aten::slice", "aten::slice_copy"}


def _has_float_input(node: torch.fx.Node, user_inputs) -> bool:
    """A genuine float side input makes insertion a float operation.

    Stop at Q/DQ: an independently quantized concat branch still needs the
    existing integer-domain legalization, including an explicit requantizer.
    """
    if node.op == "placeholder":
        return node.name in user_inputs and node.meta["val"].is_floating_point()
    return _schema_name(node) in _FLOAT_LAYOUTS and any(
        _has_float_input(source, user_inputs) for source in node.all_input_nodes
    )


def _constant_source(node, user_inputs):
    if _schema_name(node) in {
        "quantized_decomposed::dequantize_per_tensor",
        "quantized_decomposed::dequantize_per_channel",
    }:
        node = node.args[0]
    if node.op == "placeholder" and node.name not in user_inputs:
        return node
    return None


def _is_local_square_linear(node, user_inputs, tables):
    """Admit a constant feature transform, leaving shared table projections alone."""
    args = parse_node_args(LinearArgs, node)
    if args.bias is not None:
        return False
    weight = args.weight
    source = _constant_source(weight, user_inputs)
    if source is None or source in tables:
        return False
    shape = weight.meta["val"].shape
    return (
        len(shape) == 2
        and shape[0] == shape[1]
        and shape[1] == args.input.meta["val"].shape[-1]
    )


def _restore_integer_fc_layouts(
    graph: torch.fx.Graph, region: set[torch.fx.Node]
) -> None:
    """Commute per-tensor DQ past layouts on ordinary FC input edges only."""
    integer_layouts: dict[torch.fx.Node, torch.fx.Node] = {}

    def integer_source(node: torch.fx.Node) -> torch.fx.Node | None:
        if node in integer_layouts:
            return integer_layouts[node]
        if _schema_name(node) == "quantized_decomposed::dequantize_per_tensor":
            dq = parse_node_args(DequantizePerTensorArgs, node)
            quantize = dq.input
            if _schema_name(
                quantize
            ) != "quantized_decomposed::quantize_per_tensor" or not quantize.meta.get(
                EXPLICIT_EMBEDDING
            ):
                return None
            q = parse_node_args(QuantizePerTensorArgs, quantize)
            if (dq.scale, dq.zero_point, dq.quant_min, dq.quant_max, dq.dtype) != (
                q.scale,
                q.zero_p,
                q.quant_min,
                q.quant_max,
                q.dtype,
            ):
                return None
            return quantize
        if node not in region or _schema_name(node) not in _FLOAT_LAYOUTS:
            return None
        source = node.args[0]
        assert isinstance(source, torch.fx.Node)
        integer_input = integer_source(source)
        if integer_input is None:
            return None
        with graph.inserting_before(node):
            layout = graph.node_copy(
                node, lambda argument: integer_input if argument is source else argument
            )
        layout.meta.pop(EXPLICIT_EMBEDDING, None)
        layout.meta.pop(FLOAT_EMBEDDING, None)
        layout.meta[QPARAM_KEY] = copy.deepcopy(integer_input.meta[QPARAM_KEY])
        layout.meta["val"] = torch.empty_like(
            node.meta["val"], dtype=getattr(torch, layout.meta[QPARAM_KEY].dtype)
        )
        integer_layouts[node] = layout
        return layout

    for consumer in list(graph.nodes):
        if _schema_name(consumer) != "aten::linear" or consumer in region:
            continue
        args = parse_node_args(LinearArgs, consumer)
        if args.input not in region or _schema_name(args.input) not in _FLOAT_LAYOUTS:
            continue
        # A genuine float FC also lies outside the bounded feature-transform
        # region. Only an existing weight and output quantization contract
        # identifies the ordinary quantized consumer whose input we restore.
        if _schema_name(args.weight) not in {
            "quantized_decomposed::dequantize_per_tensor",
            "quantized_decomposed::dequantize_per_channel",
        } or not (
            QPARAM_KEY in consumer.meta
            or any(
                _schema_name(user) == "quantized_decomposed::quantize_per_tensor"
                for user in consumer.users
            )
        ):
            continue
        restored = integer_source(args.input)
        if restored is None:
            raise RuntimeError(
                "Cannot recover the ordinary FC input domain through an embedding "
                "float layout; an explicit input Quantize boundary is required."
            )
        consumer.replace_input_with(args.input, restored)


class PreserveEmbeddingQuantization(PassBase):
    """Retain explicit domains through a supported embedding CPU continuation.

    Match a constant UINT8 rank-two table dequantized per row (axis zero),
    followed by ATen embedding. Preserve Q/DQ, static MUL, float layouts and
    insertion concats, and independent constant square feature transforms.
    A terminal DQ is folded to its explicit Q, retaining the pre-existing
    integer output ABI. Other operators, including shared FC consumers of the
    table, keep their existing quantization policy.
    """

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph = exported_program.graph
        user_inputs = exported_program.graph_signature.user_inputs
        starts = []
        tables = set()
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
            tables.add(weight)

        allowed = {
            "quantized_decomposed::quantize_per_tensor",
            "quantized_decomposed::dequantize_per_tensor",
            "aten::mul",
            *_FLOAT_LAYOUTS,
        }
        pending = list(starts)
        seen = set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            for user in node.users:
                if user.op != "call_function":
                    continue
                name = _schema_name(user)
                value = user.meta.get("val")
                # Membership alone is not a float contract: an explicit Q
                # produces integer codes that may also feed ordinary layouts
                # or concat without an intervening DQ.
                if name not in {
                    "quantized_decomposed::quantize_per_tensor",
                    "quantized_decomposed::dequantize_per_tensor",
                } and not (
                    isinstance(value, torch.Tensor) and value.is_floating_point()
                ):
                    continue
                if user.op == "call_function" and (
                    _schema_name(user) in allowed
                    or (
                        _schema_name(user) == "aten::cat"
                        and all(
                            operand.meta["val"].is_floating_point()
                            for operand in user.all_input_nodes
                        )
                        and (
                            all(operand in seen for operand in user.all_input_nodes)
                            or any(
                                _has_float_input(operand, user_inputs)
                                for operand in user.all_input_nodes
                            )
                        )
                    )
                    or (
                        _schema_name(user) == "aten::linear"
                        and _is_local_square_linear(user, user_inputs, tables)
                    )
                    or (
                        _schema_name(user) == "aten::clamp"
                        and user.meta.get("_embedding_range_clamp_")
                    )
                ):
                    user.meta[EXPLICIT_EMBEDDING] = True
                    pending.append(user)

        # Float layout/concat side inputs (for example visual embeddings) are
        # genuine CPU values. Do not let backward propagation assign them an
        # integer grid merely because another concat input was quantized earlier.
        def preserve_float_operand(operand):
            if operand in seen:
                return
            name = _schema_name(operand)
            if name in _FLOAT_LAYOUTS:
                operand.meta[EXPLICIT_EMBEDDING] = True
                operand.meta[FLOAT_EMBEDDING] = True
                for source in operand.all_input_nodes:
                    preserve_float_operand(source)
            elif (
                operand.op == "placeholder" and operand.meta["val"].is_floating_point()
            ):
                operand.meta[FLOAT_EMBEDDING] = True
            else:
                raise RuntimeError(
                    "Unsupported explicit embedding float side operand: "
                    f"{operand.name}. Expected an island value or float "
                    "placeholder/layout; independent quantized branches "
                    "require a separately supported conversion."
                )

        for node in graph.nodes:
            if node not in seen:
                continue
            if _schema_name(node) == "aten::cat":
                for operand in node.all_input_nodes:
                    preserve_float_operand(operand)
            elif _schema_name(node) == "aten::linear":
                weight = parse_node_args(LinearArgs, node).weight
                if _schema_name(weight) in {
                    "quantized_decomposed::dequantize_per_tensor",
                    "quantized_decomposed::dequantize_per_channel",
                }:
                    weight.meta[EXPLICIT_EMBEDDING] = True
                    weight.meta[FLOAT_EMBEDDING] = True
                else:
                    preserve_float_operand(weight)

        # Complete the layout-only closure of genuine float side inputs before
        # collecting their independent quantizers. Sibling creation order must
        # not decide whether a later layout absorbs a Q/DQ's integer grid.
        float_sides = {
            node
            for node in graph.nodes
            if node not in seen and node.meta.get(FLOAT_EMBEDDING)
        }
        pending = list(float_sides)
        while pending:
            node = pending.pop()
            for user in node.users:
                value = user.meta.get("val")
                if (
                    _schema_name(user) in _FLOAT_LAYOUTS
                    and isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                    and user not in float_sides
                ):
                    user.meta[EXPLICIT_EMBEDDING] = True
                    user.meta[FLOAT_EMBEDDING] = True
                    float_sides.add(user)
                    pending.append(user)

        # Keep each branch's original Q as a boundary instead of assigning its
        # grid to the shared float producer or any intervening float layout.
        for node in float_sides:
            for user in node.users:
                if _schema_name(user) != "quantized_decomposed::quantize_per_tensor":
                    continue
                user.meta[EXPLICIT_EMBEDDING] = True
                seen.add(user)
                for dq in user.users:
                    if (
                        _schema_name(dq)
                        == "quantized_decomposed::dequantize_per_tensor"
                    ):
                        dq.meta[EXPLICIT_EMBEDDING] = True
                        seen.add(dq)

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
        _restore_integer_fc_layouts(graph, seen)
        graph.lint()
        exported_program.graph_module.recompile()
        return PassResult(False)

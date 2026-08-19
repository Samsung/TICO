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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from tico.circle._object import clone_object, ObjectFactory
from tico.circle._schema import circle_schema
from tico.circle.analysis import TensorContract
from tico.circle.builder import CircleBuilder
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleValueError
from tico.circle.graph import as_indices, as_list, CircleGraph, is_constant_tensor
from tico.circle.passes.base import CirclePass, CirclePassContext, CirclePassResult
from tico.circle.passes.optimization._utils import (
    AppendedObjectCheckpoint,
    operator_builtin_code,
    operator_is_plain,
    operator_version,
    OptimizationSchemaResolver,
    tensor_contract,
)
from tico.circle.value import TensorValue, TensorValueCodec


@dataclass(frozen=True)
class TransposeConvSliceFusionPolicy:
    """Control conservative layout and type limits for TConv-Slice fusion."""

    require_single_consumer: bool = True
    support_channel_crop: bool = True

    def __post_init__(self) -> None:
        """Normalize policy switches to plain bool values."""

        object.__setattr__(
            self,
            "require_single_consumer",
            bool(self.require_single_consumer),
        )
        object.__setattr__(
            self,
            "support_channel_crop",
            bool(self.support_channel_crop),
        )


class FuseTransposeConvSlicePass(CirclePass):
    """Fuse a static NHWC SLICE into one FLOAT32 TRANSPOSE_CONV producer."""

    def __init__(
        self,
        *,
        policy: TransposeConvSliceFusionPolicy | None = None,
        builtin_codes: Mapping[str, int] | None = None,
        builtin_options_types: Mapping[str, int] | None = None,
        tensor_types: Mapping[str, int] | None = None,
        padding_values: Mapping[str, int] | None = None,
        codec: TensorValueCodec | None = None,
        object_factory: ObjectFactory | None = None,
    ) -> None:
        """Create the fusion with injectable Circle schema values and value services."""

        self.policy = policy or TransposeConvSliceFusionPolicy()
        self.resolver = OptimizationSchemaResolver(
            builtin_codes=builtin_codes,
            builtin_options_types=builtin_options_types,
            tensor_types=tensor_types,
            object_factory=object_factory,
        )
        self.codec = codec or TensorValueCodec()
        self.object_factory = object_factory
        self.codes = {
            name: self.resolver.builtin_code(name)
            for name in ("SLICE", "TRANSPOSE_CONV")
        }
        self.options_types = {
            name: self.resolver.builtin_options_type(name)
            for name in ("SliceOptions", "TransposeConvOptions")
        }
        self.float32_type = self.resolver.tensor_type("FLOAT32")
        self.int32_type = self.resolver.tensor_type("INT32")
        configured_padding = {
            str(name).upper(): int(value)
            for name, value in (padding_values or {}).items()
        }
        self.padding_values = {
            "SAME": configured_padding.get(
                "SAME",
                _maybe_schema_enum_value("Padding", "SAME", 0),
            ),
            "VALID": configured_padding.get(
                "VALID",
                _maybe_schema_enum_value("Padding", "VALID", 1),
            ),
        }

    def run(
        self,
        document: CircleDocument,
        context: CirclePassContext,
    ) -> CirclePassResult:
        """Fuse every supported slice and leave obsolete producers for DCE."""

        del context
        changes = 0
        diagnostics: list[str] = []
        for subgraph_index, _subgraph in enumerate(as_list(document.model.subgraphs)):
            operator_index = 0
            while True:
                operators = as_list(document.subgraph(subgraph_index).operators)
                if operator_index >= len(operators):
                    break
                graph = CircleGraph(document.model, subgraph_index)
                operator = operators[operator_index]
                if (
                    operator_builtin_code(document.model, operator)
                    != self.codes["SLICE"]
                ):
                    operator_index += 1
                    continue
                if self._fuse(document, graph, operator_index):
                    changes += 1
                    diagnostics.append(
                        "Fused TRANSPOSE_CONV and SLICE at "
                        f"subgraphs[{subgraph_index}].operators[{operator_index}]."
                    )
                operator_index += 1
        return CirclePassResult(
            modified=changes > 0,
            changes=changes,
            diagnostics=tuple(diagnostics),
        )

    def _fuse(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        slice_index: int,
    ) -> bool:
        """Replace one slice with a new TConv carrying folded crop parameters."""

        operators = as_list(graph.subgraph.operators)
        slice_operator = operators[slice_index]
        if not operator_is_plain(slice_operator):
            return False
        if int(getattr(slice_operator, "builtinOptionsType", 0) or 0) != (
            self.options_types["SliceOptions"]
        ):
            return False
        slice_inputs = tuple(as_indices(slice_operator.inputs))
        slice_outputs = tuple(as_indices(slice_operator.outputs))
        if len(slice_inputs) != 3 or len(slice_outputs) != 1:
            return False
        tconv_output, begin_index, size_index = slice_inputs
        tconv_index = graph.producer(tconv_output)
        if tconv_index is None or tconv_output in graph.outputs:
            return False
        if self.policy.require_single_consumer and graph.consumers(tconv_output) != (
            slice_index,
        ):
            return False
        tconv = operators[tconv_index]
        if operator_builtin_code(document.model, tconv) != self.codes["TRANSPOSE_CONV"]:
            return False
        if not operator_is_plain(tconv):
            return False
        if (
            int(getattr(tconv, "builtinOptionsType", 0) or 0)
            != self.options_types["TransposeConvOptions"]
        ):
            return False
        tconv_inputs = tuple(as_indices(tconv.inputs))
        tconv_outputs = tuple(as_indices(tconv.outputs))
        if len(tconv_inputs) != 4 or tconv_outputs != (tconv_output,):
            return False
        shape_index, filter_index, data_index, bias_index = tconv_inputs
        constant_indices = (
            shape_index,
            filter_index,
            bias_index,
            begin_index,
            size_index,
        )
        if not all(
            is_constant_tensor(document.model, graph.subgraph, tensor_index)
            for tensor_index in constant_indices
        ):
            return False

        values = {
            tensor_index: self._decode(document, graph, tensor_index)
            for tensor_index in constant_indices
        }
        if any(value is None for value in values.values()):
            return False
        shape_value = values[shape_index]
        filter_value = values[filter_index]
        bias_value = values[bias_index]
        begin_value = values[begin_index]
        size_value = values[size_index]
        assert shape_value is not None
        assert filter_value is not None
        assert bias_value is not None
        assert begin_value is not None
        assert size_value is not None

        if shape_value.tensor_type != self.int32_type or shape_value.shape != (4,):
            return False
        if begin_value.tensor_type != self.int32_type or begin_value.shape != (4,):
            return False
        if size_value.tensor_type != self.int32_type or size_value.shape != (4,):
            return False
        if filter_value.tensor_type != self.float32_type or filter_value.data.ndim != 4:
            return False
        if bias_value.tensor_type != self.float32_type or bias_value.data.ndim != 1:
            return False
        if filter_value.quantization is not None or bias_value.quantization is not None:
            return False

        pre_shape = tuple(int(value) for value in shape_value.data.reshape(-1))
        begin = tuple(int(value) for value in begin_value.data.reshape(-1))
        output_shape = tuple(int(value) for value in size_value.data.reshape(-1))
        if any(dimension <= 0 for dimension in pre_shape + output_shape):
            return False
        if begin[0] != 0 or output_shape[0] != pre_shape[0]:
            return False
        if any(offset < 0 for offset in begin):
            return False
        if any(begin[axis] + output_shape[axis] > pre_shape[axis] for axis in range(4)):
            return False

        data_contract = tensor_contract(graph, data_index)
        intermediate_contract = tensor_contract(graph, tconv_output)
        final_contract = tensor_contract(graph, slice_outputs[0])
        if not all(
            _supported_float32(contract, self.float32_type)
            for contract in (data_contract, intermediate_contract, final_contract)
        ):
            return False
        if (
            data_contract.rank != 4
            or intermediate_contract.shape != pre_shape
            or final_contract.shape != output_shape
        ):
            return False

        filter_shape = tuple(int(value) for value in filter_value.shape)
        original_filter_shape = filter_shape
        if filter_shape[0] != pre_shape[3]:
            return False
        if filter_shape[3] != data_contract.shape[3]:
            return False
        if bias_value.shape != (pre_shape[3],):
            return False
        options = getattr(tconv, "builtinOptions", None)
        if options is None:
            return False
        stride_h = int(getattr(options, "strideH", 0) or 0)
        stride_w = int(getattr(options, "strideW", 0) or 0)
        if stride_h <= 0 or stride_w <= 0:
            return False
        source_padding = _padding_name(
            int(getattr(options, "padding", -1)),
            self.padding_values,
        )
        if source_padding is None:
            return False
        if (
            _transpose_conv_input_size(
                source_padding,
                pre_shape[1],
                filter_shape[1],
                stride_h,
            )
            != data_contract.shape[1]
            or _transpose_conv_input_size(
                source_padding,
                pre_shape[2],
                filter_shape[2],
                stride_w,
            )
            != data_contract.shape[2]
        ):
            return False
        padding = _find_representable_padding(
            input_height=data_contract.shape[1],
            input_width=data_contract.shape[2],
            output_height=output_shape[1],
            output_width=output_shape[2],
            filter_height=filter_shape[1],
            filter_width=filter_shape[2],
            stride_h=stride_h,
            stride_w=stride_w,
            crop_top=begin[1],
            crop_left=begin[2],
            padding_values=self.padding_values,
        )
        if padding is None:
            return False

        channel_offset = begin[3]
        output_channels = output_shape[3]
        if channel_offset != 0 or output_channels != pre_shape[3]:
            if not self.policy.support_channel_crop:
                return False
            if channel_offset + output_channels > pre_shape[3]:
                return False
            filter_value = TensorValue(
                tensor_type=filter_value.tensor_type,
                shape=(
                    output_channels,
                    filter_shape[1],
                    filter_shape[2],
                    filter_shape[3],
                ),
                data=filter_value.data[
                    channel_offset : channel_offset + output_channels,
                    :,
                    :,
                    :,
                ],
                quantization=None,
            )
            bias_value = TensorValue(
                tensor_type=bias_value.tensor_type,
                shape=(output_channels,),
                data=bias_value.data[channel_offset : channel_offset + output_channels],
                quantization=None,
            )

        final_shape_value = TensorValue.from_values(
            self.int32_type,
            np.asarray(output_shape, dtype=np.int32),
            dtype=np.int32,
        )
        checkpoint = AppendedObjectCheckpoint.capture(
            document,
            subgraph_index=graph.subgraph_index,
        )
        builder = CircleBuilder(
            document,
            subgraph_index=graph.subgraph_index,
            codec=self.codec,
            object_factory=self.object_factory,
        )
        try:
            final_shape_index = builder.add_constant(
                _derived_name(graph, slice_outputs[0], "tconv_shape"),
                final_shape_value,
            )
            final_filter_index = filter_index
            final_bias_index = bias_index
            if filter_value.shape != original_filter_shape:
                final_filter_index = builder.add_constant(
                    _derived_name(graph, slice_outputs[0], "tconv_filter"),
                    filter_value,
                )
                final_bias_index = builder.add_constant(
                    _derived_name(graph, slice_outputs[0], "tconv_bias"),
                    bias_value,
                )
            fused_options = clone_object(options)
            fused_options.padding = padding
            replacement = builder.make_operator(
                self.codes["TRANSPOSE_CONV"],
                inputs=(
                    final_shape_index,
                    final_filter_index,
                    data_index,
                    final_bias_index,
                ),
                outputs=slice_outputs,
                version=operator_version(document.model, tconv),
                builtin_options_type=self.options_types["TransposeConvOptions"],
                builtin_options=fused_options,
            )
            builder.replace_operator(slice_index, replacement)
        except Exception:
            checkpoint.rollback(document)
            raise
        return True

    def _decode(
        self,
        document: CircleDocument,
        graph: CircleGraph,
        tensor_index: int,
    ) -> TensorValue | None:
        """Decode one inline constant or return None for unsupported storage."""

        try:
            return self.codec.decode_tensor(
                document.model,
                subgraph_index=graph.subgraph_index,
                tensor_index=tensor_index,
            )
        except (CircleValueError, IndexError, ValueError):
            return None


def _find_representable_padding(
    *,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    filter_height: int,
    filter_width: int,
    stride_h: int,
    stride_w: int,
    crop_top: int,
    crop_left: int,
    padding_values: Mapping[str, int],
) -> int | None:
    """Find SAME or VALID whose implicit crop matches the requested top-left crop."""

    for name in ("VALID", "SAME"):
        candidate_input_h = _transpose_conv_input_size(
            name,
            output_height,
            filter_height,
            stride_h,
        )
        candidate_input_w = _transpose_conv_input_size(
            name,
            output_width,
            filter_width,
            stride_w,
        )
        if candidate_input_h != input_height or candidate_input_w != input_width:
            continue
        candidate_top = max(
            0,
            ((input_height - 1) * stride_h + filter_height - output_height) // 2,
        )
        candidate_left = max(
            0,
            ((input_width - 1) * stride_w + filter_width - output_width) // 2,
        )
        if candidate_top == crop_top and candidate_left == crop_left:
            return int(padding_values[name])
    return None


def _padding_name(
    padding_value: int,
    padding_values: Mapping[str, int],
) -> str | None:
    """Return the symbolic SAME or VALID name for one configured enum value."""

    for name in ("SAME", "VALID"):
        if int(padding_values[name]) == int(padding_value):
            return name
    return None


def _transpose_conv_input_size(
    padding_name: str,
    output_size: int,
    filter_size: int,
    stride: int,
) -> int:
    """Solve the static TConv output-size relation for one input dimension."""

    if padding_name == "SAME":
        return (output_size + stride - 1) // stride
    if padding_name == "VALID":
        return (output_size + stride - filter_size) // stride
    raise ValueError(f"Unsupported padding name: {padding_name}.")


def _supported_float32(contract: TensorContract, float32_type: int) -> bool:
    """Return whether one contract is static, dense, immutable, and plain FLOAT32."""

    signature = contract.shape_signature
    return (
        contract.tensor_type == float32_type
        and (signature is None or all(dimension >= 0 for dimension in signature))
        and not contract.is_variable
        and contract.sparsity is None
        and contract.variant_tensors is None
        and contract.quantization is None
    )


def _derived_name(graph: CircleGraph, tensor_index: int, suffix: str) -> str:
    """Create a stable constant name from a preserved output tensor."""

    tensors = as_list(graph.subgraph.tensors)
    raw_name = getattr(tensors[tensor_index], "name", None)
    if isinstance(raw_name, bytes):
        name = raw_name.decode("utf-8", errors="replace")
    else:
        name = str(raw_name or f"tensor_{tensor_index}")
    return f"{name}/{suffix}"


def _schema_enum_value(enum_name: str, member_name: str) -> int:
    """Return one generated Circle enum member by symbolic name."""

    schema = circle_schema()
    module = getattr(schema, enum_name, None)
    enum_type = getattr(module, enum_name, None) if module is not None else None
    if enum_type is None:
        enum_type = module
    if enum_type is None or not hasattr(enum_type, member_name):
        raise RuntimeError(f"Circle schema does not provide {enum_name}.{member_name}.")
    return int(getattr(enum_type, member_name))


def _maybe_schema_enum_value(
    enum_name: str,
    member_name: str,
    fallback: int,
) -> int:
    """Return a generated enum value or a stable legacy fallback."""

    try:
        return _schema_enum_value(enum_name, member_name)
    except (AttributeError, ImportError, RuntimeError):
        return int(fallback)


__all__ = [
    "FuseTransposeConvSlicePass",
    "TransposeConvSliceFusionPolicy",
]

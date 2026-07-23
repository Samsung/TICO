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

from typing import Any

from tico.circle._schema import decode_text, enum_name
from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list, is_constant_tensor
from tico.circle.inspect.summary import CircleModelSummary, summarize_document


def _enum_name_or_value(enum_type: str, value: int) -> str:
    try:
        return enum_name(enum_type, value)
    except (ImportError, ModuleNotFoundError, RuntimeError):
        return str(value)


def _operator_code_name(model: Any, opcode_index: int) -> str:
    operator_codes = as_list(model.operatorCodes)
    if opcode_index < 0 or opcode_index >= len(operator_codes):
        return f"<invalid:{opcode_index}>"
    code = operator_codes[opcode_index]
    custom_code = decode_text(getattr(code, "customCode", ""))
    if custom_code:
        return custom_code
    builtin_code = int(getattr(code, "builtinCode", -1))
    return _enum_name_or_value("BuiltinOperator", builtin_code)


def format_summary(summary: CircleModelSummary) -> str:
    """Format a model summary as compact human-readable text."""

    lines = [
        "Circle model",
        f"  source: {summary.source or '<memory>'}",
        f"  version: {summary.version}",
        f"  description: {summary.description or '<none>'}",
        f"  subgraphs: {len(summary.subgraphs)}",
        f"  operator codes: {summary.operator_codes}",
        f"  buffers: {summary.buffers} "
        f"({summary.buffers_with_payload} with payload, "
        f"{summary.buffer_bytes} payload bytes)",
        f"  signatures: {summary.signatures}",
        f"  metadata entries: {summary.metadata_entries}",
    ]
    for subgraph in summary.subgraphs:
        lines.extend(
            (
                f"  subgraph {subgraph.index}: {subgraph.name or '<unnamed>'}",
                f"    inputs/outputs: {subgraph.inputs}/{subgraph.outputs}",
                f"    tensors: {subgraph.tensors} "
                f"({subgraph.constant_tensors} constant)",
                f"    operators: {subgraph.operators}",
            )
        )
    return "\n".join(lines)


def format_document(
    document: CircleDocument,
    *,
    subgraph_index: int | None = None,
    include_tensors: bool = False,
    include_operators: bool = False,
) -> str:
    """Format a Circle document with optional tensor and operator details."""

    summary = summarize_document(document)
    lines = [format_summary(summary)]
    if not include_tensors and not include_operators:
        return lines[0]

    indices = (
        (subgraph_index,)
        if subgraph_index is not None
        else tuple(range(document.subgraph_count))
    )
    for index in indices:
        subgraph = document.subgraph(index)
        lines.append("")
        lines.append(f"Subgraph {index} details")
        if include_tensors:
            lines.append("  Tensors")
            for tensor_index, tensor in enumerate(as_list(subgraph.tensors)):
                name = decode_text(getattr(tensor, "name", "")) or "<unnamed>"
                shape = [
                    int(dimension)
                    for dimension in as_list(getattr(tensor, "shape", None))
                ]
                tensor_type = _enum_name_or_value(
                    "TensorType", int(getattr(tensor, "type", -1))
                )
                buffer_index = int(getattr(tensor, "buffer", 0) or 0)
                constant = is_constant_tensor(document.model, subgraph, tensor_index)
                lines.append(
                    f"    [{tensor_index}] {name}: shape={shape}, type={tensor_type}, "
                    f"buffer={buffer_index}, constant={constant}"
                )
        if include_operators:
            lines.append("  Operators")
            for operator_index, operator in enumerate(as_list(subgraph.operators)):
                opcode_index = int(getattr(operator, "opcodeIndex", -1))
                operator_name = _operator_code_name(document.model, opcode_index)
                inputs = as_indices(getattr(operator, "inputs", None))
                outputs = as_indices(getattr(operator, "outputs", None))
                lines.append(
                    f"    [{operator_index}] {operator_name} "
                    f"(opcode={opcode_index}): inputs={inputs}, outputs={outputs}"
                )
    return "\n".join(lines)

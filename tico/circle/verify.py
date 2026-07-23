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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from tico.circle._schema import decode_text
from tico.circle.errors import CircleError
from tico.circle.graph import (
    as_indices,
    as_list,
    has_buffer_payload,
    is_constant_tensor,
    OPTIONAL_TENSOR_INDEX,
)
from tico.circle.rewrite import iter_subgraph_references


class VerificationSeverity(str, Enum):
    """Classify a verification issue as an error or warning."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class VerificationIssue:
    """Describe one structural verification finding."""

    severity: VerificationSeverity
    code: str
    path: str
    message: str

    def format(self) -> str:
        """Format the issue as one human-readable line."""

        return (
            f"{self.severity.value.upper()} [{self.code}] "
            f"{self.path}: {self.message}"
        )


@dataclass(frozen=True)
class VerificationReport:
    """Collect internal-consistency errors and warnings for a Circle document."""

    issues: tuple[VerificationIssue, ...]

    @property
    def errors(self) -> tuple[VerificationIssue, ...]:
        """Return error findings."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity is VerificationSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[VerificationIssue, ...]:
        """Return warning findings."""

        return tuple(
            issue
            for issue in self.issues
            if issue.severity is VerificationSeverity.WARNING
        )

    @property
    def ok(self) -> bool:
        """Return whether the report contains no errors."""

        return not self.errors

    def format(self) -> str:
        """Format all findings as a multi-line diagnostic."""

        if not self.issues:
            return "Circle internal-consistency verification succeeded."
        return "\n".join(issue.format() for issue in self.issues)


class CircleVerificationError(CircleError):
    """Raised when Circle internal-consistency verification finds errors."""

    def __init__(self, report: VerificationReport):
        self.report = report
        super().__init__(report.format())


def _model_from_document(document_or_model: Any) -> Any:
    return getattr(document_or_model, "model", document_or_model)


def _issue(
    issues: list[VerificationIssue],
    severity: VerificationSeverity,
    code: str,
    path: str,
    message: str,
) -> None:
    issues.append(VerificationIssue(severity, code, path, message))


def _check_index(
    issues: list[VerificationIssue],
    index: int,
    size: int,
    path: str,
    *,
    optional: bool = False,
) -> bool:
    if optional and index == OPTIONAL_TENSOR_INDEX:
        return True
    if index < 0 or index >= size:
        _issue(
            issues,
            VerificationSeverity.ERROR,
            "INVALID_INDEX",
            path,
            f"Index {index} is outside the valid range 0..{size - 1}.",
        )
        return False
    return True


def _check_duplicates(
    issues: list[VerificationIssue],
    values: Iterable[int],
    path: str,
) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        _issue(
            issues,
            VerificationSeverity.WARNING,
            "DUPLICATE_INDEX",
            path,
            f"Contains duplicate indices {sorted(duplicates)}.",
        )


def verify_document(
    document_or_model: Any,
    *,
    raise_on_error: bool = True,
) -> VerificationReport:
    """Check internal consistency of a Circle document or Object API model.

    The check validates index ranges, graph producer and consumer bookkeeping,
    reserved buffer rules, shape-signature rank consistency, signature mappings,
    metadata buffer references, and subgraph references stored in operator options.
    It does not execute the model, compare numerical outputs, validate every
    operator-specific semantic constraint, or determine backend compatibility.
    """

    model = _model_from_document(document_or_model)
    issues: list[VerificationIssue] = []
    raw_subgraphs = getattr(model, "subgraphs", None)
    raw_buffers = getattr(model, "buffers", None)
    raw_operator_codes = getattr(model, "operatorCodes", None)

    if raw_subgraphs is None:
        _issue(
            issues,
            VerificationSeverity.ERROR,
            "MISSING_SUBGRAPHS",
            "model.subgraphs",
            "The model does not define a subgraph vector.",
        )
    subgraphs = as_list(raw_subgraphs)
    if not subgraphs:
        _issue(
            issues,
            VerificationSeverity.ERROR,
            "EMPTY_SUBGRAPHS",
            "model.subgraphs",
            "The model must contain at least one subgraph.",
        )

    buffers = as_list(raw_buffers)
    if not buffers:
        _issue(
            issues,
            VerificationSeverity.ERROR,
            "MISSING_BUFFER_ZERO",
            "model.buffers",
            "The model must contain buffer 0.",
        )
    elif has_buffer_payload(buffers[0]):
        _issue(
            issues,
            VerificationSeverity.ERROR,
            "BUFFER_ZERO_NOT_EMPTY",
            "model.buffers[0]",
            "Buffer 0 is reserved and must not contain payload data.",
        )

    if raw_operator_codes is None:
        _issue(
            issues,
            VerificationSeverity.ERROR,
            "MISSING_OPERATOR_CODES",
            "model.operatorCodes",
            "The model does not define an operator-code vector.",
        )
    operator_codes = as_list(raw_operator_codes)

    used_buffers: set[int] = {0} if buffers else set()
    used_operator_codes: set[int] = set()

    for subgraph_index, subgraph in enumerate(subgraphs):
        path = f"model.subgraphs[{subgraph_index}]"
        tensors = as_list(getattr(subgraph, "tensors", None))
        operators = as_list(getattr(subgraph, "operators", None))
        inputs = as_indices(getattr(subgraph, "inputs", None))
        outputs = as_indices(getattr(subgraph, "outputs", None))
        _check_duplicates(issues, inputs, f"{path}.inputs")
        _check_duplicates(issues, outputs, f"{path}.outputs")

        tensor_names: dict[str, int] = {}
        for tensor_index, tensor in enumerate(tensors):
            tensor_path = f"{path}.tensors[{tensor_index}]"
            buffer_index = int(getattr(tensor, "buffer", 0) or 0)
            if _check_index(
                issues,
                buffer_index,
                len(buffers),
                f"{tensor_path}.buffer",
            ):
                used_buffers.add(buffer_index)

            name = decode_text(getattr(tensor, "name", ""))
            if name:
                previous = tensor_names.get(name)
                if previous is not None:
                    _issue(
                        issues,
                        VerificationSeverity.WARNING,
                        "DUPLICATE_TENSOR_NAME",
                        f"{tensor_path}.name",
                        f"Tensor name {name!r} is also used by tensor {previous}.",
                    )
                tensor_names[name] = tensor_index

            shape = getattr(tensor, "shape", None)
            shape_signature = getattr(tensor, "shapeSignature", None)
            if shape is not None and shape_signature is not None:
                if len(shape_signature) > 0 and len(shape) != len(shape_signature):
                    _issue(
                        issues,
                        VerificationSeverity.ERROR,
                        "SHAPE_SIGNATURE_RANK",
                        f"{tensor_path}.shapeSignature",
                        "Shape and shape signature have different ranks.",
                    )

        for field_name, values in (("inputs", inputs), ("outputs", outputs)):
            for position, tensor_index in enumerate(values):
                _check_index(
                    issues,
                    tensor_index,
                    len(tensors),
                    f"{path}.{field_name}[{position}]",
                )

        producers: dict[int, int] = {}
        consumers: dict[int, list[int]] = {}
        for operator_index, operator in enumerate(operators):
            operator_path = f"{path}.operators[{operator_index}]"
            opcode_index = int(getattr(operator, "opcodeIndex", -1))
            if _check_index(
                issues,
                opcode_index,
                len(operator_codes),
                f"{operator_path}.opcodeIndex",
            ):
                used_operator_codes.add(opcode_index)

            for field_name, optional in (
                ("inputs", True),
                ("outputs", False),
                ("intermediates", True),
            ):
                for position, tensor_index in enumerate(
                    as_indices(getattr(operator, field_name, None))
                ):
                    valid = _check_index(
                        issues,
                        tensor_index,
                        len(tensors),
                        f"{operator_path}.{field_name}[{position}]",
                        optional=optional,
                    )
                    if not valid or tensor_index == OPTIONAL_TENSOR_INDEX:
                        continue
                    if field_name == "outputs":
                        previous = producers.get(tensor_index)
                        if previous is not None:
                            _issue(
                                issues,
                                VerificationSeverity.ERROR,
                                "MULTIPLE_PRODUCERS",
                                f"{operator_path}.{field_name}[{position}]",
                                f"Tensor {tensor_index} is already produced by "
                                f"operator {previous}.",
                            )
                        producers[tensor_index] = operator_index
                    elif field_name == "inputs":
                        consumers.setdefault(tensor_index, []).append(operator_index)

        for tensor_index in inputs:
            if tensor_index in producers:
                _issue(
                    issues,
                    VerificationSeverity.WARNING,
                    "INPUT_HAS_PRODUCER",
                    f"{path}.inputs",
                    f"Graph input tensor {tensor_index} is also produced by operator "
                    f"{producers[tensor_index]}.",
                )

        input_set = set(inputs)
        for tensor_index, consumer_indices in sorted(consumers.items()):
            if (
                tensor_index not in producers
                and tensor_index not in input_set
                and not is_constant_tensor(model, subgraph, tensor_index)
            ):
                _issue(
                    issues,
                    VerificationSeverity.ERROR,
                    "UNDEFINED_INPUT",
                    f"{path}.tensors[{tensor_index}]",
                    f"Tensor {tensor_index} is consumed by operators "
                    f"{consumer_indices} but has no producer and is not an "
                    "input or constant.",
                )

        for position, tensor_index in enumerate(outputs):
            if tensor_index < 0 or tensor_index >= len(tensors):
                continue
            if (
                tensor_index not in producers
                and tensor_index not in input_set
                and not is_constant_tensor(model, subgraph, tensor_index)
            ):
                _issue(
                    issues,
                    VerificationSeverity.ERROR,
                    "UNDEFINED_OUTPUT",
                    f"{path}.outputs[{position}]",
                    f"Output tensor {tensor_index} has no producer and is not "
                    "an input or constant.",
                )

        referenced_tensors = set(inputs) | set(outputs)
        referenced_tensors.update(producers)
        referenced_tensors.update(consumers)
        unused_tensors = sorted(set(range(len(tensors))) - referenced_tensors)
        if unused_tensors:
            _issue(
                issues,
                VerificationSeverity.WARNING,
                "UNUSED_TENSORS",
                f"{path}.tensors",
                f"Unused tensor indices: {unused_tensors}.",
            )

    signatures = as_list(getattr(model, "signatureDefs", None))
    signature_keys: set[str] = set()
    for signature_index, signature in enumerate(signatures):
        path = f"model.signatureDefs[{signature_index}]"
        subgraph_index = int(getattr(signature, "subgraphIndex", -1))
        valid_subgraph = _check_index(
            issues,
            subgraph_index,
            len(subgraphs),
            f"{path}.subgraphIndex",
        )
        signature_key = decode_text(getattr(signature, "signatureKey", ""))
        if signature_key and signature_key in signature_keys:
            _issue(
                issues,
                VerificationSeverity.WARNING,
                "DUPLICATE_SIGNATURE_KEY",
                f"{path}.signatureKey",
                f"Signature key {signature_key!r} is duplicated.",
            )
        signature_keys.add(signature_key)
        if not valid_subgraph:
            continue
        tensor_count = len(as_list(subgraphs[subgraph_index].tensors))
        for field_name in ("inputs", "outputs"):
            for map_index, tensor_map in enumerate(
                as_list(getattr(signature, field_name, None))
            ):
                tensor_index = int(getattr(tensor_map, "tensorIndex", -1))
                valid_tensor = _check_index(
                    issues,
                    tensor_index,
                    tensor_count,
                    f"{path}.{field_name}[{map_index}].tensorIndex",
                )
                if valid_tensor:
                    graph_io = set(
                        as_indices(getattr(subgraphs[subgraph_index], field_name, None))
                    )
                    if tensor_index not in graph_io:
                        _issue(
                            issues,
                            VerificationSeverity.ERROR,
                            "SIGNATURE_IO_MISMATCH",
                            f"{path}.{field_name}[{map_index}].tensorIndex",
                            f"Tensor {tensor_index} is not a subgraph "
                            f"{field_name[:-1]}.",
                        )

    for (
        path,
        _container,
        _field_name,
        subgraph_index,
    ) in iter_subgraph_references(model):
        _check_index(issues, subgraph_index, len(subgraphs), path)

    for index, buffer_index in enumerate(
        as_indices(getattr(model, "metadataBuffer", None))
    ):
        if _check_index(
            issues,
            int(buffer_index),
            len(buffers),
            f"model.metadataBuffer[{index}]",
        ):
            used_buffers.add(int(buffer_index))
    for index, metadata in enumerate(as_list(getattr(model, "metadata", None))):
        if not hasattr(metadata, "buffer"):
            continue
        buffer_index = int(metadata.buffer)
        if _check_index(
            issues,
            buffer_index,
            len(buffers),
            f"model.metadata[{index}].buffer",
        ):
            used_buffers.add(buffer_index)

    unused_codes = sorted(set(range(len(operator_codes))) - used_operator_codes)
    if unused_codes:
        _issue(
            issues,
            VerificationSeverity.WARNING,
            "UNUSED_OPERATOR_CODES",
            "model.operatorCodes",
            f"Unused operator-code indices: {unused_codes}.",
        )
    unused_buffers = sorted(set(range(len(buffers))) - used_buffers)
    if unused_buffers:
        _issue(
            issues,
            VerificationSeverity.WARNING,
            "UNUSED_BUFFERS",
            "model.buffers",
            f"Unused buffer indices: {unused_buffers}.",
        )

    report = VerificationReport(tuple(issues))
    if raise_on_error and not report.ok:
        raise CircleVerificationError(report)
    return report

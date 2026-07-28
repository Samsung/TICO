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

import unittest
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tico.circle._schema import decode_text
from tico.circle.document import CircleDocument
from tico.circle.graph import as_indices, as_list

from test.support.circle.evaluator import (
    CircleEvaluationResult,
    CircleReferenceEvaluator,
)


@dataclass(frozen=True)
class TensorContract:
    """Describe the semantically relevant interface fields of one tensor."""

    name: str
    shape: tuple[int, ...]
    shape_signature: tuple[int, ...] | None
    tensor_type: int
    is_variable: bool
    quantization: Any


@dataclass(frozen=True)
class SignatureContract:
    """Describe one signature without relying on mutable tensor indices."""

    key: str
    inputs: tuple[tuple[str, TensorContract], ...]
    outputs: tuple[tuple[str, TensorContract], ...]


@dataclass(frozen=True)
class GraphInterfaceContract:
    """Describe graph input, output, and signature contracts."""

    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    signatures: tuple[SignatureContract, ...]


@dataclass(frozen=True)
class CirclePassValueTestResult:
    """Return transformed artifacts and numerical results from a pass value test."""

    document: CircleDocument
    transform_result: Any
    source_evaluation: CircleEvaluationResult
    transformed_evaluation: CircleEvaluationResult


@dataclass(frozen=True)
class CircleExtractionValueTestResult:
    """Return extraction metadata and numerical results from a value test."""

    extraction_result: Any
    document: CircleDocument
    source_evaluation: CircleEvaluationResult
    extracted_evaluation: CircleEvaluationResult

    @property
    def selected_operator_indices(self) -> tuple[int, ...]:
        """Return the source operator indices selected for extraction."""

        return self.extraction_result.selected_operator_indices

    @property
    def source_boundary(self) -> Any:
        """Return the extraction boundary in the source tensor index space."""

        return self.extraction_result.source_boundary

    @property
    def boundary(self) -> Any:
        """Return the extraction boundary in the compacted tensor index space."""

        return self.extraction_result.boundary

    @property
    def removed_operators(self) -> int:
        """Return the number of operators removed by extraction."""

        return int(self.extraction_result.removed_operators)

    @property
    def rewrite_stats(self) -> Any:
        """Return the source extraction rewrite statistics."""

        return self.extraction_result.rewrite_stats


def _freeze_object(value: Any) -> Any:
    """Convert generated Object API values into recursively comparable data."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, np.ndarray):
        return (str(value.dtype), tuple(value.shape), value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_object(item) for item in value)
    attributes = getattr(value, "__dict__", None)
    if attributes is not None:
        return tuple(
            sorted(
                (name, _freeze_object(attribute))
                for name, attribute in attributes.items()
            )
        )
    return repr(value)


class CircleValueTestCase(unittest.TestCase):
    """Provide reusable numerical assertions for Circle graph transformations."""

    evaluator: CircleReferenceEvaluator

    def setUp(self) -> None:
        """Create a fresh reference evaluator for each test."""

        super().setUp()
        self.evaluator = CircleReferenceEvaluator()

    def round_trip(self, document: CircleDocument) -> CircleDocument:
        """Serialize, deserialize, and structurally verify a Circle document."""

        restored = CircleDocument.from_bytes(document.to_bytes())
        restored.verify(raise_on_error=True)
        return restored

    def assert_outputs_equal(
        self,
        expected: Sequence[np.ndarray],
        actual: Sequence[np.ndarray],
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        """Compare output count, shape, dtype, and numerical values."""

        self.assertEqual(len(expected), len(actual))
        for output_index, (expected_value, actual_value) in enumerate(
            zip(expected, actual)
        ):
            expected_array = np.asarray(expected_value)
            actual_array = np.asarray(actual_value)
            self.assertEqual(
                expected_array.shape,
                actual_array.shape,
                msg=f"Output {output_index} shape mismatch.",
            )
            self.assertEqual(
                expected_array.dtype,
                actual_array.dtype,
                msg=f"Output {output_index} dtype mismatch.",
            )
            if np.issubdtype(expected_array.dtype, np.inexact):
                np.testing.assert_allclose(
                    actual_array,
                    expected_array,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    err_msg=f"Output {output_index} value mismatch.",
                )
            else:
                np.testing.assert_array_equal(
                    actual_array,
                    expected_array,
                    err_msg=f"Output {output_index} value mismatch.",
                )

    def tensor_contract(
        self,
        document: CircleDocument,
        tensor_index: int,
        *,
        subgraph_index: int = 0,
    ) -> TensorContract:
        """Return a stable tensor contract independent of index compaction."""

        tensor = document.subgraph(subgraph_index).tensors[int(tensor_index)]
        raw_shape_signature = getattr(tensor, "shapeSignature", None)
        shape_signature = (
            None
            if raw_shape_signature is None
            else tuple(int(value) for value in as_list(raw_shape_signature))
        )
        return TensorContract(
            name=decode_text(getattr(tensor, "name", "")),
            shape=tuple(
                int(value) for value in as_list(getattr(tensor, "shape", None))
            ),
            shape_signature=shape_signature,
            tensor_type=int(getattr(tensor, "type")),
            is_variable=bool(getattr(tensor, "isVariable", False)),
            quantization=_freeze_object(getattr(tensor, "quantization", None)),
        )

    def graph_interface_contract(
        self,
        document: CircleDocument,
        *,
        subgraph_index: int = 0,
    ) -> GraphInterfaceContract:
        """Return graph interface semantics without using raw tensor indices."""

        subgraph = document.subgraph(subgraph_index)
        inputs = tuple(
            self.tensor_contract(
                document,
                tensor_index,
                subgraph_index=subgraph_index,
            )
            for tensor_index in as_indices(getattr(subgraph, "inputs", None))
        )
        outputs = tuple(
            self.tensor_contract(
                document,
                tensor_index,
                subgraph_index=subgraph_index,
            )
            for tensor_index in as_indices(getattr(subgraph, "outputs", None))
        )

        signatures: list[SignatureContract] = []
        for signature in as_list(getattr(document.model, "signatureDefs", None)):
            if int(getattr(signature, "subgraphIndex", -1)) != subgraph_index:
                continue
            signature_inputs = tuple(
                (
                    decode_text(getattr(tensor_map, "name", "")),
                    self.tensor_contract(
                        document,
                        int(getattr(tensor_map, "tensorIndex")),
                        subgraph_index=subgraph_index,
                    ),
                )
                for tensor_map in as_list(getattr(signature, "inputs", None))
            )
            signature_outputs = tuple(
                (
                    decode_text(getattr(tensor_map, "name", "")),
                    self.tensor_contract(
                        document,
                        int(getattr(tensor_map, "tensorIndex")),
                        subgraph_index=subgraph_index,
                    ),
                )
                for tensor_map in as_list(getattr(signature, "outputs", None))
            )
            signatures.append(
                SignatureContract(
                    key=decode_text(getattr(signature, "signatureKey", "")),
                    inputs=signature_inputs,
                    outputs=signature_outputs,
                )
            )

        return GraphInterfaceContract(
            inputs=inputs,
            outputs=outputs,
            signatures=tuple(signatures),
        )

    def assert_interfaces_equal(
        self,
        expected_document: CircleDocument,
        actual_document: CircleDocument,
        *,
        subgraph_index: int = 0,
        check_tensor_names: bool = False,
    ) -> None:
        """Assert that two documents expose equivalent graph interfaces."""

        expected = self.graph_interface_contract(
            expected_document,
            subgraph_index=subgraph_index,
        )
        actual = self.graph_interface_contract(
            actual_document,
            subgraph_index=subgraph_index,
        )
        self.assertEqual(len(expected.inputs), len(actual.inputs))
        self.assertEqual(len(expected.outputs), len(actual.outputs))

        for expected_tensor, actual_tensor in zip(expected.inputs, actual.inputs):
            self._assert_tensor_contracts_equal(
                expected_tensor,
                actual_tensor,
                check_name=check_tensor_names,
            )
        for expected_tensor, actual_tensor in zip(expected.outputs, actual.outputs):
            self._assert_tensor_contracts_equal(
                expected_tensor,
                actual_tensor,
                check_name=check_tensor_names,
            )

        self.assertEqual(len(expected.signatures), len(actual.signatures))
        for expected_signature, actual_signature in zip(
            expected.signatures,
            actual.signatures,
        ):
            self.assertEqual(expected_signature.key, actual_signature.key)
            self.assertEqual(
                [name for name, _ in expected_signature.inputs],
                [name for name, _ in actual_signature.inputs],
            )
            self.assertEqual(
                [name for name, _ in expected_signature.outputs],
                [name for name, _ in actual_signature.outputs],
            )
            for (_, expected_tensor), (_, actual_tensor) in zip(
                expected_signature.inputs,
                actual_signature.inputs,
            ):
                self._assert_tensor_contracts_equal(
                    expected_tensor,
                    actual_tensor,
                    check_name=check_tensor_names,
                )
            for (_, expected_tensor), (_, actual_tensor) in zip(
                expected_signature.outputs,
                actual_signature.outputs,
            ):
                self._assert_tensor_contracts_equal(
                    expected_tensor,
                    actual_tensor,
                    check_name=check_tensor_names,
                )

    def assert_tensor_contract_equal(
        self,
        expected_document: CircleDocument,
        expected_tensor_index: int,
        actual_document: CircleDocument,
        actual_tensor_index: int,
        *,
        expected_subgraph_index: int = 0,
        actual_subgraph_index: int = 0,
        check_name: bool = True,
    ) -> None:
        """Compare two tensor contracts across rewritten index spaces."""

        self._assert_tensor_contracts_equal(
            self.tensor_contract(
                expected_document,
                expected_tensor_index,
                subgraph_index=expected_subgraph_index,
            ),
            self.tensor_contract(
                actual_document,
                actual_tensor_index,
                subgraph_index=actual_subgraph_index,
            ),
            check_name=check_name,
        )

    def assert_pass_preserves_value(
        self,
        source: CircleDocument,
        inputs: tuple[np.ndarray, ...],
        transform: Callable[[CircleDocument], Any],
        *,
        expected_outputs: Sequence[np.ndarray],
        expected_modified: bool = True,
        expected_changes: int | None = None,
        compare_interface: bool = True,
        check_interface_tensor_names: bool = False,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> CirclePassValueTestResult:
        """Assert that a Circle transformation preserves a numerical golden."""

        source = self.round_trip(source)
        source_evaluation = self.evaluator.evaluate(source, inputs)
        self.assert_outputs_equal(
            expected_outputs,
            source_evaluation.outputs,
            rtol=rtol,
            atol=atol,
        )

        transformed = source.clone()
        transform_result = transform(transformed)
        if hasattr(transform_result, "modified"):
            self.assertEqual(bool(transform_result.modified), expected_modified)
        elif expected_modified:
            self.fail("Transformation result does not expose a modified property.")
        if expected_changes is not None:
            self.assertEqual(int(transform_result.changes), expected_changes)

        transformed = self.round_trip(transformed)
        if compare_interface:
            self.assert_interfaces_equal(
                source,
                transformed,
                check_tensor_names=check_interface_tensor_names,
            )

        transformed_evaluation = self.evaluator.evaluate(transformed, inputs)
        self.assert_outputs_equal(
            expected_outputs,
            transformed_evaluation.outputs,
            rtol=rtol,
            atol=atol,
        )
        self.assert_outputs_equal(
            source_evaluation.outputs,
            transformed_evaluation.outputs,
            rtol=rtol,
            atol=atol,
        )
        return CirclePassValueTestResult(
            document=transformed,
            transform_result=transform_result,
            source_evaluation=source_evaluation,
            transformed_evaluation=transformed_evaluation,
        )

    def assert_extraction_preserves_value(
        self,
        source: CircleDocument,
        inputs: tuple[np.ndarray, ...],
        extract: Callable[[CircleDocument], Any],
        *,
        expected_source_outputs: Sequence[np.ndarray] | None = None,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> CircleExtractionValueTestResult:
        """Assert that an extracted graph reproduces its source boundary values."""

        source = self.round_trip(source)
        source_evaluation = self.evaluator.evaluate(source, inputs)
        if expected_source_outputs is not None:
            self.assert_outputs_equal(
                expected_source_outputs,
                source_evaluation.outputs,
                rtol=rtol,
                atol=atol,
            )

        extraction_result = extract(source)
        extracted = self.round_trip(extraction_result.document)
        extracted_inputs = tuple(
            source_evaluation.tensor_values[tensor_index]
            for tensor_index in extraction_result.source_boundary.inputs
        )
        expected_outputs = tuple(
            source_evaluation.tensor_values[tensor_index]
            for tensor_index in extraction_result.source_boundary.outputs
        )
        extracted_evaluation = self.evaluator.evaluate(extracted, extracted_inputs)
        self.assert_outputs_equal(
            expected_outputs,
            extracted_evaluation.outputs,
            rtol=rtol,
            atol=atol,
        )

        self.assertEqual(
            len(extraction_result.source_boundary.inputs),
            len(extraction_result.boundary.inputs),
        )
        self.assertEqual(
            len(extraction_result.source_boundary.outputs),
            len(extraction_result.boundary.outputs),
        )
        for source_index, extracted_index in zip(
            extraction_result.source_boundary.inputs,
            extraction_result.boundary.inputs,
        ):
            self.assert_tensor_contract_equal(
                source,
                source_index,
                extracted,
                extracted_index,
            )
        for source_index, extracted_index in zip(
            extraction_result.source_boundary.outputs,
            extraction_result.boundary.outputs,
        ):
            self.assert_tensor_contract_equal(
                source,
                source_index,
                extracted,
                extracted_index,
            )

        return CircleExtractionValueTestResult(
            extraction_result=extraction_result,
            document=extracted,
            source_evaluation=source_evaluation,
            extracted_evaluation=extracted_evaluation,
        )

    def _assert_tensor_contracts_equal(
        self,
        expected: TensorContract,
        actual: TensorContract,
        *,
        check_name: bool,
    ) -> None:
        """Compare tensor contract fields with optional name checking."""

        if check_name:
            self.assertEqual(expected.name, actual.name)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.shape_signature, actual.shape_signature)
        self.assertEqual(expected.tensor_type, actual.tensor_type)
        self.assertEqual(expected.is_variable, actual.is_variable)
        self.assertEqual(expected.quantization, actual.quantization)

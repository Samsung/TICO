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

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.export import _interface_snapshot, optimize_for_export
from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.circle.passes.manager import CirclePassManager
from tico.circle.passes.optimization.simplify.identity_ops import (
    EliminateIdentityOpsPass,
)
from tico.circle.passes.optimization.simplify.views import SimplifyViewOpsPass
from tico.config import CompileConfigV1, get_default_config

from test.unit_test.circle.export_fixture import (
    add_signature,
    codec,
    CODES,
    constant,
    empty_document,
    factory,
    mul_document,
    operator,
    OPTIONS,
    tensor,
    TYPES,
)


def identity_pass():
    return EliminateIdentityOpsPass(
        builtin_codes=CODES,
        activation_none=0,
        codec=codec(),
        object_factory=factory,
    )


def view_pass():
    return SimplifyViewOpsPass(
        builtin_codes=CODES,
        builtin_options_types=OPTIONS,
        tensor_types=TYPES,
        codec=codec(),
        object_factory=factory,
    )


def run_pass(document, circle_pass, preserve=True):
    return CirclePassManager([circle_pass]).run(
        document, CirclePassContext(preserve_io=preserve)
    )


# (numpy dtype, Circle TensorType) pairs covering every float width.
FLOAT_DTYPES: tuple[tuple[type[np.floating], int], ...] = (
    (np.float16, 1),
    (np.float32, 0),
    (np.float64, 10),
)


def add(lhs: Any, rhs: Any) -> Any:
    return np.add(lhs, rhs)


class TestExportConfig(unittest.TestCase):
    def test_default_enabled(self):
        self.assertTrue(CompileConfigV1().circle_optimize)
        self.assertTrue(get_default_config().get("circle_optimize"))

    def test_dict_round_trip(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = CompileConfigV1.from_dict({"circle_optimize": enabled})
                self.assertIs(config.get("circle_optimize"), enabled)
                self.assertIs(config.to_dict()["circle_optimize"], enabled)

    def test_old_dict_inherits_default(self):
        self.assertTrue(CompileConfigV1.from_dict({}).get("circle_optimize"))

    def test_standalone_context_preserves_old_policy(self):
        self.assertFalse(CirclePassContext().preserve_io)


class TestMulOne(unittest.TestCase):
    def test_removes_internal_float_mul_in_either_order(self):
        for dtype, tensor_type in FLOAT_DTYPES:
            for reverse in (False, True):
                with self.subTest(dtype=dtype, reverse=reverse):
                    document = mul_document(
                        dtype=dtype, tensor_type=tensor_type, reverse=reverse
                    )
                    before = _interface_snapshot(document)
                    self.assertTrue(run_pass(document, identity_pass()).modified)
                    self.assertEqual(len(document.subgraph().operators), 1)
                    self.assertEqual(document.subgraph().operators[0].inputs, [0])
                    self.assertEqual(_interface_snapshot(document), before)
                    self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_numpy_results_match_before_and_after_rewrite(self):
        for dtype, tensor_type in FLOAT_DTYPES:
            with self.subTest(dtype=dtype):
                document = mul_document(dtype=dtype, tensor_type=tensor_type)
                x = np.array(
                    [
                        [0.0, -0.0, np.finfo(dtype).tiny],
                        [-np.finfo(dtype).tiny, np.inf, np.nan],
                    ],
                    dtype=dtype,
                )
                expected = -(x * np.array(1.0, dtype=dtype))
                run_pass(document, identity_pass())
                self.assertEqual(document.subgraph().operators[0].inputs, [0])
                actual = -x
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(
                    np.signbit(actual[:, :2]), np.signbit(expected[:, :2])
                )

    def test_retains_public_output_and_signature(self):
        document = mul_document(public_mul=True)
        add_signature(document)
        before = _interface_snapshot(document)
        self.assertFalse(run_pass(document, identity_pass()).modified)
        self.assertEqual(_interface_snapshot(document), before)

    def test_standalone_may_bypass_public_output(self):
        document = mul_document(public_mul=True)
        self.assertTrue(run_pass(document, identity_pass(), preserve=False).modified)
        self.assertEqual(document.subgraph().outputs, [0])

    def test_exact_one_not_approximate(self):
        for value in (
            0.0,
            -1.0,
            np.nextafter(np.float32(1), np.float32(2)),
            1.001,
            np.nan,
            np.inf,
            -np.inf,
        ):
            with self.subTest(value=value):
                document = mul_document(value)
                self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_scalar_shapes_without_broadcast_expansion(self):
        for shape in ((), (1,), (1, 1)):
            with self.subTest(shape=shape):
                document = mul_document(constant_shape=shape)
                self.assertTrue(run_pass(document, identity_pass()).modified)
        document = mul_document(constant_shape=(1, 1, 1))
        for t in document.subgraph().tensors[1:3]:
            t.shape = t.shapeSignature = [1, 2, 3]
        self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_nonscalar_ones(self):
        document = mul_document(constant_shape=(2, 3))
        self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_integer_and_quantized_mul(self):
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                document = mul_document(
                    2 if quantized else 1, dtype=np.int8, tensor_type=TYPES["INT8"]
                )
                if quantized:
                    qparams = factory("QuantizationParameters")
                    qparams.scale, qparams.zeroPoint = [0.5], [0]
                    for t in document.subgraph().tensors:
                        t.quantization = copy.deepcopy(qparams)
                self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_fused_activation(self):
        document = mul_document()
        document.subgraph().operators[0].builtinOptions.fusedActivationFunction = 1
        self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_dynamic_shape(self):
        document = mul_document()
        for t in document.subgraph().tensors[:3]:
            t.shape = [1, 3]
            t.shapeSignature = [-1, 3]
        self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_unsupported_tensor_contracts(self):
        contracts = (
            ("isVariable", True),
            ("sparsity", NS(traversalOrder=[0, 1], blockMap=[], dimMetadata=[])),
            ("variantTensors", [NS(type=0, shape=[1])]),
        )
        for field_name, value in contracts:
            with self.subTest(field_name=field_name):
                document = mul_document()
                for t in document.subgraph().tensors[:3]:
                    setattr(t, field_name, copy.deepcopy(value))
                self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_mismatched_dtype(self):
        document = mul_document()
        document.subgraph().tensors[3].type = TYPES["INT32"]
        self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_mutating_operator(self):
        document = mul_document()
        document.subgraph().operators[0].mutatingVariableInputs = [True, False]
        self.assertFalse(run_pass(document, identity_pass()).modified)

    def test_retains_two_constant_operands_for_constant_folder(self):
        document = mul_document()
        document.subgraph().inputs = []
        document.subgraph().tensors[0].buffer = document.subgraph().tensors[3].buffer
        document.subgraph().tensors[0].shape = [1]
        document.subgraph().tensors[0].shapeSignature = [1]
        self.assertFalse(run_pass(document, identity_pass()).modified)


class TestExportBoundaries(unittest.TestCase):
    def test_existing_identity_rules_keep_public_outputs(self):
        for name in ("ADD", "CAST", "SLICE", "STRIDED_SLICE", "SPLIT", "SPLIT_V"):
            with self.subTest(name=name):
                document = empty_document()
                graph = document.subgraph()
                graph.tensors = [tensor("x"), tensor("named_output")]
                graph.inputs, graph.outputs = [0], [1]
                if name == "ADD":
                    inputs = [0, constant(document, "zero", 0.0)]
                    options = {}
                elif name == "CAST":
                    inputs, options = [0], {"inDataType": 0, "outDataType": 0}
                elif name == "SLICE":
                    inputs = [
                        0,
                        constant(document, "begin", [0, 0], np.int32, 2),
                        constant(document, "size", [2, 3], np.int32, 2),
                    ]
                    options = {}
                elif name == "STRIDED_SLICE":
                    inputs = [
                        0,
                        constant(document, "begin", [0, 0], np.int32, 2),
                        constant(document, "end", [2, 3], np.int32, 2),
                        constant(document, "strides", [1, 1], np.int32, 2),
                    ]
                    options = {}
                else:
                    axis = constant(document, "axis", 0, np.int32, 2)
                    if name == "SPLIT":
                        inputs = [axis, 0]
                    else:
                        size = constant(document, "size", [2], np.int32, 2)
                        inputs = [0, size, axis]
                    options = {"numSplits": 1}
                operator(document, name, inputs, [1], **options)
                add_signature(document)
                before = _interface_snapshot(document)
                self.assertFalse(run_pass(document, identity_pass()).modified)
                self.assertEqual(_interface_snapshot(document), before)
                # Verify the guard, not a fixture that never matched in either mode.
                self.assertTrue(
                    run_pass(document, identity_pass(), preserve=False).modified
                )

    def test_identity_views_keep_public_outputs(self):
        for name in ("RESHAPE", "TRANSPOSE"):
            with self.subTest(name=name):
                document = empty_document()
                graph = document.subgraph()
                graph.tensors = [tensor("x"), tensor("output")]
                values = [2, 3] if name == "RESHAPE" else [0, 1]
                param = constant(document, "param", values, np.int32, 2)
                operator(document, name, [0, param], [1])
                graph.inputs, graph.outputs = [0], [1]
                add_signature(document)
                before = _interface_snapshot(document)
                self.assertFalse(run_pass(document, view_pass()).modified)
                self.assertEqual(_interface_snapshot(document), before)
                self.assertTrue(
                    run_pass(document, view_pass(), preserve=False).modified
                )

    def test_inverse_transpose_pair_keeps_output(self):
        document = empty_document()
        graph = document.subgraph()
        graph.tensors = [tensor("x"), tensor("transposed", (3, 2)), tensor("output")]
        permutation = constant(document, "permutation", [1, 0], np.int32, 2)
        operator(document, "TRANSPOSE", [0, permutation], [1])
        operator(document, "TRANSPOSE", [1, permutation], [2])
        graph.inputs, graph.outputs = [0], [2]
        before = _interface_snapshot(document)
        self.assertFalse(run_pass(document, view_pass()).modified)
        self.assertEqual(_interface_snapshot(document), before)
        self.assertTrue(run_pass(document, view_pass(), preserve=False).modified)

    def test_dce_retains_unused_inputs_only_in_preserve_mode(self):
        for preserve in (True, False):
            with self.subTest(preserve=preserve):
                document = mul_document()
                graph = document.subgraph()
                graph.tensors.append(tensor("unused"))
                graph.inputs.append(4)
                run_pass(document, DeadCodeEliminationPass(), preserve)
                self.assertEqual(graph.inputs, [0, 4] if preserve else [0])

    def test_constant_folder_keeps_unused_inputs_in_preserve_mode(self):
        from tico.circle.passes.optimization.fold.constant_subgraph import (
            FoldConstantSubgraphPass,
        )
        from tico.circle.passes.optimization.fold.evaluators import (
            BinaryElementwiseEvaluator,
            ConstantEvaluatorRegistry,
        )

        for preserve in (True, False):
            with self.subTest(preserve=preserve):
                document = empty_document()
                graph = document.subgraph()
                graph.tensors = [tensor("unused"), tensor("sum", (1,))]
                graph.inputs, graph.outputs = [0], [1]
                left = constant(document, "left", [2.0])
                right = constant(document, "right", [3.0])
                operator(document, "ADD", [left, right], [1])
                before = _interface_snapshot(document)
                registry = ConstantEvaluatorRegistry(
                    ((CODES["ADD"], BinaryElementwiseEvaluator("ADD", add)),)
                )
                folder = FoldConstantSubgraphPass(
                    evaluator_registry=registry,
                    codec=codec(),
                    object_factory=factory,
                )
                self.assertTrue(run_pass(document, folder, preserve).modified)
                self.assertEqual(graph.inputs, [0] if preserve else [])
                value = codec().decode_tensor(
                    document.model, subgraph_index=0, tensor_index=1
                )
                self.assertEqual(value.data.item(), 5.0)
                if preserve:
                    self.assertEqual(_interface_snapshot(document), before)

    def test_snapshot_detects_qparam_and_alias_changes(self):
        document = mul_document(public_mul=True)
        before = _interface_snapshot(document)
        qparams = factory("QuantizationParameters")
        qparams.scale, qparams.zeroPoint = [0.5], [0]
        document.subgraph().tensors[1].quantization = qparams
        self.assertNotEqual(_interface_snapshot(document), before)
        document = mul_document(public_mul=True)
        document.subgraph().tensors[1].name = "x"
        document.subgraph().outputs = [0, 1]
        before = _interface_snapshot(document)
        document.subgraph().outputs = [0, 0]
        self.assertNotEqual(_interface_snapshot(document), before)

    def test_compaction_preserves_names_signatures_and_aliases(self):
        document = mul_document()
        graph = document.subgraph()
        # The same input is also an output, and output order is observable.
        graph.outputs = [2, 0, 2]
        add_signature(document)
        before = _interface_snapshot(document)
        manager = CirclePassManager(
            [identity_pass(), DeadCodeEliminationPass(), CompactIndicesPass()]
        )
        manager.run(document, CirclePassContext(preserve_io=True))
        self.assertEqual(_interface_snapshot(document), before)

    def test_snapshot_detects_public_contract_changes(self):
        original = mul_document(public_mul=True)
        add_signature(original)
        before = _interface_snapshot(original)

        def reshape(d):
            d.subgraph().tensors[1].shape = [1, 3]
            d.subgraph().tensors[1].shapeSignature = [-1, 3]

        mutations = (
            lambda d: setattr(d.subgraph().tensors[1], "name", "renamed"),
            lambda d: setattr(d.subgraph().tensors[1], "type", TYPES["INT8"]),
            reshape,
            lambda d: setattr(d.subgraph(), "inputs", []),
            lambda d: setattr(d.subgraph(), "outputs", [0]),
            lambda d: setattr(d.model.signatureDefs[0], "signatureKey", "renamed"),
        )
        for mutation in mutations:
            document = original.clone()
            mutation(document)
            self.assertNotEqual(_interface_snapshot(document), before)

    def test_protection_is_subgraph_local(self):
        document = mul_document()
        document.model.subgraphs.append(copy.deepcopy(document.subgraph()))
        document.subgraph(1).outputs = [1]
        context = CirclePassContext(preserve_io=True)
        self.assertTrue(context.can_bypass_tensor(document, document.graph(0), 1))
        self.assertFalse(context.can_bypass_tensor(document, document.graph(1), 1))


class TestExportOperation(unittest.TestCase):
    """Exercise export orchestration with binary I/O and preset selection mocked."""

    @staticmethod
    def pipeline(document, context):
        manager = CirclePassManager(
            [
                identity_pass(),
                DeadCodeEliminationPass(),
                CompactIndicesPass(),
            ]
        )
        result = manager.run(document, context)
        return NS(
            modified=result.modified,
            changes=result.changes,
            phases=(NS(name="test_cleanup", result=result),),
        )

    def test_runs_preserving_pipeline_after_load_before_pack(self):
        document = mul_document()
        before = _interface_snapshot(document)
        with (
            patch.object(CircleDocument, "from_bytes", return_value=document),
            patch.object(CircleDocument, "to_bytes", return_value=b"optimized") as pack,
            patch("tico.circle.export.create_o1_pipeline") as create,
        ):
            create.return_value.run.side_effect = self.pipeline
            self.assertEqual(optimize_for_export(b"raw"), b"optimized")
            context = create.return_value.run.call_args.args[1]
            self.assertTrue(context.preserve_io)
            self.assertTrue(context.verify_after_each_pass)
            pack.assert_called_once()
        self.assertEqual(_interface_snapshot(document), before)

    def test_unchanged_pipeline_returns_original_bytes_without_pack(self):
        document = mul_document()
        with (
            patch.object(CircleDocument, "from_bytes", return_value=document),
            patch.object(CircleDocument, "to_bytes") as pack,
            patch("tico.circle.export.create_o1_pipeline") as create,
        ):
            create.return_value.run.return_value = NS(
                modified=False, changes=0, phases=()
            )
            raw = b"unchanged"
            self.assertIs(optimize_for_export(raw), raw)
            pack.assert_not_called()

    def test_pipeline_failure_propagates(self):
        with (
            patch.object(CircleDocument, "from_bytes", return_value=mul_document()),
            patch("tico.circle.export.create_o1_pipeline") as create,
        ):
            create.return_value.run.side_effect = RuntimeError("optimizer failed")
            with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
                optimize_for_export(b"raw")

    def test_interface_violation_is_not_silently_returned(self):
        document = mul_document()

        def corrupt(document, context):
            document.subgraph().tensors[2].name = "wrong_output"
            return NS(modified=True, changes=1)

        with (
            patch.object(CircleDocument, "from_bytes", return_value=document),
            patch.object(CircleDocument, "to_bytes") as pack,
            patch("tico.circle.export.create_o1_pipeline") as create,
        ):
            create.return_value.run.side_effect = corrupt
            with self.assertRaisesRegex(CircleRewriteError, "I/O or signature"):
                optimize_for_export(b"raw")
            pack.assert_not_called()

    def test_debug_dumps_are_unique_and_include_pass_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with (
                    patch.object(
                        CircleDocument, "from_bytes", return_value=mul_document()
                    ),
                    patch.object(CircleDocument, "to_bytes", return_value=b"optimized"),
                    patch("tico.circle.export.create_o1_pipeline") as create,
                ):
                    create.return_value.run.side_effect = self.pipeline
                    optimize_for_export(b"raw", dump_directory=directory)
            folders = list(Path(directory).iterdir())
            self.assertEqual(len(folders), 2)
            for folder in folders:
                self.assertEqual((folder / "before.circle").read_bytes(), b"raw")
                self.assertEqual((folder / "after.circle").read_bytes(), b"optimized")
                report = json.loads((folder / "report.json").read_text())
                self.assertTrue(report["preserve_io"])
                self.assertTrue(report["phases"][0]["executions"])

    def test_original_dump_survives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(CircleDocument, "from_bytes", return_value=mul_document()),
                patch("tico.circle.export.create_o1_pipeline") as create,
            ):
                create.return_value.run.side_effect = RuntimeError("failed")
                with self.assertRaises(RuntimeError):
                    optimize_for_export(b"raw", dump_directory=directory)
            folder = next(Path(directory).iterdir())
            self.assertEqual((folder / "before.circle").read_bytes(), b"raw")
            self.assertFalse((folder / "after.circle").exists())


if __name__ == "__main__":
    unittest.main()

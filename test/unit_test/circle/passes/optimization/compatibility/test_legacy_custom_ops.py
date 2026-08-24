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
from typing import Any

import numpy as np

from tico.circle.passes import CirclePassContext, ResolveLegacyCustomOpsPass
from tico.circle.passes.optimization._utils import operator_builtin_code

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    INT32,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._compatibility_fixture import (
    ADD,
    add_constant,
    BATCH_MATMUL,
    BUILTIN_CODES,
    BUILTIN_OPTIONS_TYPES,
    compatibility_object_factory,
    CUSTOM,
    INT64,
    make_builder,
    make_codec,
    MAX_POOL_2D,
    PADDING_VALUES,
    SPLIT_V,
    static_contract,
    TENSOR_TYPES,
)


class ResolveLegacyCustomOpsTest(unittest.TestCase):
    """Check builtin recovery and transactional compatibility behavior."""

    def setUp(self) -> None:
        """Create fake schema services and deterministic custom-option mappings."""

        self.codec = make_codec()
        self.option_maps: dict[bytes, dict[str, Any]] = {
            b"bmm": {"adj_x": True, "adj_y": False},
            b"matmul": {"transpose_a": False, "transpose_b": True},
            b"pool": {
                "ksize": [1, 1, 1, 1],
                "strides": [1, 1, 1, 1],
                "padding": "VALID",
                "include_batch_in_index": False,
            },
        }

    def _pass(self) -> ResolveLegacyCustomOpsPass:
        """Create the resolver with explicit enum values and option decoding."""

        return ResolveLegacyCustomOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            padding_values=PADDING_VALUES,
            activation_none=0,
            codec=self.codec,
            object_factory=compatibility_object_factory,
            custom_option_decoder=lambda payload: self.option_maps[payload],
        )

    @staticmethod
    def _set_custom_options(document, marker: bytes) -> None:
        """Attach one marker payload to the most recently appended operator."""

        document.subgraph().operators[-1].customOptions = np.frombuffer(
            marker,
            dtype=np.uint8,
        ).copy()

    def test_add_v2_is_recovered_as_builtin_add(self) -> None:
        """Preserve inputs and output while replacing the custom opcode."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        lhs = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="lhs",
            shape=[2],
        )
        rhs = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="rhs",
            shape=[2],
        )
        output = builder.add_operator(
            CUSTOM,
            custom_code="AddV2",
            inputs=(lhs, rhs),
            output_contracts=(static_contract((2,)),),
            output_names=("sum",),
        )[0]
        document.subgraph().outputs = [output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[0]
        self.assertEqual(operator_builtin_code(document.model, operator), ADD)
        self.assertEqual(operator.inputs, [lhs, rhs])
        self.assertEqual(operator.outputs, [output])

    def test_batch_matmul_and_matmul_decode_transpose_flags(self) -> None:
        """Map both former custom names to builtin BATCH_MATMUL options."""

        cases = (
            ("BatchMatMulV2", b"bmm", True, False),
            ("MatMul", b"matmul", False, True),
        )
        for custom_code, marker, expected_lhs, expected_rhs in cases:
            with self.subTest(custom_code=custom_code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                lhs = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="lhs",
                    shape=[3, 2] if expected_lhs else [2, 3],
                )
                rhs = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="rhs",
                    shape=[4, 3] if expected_rhs else [3, 4],
                )
                output = builder.add_operator(
                    CUSTOM,
                    custom_code=custom_code,
                    inputs=(lhs, rhs),
                    output_contracts=(static_contract((2, 4)),),
                    output_names=("matmul",),
                )[0]
                self._set_custom_options(document, marker)
                document.subgraph().outputs = [output]

                result = self._pass().run(
                    document,
                    CirclePassContext(verify_after_each_pass=False),
                )

                self.assertTrue(result.modified)
                operator = document.subgraph().operators[0]
                self.assertEqual(
                    operator_builtin_code(document.model, operator),
                    BATCH_MATMUL,
                )
                self.assertEqual(operator.builtinOptions.adjointLhs, expected_lhs)
                self.assertEqual(operator.builtinOptions.adjointRhs, expected_rhs)

    def test_split_v_narrows_int64_shape_constants(self) -> None:
        """Create INT32 compatibility constants and preserve all output tensors."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[4],
        )
        sizes = add_constant(builder, "sizes", [2, 2], INT64, np.int64)
        axis = add_constant(builder, "axis", 0, INT64, np.int64)
        outputs = builder.add_operator(
            CUSTOM,
            custom_code="SplitV",
            inputs=(source, sizes, axis),
            output_contracts=(static_contract((2,)), static_contract((2,))),
            output_names=("left", "right"),
        )
        document.subgraph().outputs = list(outputs)

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[0]
        self.assertEqual(operator_builtin_code(document.model, operator), SPLIT_V)
        size_contract = document.subgraph().tensors[operator.inputs[1]]
        axis_contract = document.subgraph().tensors[operator.inputs[2]]
        self.assertEqual(size_contract.type, INT32)
        self.assertEqual(axis_contract.type, INT32)
        self.assertEqual(operator.builtinOptions.numSplits, 2)

    def test_split_v_rolls_back_when_second_int64_constant_overflows(self) -> None:
        """Remove a first narrowed constant when a later match condition fails."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[4],
        )
        sizes = add_constant(builder, "sizes", [2, 2], INT64, np.int64)
        axis = add_constant(builder, "axis", 2**40, INT64, np.int64)
        builder.add_operator(
            CUSTOM,
            custom_code="SplitV",
            inputs=(source, sizes, axis),
            output_contracts=(static_contract((2,)), static_contract((2,))),
            output_names=("left", "right"),
        )
        tensor_count = len(document.subgraph().tensors)
        buffer_count = len(document.model.buffers)

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().tensors), tensor_count)
        self.assertEqual(len(document.model.buffers), buffer_count)
        self.assertEqual(
            operator_builtin_code(document.model, document.subgraph().operators[0]),
            CUSTOM,
        )

    def test_unit_max_pool_with_argmax_materializes_index_output(self) -> None:
        """Recover the exactly representable 1x1 pooling special case."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2, 2, 1],
        )
        value_output, index_output = builder.add_operator(
            CUSTOM,
            custom_code="MaxPoolWithArgmax",
            inputs=(source,),
            output_contracts=(
                static_contract((1, 2, 2, 1)),
                static_contract((1, 2, 2, 1), INT32),
            ),
            output_names=("values", "indices"),
        )
        self._set_custom_options(document, b"pool")
        document.subgraph().outputs = [value_output, index_output]

        result = self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        operator = document.subgraph().operators[0]
        self.assertEqual(operator_builtin_code(document.model, operator), MAX_POOL_2D)
        indices = self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=index_output,
        )
        np.testing.assert_array_equal(
            indices.data,
            np.asarray([[[[0], [1]], [[2], [3]]]], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()

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

"""Hand-computed checks for element-wise, comparison, and activation kernels."""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np
from circle_schema import circle
from tico.circle.document import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    CircleRuntimeValidationError,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)

from test.support.circle.builder import CircleModelBuilder


def _options(name: str, **fields: Any) -> Any:
    options = getattr(getattr(circle, name), f"{name}T")()
    for key, value in fields.items():
        setattr(options, key, value)
    return options


def run_unary(
    builtin: str,
    value: np.ndarray,
    *,
    options: Any = None,
    out_dtype: Any = None,
) -> np.ndarray:
    """Build ``builtin(x)`` with the value's shape and execute it."""

    builder = CircleModelBuilder()
    x = builder.input("x", list(value.shape), dtype=value.dtype)
    out = builder.activation("out", list(value.shape), dtype=out_dtype or value.dtype)
    builder.operator(builtin, [x], [out], options=options)
    builder.set_outputs(out)
    return CircleReferenceRuntime(builder.build()).run((value,)).outputs[0]


def run_binary(
    builtin: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    options: Any = None,
    out_shape: list[int] | None = None,
    out_dtype: Any = None,
) -> np.ndarray:
    """Build ``builtin(lhs, rhs)`` with broadcast output shape and execute it."""

    builder = CircleModelBuilder()
    x = builder.input("x", list(lhs.shape), dtype=lhs.dtype)
    y = builder.input("y", list(rhs.shape), dtype=rhs.dtype)
    shape = out_shape or list(np.broadcast_shapes(lhs.shape, rhs.shape))
    out = builder.activation("out", shape, dtype=out_dtype or lhs.dtype)
    builder.operator(builtin, [x, y], [out], options=options)
    builder.set_outputs(out)
    return CircleReferenceRuntime(builder.build()).run((lhs, rhs)).outputs[0]


class ArithmeticKernelTest(unittest.TestCase):
    def test_add_broadcasts_and_applies_relu6(self):
        lhs = np.array([[1.0, -4.0], [5.0, 2.0]], dtype=np.float32)
        rhs = np.array([1.0, 1.0], dtype=np.float32)
        result = run_binary(
            "ADD", lhs, rhs, options=_options("AddOptions", fusedActivationFunction=3)
        )
        np.testing.assert_array_equal(result, [[2.0, 0.0], [6.0, 3.0]])

    def test_sub_mul_div_pow_float(self):
        lhs = np.array([6.0, -3.0, 2.0], dtype=np.float32)
        rhs = np.array([2.0, 2.0, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(
            run_binary("SUB", lhs, rhs, options=_options("SubOptions")),
            [4.0, -5.0, 1.5],
        )
        np.testing.assert_array_equal(
            run_binary("MUL", lhs, rhs, options=_options("MulOptions")),
            [12.0, -6.0, 1.0],
        )
        np.testing.assert_array_equal(
            run_binary("DIV", lhs, rhs, options=_options("DivOptions")),
            [3.0, -1.5, 4.0],
        )
        np.testing.assert_allclose(
            run_binary("POW", lhs, rhs, options=_options("PowOptions")),
            [36.0, 9.0, math.sqrt(2.0)],
            rtol=1e-6,
        )

    def test_integer_add_and_div_truncate_toward_zero(self):
        lhs = np.array([7, -7, 9], dtype=np.int32)
        rhs = np.array([2, 2, -4], dtype=np.int32)
        np.testing.assert_array_equal(
            run_binary("ADD", lhs, rhs, options=_options("AddOptions")), [9, -5, 5]
        )
        result = run_binary("DIV", lhs, rhs, options=_options("DivOptions"))
        np.testing.assert_array_equal(result, [3, -3, -2])
        self.assertEqual(result.dtype, np.int32)

    def test_int64_arithmetic_keeps_dtype(self):
        lhs = np.array([2**40, -1], dtype=np.int64)
        rhs = np.array([1, 2], dtype=np.int64)
        result = run_binary("MUL", lhs, rhs, options=_options("MulOptions"))
        np.testing.assert_array_equal(result, [2**40, -2])
        self.assertEqual(result.dtype, np.int64)

    def test_mixed_dtypes_are_rejected(self):
        lhs = np.array([1.0], dtype=np.float32)
        rhs = np.array([1], dtype=np.int32)
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "share one dtype"):
            run_binary(
                "ADD", lhs, rhs, options=_options("AddOptions"), out_dtype=np.float32
            )

    def test_incompatible_broadcast_is_rejected(self):
        lhs = np.zeros((2, 3), dtype=np.float32)
        rhs = np.zeros((4,), dtype=np.float32)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "not broadcast-compatible"
        ):
            run_binary(
                "ADD", lhs, rhs, options=_options("AddOptions"), out_shape=[2, 3]
            )

    def test_maximum_minimum(self):
        lhs = np.array([1.0, 5.0, -2.0], dtype=np.float32)
        rhs = np.array([3.0], dtype=np.float32)
        np.testing.assert_array_equal(run_binary("MAXIMUM", lhs, rhs), [3.0, 5.0, 3.0])
        np.testing.assert_array_equal(run_binary("MINIMUM", lhs, rhs), [1.0, 3.0, -2.0])


class IntegerDivisionKernelTest(unittest.TestCase):
    def _runtime(self, lhs, rhs, *, activation=0):
        builder = CircleModelBuilder()
        x = builder.input("lhs", lhs.shape, dtype=lhs.dtype)
        y = builder.input("rhs", rhs.shape, dtype=rhs.dtype)
        shape = np.broadcast_shapes(lhs.shape, rhs.shape)
        out = builder.activation("out", shape, dtype=lhs.dtype)
        builder.operator(
            "DIV",
            [x, y],
            [out],
            options=_options("DivOptions", fusedActivationFunction=activation),
        )
        builder.set_outputs(out)
        document = CircleDocument.from_bytes(builder.build().to_bytes())
        self.assertTrue(document.verify(raise_on_error=False).ok)
        return CircleReferenceRuntime(document)

    def _assert_division(self, lhs, rhs, expected=None, *, activation=0):
        if expected is None:
            left, right = np.broadcast_arrays(lhs, rhs)
            # Python integers retain all bits; abs division supplies truncation
            # toward zero without any floating-point conversion.
            expected = np.array(
                [
                    (abs(int(a)) // abs(int(b)))
                    * (-1 if (int(a) < 0) != (int(b) < 0) else 1)
                    for a, b in zip(left.flat, right.flat)
                ],
                dtype=lhs.dtype,
            ).reshape(left.shape)
        original_lhs, original_rhs = lhs.copy(), rhs.copy()
        runtime = self._runtime(lhs, rhs, activation=activation)
        for _ in range(2):
            result = runtime.run((lhs, rhs)).outputs[0]
            self.assertEqual(result.dtype, lhs.dtype)
            self.assertEqual(result.shape, np.broadcast_shapes(lhs.shape, rhs.shape))
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(lhs, original_lhs)
            np.testing.assert_array_equal(rhs, original_rhs)

    def test_int64_identity_preserves_every_bit(self):
        lhs = np.array([2**53 + 1, -(2**53 + 1), 2**63 - 1, -(2**63)], np.int64)
        self._assert_division(lhs, np.ones_like(lhs))

    def test_int64_nonunit_division_truncates_all_signs(self):
        maximum, minimum = 2**63 - 1, -(2**63)
        lhs = np.array(
            [maximum, -maximum, maximum, -maximum, minimum, minimum], np.int64
        )
        rhs = np.array([3, 3, -3, -3, 3, -3], np.int64)
        self._assert_division(lhs, rhs)

    def test_uint64_identity_and_nonunit_divisors(self):
        maximum = 2**64 - 1
        lhs = np.array([2**53 + 1, maximum, maximum, maximum, maximum], np.uint64)
        rhs = np.array([1, 1, 3, 5, maximum], np.uint64)
        self._assert_division(lhs, rhs)

    def test_all_integer_dtypes_keep_exact_normal_quotients(self):
        for dtype in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            with self.subTest(dtype=dtype):
                info = np.iinfo(dtype)
                lhs = np.array([info.min, info.min + 1, info.max, 7], dtype=dtype)
                rhs = np.array([2, 2, 3, 2], dtype=dtype)
                self._assert_division(lhs, rhs)

    def test_broadcasting_and_noncontiguous_inputs(self):
        for dtype in (np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                last = -(2**53 + 3) if dtype == np.int64 else 2**53 + 3
                lhs = np.array([[2**53 + 1, 0], [last, 0]], dtype=dtype)[:, :1]
                rhs = np.array([1, -2, 3] if dtype == np.int64 else [1, 2, 3], dtype)
                self._assert_division(lhs, rhs)

    def test_scalar_operands_and_empty_shapes(self):
        for dtype in (np.int64, np.uint64):
            cases = (
                (np.array(2**53 + 1, dtype), np.array(1, dtype)),
                (np.array(2**53 + 1, dtype), np.array([1, 2, 3], dtype)),
                (np.array([2**53 + 1, 2**53 + 3], dtype), np.array(3, dtype)),
                (np.empty((0, 3), dtype), np.ones((1, 3), dtype)),
            )
            for lhs, rhs in cases:
                with self.subTest(dtype=dtype, lhs=lhs.shape, rhs=rhs.shape):
                    self._assert_division(lhs, rhs)

    def test_fused_activations_use_integer_quotients(self):
        lhs = np.array([2**63 - 1, -(2**63), 17, -17], np.int64)
        rhs = np.array([1, 1, 3, 3], np.int64)
        for activation, expected in (
            (1, [2**63 - 1, 0, 5, 0]),
            (2, [1, -1, 1, -1]),
            (3, [6, 0, 5, 0]),
        ):
            with self.subTest(activation=activation):
                self._assert_division(lhs, rhs, expected, activation=activation)

    def test_zero_divisors_preserve_legacy_cast_results(self):
        for dtype in (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ):
            maximum = np.iinfo(dtype).max
            values = np.array([0, 1, 2, maximum], dtype=dtype)
            cases = (
                (np.array(0, dtype), np.array(0, dtype)),
                (values[:2], np.zeros(2, dtype)),
                (values, np.zeros(4, dtype)),
                (values[:, None], np.zeros((1, 4), dtype)),
                (np.asfortranarray(np.tile(values, (3, 1))), np.zeros((3, 4), dtype)),
            )
            for lhs, rhs in cases:
                with self.subTest(dtype=dtype, lhs=lhs.shape, rhs=rhs.shape):
                    # Invalid float-to-integer casts depend on dtype and array
                    # layout. Preserve the existing whole-array cast contract.
                    with np.errstate(all="ignore"):
                        expected = np.trunc(
                            np.divide(lhs.astype(np.float64), rhs.astype(np.float64))
                        ).astype(dtype)
                    self._assert_division(lhs, rhs, expected)

    def test_signed_overflow_preserves_legacy_result(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                minimum = int(np.iinfo(dtype).min)
                lhs = np.array([minimum, minimum, 7, -7], dtype=dtype)
                rhs = np.array([-1, 1, 2, 2], dtype=dtype)
                self._assert_division(lhs, rhs, [minimum, minimum, 3, -3])

    def test_mixed_zero_and_valid_divisors_keep_exact_valid_lanes(self):
        for dtype in (np.int32, np.int64, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                maximum = int(np.iinfo(dtype).max)
                lhs = np.array([0, 1, 2, maximum, maximum], dtype=dtype)
                rhs = np.array([0, 0, 1, 3, 1], dtype=dtype)
                with np.errstate(all="ignore"):
                    expected = np.trunc(
                        np.divide(lhs.astype(np.float64), rhs.astype(np.float64))
                    ).astype(dtype)
                expected[2:] = [2, maximum // 3, maximum]
                self._assert_division(lhs, rhs, expected)

    def test_float_division_keeps_fractional_and_special_values(self):
        for dtype in (np.float16, np.float32):
            lhs = np.array([7, -7, 0, 1, -1, np.inf, np.nan, -0.0], dtype)
            rhs = np.array([2, 2, 0, 0, 0, 2, 1, 1], dtype)
            with self.subTest(dtype=dtype), np.errstate(all="ignore"):
                expected = np.divide(lhs, rhs)
                self._assert_division(lhs, rhs, expected)
                result = self._runtime(lhs, rhs).run((lhs, rhs)).outputs[0]
                np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))


class ComparisonAndLogicalKernelTest(unittest.TestCase):
    def test_comparisons_return_bool(self):
        lhs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rhs = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        cases = {
            "EQUAL": [False, True, False],
            "NOT_EQUAL": [True, False, True],
            "GREATER": [False, False, True],
            "GREATER_EQUAL": [False, True, True],
            "LESS": [True, False, False],
            "LESS_EQUAL": [True, True, False],
        }
        for builtin, expected in cases.items():
            result = run_binary(builtin, lhs, rhs, out_dtype=np.bool_)
            self.assertEqual(result.dtype, np.bool_, builtin)
            np.testing.assert_array_equal(result, expected, err_msg=builtin)

    def test_int64_not_equal(self):
        lhs = np.array([1, 2], dtype=np.int64)
        rhs = np.array([1, 3], dtype=np.int64)
        np.testing.assert_array_equal(
            run_binary("NOT_EQUAL", lhs, rhs, out_dtype=np.bool_), [False, True]
        )

    def test_logical_and_not(self):
        lhs = np.array([True, True, False], dtype=np.bool_)
        rhs = np.array([True, False, False], dtype=np.bool_)
        np.testing.assert_array_equal(
            run_binary("LOGICAL_AND", lhs, rhs), [True, False, False]
        )
        np.testing.assert_array_equal(
            run_unary("LOGICAL_NOT", lhs), [False, False, True]
        )

    def test_logical_and_rejects_non_bool(self):
        lhs = np.array([1, 0], dtype=np.int32)
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "requires BOOL"):
            run_binary("LOGICAL_AND", lhs, lhs, out_dtype=np.bool_)

    def test_select_v2_broadcasts_condition(self):
        builder = CircleModelBuilder()
        cond = builder.input("cond", [2, 1], dtype=np.bool_)
        a = builder.input("a", [2, 3])
        b = builder.input("b", [3])
        out = builder.activation("out", [2, 3])
        builder.operator(
            "SELECT_V2", [cond, a, b], [out], options=_options("SelectV2Options")
        )
        builder.set_outputs(out)
        result = CircleReferenceRuntime(builder.build()).run(
            (
                np.array([[True], [False]]),
                np.ones((2, 3), dtype=np.float32),
                np.array([7.0, 8.0, 9.0], dtype=np.float32),
            )
        )
        np.testing.assert_array_equal(
            result.outputs[0], [[1.0, 1.0, 1.0], [7.0, 8.0, 9.0]]
        )


class SelectKernelTest(unittest.TestCase):
    def _runtime(
        self,
        condition,
        on_true,
        on_false,
        *,
        builtin="SELECT",
        output_shape=None,
        quantized=False,
    ):
        builder = CircleModelBuilder()
        inputs = [
            builder.input(name, value.shape, dtype=value.dtype)
            for name, value in zip(
                ("condition", "on_true", "on_false"), (condition, on_true, on_false)
            )
        ]
        out = builder.activation(
            "out",
            on_true.shape if output_shape is None else output_shape,
            dtype=on_true.dtype,
        )
        if quantized:
            for index in (*inputs[1:], out):
                builder.subgraph.tensors[index].quantization = builder.quantization(
                    [0.5], [0]
                )
        options = "SelectOptions" if builtin == "SELECT" else "SelectV2Options"
        builder.operator(builtin, inputs, [out], options=_options(options))
        builder.set_outputs(out)
        # Exercise the serialized schema and verifier, not just Object API fixtures.
        return CircleReferenceRuntime(builder.build().to_bytes(), verify=True)

    def test_rank_one_condition_selects_rows_and_preserves_inputs(self):
        for dtype in (np.int8, np.int32, np.int64, np.float32):
            with self.subTest(dtype=dtype):
                inputs: tuple[np.ndarray, ...] = (
                    np.array([True, False]),
                    np.array([[1, 2], [3, 4]], dtype=dtype),
                    np.array([[10, 20], [30, 40]], dtype=dtype),
                )
                originals = tuple(value.copy() for value in inputs)
                runtime = self._runtime(*inputs)
                for _ in range(2):
                    result = runtime.run(inputs).outputs[0]
                    np.testing.assert_array_equal(result, [[1, 2], [30, 40]])
                    self.assertEqual(result.dtype, dtype)
                    for value, original in zip(inputs, originals):
                        np.testing.assert_array_equal(value, original)
                        self.assertTrue(value.flags.writeable)
                        self.assertFalse(np.shares_memory(result, value))

    def test_rank_one_condition_selects_rectangular_and_higher_rank_slices(self):
        for shape in ((2, 3), (2, 1, 3)):
            with self.subTest(shape=shape):
                inputs = (
                    np.array([True, False]),
                    np.reshape(np.array([[1, 2, 3], [4, 5, 6]], np.float32), shape),
                    np.reshape(
                        np.array([[10, 20, 30], [40, 50, 60]], np.float32), shape
                    ),
                )
                result = self._runtime(*inputs).run(inputs).outputs[0]
                expected = np.reshape(np.array([[1, 2, 3], [40, 50, 60]]), shape)
                np.testing.assert_array_equal(result, expected)

    def test_scalar_and_equal_shape_conditions(self):
        on_true = np.array([[1, 2], [3, 4]], np.float32)
        on_false = np.array([[10, 20], [30, 40]], np.float32)
        for condition, expected in (
            (np.array(True), on_true),
            (np.array(False), on_false),
            (np.array([[True, False], [False, True]]), [[1, 20], [30, 4]]),
        ):
            with self.subTest(condition=condition):
                inputs = (condition, on_true, on_false)
                np.testing.assert_array_equal(
                    self._runtime(*inputs).run(inputs).outputs[0], expected
                )
        inputs = (np.array(False), np.array(True), np.array(False))
        result = self._runtime(*inputs).run(inputs).outputs[0]
        self.assertEqual(result.shape, ())
        self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(result, False)

    def test_mixed_single_element_shapes_keep_declared_output_shape(self):
        # TFLite SelectPrepare/Select explicitly allow mixed scalar representations.
        for condition in (True, False):
            for shape in ((), (1,), (1, 1)):
                with self.subTest(condition=condition, shape=shape):
                    inputs = (
                        np.array([condition]),
                        np.array(7, np.int32),
                        np.array([[9]], np.int32),
                    )
                    result = self._runtime(*inputs, output_shape=shape).run(inputs)
                    np.testing.assert_array_equal(
                        result.outputs[0], np.full(shape, 7 if condition else 9)
                    )

    def test_general_broadcasting_is_rejected_by_legacy_select(self):
        for condition_shape, true_shape, false_shape in (
            ((2, 1), (2, 3), (2, 3)),
            ((3,), (2, 3), (2, 3)),
            ((1, 2), (2, 2), (2, 2)),
            ((), (2, 3), (3,)),
        ):
            with self.subTest(shapes=(condition_shape, true_shape, false_shape)):
                inputs = (
                    np.ones(condition_shape, np.bool_),
                    np.ones(true_shape, np.float32),
                    np.zeros(false_shape, np.float32),
                )
                with self.assertRaisesRegex(
                    CircleRuntimeValidationError, "SELECT.*shape"
                ):
                    self._runtime(*inputs).run(inputs)

    def test_empty_slices_keep_the_data_shape(self):
        for condition, shape in (
            (np.array([], np.bool_), (0, 3)),
            (np.array([True, False]), (2, 0, 3)),
            (np.empty((0, 3), np.bool_), (0, 3)),
            (np.array(True), (0, 3)),
        ):
            with self.subTest(condition_shape=condition.shape, shape=shape):
                inputs = (
                    condition,
                    np.empty(shape, np.int32),
                    np.empty(shape, np.int32),
                )
                result = self._runtime(*inputs).run(inputs).outputs[0]
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, np.int32)
                self.assertEqual(result.size, 0)

    def test_quantized_rows_preserve_values_and_qparams_in_both_modes(self):
        inputs = (
            np.array([True, False]),
            np.array([[2, 4], [6, 8]], np.int8),
            np.array([[10, 12], [14, 16]], np.int8),
        )
        runtime = self._runtime(*inputs, quantized=True)
        qparams = tuple(runtime.program.tensor(i).quantization for i in (1, 2, 3))
        for mode in (ExecutionMode.NATIVE, ExecutionMode.FAKE_QUANTIZE):
            with self.subTest(mode=mode):
                native = mode is ExecutionMode.NATIVE
                values = (
                    inputs
                    if native
                    else (
                        inputs[0],
                        *(value.astype(np.float32) * 0.5 for value in inputs[1:]),
                    )
                )
                result = runtime.run(values, mode=mode).outputs[0]
                np.testing.assert_array_equal(
                    result, [[2, 4], [14, 16]] if native else [[1, 2], [7, 8]]
                )
                self.assertEqual(result.dtype, np.int8 if native else np.float32)
                self.assertEqual(
                    tuple(runtime.program.tensor(i).quantization for i in (1, 2, 3)),
                    qparams,
                )

    def test_select_v2_keeps_trailing_dimension_broadcasting(self):
        inputs = (
            np.array([True, False]),
            np.array([[1, 2], [3, 4]], np.float32),
            np.array([[10, 20], [30, 40]], np.float32),
        )
        result = self._runtime(*inputs, builtin="SELECT_V2").run(inputs).outputs[0]
        np.testing.assert_array_equal(result, [[1, 20], [3, 40]])

    def test_dtype_validation_is_preserved(self):
        for builtin in ("SELECT", "SELECT_V2"):
            for inputs, message in (
                ((np.array([1]), np.array([1]), np.array([2])), "must be a BOOL"),
                ((np.array([True]), np.array([1]), np.array([2.0])), "share one dtype"),
            ):
                with self.subTest(builtin=builtin, message=message):
                    with self.assertRaisesRegex(
                        UnsupportedCircleOperatorError, message
                    ):
                        self._runtime(*inputs, builtin=builtin).run(inputs)

    def test_incorrect_output_shape_is_not_repaired(self):
        inputs = (
            np.array(True),
            np.ones((2, 2), np.float32),
            np.zeros((2, 2), np.float32),
        )
        with self.assertRaisesRegex(CircleRuntimeValidationError, "shape"):
            self._runtime(*inputs, output_shape=(4,)).run(inputs)


class UnaryKernelTest(unittest.TestCase):
    def test_float_math(self):
        x = np.array([0.25, 1.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(run_unary("SQRT", x), [0.5, 1.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(run_unary("RSQRT", x), [2.0, 1.0, 0.5], rtol=1e-6)
        np.testing.assert_allclose(run_unary("LOG", x), np.log(x), rtol=1e-6)
        np.testing.assert_allclose(run_unary("EXP", x), np.exp(x), rtol=1e-6)
        np.testing.assert_allclose(run_unary("NEG", x), [-0.25, -1.0, -4.0])
        np.testing.assert_allclose(run_unary("ABS", -x), x)
        np.testing.assert_allclose(run_unary("SIN", x), np.sin(x), rtol=1e-6)
        np.testing.assert_allclose(run_unary("COS", x), np.cos(x), rtol=1e-6)
        np.testing.assert_allclose(run_unary("TANH", x), np.tanh(x), rtol=1e-6)
        np.testing.assert_allclose(
            run_unary("LOGISTIC", np.array([0.0], dtype=np.float32)), [0.5], rtol=1e-6
        )

    def test_round_uses_half_to_even_like_tflite(self):
        x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, 2.4, 2.6], dtype=np.float32)
        np.testing.assert_array_equal(
            run_unary("ROUND", x), [0.0, 2.0, 2.0, -0.0, -2.0, 2.0, 3.0]
        )

    def test_relu_family(self):
        x = np.array([-2.0, -0.5, 0.5, 7.0], dtype=np.float32)
        np.testing.assert_array_equal(run_unary("RELU", x), [0.0, 0.0, 0.5, 7.0])
        np.testing.assert_array_equal(run_unary("RELU6", x), [0.0, 0.0, 0.5, 6.0])
        np.testing.assert_array_equal(
            run_unary("RELU_N1_TO_1", x), [-1.0, -0.5, 0.5, 1.0]
        )
        np.testing.assert_allclose(
            run_unary("LEAKY_RELU", x, options=_options("LeakyReluOptions", alpha=0.1)),
            [-0.2, -0.05, 0.5, 7.0],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            run_unary("ELU", x),
            [math.exp(-2.0) - 1.0, math.exp(-0.5) - 1.0, 0.5, 7.0],
            rtol=1e-6,
        )

    def test_gelu_exact_and_tanh_approximation(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        exact = [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in x]
        approx = [
            0.5
            * v
            * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v**3)))
            for v in x
        ]
        np.testing.assert_allclose(
            run_unary("GELU", x, options=_options("GeluOptions", approximate=False)),
            exact,
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            run_unary("GELU", x, options=_options("GeluOptions", approximate=True)),
            approx,
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertGreater(abs(exact[3] - approx[3]), 1e-6)

    def test_prelu_with_per_channel_alpha(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2, 3])
        alpha = builder.const_f32("alpha", [0.5, 2.0, -1.0])
        out = builder.activation("out", [1, 2, 3])
        builder.operator("PRELU", [x, alpha], [out])
        builder.set_outputs(out)
        value = np.array([[[-2.0, -2.0, -2.0], [1.0, 2.0, 3.0]]], dtype=np.float32)
        result = CircleReferenceRuntime(builder.build()).run((value,)).outputs[0]
        np.testing.assert_array_equal(result, [[[-1.0, -4.0, 2.0], [1.0, 2.0, 3.0]]])

    def test_prelu_alpha_must_broadcast_to_input(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2])
        alpha = builder.const_f32("alpha", [[0.5], [0.5]])
        out = builder.activation("out", [2])
        builder.operator("PRELU", [x, alpha], [out])
        builder.set_outputs(out)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "must broadcast to the input shape"
        ):
            CircleReferenceRuntime(builder.build()).run(
                (np.zeros(2, dtype=np.float32),)
            )

    def test_float_only_unary_rejects_integers(self):
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "not defined for dtype int32"
        ):
            run_unary("EXP", np.array([1, 2], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()

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
from tico.circle.runtime import (
    CircleReferenceRuntime,
    CircleRuntimeValidationError,
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

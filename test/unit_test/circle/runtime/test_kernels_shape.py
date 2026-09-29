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

"""Hand-computed checks for data-movement, reduction, and index kernels."""

from __future__ import annotations

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


class _Fixture:
    """Small helper that builds one operator over one input tensor."""

    def __init__(self, value: np.ndarray, *, dynamic: bool = False) -> None:
        self.builder = CircleModelBuilder()
        signature = [-1] + list(value.shape[1:]) if dynamic else None
        placeholder = [1] + list(value.shape[1:]) if dynamic else list(value.shape)
        self.x = self.builder.input(
            "x", placeholder, dtype=value.dtype, shape_signature=signature
        )
        self.value = value

    def const(self, name: str, values: Any, dtype: Any = np.int32) -> int:
        return self.builder.constant(name, values, dtype=dtype)

    def run(
        self,
        builtin: str,
        inputs: list[int],
        out_shape: list[int],
        *,
        options: Any = None,
        out_dtype: Any = None,
        out_signature: list[int] | None = None,
        extra_inputs: tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        out = self.builder.activation(
            "out",
            out_shape,
            dtype=out_dtype or self.value.dtype,
            shape_signature=out_signature,
        )
        self.builder.operator(builtin, inputs, [out], options=options)
        self.builder.set_outputs(out)
        runtime = CircleReferenceRuntime(self.builder.build())
        return runtime.run((self.value,) + extra_inputs).outputs[0]


class ReshapeTransposeTest(unittest.TestCase):
    def test_reshape_infers_minus_one_from_shape_input(self):
        fixture = _Fixture(np.arange(6, dtype=np.float32).reshape(2, 3))
        shape = fixture.const("shape", [3, -1])
        result = fixture.run(
            "RESHAPE",
            [fixture.x, shape],
            [3, 2],
            options=_options("ReshapeOptions", newShape=[3, -1]),
        )
        np.testing.assert_array_equal(result, [[0, 1], [2, 3], [4, 5]])

    def test_reshape_uses_options_when_shape_input_is_absent(self):
        fixture = _Fixture(np.arange(6, dtype=np.float32).reshape(2, 3))
        result = fixture.run(
            "RESHAPE",
            [fixture.x],
            [6],
            options=_options("ReshapeOptions", newShape=[6]),
        )
        np.testing.assert_array_equal(result, np.arange(6, dtype=np.float32))

    def test_reshape_with_wrong_element_count_is_rejected(self):
        fixture = _Fixture(np.arange(6, dtype=np.float32).reshape(2, 3))
        shape = fixture.const("shape", [4, 2])
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "has 8 elements but the input has 6"
        ):
            fixture.run(
                "RESHAPE",
                [fixture.x, shape],
                [4, 2],
                options=_options("ReshapeOptions"),
            )

    def test_transpose_squeeze_expand_dims(self):
        fixture = _Fixture(np.arange(6, dtype=np.float32).reshape(1, 2, 3))
        perm = fixture.const("perm", [2, 0, 1])
        result = fixture.run(
            "TRANSPOSE",
            [fixture.x, perm],
            [3, 1, 2],
            options=_options("TransposeOptions"),
        )
        np.testing.assert_array_equal(result, np.transpose(fixture.value, (2, 0, 1)))

        fixture = _Fixture(np.arange(6, dtype=np.float32).reshape(1, 2, 3))
        result = fixture.run(
            "SQUEEZE",
            [fixture.x],
            [2, 3],
            options=_options("SqueezeOptions", squeezeDims=[0]),
        )
        np.testing.assert_array_equal(result, fixture.value[0])

        fixture = _Fixture(np.arange(6, dtype=np.float32).reshape(2, 3))
        axis = fixture.const("axis", -1)
        result = fixture.run(
            "EXPAND_DIMS",
            [fixture.x, axis],
            [2, 3, 1],
            options=_options("ExpandDimsOptions"),
        )
        np.testing.assert_array_equal(result, fixture.value[:, :, None])

    def test_squeeze_of_non_unit_axis_is_rejected(self):
        fixture = _Fixture(np.zeros((1, 2), dtype=np.float32))
        with self.assertRaisesRegex(CircleRuntimeValidationError, "axis 1 has size 2"):
            fixture.run(
                "SQUEEZE",
                [fixture.x],
                [1],
                options=_options("SqueezeOptions", squeezeDims=[1]),
            )

    def test_broadcast_to(self):
        fixture = _Fixture(np.array([[1.0], [2.0]], dtype=np.float32))
        shape = fixture.const("shape", [2, 3])
        result = fixture.run(
            "BROADCAST_TO",
            [fixture.x, shape],
            [2, 3],
            options=_options("BroadcastToOptions"),
        )
        np.testing.assert_array_equal(result, [[1, 1, 1], [2, 2, 2]])


class ConcatSplitSliceTest(unittest.TestCase):
    def test_concatenation_on_negative_axis(self):
        builder = CircleModelBuilder()
        a = builder.input("a", [2, 1])
        b = builder.input("b", [2, 2])
        out = builder.activation("out", [2, 3])
        builder.operator(
            "CONCATENATION",
            [a, b],
            [out],
            options=_options("ConcatenationOptions", axis=-1),
        )
        builder.set_outputs(out)
        result = CircleReferenceRuntime(builder.build()).run(
            (
                np.array([[1.0], [2.0]], dtype=np.float32),
                np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            )
        )
        np.testing.assert_array_equal(result.outputs[0], [[1, 3, 4], [2, 5, 6]])

    def test_concatenation_shape_mismatch_is_rejected(self):
        builder = CircleModelBuilder()
        a = builder.input("a", [2, 1])
        b = builder.input("b", [3, 2])
        out = builder.activation("out", [2, 3])
        builder.operator(
            "CONCATENATION",
            [a, b],
            [out],
            options=_options("ConcatenationOptions", axis=1),
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(CircleRuntimeValidationError, "outside axis 1"):
            CircleReferenceRuntime(builder.build()).run(
                (np.zeros((2, 1), dtype=np.float32), np.zeros((3, 2), dtype=np.float32))
            )

    def test_split_v_with_inferred_size_and_all_outputs(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [5, 2])
        sizes = builder.const_i32("sizes", [2, -1])
        axis = builder.const_i32("axis", 0)
        first = builder.activation("first", [2, 2])
        second = builder.activation("second", [3, 2])
        builder.operator(
            "SPLIT_V",
            [x, sizes, axis],
            [first, second],
            options=_options("SplitVOptions", numSplits=2),
        )
        builder.set_outputs(second, first)
        value = np.arange(10, dtype=np.float32).reshape(5, 2)
        result = CircleReferenceRuntime(builder.build()).run((value,))
        np.testing.assert_array_equal(result.outputs[0], value[2:])
        np.testing.assert_array_equal(result.outputs[1], value[:2])

    def test_slice_with_minus_one_size(self):
        fixture = _Fixture(np.arange(12, dtype=np.float32).reshape(3, 4))
        begin = fixture.const("begin", [1, 1])
        size = fixture.const("size", [-1, 2])
        result = fixture.run(
            "SLICE", [fixture.x, begin, size], [2, 2], options=_options("SliceOptions")
        )
        np.testing.assert_array_equal(result, [[5, 6], [9, 10]])

    def test_strided_slice_with_negative_indices_and_stride(self):
        fixture = _Fixture(np.arange(10, dtype=np.float32))
        begin = fixture.const("begin", [-8])
        end = fixture.const("end", [100])
        strides = fixture.const("strides", [3])
        result = fixture.run(
            "STRIDED_SLICE",
            [fixture.x, begin, end, strides],
            [3],
            options=_options("StridedSliceOptions"),
        )
        np.testing.assert_array_equal(result, [2, 5, 8])

    def test_strided_slice_masks(self):
        fixture = _Fixture(np.arange(12, dtype=np.float32).reshape(3, 4))
        begin = fixture.const("begin", [1, 2])
        end = fixture.const("end", [0, 0])
        strides = fixture.const("strides", [1, 1])
        # endMask on axis 1 keeps the tail; shrinkAxisMask on axis 0 drops the axis.
        result = fixture.run(
            "STRIDED_SLICE",
            [fixture.x, begin, end, strides],
            [2],
            options=_options("StridedSliceOptions", endMask=0b10, shrinkAxisMask=0b01),
        )
        np.testing.assert_array_equal(result, [6, 7])

    def test_strided_slice_new_axis_mask_is_rejected(self):
        fixture = _Fixture(np.arange(4, dtype=np.float32))
        begin = fixture.const("begin", [0])
        end = fixture.const("end", [4])
        strides = fixture.const("strides", [1])
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "newAxisMask"):
            fixture.run(
                "STRIDED_SLICE",
                [fixture.x, begin, end, strides],
                [1, 4],
                options=_options("StridedSliceOptions", newAxisMask=1),
            )

    def test_pad_and_pad_v2(self):
        fixture = _Fixture(np.array([[1.0, 2.0]], dtype=np.float32))
        paddings = fixture.const("paddings", [[1, 0], [0, 1]])
        result = fixture.run(
            "PAD", [fixture.x, paddings], [2, 3], options=_options("PadOptions")
        )
        np.testing.assert_array_equal(result, [[0, 0, 0], [1, 2, 0]])

        fixture = _Fixture(np.array([[1.0, 2.0]], dtype=np.float32))
        paddings = fixture.const("paddings", [[1, 0], [0, 1]])
        constant = fixture.const("value", -9.0, dtype=np.float32)
        result = fixture.run(
            "PADV2",
            [fixture.x, paddings, constant],
            [2, 3],
            options=_options("PadV2Options"),
        )
        np.testing.assert_array_equal(result, [[-9, -9, -9], [1, 2, -9]])

    def test_pad_v2_constant_dtype_must_match(self):
        fixture = _Fixture(np.array([[1.0, 2.0]], dtype=np.float32))
        paddings = fixture.const("paddings", [[1, 0], [0, 1]])
        constant = fixture.const("value", 0, dtype=np.int32)
        with self.assertRaisesRegex(
            UnsupportedCircleOperatorError, "constant dtype int32"
        ):
            fixture.run(
                "PADV2",
                [fixture.x, paddings, constant],
                [2, 3],
                options=_options("PadV2Options"),
            )


class GatherTest(unittest.TestCase):
    def test_gather_on_axis_with_int64_indices(self):
        fixture = _Fixture(np.arange(12, dtype=np.float32).reshape(3, 4))
        indices = fixture.const("indices", [[3, 0]], dtype=np.int64)
        result = fixture.run(
            "GATHER",
            [fixture.x, indices],
            [3, 1, 2],
            options=_options("GatherOptions", axis=-1),
        )
        np.testing.assert_array_equal(result, fixture.value[:, [[3, 0]]])

    def test_gather_with_batch_dims(self):
        fixture = _Fixture(np.arange(12, dtype=np.float32).reshape(2, 3, 2))
        indices = fixture.const("indices", [[2, 0], [1, 1]], dtype=np.int32)
        result = fixture.run(
            "GATHER",
            [fixture.x, indices],
            [2, 2, 2],
            options=_options("GatherOptions", axis=1, batchDims=1),
        )
        expected = np.stack([fixture.value[0][[2, 0]], fixture.value[1][[1, 1]]])
        np.testing.assert_array_equal(result, expected)

    def test_gather_out_of_range_index_is_rejected(self):
        fixture = _Fixture(np.arange(4, dtype=np.float32))
        indices = fixture.const("indices", [4], dtype=np.int32)
        with self.assertRaisesRegex(CircleRuntimeValidationError, r"outside \[0, 4\)"):
            fixture.run(
                "GATHER",
                [fixture.x, indices],
                [1],
                options=_options("GatherOptions", axis=0),
            )

    def test_gather_nd(self):
        fixture = _Fixture(np.arange(12, dtype=np.float32).reshape(3, 4))
        indices = fixture.const("indices", [[0, 1], [2, 3]], dtype=np.int32)
        result = fixture.run(
            "GATHER_ND", [fixture.x, indices], [2], options=_options("GatherNdOptions")
        )
        np.testing.assert_array_equal(result, [1, 11])

        fixture = _Fixture(np.arange(12, dtype=np.float32).reshape(3, 4))
        indices = fixture.const("indices", [[2], [0]], dtype=np.int32)
        result = fixture.run(
            "GATHER_ND",
            [fixture.x, indices],
            [2, 4],
            options=_options("GatherNdOptions"),
        )
        np.testing.assert_array_equal(result, fixture.value[[2, 0]])


class ShapeCastTest(unittest.TestCase):
    def test_shape_of_dynamic_input_follows_runtime_value(self):
        fixture = _Fixture(np.zeros((5, 3), dtype=np.float32), dynamic=True)
        result = fixture.run(
            "SHAPE",
            [fixture.x],
            [2],
            options=_options(
                "ShapeOptions", outType=circle.TensorType.TensorType.INT32
            ),
            out_dtype=np.int32,
        )
        np.testing.assert_array_equal(result, np.array([5, 3], dtype=np.int32))
        self.assertEqual(result.dtype, np.int32)

    def test_shape_out_type_must_match_tensor_type(self):
        fixture = _Fixture(np.zeros((2,), dtype=np.float32))
        with self.assertRaisesRegex(CircleRuntimeValidationError, "outType"):
            fixture.run(
                "SHAPE",
                [fixture.x],
                [1],
                options=_options(
                    "ShapeOptions", outType=circle.TensorType.TensorType.INT64
                ),
                out_dtype=np.int32,
            )

    def test_cast_float_to_int_truncates_and_to_bool_tests_nonzero(self):
        fixture = _Fixture(np.array([-1.7, 0.0, 2.9], dtype=np.float32))
        result = fixture.run(
            "CAST",
            [fixture.x],
            [3],
            options=_options(
                "CastOptions",
                inDataType=circle.TensorType.TensorType.FLOAT32,
                outDataType=circle.TensorType.TensorType.INT32,
            ),
            out_dtype=np.int32,
        )
        np.testing.assert_array_equal(result, [-1, 0, 2])
        self.assertEqual(result.dtype, np.int32)

        fixture = _Fixture(np.array([-1.7, 0.0, 2.9], dtype=np.float32))
        result = fixture.run("CAST", [fixture.x], [3], out_dtype=np.bool_)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_cast_options_must_agree_with_tensor_types(self):
        fixture = _Fixture(np.array([1.0], dtype=np.float32))
        with self.assertRaisesRegex(CircleRuntimeValidationError, "outDataType"):
            fixture.run(
                "CAST",
                [fixture.x],
                [1],
                options=_options(
                    "CastOptions",
                    inDataType=circle.TensorType.TensorType.FLOAT32,
                    outDataType=circle.TensorType.TensorType.INT64,
                ),
                out_dtype=np.int32,
            )


class ReductionTest(unittest.TestCase):
    def test_mean_sum_reduce_max_with_negative_axes_and_keep_dims(self):
        value = np.array([[1.0, 2.0, 6.0], [4.0, 5.0, 3.0]], dtype=np.float32)
        fixture = _Fixture(value)
        axes = fixture.const("axes", [-1])
        result = fixture.run(
            "MEAN",
            [fixture.x, axes],
            [2, 1],
            options=_options("ReducerOptions", keepDims=True),
        )
        np.testing.assert_array_equal(result, [[3.0], [4.0]])

        fixture = _Fixture(value)
        axes = fixture.const("axes", [0, 1])
        result = fixture.run(
            "SUM",
            [fixture.x, axes],
            [],
            options=_options("ReducerOptions", keepDims=False),
        )
        self.assertEqual(result.shape, ())
        self.assertEqual(float(result), 21.0)

        fixture = _Fixture(value)
        axes = fixture.const("axes", [0])
        result = fixture.run(
            "REDUCE_MAX", [fixture.x, axes], [3], options=_options("ReducerOptions")
        )
        np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])

    def test_reduce_max_on_bool_acts_as_any(self):
        fixture = _Fixture(np.array([[False, False], [False, True]], dtype=np.bool_))
        axes = fixture.const("axes", [1])
        result = fixture.run(
            "REDUCE_MAX", [fixture.x, axes], [2], options=_options("ReducerOptions")
        )
        np.testing.assert_array_equal(result, [False, True])
        self.assertEqual(result.dtype, np.bool_)

    def test_arg_max_returns_first_maximum_in_requested_dtype(self):
        fixture = _Fixture(
            np.array([[1.0, 5.0, 5.0], [7.0, 0.0, 7.0]], dtype=np.float32)
        )
        axis = fixture.const("axis", 1)
        result = fixture.run(
            "ARG_MAX",
            [fixture.x, axis],
            [2],
            options=_options(
                "ArgMaxOptions", outputType=circle.TensorType.TensorType.INT64
            ),
            out_dtype=np.int64,
        )
        np.testing.assert_array_equal(result, [1, 0])
        self.assertEqual(result.dtype, np.int64)

    def test_cumsum_exclusive_and_reverse(self):
        value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        fixture = _Fixture(value)
        axis = fixture.const("axis", 0)
        result = fixture.run(
            "CUMSUM", [fixture.x, axis], [4], options=_options("CumsumOptions")
        )
        np.testing.assert_array_equal(result, [1, 3, 6, 10])

        fixture = _Fixture(value)
        axis = fixture.const("axis", 0)
        result = fixture.run(
            "CUMSUM",
            [fixture.x, axis],
            [4],
            options=_options("CumsumOptions", exclusive=True, reverse=True),
        )
        np.testing.assert_array_equal(result, [9, 7, 4, 0])

    def test_cumsum_int64(self):
        fixture = _Fixture(np.array([[1, 2], [3, 4]], dtype=np.int64))
        axis = fixture.const("axis", 1)
        result = fixture.run(
            "CUMSUM", [fixture.x, axis], [2, 2], options=_options("CumsumOptions")
        )
        np.testing.assert_array_equal(result, [[1, 3], [3, 7]])
        self.assertEqual(result.dtype, np.int64)

    def test_softmax_with_beta(self):
        fixture = _Fixture(np.array([[0.0, math_log(3.0)]], dtype=np.float32))
        result = fixture.run(
            "SOFTMAX", [fixture.x], [1, 2], options=_options("SoftmaxOptions", beta=1.0)
        )
        np.testing.assert_allclose(result, [[0.25, 0.75]], rtol=1e-6)

        fixture = _Fixture(np.array([[0.0, math_log(3.0)]], dtype=np.float32))
        result = fixture.run(
            "SOFTMAX", [fixture.x], [1, 2], options=_options("SoftmaxOptions", beta=2.0)
        )
        np.testing.assert_allclose(result, [[0.1, 0.9]], rtol=1e-6)


def math_log(value: float) -> float:
    import math

    return math.log(value)


if __name__ == "__main__":
    unittest.main()

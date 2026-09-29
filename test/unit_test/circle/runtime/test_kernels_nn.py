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

"""Hand-computed checks for convolution, pooling, matrix, resize, and norm kernels.

Fixtures use NHWC activations and Circle filter layouts. Expected values are
derived by hand (or with an independent scalar formula written in the test),
never by calling the runtime.
"""

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

SAME = 0
VALID = 1


def _options(name: str, **fields: Any) -> Any:
    options = getattr(getattr(circle, name), f"{name}T")()
    for key, value in fields.items():
        setattr(options, key, value)
    return options


def _run(builder: CircleModelBuilder, *inputs: np.ndarray) -> np.ndarray:
    return CircleReferenceRuntime(builder.build()).run(inputs).outputs[0]


class Conv2DTest(unittest.TestCase):
    def test_valid_conv_with_stride_and_bias(self):
        """3x3 single-channel image, 2x2 filter of ones, stride 2 -> 1x1 output."""

        builder = CircleModelBuilder()
        x = builder.input("x", [1, 3, 3, 1])
        filters = builder.const_f32("filter", np.ones((1, 2, 2, 1), dtype=np.float32))
        bias = builder.const_f32("bias", [0.5])
        out = builder.activation("out", [1, 1, 1, 1])
        builder.operator(
            "CONV_2D",
            [x, filters, bias],
            [out],
            options=_options(
                "Conv2DOptions",
                padding=VALID,
                strideH=2,
                strideW=2,
                dilationHFactor=1,
                dilationWFactor=1,
            ),
        )
        builder.set_outputs(out)
        image = np.arange(9, dtype=np.float32).reshape((1, 3, 3, 1))
        # window [[0,1],[3,4]] -> 8, plus bias 0.5
        np.testing.assert_array_equal(_run(builder, image), [[[[8.5]]]])

    def test_same_padding_puts_extra_padding_at_the_end(self):
        """2x2 image, 2x2 filter, stride 1, SAME: total pad 1 -> bottom/right only."""

        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2, 2, 1])
        filters = builder.const_f32("filter", np.ones((1, 2, 2, 1), dtype=np.float32))
        out = builder.activation("out", [1, 2, 2, 1])
        builder.operator(
            "CONV_2D",
            [x, filters, -1],
            [out],
            options=_options(
                "Conv2DOptions",
                padding=SAME,
                strideH=1,
                strideW=1,
                dilationHFactor=1,
                dilationWFactor=1,
            ),
        )
        builder.set_outputs(out)
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        expected = np.array([[10.0, 6.0], [7.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        np.testing.assert_array_equal(_run(builder, image), expected)

    def test_multi_channel_filter_layout_ohwi_and_fused_relu(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 1, 1, 2])
        # Two output channels: [ic0 + 2*ic1, -(ic0 + ic1)]
        filters = builder.const_f32(
            "filter", np.array([[[[1.0, 2.0]]], [[[-1.0, -1.0]]]], dtype=np.float32)
        )
        bias = builder.const_f32("bias", [0.0, 0.0])
        out = builder.activation("out", [1, 1, 1, 2])
        builder.operator(
            "CONV_2D",
            [x, filters, bias],
            [out],
            options=_options(
                "Conv2DOptions",
                padding=VALID,
                strideH=1,
                strideW=1,
                dilationHFactor=1,
                dilationWFactor=1,
                fusedActivationFunction=1,
            ),
        )
        builder.set_outputs(out)
        image = np.array([[[[3.0, 5.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(_run(builder, image), [[[[13.0, 0.0]]]])

    def test_dilation(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 3, 3, 1])
        filters = builder.const_f32("filter", np.ones((1, 2, 2, 1), dtype=np.float32))
        out = builder.activation("out", [1, 1, 1, 1])
        builder.operator(
            "CONV_2D",
            [x, filters, -1],
            [out],
            options=_options(
                "Conv2DOptions",
                padding=VALID,
                strideH=1,
                strideW=1,
                dilationHFactor=2,
                dilationWFactor=2,
            ),
        )
        builder.set_outputs(out)
        image = np.arange(9, dtype=np.float32).reshape((1, 3, 3, 1))
        # corners: 0 + 2 + 6 + 8
        np.testing.assert_array_equal(_run(builder, image), [[[[16.0]]]])

    def test_channel_mismatch_is_rejected(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2, 2, 3])
        filters = builder.const_f32("filter", np.ones((1, 1, 1, 2), dtype=np.float32))
        out = builder.activation("out", [1, 2, 2, 1])
        builder.operator(
            "CONV_2D",
            [x, filters, -1],
            [out],
            options=_options("Conv2DOptions", padding=VALID, strideH=1, strideW=1),
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "filter input channels 2"
        ):
            _run(builder, np.zeros((1, 2, 2, 3), dtype=np.float32))


class DepthwiseConv2DTest(unittest.TestCase):
    def test_depth_multiplier_channel_order(self):
        """Two input channels, multiplier 2: output channel oc = ic * 2 + m."""

        builder = CircleModelBuilder()
        x = builder.input("x", [1, 1, 1, 2])
        # filter [1, 1, 1, 4]: oc0 = ic0*1, oc1 = ic0*10, oc2 = ic1*100, oc3 = ic1*1000
        filters = builder.const_f32(
            "filter", np.array([[[[1.0, 10.0, 100.0, 1000.0]]]], dtype=np.float32)
        )
        bias = builder.const_f32("bias", [0.0, 0.0, 0.0, 0.0])
        out = builder.activation("out", [1, 1, 1, 4])
        builder.operator(
            "DEPTHWISE_CONV_2D",
            [x, filters, bias],
            [out],
            options=_options(
                "DepthwiseConv2DOptions",
                padding=VALID,
                strideH=1,
                strideW=1,
                depthMultiplier=2,
                dilationHFactor=1,
                dilationWFactor=1,
            ),
        )
        builder.set_outputs(out)
        image = np.array([[[[2.0, 3.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(
            _run(builder, image), [[[[2.0, 20.0, 300.0, 3000.0]]]]
        )

    def test_same_padding_spatial_window(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2, 2, 1])
        filters = builder.const_f32("filter", np.ones((1, 3, 3, 1), dtype=np.float32))
        bias = builder.const_f32("bias", [1.0])
        out = builder.activation("out", [1, 2, 2, 1])
        builder.operator(
            "DEPTHWISE_CONV_2D",
            [x, filters, bias],
            [out],
            options=_options(
                "DepthwiseConv2DOptions",
                padding=SAME,
                strideH=1,
                strideW=1,
                depthMultiplier=1,
            ),
        )
        builder.set_outputs(out)
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        # every 3x3 window covers the whole image (pad 1 on each side): sum 10 + bias 1
        np.testing.assert_array_equal(
            _run(builder, image), np.full((1, 2, 2, 1), 11.0, dtype=np.float32)
        )


class TransposeConvTest(unittest.TestCase):
    def _transpose_conv(
        self,
        image: np.ndarray,
        filters: np.ndarray,
        output_shape: list[int],
        *,
        stride: int,
        padding: int,
        bias: list[float] | None = None,
    ) -> np.ndarray:
        builder = CircleModelBuilder()
        shape = builder.const_i32("output_shape", output_shape)
        weight = builder.const_f32("filter", filters)
        x = builder.input("x", list(image.shape))
        inputs = [shape, weight, x]
        if bias is not None:
            inputs.append(builder.const_f32("bias", bias))
        else:
            inputs.append(-1)
        out = builder.activation("out", output_shape)
        builder.operator(
            "TRANSPOSE_CONV",
            inputs,
            [out],
            options=_options(
                "TransposeConvOptions", padding=padding, strideH=stride, strideW=stride
            ),
        )
        builder.set_outputs(out)
        return _run(builder, image)

    def test_valid_stride_two_scatter(self):
        """1x2 input, 2x2 filter of ones, stride 2 -> each input pixel fills a 2x2 block."""

        image = np.array([[[[1.0], [2.0]]]], dtype=np.float32)  # [1, 1, 2, 1]
        filters = np.ones((1, 2, 2, 1), dtype=np.float32)  # [oc, kh, kw, ic]
        result = self._transpose_conv(
            image, filters, [1, 2, 4, 1], stride=2, padding=VALID, bias=[0.5]
        )
        expected = np.array(
            [[[1.5], [1.5], [2.5], [2.5]], [[1.5], [1.5], [2.5], [2.5]]],
            dtype=np.float32,
        )[None]
        np.testing.assert_array_equal(result, expected)

    def test_same_padding_crops_symmetrically(self):
        """2x2 input, 3x3 filter of ones, stride 1, SAME output 2x2 -> pad 1 on each side."""

        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        filters = np.ones((1, 3, 3, 1), dtype=np.float32)
        result = self._transpose_conv(
            image, filters, [1, 2, 2, 1], stride=1, padding=SAME
        )
        # Full 4x4 scatter of a 2x2 image with a 3x3 ones filter, cropped by one
        # on each side, leaves every position covering the entire image: 10.
        np.testing.assert_array_equal(
            result, np.full((1, 2, 2, 1), 10.0, dtype=np.float32)
        )

    def test_output_shape_must_match_batch_and_channels(self):
        image = np.zeros((1, 1, 1, 1), dtype=np.float32)
        filters = np.ones((2, 1, 1, 1), dtype=np.float32)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError,
            "does not match batch 1 and filter output channels 2",
        ):
            self._transpose_conv(image, filters, [1, 1, 1, 1], stride=1, padding=VALID)


class PoolingTest(unittest.TestCase):
    def _pool(
        self, builtin: str, image: np.ndarray, out_shape: list[int], **fields: Any
    ) -> np.ndarray:
        builder = CircleModelBuilder()
        x = builder.input("x", list(image.shape))
        out = builder.activation("out", out_shape)
        builder.operator(
            builtin, [x], [out], options=_options("Pool2DOptions", **fields)
        )
        builder.set_outputs(out)
        return _run(builder, image)

    def test_average_pool_same_padding_excludes_padded_cells(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        result = self._pool(
            "AVERAGE_POOL_2D",
            image,
            [1, 2, 2, 1],
            padding=SAME,
            strideH=1,
            strideW=1,
            filterHeight=2,
            filterWidth=2,
        )
        # windows: {1,2,3,4}->2.5, {2,4}->3, {3,4}->3.5, {4}->4
        expected = np.array([[2.5, 3.0], [3.5, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        np.testing.assert_array_equal(result, expected)

    def test_max_pool_valid_with_stride(self):
        image = np.arange(16, dtype=np.float32).reshape((1, 4, 4, 1))
        result = self._pool(
            "MAX_POOL_2D",
            image,
            [1, 2, 2, 1],
            padding=VALID,
            strideH=2,
            strideW=2,
            filterHeight=2,
            filterWidth=2,
        )
        expected = np.array([[5.0, 7.0], [13.0, 15.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        np.testing.assert_array_equal(result, expected)

    def test_max_pool_same_padding_ignores_padding(self):
        image = np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        result = self._pool(
            "MAX_POOL_2D",
            image,
            [1, 1, 1, 1],
            padding=SAME,
            strideH=2,
            strideW=2,
            filterHeight=3,
            filterWidth=3,
        )
        np.testing.assert_array_equal(result, [[[[-1.0]]]])


class MatrixTest(unittest.TestCase):
    def test_fully_connected_keep_num_dims_and_optional_bias(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [2, 1, 2])
        weights = builder.const_f32("weights", [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = builder.activation("out", [2, 1, 3])
        builder.operator(
            "FULLY_CONNECTED",
            [x, weights, -1],
            [out],
            options=_options(
                "FullyConnectedOptions", keepNumDims=True, weightsFormat=0
            ),
        )
        builder.set_outputs(out)
        value = np.array([[[1.0, 1.0]], [[0.0, 2.0]]], dtype=np.float32)
        np.testing.assert_array_equal(
            _run(builder, value), [[[3.0, 7.0, 11.0]], [[4.0, 8.0, 12.0]]]
        )

    def test_fully_connected_flattens_without_keep_num_dims(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2, 2])
        weights = builder.const_f32("weights", [[1.0, 0.0]])
        bias = builder.const_f32("bias", [10.0])
        out = builder.activation("out", [2, 1])
        builder.operator(
            "FULLY_CONNECTED",
            [x, weights, bias],
            [out],
            options=_options("FullyConnectedOptions", keepNumDims=False),
        )
        builder.set_outputs(out)
        value = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        np.testing.assert_array_equal(_run(builder, value), [[11.0], [13.0]])

    def test_fully_connected_rejects_shuffled_weights_format(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2])
        weights = builder.const_f32("weights", [[1.0, 0.0]])
        out = builder.activation("out", [1, 1])
        builder.operator(
            "FULLY_CONNECTED",
            [x, weights, -1],
            [out],
            options=_options("FullyConnectedOptions", weightsFormat=1),
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "weightsFormat 1"):
            _run(builder, np.zeros((1, 2), dtype=np.float32))

    def test_batch_matmul_with_broadcast_and_adjoint(self):
        builder = CircleModelBuilder()
        lhs = builder.input("lhs", [2, 2, 3])
        rhs = builder.const_f32("rhs", np.arange(6, dtype=np.float32).reshape(1, 2, 3))
        out = builder.activation("out", [2, 2, 2])
        builder.operator(
            "BATCH_MATMUL",
            [lhs, rhs],
            [out],
            options=_options("BatchMatMulOptions", adjointLhs=False, adjointRhs=True),
        )
        builder.set_outputs(out)
        value = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        expected = np.matmul(
            value, np.swapaxes(np.arange(6, dtype=np.float32).reshape(1, 2, 3), -1, -2)
        )
        np.testing.assert_array_equal(_run(builder, value), expected)

    def test_batch_matmul_contraction_mismatch_is_rejected(self):
        builder = CircleModelBuilder()
        lhs = builder.input("lhs", [2, 3])
        rhs = builder.const_f32("rhs", np.zeros((2, 3), dtype=np.float32))
        out = builder.activation("out", [2, 3])
        builder.operator(
            "BATCH_MATMUL", [lhs, rhs], [out], options=_options("BatchMatMulOptions")
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "contraction sizes differ"
        ):
            _run(builder, np.zeros((2, 3), dtype=np.float32))


class ResizeTest(unittest.TestCase):
    def _resize(
        self,
        builtin: str,
        image: np.ndarray,
        size: list[int],
        *,
        align_corners: bool,
        half_pixel: bool,
    ) -> np.ndarray:
        builder = CircleModelBuilder()
        x = builder.input("x", list(image.shape))
        size_tensor = builder.const_i32("size", size)
        out = builder.activation(
            "out", [image.shape[0], size[0], size[1], image.shape[3]]
        )
        options_name = (
            "ResizeBilinearOptions"
            if builtin == "RESIZE_BILINEAR"
            else "ResizeNearestNeighborOptions"
        )
        builder.operator(
            builtin,
            [x, size_tensor],
            [out],
            options=_options(
                options_name, alignCorners=align_corners, halfPixelCenters=half_pixel
            ),
        )
        builder.set_outputs(out)
        return _run(builder, image)

    def test_bilinear_asymmetric_legacy(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        result = self._resize(
            "RESIZE_BILINEAR", image, [4, 4], align_corners=False, half_pixel=False
        )
        # scale 0.5: source coordinate y*0.5 -> rows [1,1.5,2,2] etc.
        expected_row0 = [1.0, 1.5, 2.0, 2.0]
        np.testing.assert_allclose(result[0, 0, :, 0], expected_row0, rtol=1e-6)
        np.testing.assert_allclose(result[0, 1, :, 0], [2.0, 2.5, 3.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(result[0, 3, :, 0], [3.0, 3.5, 4.0, 4.0], rtol=1e-6)

    def test_bilinear_half_pixel_centers(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32).reshape(
            (1, 2, 2, 1)
        )
        result = self._resize(
            "RESIZE_BILINEAR", image, [4, 4], align_corners=False, half_pixel=True
        )
        # source x = (dst + 0.5) * 0.5 - 0.5 -> [-0.25, 0.25, 0.75, 1.25] clamped to [0, 1]
        np.testing.assert_allclose(
            result[0, 0, :, 0], [1.0, 1.25, 1.75, 2.0], rtol=1e-6
        )

    def test_bilinear_align_corners(self):
        image = np.array([[1.0, 4.0]], dtype=np.float32).reshape((1, 1, 2, 1))
        result = self._resize(
            "RESIZE_BILINEAR", image, [1, 4], align_corners=True, half_pixel=False
        )
        np.testing.assert_allclose(result[0, 0, :, 0], [1.0, 2.0, 3.0, 4.0], rtol=1e-6)

    def test_bilinear_rejects_both_flags(self):
        image = np.zeros((1, 1, 2, 1), dtype=np.float32)
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "both alignCorners and halfPixelCenters"
        ):
            self._resize(
                "RESIZE_BILINEAR", image, [1, 4], align_corners=True, half_pixel=True
            )

    def test_nearest_neighbor_floor_and_half_pixel(self):
        image = np.array([[10.0, 20.0, 30.0]], dtype=np.float32).reshape((1, 1, 3, 1))
        legacy = self._resize(
            "RESIZE_NEAREST_NEIGHBOR",
            image,
            [1, 5],
            align_corners=False,
            half_pixel=False,
        )
        # floor(x * 0.6): 0,0,1,1,2
        np.testing.assert_array_equal(
            legacy[0, 0, :, 0], [10.0, 10.0, 20.0, 20.0, 30.0]
        )
        half = self._resize(
            "RESIZE_NEAREST_NEIGHBOR",
            image,
            [1, 5],
            align_corners=False,
            half_pixel=True,
        )
        # floor((x + 0.5) * 0.6): 0,0,1,2,2
        np.testing.assert_array_equal(half[0, 0, :, 0], [10.0, 10.0, 20.0, 30.0, 30.0])


class NormalizationTest(unittest.TestCase):
    def test_instance_norm_nhwc(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 1, 4, 1])
        gamma = builder.const_f32("gamma", [2.0])
        beta = builder.const_f32("beta", [1.0])
        out = builder.activation("out", [1, 1, 4, 1])
        builder.operator(
            "INSTANCE_NORM",
            [x, gamma, beta],
            [out],
            options=_options("InstanceNormOptions", epsilon=0.0),
        )
        builder.set_outputs(out)
        value = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape((1, 1, 4, 1))
        mean, var = 2.5, 1.25
        expected = [
            (v - mean) / math.sqrt(var) * 2.0 + 1.0 for v in [1.0, 2.0, 3.0, 4.0]
        ]
        np.testing.assert_allclose(
            _run(builder, value)[0, 0, :, 0], expected, rtol=1e-6
        )

    def test_rms_norm_last_axis_with_gamma(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 1, 2])
        gamma = builder.const_f32("gamma", [1.0, 2.0])
        out = builder.activation("out", [1, 1, 2])
        builder.operator(
            "RMS_NORM",
            [x, gamma],
            [out],
            options=_options("RmsNormOptions", epsilon=1e-6),
        )
        builder.set_outputs(out)
        value = np.array([[[3.0, 4.0]]], dtype=np.float32)
        rms = math.sqrt((9.0 + 16.0) / 2.0 + 1e-6)
        np.testing.assert_allclose(
            _run(builder, value), [[[3.0 / rms, 8.0 / rms]]], rtol=1e-6
        )

    def test_rms_norm_rejects_rank_2(self):
        builder = CircleModelBuilder()
        x = builder.input("x", [1, 2])
        gamma = builder.const_f32("gamma", [1.0, 1.0])
        out = builder.activation("out", [1, 2])
        builder.operator(
            "RMS_NORM",
            [x, gamma],
            [out],
            options=_options("RmsNormOptions", epsilon=1e-6),
        )
        builder.set_outputs(out)
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "rank 3 or 4"):
            _run(builder, np.zeros((1, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

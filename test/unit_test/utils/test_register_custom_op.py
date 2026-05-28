# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import unittest
from unittest.mock import MagicMock, patch

import tico.utils.register_custom_op as register_custom_op
import torch


class TestRegisterCustomOp(unittest.TestCase):
    """Test cases for tico.utils.register_custom_op module"""

    def setUp(self):
        """Set up test fixtures"""
        # Register all custom ops
        register_custom_op.RegisterOps()

    def test_circle_resize_nearest_neighbor_basic(self):
        """Test CircleResizeNearestNeighbor basic functionality"""
        input_tensor = torch.randn(1, 32, 32, 3)
        size = [1, 64, 64, 3]

        result = torch.ops.circle_custom.resize_nearest_neighbor(input_tensor, size)

        # Check output shape
        self.assertEqual(result.shape, torch.Size(size))

        # Check that scale factors are equal
        H_scale_factor = size[1] / input_tensor.size(1)
        W_scale_factor = size[2] / input_tensor.size(2)
        self.assertEqual(H_scale_factor, W_scale_factor)

    def test_circle_resize_nearest_neighbor_unequal_scale_factors(self):
        """Test CircleResizeNearestNeighbor with unequal scale factors"""
        input_tensor = torch.randn(1, 32, 32, 3)
        size = [1, 64, 48, 3]  # Unequal scale factors

        with self.assertRaises(RuntimeError) as context:
            torch.ops.circle_custom.resize_nearest_neighbor(input_tensor, size)

        self.assertIn("Scale factor of H and W should be same", str(context.exception))

    def test_circle_conv2d_with_defaults(self):
        """Test CircleConv2d with default parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)  # NHWC
        weight_tensor = torch.randn(16, 3, 3, 3)  # OHWI

        result = torch.ops.circle_custom.conv2d(input_tensor, weight_tensor)

        # Check output shape
        expected_shape = [1, 30, 30, 16]  # NHWC with no padding
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_conv2d_with_custom_params(self):
        """Test CircleConv2d with custom parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)
        weight_tensor = torch.randn(16, 3, 3, 3)
        bias_tensor = torch.randn(16)
        stride = [2, 2]
        padding = [1, 1]

        result = torch.ops.circle_custom.conv2d(
            input_tensor, weight_tensor, bias_tensor, stride, padding
        )

        # Check output shape with custom params
        expected_shape = [1, 16, 16, 16]  # NHWC with stride=2, padding=1
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_conv2d_invalid_groups(self):
        """Test CircleConv2d with invalid groups parameter"""
        input_tensor = torch.randn(1, 32, 32, 3)
        weight_tensor = torch.randn(16, 3, 3, 3)

        with self.assertRaises(RuntimeError) as context:
            torch.ops.circle_custom.conv2d(input_tensor, weight_tensor, groups=2)

        self.assertIn("CircleConv2d only supports 1 'groups'", str(context.exception))

    def test_circle_conv2d_padding_with_defaults(self):
        """Test CircleConv2dPadding with default parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)
        weight_tensor = torch.randn(16, 3, 3, 3)

        result = torch.ops.circle_custom.conv2d.padding(input_tensor, weight_tensor)

        # Check output shape
        expected_shape = [1, 30, 30, 16]  # NHWC with "valid" padding
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_conv2d_padding_with_string_padding(self):
        """Test CircleConv2dPadding with string padding"""
        input_tensor = torch.randn(1, 32, 32, 3)
        weight_tensor = torch.randn(16, 3, 3, 3)

        result = torch.ops.circle_custom.conv2d.padding(
            input_tensor, weight_tensor, padding="same"
        )

        # Check output shape with "same" padding
        expected_shape = [1, 32, 32, 16]  # NHWC with "same" padding
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_depthwise_conv2d_basic(self):
        """Test CircleDepthwiseConv2d basic functionality"""
        input_tensor = torch.randn(1, 32, 32, 16)  # NHWC, 16 channels
        weight_tensor = torch.randn(1, 3, 3, 16)  # OHWI format
        groups = 16

        result = torch.ops.circle_custom.depthwise_conv2d(
            input_tensor, weight_tensor, groups=groups
        )

        # Check output shape
        expected_shape = [1, 30, 30, 16]  # NHWC with no padding
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_depthwise_conv2d_with_custom_params(self):
        """Test CircleDepthwiseConv2d with custom parameters"""
        input_tensor = torch.randn(1, 32, 32, 16)
        weight_tensor = torch.randn(1, 3, 3, 16)
        bias_tensor = torch.randn(16)
        stride = [2, 2]
        padding = [1, 1]
        groups = 16

        result = torch.ops.circle_custom.depthwise_conv2d(
            input_tensor, weight_tensor, bias_tensor, stride, padding, None, groups
        )

        # Check output shape with custom params
        expected_shape = [1, 16, 16, 16]  # NHWC with stride=2, padding=1
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_depthwise_conv2d_invalid_groups_assertion(self):
        """Test CircleDepthwiseConv2d with invalid groups assertion"""
        input_tensor = torch.randn(1, 32, 32, 16)
        weight_tensor = torch.randn(1, 3, 3, 16)

        with self.assertRaises(AssertionError):
            torch.ops.circle_custom.depthwise_conv2d(
                input_tensor, weight_tensor, groups=1
            )

    def test_circle_depthwise_conv2d_padding_basic(self):
        """Test CircleDepthwiseConv2dPadding basic functionality"""
        input_tensor = torch.randn(1, 32, 32, 16)
        weight_tensor = torch.randn(1, 3, 3, 16)
        groups = 16

        result = torch.ops.circle_custom.depthwise_conv2d.padding(
            input_tensor, weight_tensor, groups=groups
        )

        # Check output shape
        expected_shape = [1, 30, 30, 16]  # NHWC with "valid" padding
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_transpose_conv_basic(self):
        """Test CircleTransposeConv basic functionality"""
        input_tensor = torch.randn(1, 16, 16, 16)  # NHWC
        weight_tensor = torch.randn(16, 3, 3, 16)  # OHWI format

        result = torch.ops.circle_custom.transpose_conv(input_tensor, weight_tensor)

        # Check output shape
        expected_shape = [1, 18, 18, 16]  # NHWC with no padding
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_transpose_conv_with_custom_params(self):
        """Test CircleTransposeConv with custom parameters"""
        input_tensor = torch.randn(1, 16, 16, 16)
        weight_tensor = torch.randn(16, 3, 3, 16)  # Output channels is 16
        bias_tensor = torch.randn(16)  # Bias size should match output channels
        stride = [2, 2]
        padding = [1, 1]
        output_padding = [1, 1]

        result = torch.ops.circle_custom.transpose_conv(
            input_tensor, weight_tensor, bias_tensor, stride, padding, output_padding
        )

        # Check output shape with custom params
        expected_shape = [
            1,
            32,
            32,
            16,
        ]  # NHWC with stride=2, padding=1, output_padding=1
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_transpose_conv_invalid_groups(self):
        """Test CircleTransposeConv with invalid groups parameter"""
        input_tensor = torch.randn(1, 16, 16, 16)
        weight_tensor = torch.randn(16, 3, 3, 16)

        with self.assertRaises(RuntimeError) as context:
            torch.ops.circle_custom.transpose_conv(
                input_tensor, weight_tensor, groups=2
            )

        self.assertIn(
            "CircleTransposeConv only supports 1 'groups'", str(context.exception)
        )

    def test_circle_maxpool2d_with_defaults(self):
        """Test CircleMaxPool2D with default parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)
        kernel_size = [2, 2]

        result = torch.ops.circle_custom.maxpool2d(input_tensor, kernel_size)

        # Check output shape
        expected_shape = [1, 16, 16, 3]  # NHWC with kernel_size=2, stride=2
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_maxpool2d_with_custom_params(self):
        """Test CircleMaxPool2D with custom parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)
        kernel_size = [3, 3]
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        ceil_mode = True

        result = torch.ops.circle_custom.maxpool2d(
            input_tensor, kernel_size, stride, padding, dilation, ceil_mode
        )

        # Check output shape with custom params
        expected_shape = [1, 32, 32, 3]
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_avgpool2d_with_defaults(self):
        """Test CircleAvgPool2D with default parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)
        kernel_size = [2, 2]

        result = torch.ops.circle_custom.avgpool2d(input_tensor, kernel_size)

        # Check output shape
        expected_shape = [1, 16, 16, 3]  # NHWC with kernel_size=2, stride=2
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_avgpool2d_with_custom_params(self):
        """Test CircleAvgPool2D with custom parameters"""
        input_tensor = torch.randn(1, 32, 32, 3)
        kernel_size = [3, 3]
        stride = [1, 1]
        padding = [1, 1]
        ceil_mode = True
        count_include_pad = False
        divisor_override = 4

        result = torch.ops.circle_custom.avgpool2d(
            input_tensor,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

        # Check output shape with custom params
        expected_shape = [1, 32, 32, 3]  # NHWC with kernel_size=3, stride=1, padding=1
        self.assertEqual(list(result.shape), expected_shape)

    def test_circle_instance_norm_basic(self):
        """Test CircleInstanceNorm basic functionality"""
        input_tensor = torch.randn(2, 32, 32, 3)  # NHWC
        weight_tensor = torch.randn(3)
        bias_tensor = torch.randn(3)

        result = torch.ops.circle_custom.instance_norm(
            input_tensor, weight_tensor, bias_tensor
        )

        # Check output shape
        self.assertEqual(list(result.shape), list(input_tensor.shape))

    def test_circle_instance_norm_with_custom_params(self):
        """Test CircleInstanceNorm with custom parameters"""
        input_tensor = torch.randn(2, 32, 32, 3)
        weight_tensor = torch.randn(3)
        bias_tensor = torch.randn(3)
        momentum = 0.2
        eps = 1e-4

        result = torch.ops.circle_custom.instance_norm(
            input_tensor,
            weight_tensor,
            bias_tensor,
            None,
            None,
            False,
            momentum,
            eps,
            False,
        )

        # Check output shape
        self.assertEqual(list(result.shape), list(input_tensor.shape))

    def test_circle_quantize_mx_int8(self):
        """Test CircleQuantizeMX with int8 format"""
        input_tensor = torch.randn(2, 32, 32, 3)
        elem_format = "int8"
        axis = -1

        result = torch.ops.circle_custom.quantize_mx(input_tensor, elem_format, axis)

        # Check output shape
        self.assertEqual(list(result.shape), list(input_tensor.shape))

    def test_circle_quantize_mx_unsupported_format(self):
        """Test CircleQuantizeMX with unsupported format"""
        input_tensor = torch.randn(2, 32, 32, 3)
        elem_format = "unsupported_format"
        axis = -1

        with self.assertRaises(RuntimeError) as context:
            torch.ops.circle_custom.quantize_mx(input_tensor, elem_format, axis)

        self.assertIn("Unsupported elem_format in quantize_mx", str(context.exception))

    def test_circle_quantize_mx_with_custom_params(self):
        """Test CircleQuantizeMX with custom parameters"""
        input_tensor = torch.randn(2, 32, 32, 3)
        elem_format = "int8"
        axis = -1
        shared_exp_method = "max"
        round_method = "nearest"

        result = torch.ops.circle_custom.quantize_mx(
            input_tensor, elem_format, axis, shared_exp_method, round_method
        )

        # Check output shape
        self.assertEqual(list(result.shape), list(input_tensor.shape))

    def test_circle_rms_norm_basic(self):
        """Test CircleRMSNorm basic functionality"""
        hidden_states = torch.randn(2, 32, 3)
        weight = torch.randn(3)

        result = torch.ops.circle_custom.rms_norm(hidden_states, weight)

        # Check output shape
        self.assertEqual(list(result.shape), list(hidden_states.shape))

    def test_circle_rms_norm_with_custom_eps(self):
        """Test CircleRMSNorm with custom epsilon"""
        hidden_states = torch.randn(2, 32, 3)
        weight = torch.randn(3)
        eps = 1e-4

        result = torch.ops.circle_custom.rms_norm(hidden_states, weight, eps)

        # Check output shape
        self.assertEqual(list(result.shape), list(hidden_states.shape))

    def test_custom_ops_with_different_tensor_shapes(self):
        """Test custom ops with different tensor shapes"""
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            input_tensor = torch.randn(batch_size, 16, 16, 3)
            weight_tensor = torch.randn(8, 3, 3, 3)

            result = torch.ops.circle_custom.conv2d(input_tensor, weight_tensor)
            expected_batch_size = batch_size
            self.assertEqual(result.shape[0], expected_batch_size)

    def test_custom_ops_with_conv2d_different_data_types(self):
        """Test custom ops with conv2d different data types"""
        # Test with different data types
        for dtype in [torch.float32, torch.float16]:
            input_tensor = torch.randn(1, 16, 16, 3, dtype=dtype)
            weight_tensor = torch.randn(8, 3, 3, 3, dtype=dtype)

            result = torch.ops.circle_custom.conv2d(input_tensor, weight_tensor)
            self.assertEqual(result.dtype, dtype)

    @patch("tico.utils.register_custom_op._quantize_mx")
    def test_circle_quantize_mx_mocked(self, mock_quantize_mx):
        """Test CircleQuantizeMX with mocked _quantize_mx function"""
        mock_quantize_mx.return_value = torch.randn(2, 32, 32, 3)

        input_tensor = torch.randn(2, 32, 32, 3)
        elem_format = "int8"
        axis = -1

        result = torch.ops.circle_custom.quantize_mx(input_tensor, elem_format, axis)

        # Check that _quantize_mx was called with correct parameters
        mock_quantize_mx.assert_called_once_with(
            input_tensor,
            scale_bits=8,
            elem_format=elem_format,
            axes=[axis],
            block_size=32,
            shared_exp_method="max",
            round="nearest",
        )


if __name__ == "__main__":
    unittest.main()

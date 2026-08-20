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

"""Tests for backend-specific data-operator qparam verification."""

import unittest

from examples.hand_detector._support.verify_quantized_circle import (
    _require_data_operator_qparams,
    CircleTensorInfo,
    MAX_POOL_2D,
    PAD,
    QuantizationInfo,
    TENSOR_UINT8,
)


def _tensor(name: str, scale: float, zero_point: int) -> CircleTensorInfo:
    return CircleTensorInfo(
        shape=(1, 4, 4, 3),
        tensor_type=TENSOR_UINT8,
        buffer_index=0,
        name=name,
        quantization=QuantizationInfo(
            scales=(scale,),
            zero_points=(zero_point,),
            quantized_dimension=0,
        ),
    )


class HandDetectorQuantizedCircleVerifierTest(unittest.TestCase):
    """Verify that only backend-constrained operators require shared qparams."""

    def test_max_pool_accepts_distinct_input_and_output_qparams(self) -> None:
        """Allow independent per-tensor affine domains across MaxPool2D."""
        _require_data_operator_qparams(
            MAX_POOL_2D,
            (_tensor("input", 0.125, 117), _tensor("output", 0.25, 103)),
            context="MAX_POOL_2D",
        )

    def test_pad_still_requires_identical_input_and_output_qparams(self) -> None:
        """Retain the existing shared-domain constraint for zero padding."""
        with self.assertRaisesRegex(RuntimeError, "identical scale and zero point"):
            _require_data_operator_qparams(
                PAD,
                (_tensor("input", 0.125, 117), _tensor("output", 0.25, 103)),
                context="PAD",
            )


if __name__ == "__main__":
    unittest.main()

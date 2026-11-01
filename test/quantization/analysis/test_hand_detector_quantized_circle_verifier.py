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
    _require_same_qparams,
    _require_supported_quantize_transition,
    CircleTensorInfo,
    MAX_POOL_2D,
    PAD,
    QuantizationInfo,
    TENSOR_INT16,
    TENSOR_UINT8,
)


def _tensor(
    name: str, scale: float, zero_point: int, tensor_type: int = TENSOR_UINT8
) -> CircleTensorInfo:
    return CircleTensorInfo(
        shape=(1, 4, 4, 3),
        tensor_type=tensor_type,
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

    def test_concatenation_requires_one_domain_on_every_axis(self) -> None:
        """Concat inputs and output share one grid; export converts the rest."""
        _require_same_qparams(
            (
                _tensor("lhs", 0.125, 117),
                _tensor("rhs", 0.125, 117),
                _tensor("out", 0.125, 117),
            ),
            context="CONCATENATION@0",
        )
        with self.assertRaisesRegex(RuntimeError, "identical scale and zero point"):
            _require_same_qparams(
                (
                    _tensor("lhs", 0.125, 117),
                    _tensor("rhs", 0.25, 117),
                    _tensor("out", 0.125, 117),
                ),
                context="CONCATENATION@0",
            )

    def test_quantize_transitions_follow_measured_triv24_support(self) -> None:
        """Accept Q8<->Q16 and Q16->Q16 grid changes; reject Q8->Q8 and identity."""
        int16 = TENSOR_INT16
        self.assertEqual(
            _require_supported_quantize_transition(
                _tensor("in", 0.125, 117),
                _tensor("out", 0.001, 0, int16),
                context="QUANTIZE@0",
            ),
            (TENSOR_UINT8, int16),
        )
        self.assertEqual(
            _require_supported_quantize_transition(
                _tensor("in", 0.001, 0, int16),
                _tensor("out", 0.002, 0, int16),
                context="QUANTIZE@0",
            ),
            (int16, int16),
        )
        with self.assertRaisesRegex(RuntimeError, "identity requantization"):
            _require_supported_quantize_transition(
                _tensor("in", 0.001, 0, int16),
                _tensor("out", 0.001, 0, int16),
                context="QUANTIZE@0",
            )
        with self.assertRaisesRegex(RuntimeError, "UINT8->UINT8 requantization"):
            _require_supported_quantize_transition(
                _tensor("in", 0.125, 117),
                _tensor("out", 0.25, 103),
                context="QUANTIZE@0",
            )


if __name__ == "__main__":
    unittest.main()

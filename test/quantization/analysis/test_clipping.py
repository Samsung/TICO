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

"""Tests for model-independent output clipping analysis."""

import unittest

import torch

from tico.quantization.analysis import (
    AffineQuantizationPolicy,
    build_clipping_candidates,
    collect_output_calibration_data,
    evaluate_clipping_candidates,
    make_output_adapter,
    metric_float,
)
from torch import nn


class IdentityTuple(nn.Module):
    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor]:
        return (input_,)


class OutputClippingTest(unittest.TestCase):
    def test_percentile_can_reduce_error_for_rare_outlier(self) -> None:
        model = IdentityTuple()
        values = torch.cat((torch.linspace(-1.0, 1.0, 100_000), torch.tensor([100.0])))
        samples = [values]
        adapter = make_output_adapter(("value",))
        data = collect_output_calibration_data(
            model,
            samples,
            output_adapter=adapter,
            max_values_per_output=2000,
        )
        candidates = build_clipping_candidates(
            data[0],
            AffineQuantizationPolicy.uint8(),
            percentiles=(99.9,),
            tail_percentages=(0.0, 0.1),
            include_l1_search=True,
        )
        evaluated = evaluate_clipping_candidates(
            model,
            samples,
            data,
            {"value": candidates},
            AffineQuantizationPolicy.uint8(),
            output_adapter=adapter,
        )["value"]
        by_name = {item.candidate.name: item for item in evaluated}
        self.assertLess(
            metric_float(by_name["p99_9"].evaluation_error, "mae"),
            metric_float(by_name["minmax"].evaluation_error, "mae"),
        )
        self.assertIn("l1_optimal", by_name)

    def test_int16_policy_has_finer_scale_than_uint8(self) -> None:
        from tico.quantization.analysis import OutputTensorQuantizer

        uint8 = OutputTensorQuantizer.from_range(
            "x", AffineQuantizationPolicy.uint8(), -10.0, 20.0
        )
        int16 = OutputTensorQuantizer.from_range(
            "x", AffineQuantizationPolicy.int16(), -10.0, 20.0
        )
        self.assertLess(int16.qparams.scale, uint8.qparams.scale)


if __name__ == "__main__":
    unittest.main()

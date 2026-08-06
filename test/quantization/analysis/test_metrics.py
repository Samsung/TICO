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

"""Tests for reusable output normalization and numerical metrics."""

import unittest

import torch

from tico.quantization.analysis import (
    evaluate_models,
    make_output_adapter,
    metric_float,
    ModelInvocation,
    normalize_outputs,
    TensorErrorMetrics,
)
from torch import nn


class AddModel(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return input_ + self.offset, input_ * 2.0 + self.offset


class TensorErrorMetricsTest(unittest.TestCase):
    def test_summary_matches_known_values(self) -> None:
        metrics = TensorErrorMetrics()
        metrics.update(torch.tensor([0.0, 2.0]), torch.tensor([1.0, 4.0]))
        summary = metrics.summary()
        self.assertAlmostEqual(metric_float(summary, "mae"), 1.5)
        self.assertAlmostEqual(metric_float(summary, "mse"), 2.5)
        self.assertAlmostEqual(metric_float(summary, "rmse"), 2.5**0.5)
        self.assertEqual(summary["count"], 2)

    def test_evaluate_models_supports_named_sequence_outputs(self) -> None:
        reference = AddModel(0.0)
        candidate = AddModel(0.5)
        samples = [torch.ones(2), torch.zeros(2)]
        report = evaluate_models(
            reference,
            candidate,
            samples,
            output_adapter=make_output_adapter(("first", "second")),
        )
        self.assertAlmostEqual(metric_float(report["first"], "mae"), 0.5)
        self.assertAlmostEqual(metric_float(report["second"], "mae"), 0.5)

    def test_model_invocation_supports_keyword_inputs(self) -> None:
        class KeywordModel(nn.Module):
            def forward(self, *, value: torch.Tensor) -> torch.Tensor:
                return value

        sample = ModelInvocation(kwargs={"value": torch.ones(3)})
        report = evaluate_models(KeywordModel(), KeywordModel(), [sample])
        self.assertEqual(metric_float(report["output"], "mae"), 0.0)

    def test_mapping_output_validation(self) -> None:
        result = normalize_outputs({"x": torch.ones(1)})
        self.assertEqual(tuple(result), ("x",))
        with self.assertRaises(TypeError):
            normalize_outputs({"x": 1.0})


if __name__ == "__main__":
    unittest.main()

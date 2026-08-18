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

"""Tests for static palm-detector subgraph execution used by reconstruction."""

from __future__ import annotations

import unittest

import torch

from examples.hand_detector.hand_detector import HandDetector


class HandDetectorSegmentTest(unittest.TestCase):
    def _model(self) -> HandDetector:
        specification = {
            "inputs": [0],
            "outputs": [2, 2],
            "operations": [
                {
                    "index": 1,
                    "name": "PRELU",
                    "inputs": [0],
                    "outputs": [1],
                    "config": {"channels": 1},
                },
                {
                    "index": 2,
                    "name": "PRELU",
                    "inputs": [1],
                    "outputs": [2],
                    "config": {"channels": 1},
                },
            ],
        }
        model = HandDetector(specification)
        for layer in model.layers:
            layer.weight.data.fill_(1.0)
        return model

    def test_segment_matches_full_graph_suffix(self) -> None:
        model = self._model()
        input_ = torch.tensor([[[[-2.0, 3.0]]]])
        full_values = model.forward_values(input_)
        suffix_values = model.execute_segment({1: full_values[1]}, (1,))
        torch.testing.assert_close(suffix_values[2], full_values[2])

    def test_segment_rejects_missing_boundary_tensor(self) -> None:
        model = self._model()
        with self.assertRaisesRegex(KeyError, "missing input tensors"):
            model.execute_segment({}, (1,))

    def test_segment_rejects_invalid_operation_position(self) -> None:
        model = self._model()
        with self.assertRaisesRegex(IndexError, "out of range"):
            model.execute_segment({0: torch.zeros(1, 1, 1, 1)}, (9,))


if __name__ == "__main__":
    unittest.main()

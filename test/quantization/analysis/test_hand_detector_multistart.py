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

"""Tests for the three-way hand-detector reconstruction split."""

from __future__ import annotations

import unittest

import torch

from examples.hand_detector._support.multistart_reconstruction import (
    split_reconstruction_samples_three_way,
)


class HandDetectorMultiStartTest(unittest.TestCase):
    def test_three_way_split_is_deterministic_and_disjoint(self) -> None:
        samples = tuple(torch.tensor([index]) for index in range(20))
        first = split_reconstruction_samples_three_way(
            samples,
            selection_count=4,
            acceptance_count=5,
            seed=17,
        )
        second = split_reconstruction_samples_three_way(
            samples,
            selection_count=4,
            acceptance_count=5,
            seed=17,
        )
        self.assertEqual(
            first.to_dict(),
            {
                "train_count": 11,
                "selection_count": 4,
                "acceptance_count": 5,
            },
        )
        self.assertEqual(
            tuple(int(value.item()) for value in first.train),
            tuple(int(value.item()) for value in second.train),
        )
        train = {int(value.item()) for value in first.train}
        selection = {int(value.item()) for value in first.selection}
        acceptance = {int(value.item()) for value in first.acceptance}
        self.assertFalse(train.intersection(selection))
        self.assertFalse(train.intersection(acceptance))
        self.assertFalse(selection.intersection(acceptance))
        self.assertEqual(train | selection | acceptance, set(range(20)))

    def test_three_way_split_rejects_exhausting_calibration_data(self) -> None:
        samples = tuple(torch.tensor([index]) for index in range(8))
        with self.assertRaisesRegex(ValueError, "smaller than"):
            split_reconstruction_samples_three_way(
                samples,
                selection_count=4,
                acceptance_count=4,
            )


if __name__ == "__main__":
    unittest.main()

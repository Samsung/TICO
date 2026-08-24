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

"""Tests for element-wise QDrop during block reconstruction."""

from __future__ import annotations

import unittest

import torch

from tico.quantization.algorithm.block_reconstruction import (
    BlockInvocation,
    BlockReconstructionConfig,
    qdrop_context,
    QDropController,
)
from tico.quantization.algorithm.block_reconstruction.qdrop import (
    maybe_qdrop_activation,
)


class QDropTest(unittest.TestCase):
    """Verify deterministic masks, edge probabilities, and scoped behavior."""

    def test_zero_probability_preserves_quantized_values(self) -> None:
        controller = QDropController(0.0, seed=7)
        float_value = torch.tensor([[1.0, 2.0]])
        quantized_value = torch.tensor([[10.0, 20.0]])
        mixed = controller.mix_invocations(
            BlockInvocation(args=(float_value,)),
            BlockInvocation(args=(quantized_value,)),
        )
        torch.testing.assert_close(mixed.args[0], quantized_value)

    def test_one_probability_uses_float_values(self) -> None:
        controller = QDropController(1.0, seed=7)
        float_value = torch.tensor([[1.0, 2.0]])
        quantized_value = torch.tensor([[10.0, 20.0]])
        mixed = controller.mix_invocations(
            BlockInvocation(args=(float_value,)),
            BlockInvocation(args=(quantized_value,)),
        )
        torch.testing.assert_close(mixed.args[0], float_value)

    def test_full_drop_preserves_a_zero_gradient_qparam_path(self) -> None:
        controller = QDropController(1.0, seed=11)
        float_value = torch.tensor([1.0])
        quantized_value = torch.tensor([2.0], requires_grad=True)
        with qdrop_context(controller):
            output = maybe_qdrop_activation(float_value, quantized_value)
        output.sum().backward()
        torch.testing.assert_close(output, float_value)
        torch.testing.assert_close(
            quantized_value.grad,
            torch.zeros_like(quantized_value),
        )

    def test_masks_are_deterministic_for_one_seed(self) -> None:
        float_value = torch.zeros(128)
        quantized_value = torch.ones(128)
        first = QDropController(0.5, seed=17).mix_invocations(
            BlockInvocation(args=(float_value,)),
            BlockInvocation(args=(quantized_value,)),
        )
        second = QDropController(0.5, seed=17).mix_invocations(
            BlockInvocation(args=(float_value,)),
            BlockInvocation(args=(quantized_value,)),
        )
        torch.testing.assert_close(first.args[0], second.args[0])
        zero_count = (first.args[0] == 0).sum()  # type: ignore[union-attr]
        one_count = (first.args[0] == 1).sum()  # type: ignore[union-attr]
        self.assertGreater(int(zero_count), 0)
        self.assertGreater(int(one_count), 0)

    def test_context_does_not_leak_into_selection_forward(self) -> None:
        controller = QDropController(1.0, seed=23)
        float_value = torch.tensor([1.0])
        quantized_value = torch.tensor([2.0])
        with qdrop_context(controller):
            training = maybe_qdrop_activation(float_value, quantized_value)
        selection = maybe_qdrop_activation(float_value, quantized_value)
        torch.testing.assert_close(training, float_value)
        torch.testing.assert_close(selection, quantized_value)

    def test_statistics_cover_input_and_internal_activation(self) -> None:
        controller = QDropController(0.5, seed=31)
        value = torch.zeros(2, 3)
        controller.mix_invocations(
            BlockInvocation(args=(value,)),
            BlockInvocation(args=(value + 1,)),
        )
        with qdrop_context(controller):
            maybe_qdrop_activation(value, value + 1)
        statistics = controller.statistics()
        self.assertEqual(statistics.input_tensor_count, 1)
        self.assertEqual(statistics.activation_tensor_count, 1)
        self.assertEqual(statistics.total_element_count, 12)
        self.assertEqual(statistics.expected_dropped_element_count, 6.0)

    def test_config_rejects_invalid_probability(self) -> None:
        for probability in (-0.1, 1.1, float("nan")):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(ValueError, "QDrop probability"):
                    BlockReconstructionConfig(
                        qdrop_probability=probability,
                    ).validate()


if __name__ == "__main__":
    unittest.main()

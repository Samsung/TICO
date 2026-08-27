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

"""Unit tests for affine input folding into linear layers."""

import unittest

import torch
import torch.nn as nn

from tico.quantization.wrapq.utils.linear_folding import (
    fold_input_affine_into_linear,
    reused_weight_qparam_scale_multiplier,
)


class LinearFoldingTest(unittest.TestCase):
    """Validate parameter rewriting and conservative input checks."""

    def test_scalar_affine_without_bias_matches_reference(self) -> None:
        """Fold scalar normalization and synthesize the required bias."""
        linear = nn.Linear(3, 2, bias=False).eval()
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
                )
            )
        original_weight = linear.weight.detach().clone()

        folded = fold_input_affine_into_linear(
            linear,
            scale=2.0,
            shift=-1.0,
        )
        input_ = torch.randn(2, 4, 3)

        torch.testing.assert_close(
            folded(input_),
            linear(input_ * 2.0 - 1.0),
        )
        torch.testing.assert_close(folded.weight, original_weight * 2.0)
        self.assertIsNotNone(folded.bias)
        torch.testing.assert_close(folded.bias, -original_weight.sum(dim=1))
        torch.testing.assert_close(linear.weight, original_weight)
        self.assertIsNone(linear.bias)
        self.assertIsNot(folded, linear)

    def test_featurewise_affine_preserves_bias_and_module_state(self) -> None:
        """Fold feature-wise values while preserving state and gradient flags."""
        linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            linear.bias.copy_(torch.tensor([0.5, -0.5]))
        linear.weight.requires_grad_(False)
        linear.bias.requires_grad_(True)
        linear.train()

        folded = fold_input_affine_into_linear(
            linear,
            scale=torch.tensor([2.0, 3.0]),
            shift=torch.tensor([1.0, -1.0]),
        )
        input_ = torch.randn(5, 2)

        torch.testing.assert_close(
            folded(input_),
            linear(input_ * torch.tensor([2.0, 3.0]) + torch.tensor([1.0, -1.0])),
        )
        torch.testing.assert_close(
            folded.weight,
            torch.tensor([[2.0, 6.0], [6.0, 12.0]]),
        )
        torch.testing.assert_close(folded.bias, torch.tensor([-0.5, -1.5]))
        self.assertTrue(folded.training)
        self.assertFalse(folded.weight.requires_grad)
        self.assertTrue(folded.bias.requires_grad)

    def test_zero_shift_preserves_absent_bias(self) -> None:
        """Avoid manufacturing a zero bias when the affine shift is zero."""
        linear = nn.Linear(3, 2, bias=False).eval()

        folded = fold_input_affine_into_linear(
            linear,
            scale=3.0,
            shift=0.0,
        )

        self.assertIsNone(folded.bias)
        torch.testing.assert_close(folded.weight, linear.weight * 3.0)

    def test_uniform_positive_scale_supports_qparam_reuse(self) -> None:
        """Scale reused weight qparams with the folded scalar multiplier."""
        linear = nn.Linear(3, 2, bias=False)

        folded = fold_input_affine_into_linear(
            linear,
            scale=2.0,
            shift=-1.0,
        )

        self.assertEqual(reused_weight_qparam_scale_multiplier(folded), 2.0)

    def test_nonuniform_scale_rejects_qparam_reuse(self) -> None:
        """Require qparam recomputation for non-uniform folded scaling."""
        linear = nn.Linear(3, 2, bias=False)
        folded = fold_input_affine_into_linear(
            linear,
            scale=torch.tensor([1.0, 2.0, 3.0]),
            shift=0.0,
        )

        with self.assertRaisesRegex(RuntimeError, "Recompute"):
            reused_weight_qparam_scale_multiplier(folded)

    def test_unfolded_module_keeps_qparam_scale(self) -> None:
        """Leave reused weight qparams unchanged for ordinary modules."""
        linear = nn.Linear(3, 2, bias=False)

        self.assertEqual(reused_weight_qparam_scale_multiplier(linear), 1.0)

    def test_rejects_non_feature_vector(self) -> None:
        """Reject affine tensors that cannot broadcast over the feature axis."""
        linear = nn.Linear(3, 2)

        with self.assertRaisesRegex(ValueError, r"shape \(3,\)"):
            fold_input_affine_into_linear(
                linear,
                scale=torch.ones(1, 3),
                shift=0.0,
            )

    def test_rejects_non_finite_values(self) -> None:
        """Reject non-finite affine parameters before creating a new layer."""
        linear = nn.Linear(3, 2)

        with self.assertRaisesRegex(ValueError, "finite"):
            fold_input_affine_into_linear(
                linear,
                scale=float("inf"),
                shift=0.0,
            )


if __name__ == "__main__":
    unittest.main()

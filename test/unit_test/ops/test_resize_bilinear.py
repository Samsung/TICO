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

"""Unit tests for the public ResizeBilinear operation facade."""

from __future__ import annotations

import unittest

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from tico.ops import ResizeBilinear2d


def _scalar_resize_bilinear_asymmetric(
    input_: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute false/false ResizeBilinear with scalar loops."""
    batch_size, input_height, input_width, channels = input_.shape
    output_height, output_width = output_size
    output = np.empty(
        (batch_size, output_height, output_width, channels),
        dtype=np.float32,
    )
    height_scale = input_height / output_height
    width_scale = input_width / output_width

    for batch in range(batch_size):
        for output_y in range(output_height):
            source_y = output_y * height_scale
            y0 = int(np.floor(source_y))
            y1 = min(y0 + 1, input_height - 1)
            y_weight = source_y - y0
            for output_x in range(output_width):
                source_x = output_x * width_scale
                x0 = int(np.floor(source_x))
                x1 = min(x0 + 1, input_width - 1)
                x_weight = source_x - x0
                top = (
                    input_[batch, y0, x0] * (1.0 - x_weight)
                    + input_[batch, y0, x1] * x_weight
                )
                bottom = (
                    input_[batch, y1, x0] * (1.0 - x_weight)
                    + input_[batch, y1, x1] * x_weight
                )
                output[batch, output_y, output_x] = (
                    top * (1.0 - y_weight) + bottom * y_weight
                )
    return output


def _run_custom_resize(
    input_: torch.Tensor,
    size: tuple[int, int],
    *,
    align_corners: bool,
    half_pixel_centers: bool,
) -> torch.Tensor:
    """Run the centrally registered NHWC ResizeBilinear custom operator."""
    return torch.ops.circle_custom.resize_bilinear.default(
        input_,
        [size[0], size[1]],
        align_corners,
        half_pixel_centers,
    )


class ResizeBilinear2dTest(unittest.TestCase):
    """Verify eager semantics and the public torch.export representation."""

    def test_asymmetric_custom_op(self) -> None:
        """Match a scalar implementation of the legacy coordinate mode."""
        generator = np.random.default_rng(20260728)
        source = generator.standard_normal((1, 3, 4, 2), dtype=np.float32)
        expected = _scalar_resize_bilinear_asymmetric(source, (6, 8))
        actual = _run_custom_resize(
            torch.from_numpy(source),
            (6, 8),
            align_corners=False,
            half_pixel_centers=False,
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)

    def test_half_pixel_custom_op_matches_pytorch(self) -> None:
        """Match PyTorch interpolation for the source-model coordinate mode."""
        source = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        actual = _run_custom_resize(
            source.permute(0, 2, 3, 1),
            (8, 10),
            align_corners=False,
            half_pixel_centers=True,
        ).permute(0, 3, 1, 2)
        expected = F.interpolate(
            source,
            size=(8, 10),
            mode="bilinear",
            align_corners=False,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_align_corners_custom_op_matches_pytorch(self) -> None:
        """Match PyTorch bilinear interpolation with aligned corners."""
        source = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        actual = _run_custom_resize(
            source.permute(0, 2, 3, 1),
            (8, 10),
            align_corners=True,
            half_pixel_centers=False,
        ).permute(0, 3, 1, 2)
        expected = F.interpolate(
            source,
            size=(8, 10),
            mode="bilinear",
            align_corners=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_torch_export_keeps_one_custom_operator(self) -> None:
        """Keep one opaque resize node between two layout permutations."""
        module = ResizeBilinear2d(
            (6, 8),
            align_corners=False,
            half_pixel_centers=True,
        ).eval()
        exported = torch.export.export(
            module,
            (torch.randn(1, 2, 3, 4),),
            strict=True,
        )
        targets = Counter(
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        )
        self.assertEqual(targets["circle_custom.resize_bilinear.default"], 1)
        self.assertEqual(targets["aten.permute.default"], 2)

    def test_module_rejects_invalid_coordinate_options(self) -> None:
        """Reject the option pair forbidden by the Circle contract."""
        with self.assertRaisesRegex(ValueError, "does not allow"):
            ResizeBilinear2d(
                (6, 8),
                align_corners=True,
                half_pixel_centers=True,
            )

    def test_custom_op_rejects_invalid_coordinate_options(self) -> None:
        """Reject invalid options in the central eager custom operator."""
        with self.assertRaisesRegex(RuntimeError, "does not allow"):
            _run_custom_resize(
                torch.randn(1, 3, 4, 2),
                (6, 8),
                align_corners=True,
                half_pixel_centers=True,
            )


if __name__ == "__main__":
    unittest.main()

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

"""Unit tests for the converted hand detector and NHWC export adapter."""

from __future__ import annotations

import json
import unittest

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tico.ops import Concat, ResizeBilinear2d, SamePaddingConv2d

from examples.hand_detector.hand_detector import (
    HandDetector,
    load_hand_detector,
    load_nhwc_hand_detector,
    lower_resize_bilinear_to_tconv,
    NHWCInputAdapter,
    ResizeBilinearTConv,
)


DIRECTORY = Path(__file__).resolve().parent


def scalar_resize_bilinear_asymmetric(
    input_: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Compute TFLite false/false ResizeBilinear with scalar loops."""

    batch_size, input_h, input_w, channels = input_.shape
    output_h, output_w = output_size
    output = np.empty(
        (batch_size, output_h, output_w, channels),
        dtype=np.float32,
    )
    height_scale = input_h / output_h
    width_scale = input_w / output_w
    for batch in range(batch_size):
        for output_y in range(output_h):
            source_y = output_y * height_scale
            y0 = int(np.floor(source_y))
            y1 = min(y0 + 1, input_h - 1)
            y_weight = source_y - y0
            for output_x in range(output_w):
                source_x = output_x * width_scale
                x0 = int(np.floor(source_x))
                x1 = min(x0 + 1, input_w - 1)
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


class ResizeBilinearTest(unittest.TestCase):
    """Validate eager semantics and the opaque torch.export representation."""

    def test_custom_op_matches_scalar_asymmetric_coordinates(self) -> None:
        """Compare the central custom op with a scalar TFLite reference."""

        generator = np.random.default_rng(20260728)
        source = generator.standard_normal((1, 3, 4, 2), dtype=np.float32)
        expected = scalar_resize_bilinear_asymmetric(source, (6, 8))
        actual = torch.ops.circle_custom.resize_bilinear.default(
            torch.from_numpy(source),
            [6, 8],
            False,
            False,
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)

    def test_module_exports_as_one_custom_operator(self) -> None:
        """Check that one facade module becomes one opaque custom operator."""

        module = ResizeBilinear2d((12, 12)).eval()
        exported = torch.export.export(
            module,
            (torch.zeros(1, 8, 6, 6),),
            strict=True,
        )
        targets = [
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        ]
        self.assertEqual(targets.count("circle_custom.resize_bilinear.default"), 1)


class ResizeBilinearTConvTest(unittest.TestCase):
    """Validate the TransposeConv lowering of 2x half-pixel RESIZE_BILINEAR."""

    def test_matches_half_pixel_resize_bilinear(self) -> None:
        """Match ResizeBilinear2d output within float rounding error."""

        torch.manual_seed(20260828)
        for height, width, channels, groups in (
            (6, 6, 8, 1),
            (6, 6, 8, 4),
            (12, 12, 4, 2),
            (5, 7, 3, 1),
        ):
            with self.subTest(
                height=height, width=width, channels=channels, groups=groups
            ):
                reference = ResizeBilinear2d(
                    (2 * height, 2 * width),
                    align_corners=False,
                    half_pixel_centers=True,
                ).eval()
                lowered = ResizeBilinearTConv(channels, groups=groups).eval()
                source = torch.randn(1, channels, height, width)
                with torch.inference_mode():
                    expected = reference(source)
                    actual = lowered(source)
                torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)

    def test_rejects_indivisible_groups(self) -> None:
        """Reject a group count that does not evenly divide the channels."""

        with self.assertRaises(ValueError):
            ResizeBilinearTConv(8, groups=3)

    def test_lowering_replaces_every_resize_and_preserves_outputs(self) -> None:
        """Replace both detector resizes without changing model outputs."""

        detector = load_hand_detector(
            DIRECTORY / "hand_detector_float.pt",
            DIRECTORY / "hand_detector_spec.json",
        ).eval()
        reference = load_hand_detector(
            DIRECTORY / "hand_detector_float.pt",
            DIRECTORY / "hand_detector_spec.json",
        ).eval()
        replaced = lower_resize_bilinear_to_tconv(detector)
        self.assertEqual(len(replaced), 2)
        for position in replaced:
            self.assertEqual(detector.operations[position]["name"], "RESIZE_BILINEAR")
            self.assertIsInstance(detector.layers[position], ResizeBilinearTConv)

        torch.manual_seed(20260828)
        source = torch.rand(1, 3, 192, 192)
        with torch.inference_mode():
            expected = reference(source)
            actual = detector(source)
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(
                actual_output, expected_output, rtol=0.0, atol=1e-4
            )


class HandDetectorTest(unittest.TestCase):
    """Validate model structure, outputs, and the NHWC adapter."""

    model: HandDetector
    nhwc_model: NHWCInputAdapter

    @classmethod
    def setUpClass(cls) -> None:
        """Load the NCHW detector and NHWC adapter once for all tests."""

        cls.model = load_hand_detector(
            DIRECTORY / "hand_detector_float.pt",
            DIRECTORY / "hand_detector_spec.json",
        ).eval()
        cls.nhwc_model = load_nhwc_hand_detector(
            DIRECTORY / "hand_detector_float.pt",
            DIRECTORY / "hand_detector_spec.json",
        ).eval()

    def test_model_shapes_and_parameter_count(self) -> None:
        """Check detector output shapes and the converted parameter count."""

        with torch.inference_mode():
            regressors, classifiers = self.model(torch.zeros(1, 3, 192, 192))
        self.assertEqual(tuple(regressors.shape), (1, 2016, 18))
        self.assertEqual(tuple(classifiers.shape), (1, 2016, 1))
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            1_136_248,
        )

    def test_nhwc_adapter_matches_nchw_detector(self) -> None:
        """Check that the input adapter changes only the external memory order."""

        generator = torch.Generator().manual_seed(20260730)
        input_nhwc = torch.rand(1, 192, 192, 3, generator=generator)
        input_nchw = input_nhwc.permute(0, 3, 1, 2)
        with torch.inference_mode():
            expected = self.model(input_nchw)
            actual = self.nhwc_model(input_nhwc)
        for expected_output, actual_output in zip(expected, actual):
            torch.testing.assert_close(actual_output, expected_output)

    def test_nhwc_adapter_exposes_nhwc_example_input(self) -> None:
        """Check the example input used as the exported Circle input ABI."""

        example_inputs = self.nhwc_model.get_example_inputs()
        self.assertEqual(len(example_inputs), 1)
        self.assertEqual(tuple(example_inputs[0].shape), (1, 192, 192, 3))

    def test_model_uses_quantizable_pool_and_concat_boundaries(self) -> None:
        """Expose native MaxPool2d and TICO Concat modules to WrapQ."""

        self.assertEqual(
            sum(
                isinstance(module, torch.nn.MaxPool2d)
                for module in self.model.modules()
            ),
            4,
        )
        self.assertEqual(
            sum(isinstance(module, Concat) for module in self.model.modules()),
            2,
        )
        self.assertEqual(
            sum(
                isinstance(module, SamePaddingConv2d) for module in self.model.modules()
            ),
            33,
        )

    def test_specification_contains_original_convolution_padding(self) -> None:
        """Preserve every source Conv2D SAME or VALID padding option."""

        specification = json.loads(
            (DIRECTORY / "hand_detector_spec.json").read_text(encoding="utf-8")
        )
        convolution_operations = [
            operation
            for operation in specification["operations"]
            if operation["name"] in {"CONV_2D", "DEPTHWISE_CONV_2D"}
        ]
        paddings = [
            operation["config"]["padding"] for operation in convolution_operations
        ]
        self.assertEqual(paddings.count("same"), 33)
        self.assertEqual(paddings.count("valid"), 30)

    def test_specification_contains_original_resize_options(self) -> None:
        """Check both source ResizeBilinear nodes and their exact options."""

        specification = json.loads(
            (DIRECTORY / "hand_detector_spec.json").read_text(encoding="utf-8")
        )
        resize_operations = [
            operation
            for operation in specification["operations"]
            if operation["name"] == "RESIZE_BILINEAR"
        ]
        self.assertEqual(len(resize_operations), 2)
        self.assertEqual(
            [operation["config"]["size"] for operation in resize_operations],
            [[12, 12], [24, 24]],
        )
        for operation in resize_operations:
            self.assertFalse(operation["config"]["align_corners"])
            self.assertTrue(operation["config"]["half_pixel_centers"])

    def test_nhwc_export_contains_expected_input_and_resize_nodes(self) -> None:
        """Check the NHWC placeholder and both custom ResizeBilinear nodes."""

        exported = torch.export.export(
            self.nhwc_model,
            self.nhwc_model.get_example_inputs(),
            strict=True,
        )
        user_input_name = exported.graph_signature.user_inputs[0]
        user_input = next(
            node for node in exported.graph.nodes if node.name == user_input_name
        )
        self.assertEqual(tuple(user_input.meta["val"].shape), (1, 192, 192, 3))

        call_nodes = [
            node for node in exported.graph.nodes if node.op == "call_function"
        ]
        counts = Counter(str(node.target) for node in call_nodes)
        self.assertEqual(counts["circle_custom.resize_bilinear.default"], 2)
        self.assertEqual(counts["circle_custom.conv2d.padding"], 5)
        self.assertEqual(
            counts["circle_custom.depthwise_conv2d.padding"],
            28,
        )
        self.assertEqual(counts["aten.pad.default"], 3)
        self.assertEqual(counts["aten.cat.default"], 2)
        resize_nodes = [
            node
            for node in call_nodes
            if str(node.target) == "circle_custom.resize_bilinear.default"
        ]
        self.assertEqual(
            [
                (list(node.args[1]), bool(node.args[2]), bool(node.args[3]))
                for node in resize_nodes
            ],
            [([12, 12], False, True), ([24, 24], False, True)],
        )
        self.assertEqual(counts["aten.slice.Tensor"], 0)
        self.assertEqual(counts["aten.mul.Tensor"], 0)


class HandLandmarkSpecTest(unittest.TestCase):
    """Validate hand-landmark executor construction from its specification."""

    SPEC_PATH = DIRECTORY / "hand_landmark_spec.json"

    def setUp(self) -> None:
        """Skip when the converted landmark specification is unavailable."""

        if not self.SPEC_PATH.exists():
            self.skipTest("hand_landmark_spec.json has not been converted.")
        self.spec = json.loads(self.SPEC_PATH.read_text(encoding="utf-8"))

    def test_specification_declares_224_input_and_four_outputs(self) -> None:
        """Check the converted landmark graph interface."""

        self.assertEqual(self.spec["input_shape"], [1, 224, 224, 3])
        self.assertEqual(len(self.spec["outputs"]), 4)
        names = {operation["name"] for operation in self.spec["operations"]}
        self.assertIn("RELU6", names)
        self.assertIn("MEAN", names)
        self.assertIn("LOGISTIC", names)
        self.assertNotIn("FULLY_CONNECTED", names)

    def test_randomly_initialized_executor_produces_output_shapes(self) -> None:
        """Run the executor graph without converted weights."""

        model = HandDetector(self.spec).eval()
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 224, 224))
        self.assertEqual(
            [tuple(value.shape) for value in outputs],
            [(1, 63), (1, 1), (1, 1), (1, 63)],
        )
        example = model.get_example_inputs()[0]
        self.assertEqual(tuple(example.shape), (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()

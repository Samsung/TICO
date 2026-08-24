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

"""Tests for contiguous detector windows and held-out calibration splits."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest.mock import patch

import torch

from examples.hand_detector._support.reconstruction import (
    _site_tensor_domain,
    _window_boundaries,
    build_reconstruction_windows,
    build_window_observer_groups,
    DetectorWindow,
    ReconstructionWindow,
    split_reconstruction_samples,
)
from examples.hand_detector._support.sensitivity import ActivationSensitivityGroup
from tico.quantization.analysis import (
    QuantizationBoundaries,
    QuantizationGroup,
    SiteSelector,
)
from tico.quantization.wrapq.control import SiteRole

from torch import nn


def _operation(name: str, input_id: int, output_id: int):
    return {
        "index": output_id,
        "name": name,
        "inputs": [input_id],
        "outputs": [output_id],
        "config": {},
    }


def _detector():
    return SimpleNamespace(
        operations=(
            _operation("PRELU", 0, 1),
            _operation("PRELU", 1, 2),
            _operation("PRELU", 2, 3),
        ),
        layers=nn.ModuleList((nn.Identity(), nn.Identity(), nn.Identity())),
        output_tensors=(3,),
    )


def _group(name: str, position: int) -> ActivationSensitivityGroup:
    return ActivationSensitivityGroup(
        group=QuantizationGroup(name, SiteSelector.none()),
        kind="stem" if position == 0 else "feature",
        operation_positions=(position,),
        operation_indices=(position,),
        operation_names=("PRELU",),
        tensor_ids=(position + 1,),
        site_paths=(f"layers.{position}.act_out",),
    )


class HandDetectorJointWindowTest(unittest.TestCase):
    def test_selection_split_is_deterministic_and_disjoint(self) -> None:
        values = tuple(torch.tensor([index]) for index in range(10))
        first_train, first_selection = split_reconstruction_samples(
            values,
            3,
            seed=17,
        )
        second_train, second_selection = split_reconstruction_samples(
            values,
            3,
            seed=17,
        )

        self.assertEqual(len(first_train), 7)
        self.assertEqual(len(first_selection), 3)
        self.assertEqual(
            [int(value) for value in first_train],
            [int(value) for value in second_train],
        )
        self.assertEqual(
            [int(value) for value in first_selection],
            [int(value) for value in second_selection],
        )
        self.assertFalse(
            set(int(value) for value in first_train).intersection(
                int(value) for value in first_selection
            )
        )

    def test_joint_window_executes_all_positions_and_returns_live_out(self) -> None:
        detector = SimpleNamespace(
            operations=(
                {"name": "PRELU", "inputs": [0], "outputs": [1], "config": {}},
                {"name": "PRELU", "inputs": [1], "outputs": [2], "config": {}},
                {"name": "ADD", "inputs": [2, 0], "outputs": [3], "config": {}},
            ),
            layers=nn.ModuleList((nn.Identity(), nn.Identity(), nn.Identity())),
        )
        window = ReconstructionWindow(
            name="first+second",
            group_names=("first", "second"),
            operation_positions=(0, 1, 2),
            input_tensor_ids=(0,),
            output_tensor_ids=(3,),
            site_paths=("site0", "site1"),
        )
        output = DetectorWindow(detector, window)(  # type: ignore[arg-type]
            torch.tensor([[2.0]])
        )
        torch.testing.assert_close(output, torch.tensor([[4.0]]))

    def test_boundary_analysis_keeps_multiple_live_ins_and_outs(self) -> None:
        detector = SimpleNamespace(
            operations=(
                {"name": "PRELU", "inputs": [0], "outputs": [1], "config": {}},
                {"name": "ADD", "inputs": [1, 9], "outputs": [2], "config": {}},
                {"name": "PRELU", "inputs": [2], "outputs": [3], "config": {}},
                {"name": "ADD", "inputs": [3, 8], "outputs": [4], "config": {}},
            ),
            output_tensors=(4,),
        )
        inputs, outputs = _window_boundaries(
            detector, (0, 1, 2)  # type: ignore[arg-type]
        )
        self.assertEqual(inputs, (0, 9))
        self.assertEqual(outputs, (3,))

    def test_boundary_analysis_ignores_embedded_constant_inputs(self) -> None:
        detector = SimpleNamespace(
            operations=(
                {
                    "name": "CONV_2D",
                    "inputs": [0, 387, 346],
                    "outputs": [141],
                    "config": {},
                },
                {
                    "name": "PRELU",
                    "inputs": [141, 344],
                    "outputs": [142],
                    "config": {},
                },
                {
                    "name": "DEPTHWISE_CONV_2D",
                    "inputs": [142, 353, 408],
                    "outputs": [143],
                    "config": {},
                },
            ),
            output_tensors=(143,),
        )

        inputs, outputs = _window_boundaries(detector, (0, 1))  # type: ignore[arg-type]

        self.assertEqual(inputs, (0,))
        self.assertEqual(outputs, (142,))

    def test_activation_site_domain_excludes_parameter_tensors(self) -> None:
        detector = SimpleNamespace(
            operations=(
                {
                    "name": "CONV_2D",
                    "inputs": [0, 387, 346],
                    "outputs": [141],
                    "config": {},
                },
            ),
        )
        site = SimpleNamespace(
            path="conv.act_in",
            module=SimpleNamespace(fp_name="detector.layers.0"),
            module_path="detector.layers.0",
            role=SiteRole.ACTIVATION_INPUT,
        )

        self.assertEqual(
            _site_tensor_domain(site, detector), (0,)  # type: ignore[arg-type]
        )

    def test_max_pool_input_and_output_use_distinct_tensor_domains(self) -> None:
        detector = SimpleNamespace(
            operations=(
                {"name": "PRELU", "inputs": [0], "outputs": [1], "config": {}},
                {
                    "name": "DEPTHWISE_CONV_2D",
                    "inputs": [1, 90, 91],
                    "outputs": [2],
                    "config": {},
                },
                {
                    "name": "MAX_POOL_2D",
                    "inputs": [1],
                    "outputs": [3],
                    "config": {},
                },
            ),
        )
        window = ReconstructionWindow(
            name="independent-pool-domains",
            group_names=("independent-pool-domains",),
            operation_positions=(0, 1, 2),
            input_tensor_ids=(0,),
            output_tensor_ids=(2, 3),
            site_paths=(
                "producer",
                "consumer",
                "pool_input",
                "pool_output",
            ),
        )
        sites = (
            SimpleNamespace(
                path="producer",
                module=SimpleNamespace(fp_name="detector.layers.0"),
                module_path="detector.layers.0",
                role=SiteRole.ACTIVATION_OUTPUT,
            ),
            SimpleNamespace(
                path="consumer",
                module=SimpleNamespace(fp_name="detector.layers.1"),
                module_path="detector.layers.1",
                role=SiteRole.ACTIVATION_INPUT,
            ),
            SimpleNamespace(
                path="pool_input",
                module=SimpleNamespace(fp_name="detector.layers.2"),
                module_path="detector.layers.2",
                role=SiteRole.ACTIVATION_INPUT,
            ),
            SimpleNamespace(
                path="pool_output",
                module=SimpleNamespace(fp_name="detector.layers.2"),
                module_path="detector.layers.2",
                role=SiteRole.ACTIVATION_OUTPUT,
            ),
        )
        with patch(
            "examples.hand_detector._support.reconstruction._find_detector",
            return_value=detector,
        ), patch(
            "examples.hand_detector._support.reconstruction." "iter_quantization_sites",
            return_value=sites,
        ):
            groups = build_window_observer_groups(nn.Identity(), window)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].name, "tensor_1")
        self.assertEqual(
            groups[0].site_paths,
            ("consumer", "pool_input", "producer"),
        )
        self.assertEqual(groups[1].name, "tensor_3")
        self.assertEqual(groups[1].site_paths, ("pool_output",))

    def test_builder_accepts_consecutive_joint_groups(self) -> None:
        groups = (
            _group("stem", 0),
            _group("feature_block_00", 1),
            _group("feature_block_01", 2),
        )
        boundaries = QuantizationBoundaries(outputs=SiteSelector.none())
        with patch(
            "examples.hand_detector._support.reconstruction."
            "build_activation_sensitivity_groups",
            return_value=groups,
        ), patch(
            "examples.hand_detector._support.reconstruction._find_detector",
            return_value=_detector(),
        ):
            windows = build_reconstruction_windows(
                nn.Identity(),
                boundaries,
                windows=("stem+feature_block_00",),
            )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].group_names, ("stem", "feature_block_00"))
        self.assertEqual(windows[0].operation_positions, (0, 1))
        self.assertEqual(windows[0].input_tensor_ids, (0,))
        self.assertEqual(windows[0].output_tensor_ids, (2,))

    def test_builder_rejects_nonconsecutive_joint_groups(self) -> None:
        groups = (
            _group("stem", 0),
            _group("feature_block_00", 1),
            _group("feature_block_01", 2),
        )
        boundaries = QuantizationBoundaries(outputs=SiteSelector.none())
        with patch(
            "examples.hand_detector._support.reconstruction."
            "build_activation_sensitivity_groups",
            return_value=groups,
        ), patch(
            "examples.hand_detector._support.reconstruction._find_detector",
            return_value=_detector(),
        ):
            with self.assertRaisesRegex(ValueError, "consecutive groups"):
                build_reconstruction_windows(
                    nn.Identity(),
                    boundaries,
                    windows=("stem+feature_block_01",),
                )


if __name__ == "__main__":
    unittest.main()

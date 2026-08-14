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

"""Tests for palm-detector activation sensitivity grouping."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest.mock import patch

from examples.hand_detector._support.sensitivity import (
    build_activation_sensitivity_groups,
)
from examples.hand_detector.hand_detector import HandDetector, NHWCInputAdapter
from tico.quantization.analysis import QuantizationBoundaries, SiteSelector
from tico.quantization.wrapq.control import QuantizationSite, SiteRole


class HandDetectorSensitivityGroupsTest(unittest.TestCase):
    """Verify logical-domain coverage and semantic head partitioning."""

    def test_internal_activation_sites_are_grouped_exactly_once(self) -> None:
        model = NHWCInputAdapter(HandDetector(_detector_spec()))
        sites = _activation_sites()
        boundaries = QuantizationBoundaries(
            outputs=SiteSelector.paths(
                "detector.layers.11.act_out",
                "detector.layers.14.act_out",
            )
        )

        with patch(
            "examples.hand_detector._support.sensitivity.iter_quantization_sites",
            return_value=iter(sites),
        ):
            groups = build_activation_sensitivity_groups(model, boundaries)

        names = tuple(group.name for group in groups)
        self.assertEqual(
            names,
            (
                "input_boundary",
                "stem",
                "feature_block_00",
                "classifiers_low_resolution_head",
                "regressors_low_resolution_head",
                "classifiers_high_resolution_head",
                "regressors_high_resolution_head",
            ),
        )

        grouped_paths = [path for group in groups for path in group.site_paths]
        internal_paths = [site.path for site in sites if not boundaries.outputs(site)]
        self.assertEqual(sorted(grouped_paths), sorted(internal_paths))
        self.assertEqual(len(grouped_paths), len(set(grouped_paths)))
        self.assertNotIn("detector.layers.11.act_out", grouped_paths)
        self.assertNotIn("detector.layers.14.act_out", grouped_paths)

    def test_producer_and_consumer_observers_share_a_group(self) -> None:
        model = NHWCInputAdapter(HandDetector(_detector_spec()))
        sites = _activation_sites()
        boundaries = QuantizationBoundaries(
            outputs=SiteSelector.paths(
                "detector.layers.11.act_out",
                "detector.layers.14.act_out",
            )
        )

        with patch(
            "examples.hand_detector._support.sensitivity.iter_quantization_sites",
            return_value=iter(sites),
        ):
            groups = build_activation_sensitivity_groups(model, boundaries)

        group_by_path = {
            path: group.name for group in groups for path in group.site_paths
        }
        self.assertEqual(
            group_by_path["detector.layers.1.act_out"],
            group_by_path["detector.layers.2.conv.act_in"],
        )
        self.assertEqual(
            group_by_path["detector.layers.8.act_out"],
            group_by_path["detector.layers.9.conv.act_in"],
        )
        self.assertEqual(
            group_by_path["detector.layers.2.conv.act_out"],
            "classifiers_low_resolution_head",
        )


def _activation_sites() -> tuple[QuantizationSite, ...]:
    sites: list[QuantizationSite] = [
        _site(
            "detector.input_quantizer.act_out",
            "detector.input_quantizer",
            "act_out",
            SiteRole.ACTIVATION_OUTPUT,
        )
    ]
    operation_names = (
        "CONV_2D",
        "PRELU",
        "CONV_2D",
        "RESHAPE",
        "CONV_2D",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "CONV_2D",
        "PRELU",
        "CONV_2D",
        "RESHAPE",
        "CONCATENATION",
        "CONV_2D",
        "RESHAPE",
        "CONCATENATION",
    )
    for position, name in enumerate(operation_names):
        if name == "CONV_2D":
            module_path = f"detector.layers.{position}.conv"
            sites.extend(
                (
                    _site(
                        f"{module_path}.act_in",
                        module_path,
                        "act_in",
                        SiteRole.ACTIVATION_INPUT,
                    ),
                    _site(
                        f"{module_path}.act_out",
                        module_path,
                        "act_out",
                        SiteRole.ACTIVATION_OUTPUT,
                    ),
                )
            )
        elif name in {"PRELU", "RESIZE_BILINEAR"}:
            module_path = f"detector.layers.{position}"
            sites.extend(
                (
                    _site(
                        f"{module_path}.act_in",
                        module_path,
                        "act_in",
                        SiteRole.ACTIVATION_INPUT,
                    ),
                    _site(
                        f"{module_path}.act_out",
                        module_path,
                        "act_out",
                        SiteRole.ACTIVATION_OUTPUT,
                    ),
                )
            )
        elif name == "CONCATENATION":
            module_path = f"detector.layers.{position}"
            sites.append(
                _site(
                    f"{module_path}.act_out",
                    module_path,
                    "act_out",
                    SiteRole.ACTIVATION_OUTPUT,
                )
            )
    return tuple(sites)


def _site(
    path: str,
    fp_name: str,
    observer_name: str,
    role: SiteRole,
) -> QuantizationSite:
    module = SimpleNamespace(fp_name=fp_name)
    return QuantizationSite(
        path=path,
        module_path=fp_name,
        observer_name=observer_name,
        role=role,
        module=module,  # type: ignore[arg-type]
        observer=SimpleNamespace(),  # type: ignore[arg-type]
    )


def _detector_spec() -> dict[str, object]:
    return {
        "inputs": [0],
        "outputs": [15, 12],
        "operations": [
            _conv(0, 1, index=0),
            _prelu(1, 2, index=1),
            _conv(2, 3, index=2),
            _reshape(3, 4, (1, -1, 1), index=3),
            _conv(2, 5, index=4, out_channels=18),
            _reshape(5, 6, (1, -1, 18), index=5),
            {
                "index": 6,
                "name": "RESIZE_BILINEAR",
                "inputs": [2],
                "outputs": [7],
                "config": {
                    "size": [4, 4],
                    "align_corners": False,
                    "half_pixel_centers": True,
                },
            },
            _conv(7, 8, index=7),
            _prelu(8, 9, index=8),
            _conv(9, 10, index=9),
            _reshape(10, 11, (1, -1, 1), index=10),
            {
                "index": 11,
                "name": "CONCATENATION",
                "inputs": [11, 4],
                "outputs": [12],
                "config": {"axis": 1},
            },
            _conv(9, 13, index=12, out_channels=18),
            _reshape(13, 14, (1, -1, 18), index=13),
            {
                "index": 14,
                "name": "CONCATENATION",
                "inputs": [14, 6],
                "outputs": [15],
                "config": {"axis": 1},
            },
        ],
    }


def _conv(
    input_tensor: int,
    output_tensor: int,
    *,
    index: int,
    out_channels: int = 1,
) -> dict[str, object]:
    return {
        "index": index,
        "name": "CONV_2D",
        "inputs": [input_tensor],
        "outputs": [output_tensor],
        "config": {
            "in_channels": 1,
            "out_channels": out_channels,
            "kernel_size": [1, 1],
            "stride": [1, 1],
            "dilation": [1, 1],
            "groups": 1,
            "has_bias": True,
            "padding": "valid",
            "pad": [0, 0, 0, 0],
        },
    }


def _prelu(
    input_tensor: int,
    output_tensor: int,
    *,
    index: int,
) -> dict[str, object]:
    return {
        "index": index,
        "name": "PRELU",
        "inputs": [input_tensor],
        "outputs": [output_tensor],
        "config": {"channels": 1},
    }


def _reshape(
    input_tensor: int,
    output_tensor: int,
    shape: tuple[int, ...],
    *,
    index: int,
) -> dict[str, object]:
    return {
        "index": index,
        "name": "RESHAPE",
        "inputs": [input_tensor],
        "outputs": [output_tensor],
        "config": {
            "shape": list(shape),
            "nhwc_memory_order": True,
        },
    }


if __name__ == "__main__":
    unittest.main()

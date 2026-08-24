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

"""Tests for hand-detector global Conv weight group construction."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

from examples.hand_detector._support import global_weight_refinement as module
from examples.hand_detector._support.global_weight_refinement import (
    build_global_conv_weight_definitions,
    run_hand_detector_global_weight_refinement,
)

from torch import nn


class HandDetectorGlobalWeightRefinementTest(unittest.TestCase):
    def _site_group(
        self,
        name: str,
        semantic_group: str,
        kind: str,
        position: int,
        path: str,
        elements: int,
    ):
        return SimpleNamespace(
            name=name,
            semantic_group=semantic_group,
            site_count=1,
            site_paths=(path,),
            parameter_element_count=elements,
            parameter_breakdown=(SimpleNamespace(kind=kind),),
            operation_positions=(position,),
            operation_indices=(position + 100,),
            operation_names=("CONV_2D",),
        )

    def test_builder_includes_regular_and_depthwise_but_not_prelu(self) -> None:
        groups = (
            self._site_group(
                "layer_000_conv2d_weight",
                "stem",
                "conv2d_weight",
                0,
                "detector.layers.0.conv.weight",
                16,
            ),
            self._site_group(
                "layer_001_depthwise_conv2d_weight",
                "feature_block_00",
                "depthwise_conv2d_weight",
                1,
                "detector.layers.1.conv.weight",
                8,
            ),
            self._site_group(
                "layer_002_prelu_slope",
                "feature_block_00",
                "prelu_slope",
                2,
                "detector.layers.2.weight",
                4,
            ),
        )
        detector = SimpleNamespace(
            layers=(
                SimpleNamespace(conv=nn.Conv2d(1, 1, 1)),
                SimpleNamespace(conv=nn.Conv2d(1, 1, 1, groups=1)),
                nn.PReLU(1),
            )
        )
        with (
            mock.patch.object(
                module,
                "build_weight_sensitivity_groups",
                return_value=groups,
            ),
            mock.patch.object(module, "_find_detector", return_value=detector),
        ):
            definitions = build_global_conv_weight_definitions(
                object(),
                object(),
            )
        self.assertEqual(len(definitions), 2)
        self.assertEqual(definitions[0].group.family, "regular_conv")
        self.assertEqual(definitions[1].group.family, "depthwise_conv")
        self.assertEqual(definitions[0].source_weight.numel(), 1)

    def test_requested_semantic_subset_is_honored(self) -> None:
        groups = (
            self._site_group(
                "layer_000_conv2d_weight",
                "stem",
                "conv2d_weight",
                0,
                "a.weight",
                1,
            ),
            self._site_group(
                "layer_001_conv2d_weight",
                "feature_block_00",
                "conv2d_weight",
                1,
                "b.weight",
                1,
            ),
        )
        detector = SimpleNamespace(
            layers=(
                SimpleNamespace(conv=nn.Conv2d(1, 1, 1)),
                SimpleNamespace(conv=nn.Conv2d(1, 1, 1)),
            )
        )
        with (
            mock.patch.object(
                module,
                "build_weight_sensitivity_groups",
                return_value=groups,
            ),
            mock.patch.object(module, "_find_detector", return_value=detector),
        ):
            definitions = build_global_conv_weight_definitions(
                object(),
                object(),
                requested_groups=("feature_block_00",),
            )
        self.assertEqual(
            [value.semantic_group for value in definitions],
            ["feature_block_00"],
        )

    def test_unknown_requested_group_fails(self) -> None:
        groups = (
            self._site_group(
                "layer_000_conv2d_weight",
                "stem",
                "conv2d_weight",
                0,
                "a.weight",
                1,
            ),
        )
        detector = SimpleNamespace(layers=(SimpleNamespace(conv=nn.Conv2d(1, 1, 1)),))
        with (
            mock.patch.object(
                module,
                "build_weight_sensitivity_groups",
                return_value=groups,
            ),
            mock.patch.object(module, "_find_detector", return_value=detector),
        ):
            with self.assertRaises(KeyError):
                build_global_conv_weight_definitions(
                    object(),
                    object(),
                    requested_groups=("missing",),
                )

    def test_runner_forwards_progress_callback(self) -> None:
        definition = SimpleNamespace(
            group=SimpleNamespace(
                name="layer_000_conv2d_weight",
                site_path="a.weight",
                family="regular_conv",
            ),
            semantic_group="stem",
            parameter_element_count=16,
            source_weight=mock.sentinel.source_weight,
            to_dict=lambda: {"group": "layer_000_conv2d_weight"},
        )
        outputs = {
            "regressors": {"mae": 0.2},
            "classifiers": {"mae": 0.08},
        }
        result = SimpleNamespace(
            to_dict=lambda: {"accepted": True},
            entry_selection_outputs=outputs,
            entry_acceptance_outputs=outputs,
            entry_evaluation_outputs=outputs,
            scale_only_selection_outputs=outputs,
            scale_only_acceptance_outputs=outputs,
            scale_only_evaluation_outputs=outputs,
            selected_outputs=outputs,
            acceptance_outputs=outputs,
            selected_evaluation_outputs=outputs,
            final_evaluation_outputs=outputs,
        )
        split = SimpleNamespace(
            train=(mock.sentinel.train,),
            selection=(mock.sentinel.selection,),
            acceptance=(mock.sentinel.acceptance,),
            to_dict=lambda: {
                "train_count": 1,
                "selection_count": 1,
                "acceptance_count": 1,
            },
        )
        progress = mock.Mock()
        runner = mock.Mock()
        runner.refine.return_value = result
        with (
            mock.patch.object(
                module,
                "build_global_conv_weight_definitions",
                return_value=(definition,),
            ),
            mock.patch.object(
                module,
                "GlobalWeightRefinementRunner",
                return_value=runner,
            ),
        ):
            report = run_hand_detector_global_weight_refinement(
                mock.sentinel.reference_model,
                mock.sentinel.candidate_model,
                data_split=split,  # type: ignore[arg-type]
                evaluation_samples=(mock.sentinel.evaluation,),
                config=mock.sentinel.config,
                selection_objective=mock.sentinel.selection_objective,
                acceptance_objective=mock.sentinel.acceptance_objective,
                output_adapter=mock.sentinel.output_adapter,
                progress_callback=progress,
            )
        self.assertTrue(report["global_refinement"]["accepted"])
        self.assertIs(
            runner.refine.call_args.kwargs["progress_callback"],
            progress,
        )


if __name__ == "__main__":
    unittest.main()

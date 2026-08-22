# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0

"""Tests for hand-detector joint DW/PW AdaRound grouping."""

from __future__ import annotations

import tempfile
import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from examples.hand_detector._support import (
    joint_adaround as module,
    reconstruction as reconstruction_module,
)
from examples.hand_detector._support.joint_adaround import (
    ALL_CONV_JOINT_GROUPS,
    apply_joint_adaround_checkpoint,
    build_joint_window_weight_groups,
    PRIORITY_JOINT_GROUPS,
    save_joint_adaround_checkpoint,
)
from examples.hand_detector._support.reconstruction import ReconstructionWindow
from tico.quantization.wrapq.control import SiteRole
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


def _site(path: str, position: int, wrapped: nn.Module):
    return SimpleNamespace(
        path=path,
        module_path=f"detector.layers.{position}.conv",
        observer_name="weight",
        role=SiteRole.PARAMETER,
        module=SimpleNamespace(module=wrapped),
    )


class HandDetectorJointAdaRoundTest(unittest.TestCase):
    def test_window_groups_depthwise_and_regular_weights_together(self) -> None:
        depthwise = nn.Conv2d(4, 4, 3, groups=4, bias=False)
        regular = nn.Conv2d(4, 8, 1, bias=False)
        prelu = nn.PReLU(8)
        sites = (
            _site("detector.layers.2.conv.weight", 2, depthwise),
            _site("detector.layers.3.conv.weight", 3, regular),
            _site("detector.layers.5.weight", 5, prelu),
        )
        window = ReconstructionWindow(
            name="feature_block_00",
            group_names=("feature_block_00",),
            operation_positions=(2, 3, 4, 5),
            input_tensor_ids=(1,),
            output_tensor_ids=(2,),
            site_paths=(),
        )
        with mock.patch.object(
            module,
            "iter_quantization_sites",
            return_value=sites,
        ):
            groups = build_joint_window_weight_groups(object(), window)
        self.assertEqual(len(groups), 2)
        self.assertEqual(
            [group.family for group in groups],
            ["depthwise_conv", "regular_conv"],
        )
        self.assertEqual(
            [group.name for group in groups],
            ["layer_002_depthwise_conv", "layer_003_regular_conv"],
        )

    def test_stem_supports_regular_only_window(self) -> None:
        regular = nn.Conv2d(3, 32, 5, bias=False)
        window = ReconstructionWindow(
            name="stem",
            group_names=("stem",),
            operation_positions=(0, 1),
            input_tensor_ids=(1,),
            output_tensor_ids=(2,),
            site_paths=(),
        )
        with mock.patch.object(
            module,
            "iter_quantization_sites",
            return_value=(_site("detector.layers.0.conv.weight", 0, regular),),
        ):
            groups = build_joint_window_weight_groups(object(), window)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].family, "regular_conv")

    def test_all_conv_preset_covers_every_feature_block(self) -> None:
        self.assertEqual(ALL_CONV_JOINT_GROUPS[0], "stem")
        self.assertEqual(
            ALL_CONV_JOINT_GROUPS[1:31],
            tuple(f"feature_block_{index:02d}" for index in range(30)),
        )
        self.assertEqual(
            ALL_CONV_JOINT_GROUPS[-2:],
            (
                "regressors_low_resolution_head",
                "regressors_high_resolution_head",
            ),
        )
        self.assertEqual(
            len(ALL_CONV_JOINT_GROUPS),
            len(set(ALL_CONV_JOINT_GROUPS)),
        )

    def test_checkpoint_restores_weight_and_nonpersistent_qparams(self) -> None:
        class Candidate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
                self.observer = MinMaxObserver(
                    name="weight",
                    dtype=DType.uint(8),
                    qscheme=QScheme.PER_CHANNEL_ASYMM,
                    channel_axis=0,
                )
                self.observer.load_qparams(
                    torch.tensor([0.1, 0.2]),
                    torch.tensor([3, 4], dtype=torch.int),
                    lock=True,
                )

        model = Candidate()
        site = SimpleNamespace(
            path="candidate.weight",
            role=SiteRole.PARAMETER,
            observer=model.observer,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/checkpoint.pt"
            with mock.patch.object(
                module,
                "iter_quantization_sites",
                return_value=(site,),
            ):
                save_joint_adaround_checkpoint(model, path)
                model.weight.data.zero_()
                model.observer.load_qparams(
                    torch.tensor([9.0, 9.0]),
                    torch.tensor([0, 0], dtype=torch.int),
                    lock=True,
                )
                result = apply_joint_adaround_checkpoint(model, path)
        torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
        scale, zero_point = model.observer.compute_qparams()
        torch.testing.assert_close(scale, torch.tensor([0.1, 0.2]))
        torch.testing.assert_close(
            zero_point,
            torch.tensor([3, 4], dtype=torch.int),
        )
        self.assertEqual(result["affine_site_count"], 1)

    def test_static_b1_reshape_preserves_reconstruction_minibatch(self) -> None:
        source = torch.arange(16 * 18 * 24 * 36, dtype=torch.float32).reshape(
            16, 18, 24, 36
        )
        values = {1: source}
        operation = {
            "name": "RESHAPE",
            "inputs": [1],
            "outputs": [2],
            "config": {
                "nhwc_memory_order": True,
                "shape": [1, -1, 18],
            },
        }
        reconstruction_module._execute_operation(
            operation,
            nn.Identity(),
            values,
        )
        expected = source.permute(0, 2, 3, 1).reshape(16, 864, 18)
        self.assertEqual(tuple(values[2].shape), (16, 864, 18))
        torch.testing.assert_close(values[2], expected)

    def test_priority_preset_is_unique_and_execution_ordered(self) -> None:
        self.assertEqual(len(PRIORITY_JOINT_GROUPS), len(set(PRIORITY_JOINT_GROUPS)))
        self.assertEqual(PRIORITY_JOINT_GROUPS[0], "stem")
        self.assertEqual(
            PRIORITY_JOINT_GROUPS[-2:],
            (
                "regressors_low_resolution_head",
                "regressors_high_resolution_head",
            ),
        )


if __name__ == "__main__":
    unittest.main()

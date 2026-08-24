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

"""Tests for hand-detector AdaRound group selection and activation replay."""

from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

import torch

from examples.hand_detector._support.adaround import (
    apply_activation_reconstruction_report,
    build_window_weight_groups,
)
from examples.hand_detector._support.reconstruction import ReconstructionWindow
from tico.quantization.wrapq.control import iter_quantization_sites
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d
from torch import nn


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = QuantConv2d(nn.Conv2d(1, 1, kernel_size=1, bias=False))


class _Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList((_Layer(),))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detector = _Detector()


class HandDetectorAdaRoundTest(unittest.TestCase):
    def _model(self) -> _Model:
        model = _Model()
        wrapper = model.detector.layers[0].conv
        wrapper.enable_calibration()
        sample = torch.ones(1, 1, 1, 1)
        wrapper(sample)
        wrapper.freeze_qparams()
        return model

    def test_window_selects_only_conv_weight_sites(self) -> None:
        model = self._model()
        window = ReconstructionWindow(
            name="stem",
            group_names=("stem",),
            operation_positions=(0,),
            input_tensor_ids=(0,),
            output_tensor_ids=(1,),
            site_paths=(),
        )
        groups = build_window_weight_groups(model, window)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0].site_path.endswith(".weight"))

    def test_replays_only_accepted_activation_qparams(self) -> None:
        model = self._model()
        activation_site = next(
            site
            for site in iter_quantization_sites(model)
            if site.observer_name == "act_out"
        )
        report = {
            "analysis": "simplified_qdrop_multistart_recipe",
            "metadata": {
                "global_percentile": 99.99,
                "observer_max_samples": 524288,
            },
            "steps": [
                {
                    "window": {"name": "stem"},
                    "observer_groups": [
                        {
                            "name": "tensor_1",
                            "site_paths": [activation_site.path],
                        }
                    ],
                    "reconstruction": {
                        "accepted": True,
                        "qparams": {"tensor_1": {"scale": 0.25, "zero_point": 7}},
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activation.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            summary = apply_activation_reconstruction_report(
                model,
                path,
                expected_percentile=99.99,
                expected_max_samples=524288,
            )
        qparams = activation_site.observer.compute_qparams()
        assert qparams is not None
        scale, zero_point = qparams
        self.assertAlmostEqual(float(scale), 0.25)
        self.assertEqual(int(zero_point), 7)
        self.assertEqual(summary["accepted_step_count"], 1)

    def test_rejects_parameter_site_in_activation_report(self) -> None:
        model = self._model()
        weight_site = next(
            site
            for site in iter_quantization_sites(model)
            if site.observer_name == "weight"
        )
        report = {
            "metadata": {"percentile": 99.99, "max_samples": 524288},
            "steps": [
                {
                    "window": {"name": "bad"},
                    "observer_groups": [
                        {"name": "weight", "site_paths": [weight_site.path]}
                    ],
                    "reconstruction": {
                        "accepted": True,
                        "qparams": {"weight": {"scale": 1.0, "zero_point": 0}},
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activation.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameter site"):
                apply_activation_reconstruction_report(model, path)


if __name__ == "__main__":
    unittest.main()

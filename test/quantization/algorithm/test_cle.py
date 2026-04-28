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

import unittest

import numpy as np
import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.algorithm.cle.cle import (
    _expand_pair_pattern,
    _expand_pairs,
    apply_cross_layer_equalization,
    equalize_layer_pair,
)
from tico.quantization.config.cle import CLEConfig


class TinyLinearPair(nn.Module):
    """Tiny model with two adjacent linear layers for CLE tests."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=True)
        self.fc2 = nn.Linear(3, 2, bias=True)

        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -2.0, 0.5, 0.25],
                        [0.1, -0.2, 0.3, -0.4],
                        [4.0, -1.0, 2.0, -3.0],
                    ]
                )
            )
            self.fc1.bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
            self.fc2.weight.copy_(
                torch.tensor(
                    [
                        [0.25, -8.0, 0.5],
                        [0.75, 4.0, -1.5],
                    ]
                )
            )
            self.fc2.bias.copy_(torch.tensor([0.25, -0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two adjacent linear layers without a nonlinearity."""
        return self.fc2(self.fc1(x))


class TinyConvPair(nn.Module):
    """Tiny model with two adjacent convolution layers for CLE tests."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 3, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=1, bias=True)

        with torch.no_grad():
            self.conv1.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[-2.0]]],
                        [[[0.25]], [[-0.5]]],
                        [[[3.0]], [[-1.0]]],
                    ]
                )
            )
            self.conv1.bias.copy_(torch.tensor([0.5, -1.0, 2.0]))
            self.conv2.weight.copy_(
                torch.tensor(
                    [
                        [[[0.5]], [[-4.0]], [[1.0]]],
                        [[[1.5]], [[2.0]], [[-2.0]]],
                        [[[0.25]], [[-1.0]], [[0.5]]],
                        [[[2.0]], [[0.75]], [[-3.0]]],
                    ]
                )
            )
            self.conv2.bias.copy_(torch.tensor([0.0, 0.25, -0.5, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two adjacent convolution layers without a nonlinearity."""
        return self.conv2(self.conv1(x))


class TinyStack(nn.Module):
    """Stacked model used to test wildcard CLE pair expansion."""

    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "up_proj": nn.Linear(4, 6),
                        "down_proj": nn.Linear(6, 4),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all stacked linear projection pairs."""
        for layer in self.layers:
            x = layer["down_proj"](layer["up_proj"](x))
        return x


class CLETest(unittest.TestCase):
    """Unit tests for Cross-Layer Equalization."""

    def setUp(self):
        """Make each test deterministic."""
        torch.manual_seed(0)
        np.random.seed(0)

    @torch.inference_mode()
    def test_equalize_linear_pair_preserves_output_and_updates_weights(self):
        """Verify that CLE preserves a linear pair output and changes weights."""
        model = TinyLinearPair().eval()
        x = torch.randn(5, 4)

        base_out = model(x)
        base_fc1_weight = model.fc1.weight.detach().clone()
        base_fc2_weight = model.fc2.weight.detach().clone()
        base_fc1_bias = model.fc1.bias.detach().clone()

        scale = equalize_layer_pair(model.fc1, model.fc2)
        target_out = model(x)

        np.testing.assert_allclose(
            actual=target_out.numpy(),
            desired=base_out.numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Output mismatch after CLE for a linear pair.",
        )
        self.assertEqual(scale.shape, torch.Size([3]))
        self.assertFalse(torch.allclose(base_fc1_weight, model.fc1.weight))
        self.assertFalse(torch.allclose(base_fc2_weight, model.fc2.weight))
        self.assertFalse(torch.allclose(base_fc1_bias, model.fc1.bias))
        self.assertTrue(torch.isfinite(model.fc1.weight).all())
        self.assertTrue(torch.isfinite(model.fc2.weight).all())

    @torch.inference_mode()
    def test_equalize_conv_pair_preserves_output_and_updates_weights(self):
        """Verify that CLE preserves a Conv2d pair output and changes weights."""
        model = TinyConvPair().eval()
        x = torch.randn(2, 2, 5, 5)

        base_out = model(x)
        base_conv1_weight = model.conv1.weight.detach().clone()
        base_conv2_weight = model.conv2.weight.detach().clone()
        base_conv1_bias = model.conv1.bias.detach().clone()

        scale = equalize_layer_pair(model.conv1, model.conv2)
        target_out = model(x)

        np.testing.assert_allclose(
            actual=target_out.numpy(),
            desired=base_out.numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Output mismatch after CLE for a Conv2d pair.",
        )
        self.assertEqual(scale.shape, torch.Size([3]))
        self.assertFalse(torch.allclose(base_conv1_weight, model.conv1.weight))
        self.assertFalse(torch.allclose(base_conv2_weight, model.conv2.weight))
        self.assertFalse(torch.allclose(base_conv1_bias, model.conv1.bias))
        self.assertTrue(torch.isfinite(model.conv1.weight).all())
        self.assertTrue(torch.isfinite(model.conv2.weight).all())

    @torch.inference_mode()
    def test_apply_cross_layer_equalization_returns_applied_scales(self):
        """Verify that model-level CLE applies configured pairs and returns scales."""
        model = TinyLinearPair().eval()
        x = torch.randn(5, 4)
        base_out = model(x)

        cfg = CLEConfig(
            pairs=[("fc1", "fc2")],
            show_progress=False,
        )
        applied_scales = apply_cross_layer_equalization(model, cfg)
        target_out = model(x)

        self.assertEqual(set(applied_scales.keys()), {("fc1", "fc2")})
        self.assertEqual(applied_scales[("fc1", "fc2")].shape, torch.Size([3]))
        np.testing.assert_allclose(
            actual=target_out.numpy(),
            desired=base_out.numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Output mismatch after model-level CLE.",
        )

    @torch.inference_mode()
    def test_prepare_convert_integration(self):
        """Verify that CLE works through the public prepare/convert flow."""
        model = TinyLinearPair().eval()
        x = torch.randn(5, 4)

        base_out = model(x)
        base_fc1_weight = model.fc1.weight.detach().clone()

        q_m = prepare(
            model,
            CLEConfig(
                pairs=[("fc1", "fc2")],
                show_progress=False,
            ),
        )
        self.assertIs(q_m, model)

        q_m = convert(q_m)
        target_out = q_m(x)

        self.assertFalse(torch.allclose(base_fc1_weight, q_m.fc1.weight))
        np.testing.assert_allclose(
            actual=target_out.numpy(),
            desired=base_out.numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Output mismatch after CLE prepare/convert flow.",
        )

    def test_expand_pair_pattern_matches_same_wildcard_capture(self):
        """Verify that wildcard pair expansion only pairs matching layer indices."""
        module_names = [
            "layers",
            "layers.0",
            "layers.0.up_proj",
            "layers.0.down_proj",
            "layers.1",
            "layers.1.up_proj",
            "layers.1.down_proj",
        ]

        pairs = _expand_pair_pattern(
            module_names,
            "layers.*.up_proj",
            "layers.*.down_proj",
        )

        self.assertEqual(
            pairs,
            [
                ("layers.0.up_proj", "layers.0.down_proj"),
                ("layers.1.up_proj", "layers.1.down_proj"),
            ],
        )

    def test_expand_pairs_supports_mixed_exact_and_wildcard_specs(self):
        """Verify that exact and wildcard CLE pair specs can be mixed."""
        model = TinyStack(num_layers=3)

        pairs = _expand_pairs(
            model,
            [
                ("layers.0.up_proj", "layers.0.down_proj"),
                ("layers.*.up_proj", "layers.*.down_proj"),
            ],
        )

        self.assertEqual(
            pairs,
            [
                ("layers.0.up_proj", "layers.0.down_proj"),
                ("layers.1.up_proj", "layers.1.down_proj"),
                ("layers.2.up_proj", "layers.2.down_proj"),
            ],
        )

    def test_expand_pair_pattern_rejects_one_sided_wildcard(self):
        """Verify that pair specs cannot use a wildcard on only one side."""
        module_names = ["layers.0.up_proj", "layers.0.down_proj"]

        with self.assertRaisesRegex(ValueError, "both use wildcards or both be exact"):
            _expand_pair_pattern(
                module_names,
                "layers.*.up_proj",
                "layers.0.down_proj",
            )

    def test_expand_pair_pattern_rejects_missing_concrete_pair(self):
        """Verify that wildcard pair expansion fails when no pair is found."""
        module_names = ["layers.0.up_proj"]

        with self.assertRaisesRegex(ValueError, "No concrete CLE layer pairs"):
            _expand_pair_pattern(
                module_names,
                "layers.*.up_proj",
                "layers.*.down_proj",
            )

    def test_equalize_layer_pair_rejects_invalid_channel_shape(self):
        """Verify that CLE rejects pairs with incompatible channel dimensions."""
        first = nn.Linear(4, 3)
        second = nn.Linear(4, 2)

        with self.assertRaisesRegex(ValueError, "first.out_channels"):
            equalize_layer_pair(first, second)

    def test_equalize_layer_pair_rejects_unsupported_layer(self):
        """Verify that CLE rejects unsupported layer types."""
        first = nn.BatchNorm1d(3)
        second = nn.Linear(3, 2)

        with self.assertRaisesRegex(TypeError, "first CLE layer"):
            equalize_layer_pair(first, second)  # type: ignore[arg-type]

    @torch.inference_mode()
    def test_equalize_bias_option_keeps_first_bias_when_disabled(self):
        """Verify that equalize_bias=False leaves the first layer bias unchanged."""
        model = TinyLinearPair().eval()
        base_bias = model.fc1.bias.detach().clone()

        equalize_layer_pair(model.fc1, model.fc2, equalize_bias=False)

        self.assertTrue(torch.allclose(base_bias, model.fc1.bias))

    @torch.inference_mode()
    def test_range_method_is_supported(self):
        """Verify that the range-based CLE method applies finite scales."""
        model = TinyLinearPair().eval()

        scale = equalize_layer_pair(model.fc1, model.fc2, method="range")

        self.assertEqual(scale.shape, torch.Size([3]))
        self.assertTrue(torch.isfinite(scale).all())

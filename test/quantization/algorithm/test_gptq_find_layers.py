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

"""Characterize layer selection and traversal through existing GPTQ utility APIs."""

import unittest

import torch
import torch.nn as nn

from tico.quantization.algorithm.gemma4_gptq.utils import (
    find_layers as find_gemma4_layers,
    get_quantizable_layers as get_gemma4_layers,
)
from tico.quantization.algorithm.gptq.utils import find_layers as find_gptq_layers
from tico.quantization.algorithm.qwen3_vl_gptq.utils import (
    find_layers as find_qwen3_vl_layers,
    get_quantizable_layers as get_qwen3_vl_layers,
)

_FINDERS = {
    "gptq": find_gptq_layers,
    "qwen3_vl": find_qwen3_vl_layers,
    "gemma4": find_gemma4_layers,
}


class _DerivedLinear(nn.Linear):
    pass


class TestGPTQFindLayers(unittest.TestCase):
    def setUp(self):
        rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng_state)
        torch.random.default_generator.manual_seed(0)

    def assert_layers(self, actual, expected):
        self.assertEqual(list(actual), list(expected))
        for name, module in expected.items():
            self.assertIs(actual[name], module)

    def _layer_types(self):
        return nn.ModuleDict(
            {
                "linear": nn.Linear(1, 1),
                "conv1d": nn.Conv1d(1, 1, 1),
                "conv2d": nn.Conv2d(1, 1, 1),
                "conv3d": nn.Conv3d(1, 1, 1),
                "transpose2d": nn.ConvTranspose2d(1, 1, 1),
                "transpose1d": nn.ConvTranspose1d(1, 1, 1),
                "transpose3d": nn.ConvTranspose3d(1, 1, 1),
                "embedding": nn.Embedding(1, 1),
                "derived": _DerivedLinear(1, 1),
            }
        )

    def test_default_layer_types(self):
        model = self._layer_types()
        for family, finder in _FINDERS.items():
            with self.subTest(family=family):
                names = (
                    ["linear"]
                    if family == "gptq"
                    else ["linear", "conv1d", "conv2d", "conv3d", "transpose2d"]
                )
                self.assert_layers(finder(model), {name: model[name] for name in names})

    def test_explicit_none_preserves_family_defaults(self):
        model = self._layer_types()
        with self.assertRaisesRegex(TypeError, "NoneType"):
            find_gptq_layers(model, layers=None)

        expected = {
            name: model[name]
            for name in ["linear", "conv1d", "conv2d", "conv3d", "transpose2d"]
        }
        for finder in (find_qwen3_vl_layers, find_gemma4_layers):
            with self.subTest(finder=finder.__module__):
                self.assert_layers(finder(model, layers=None), expected)

    def test_explicit_empty_types_select_nothing(self):
        model = nn.Sequential(nn.Linear(1, 1))
        empty_list: list[type[nn.Module]] = []
        for family, finder in _FINDERS.items():
            for layers in (empty_list, ()):
                with self.subTest(family=family, layers=layers):
                    before = layers[:]
                    self.assertEqual(finder(model, layers=layers), {})
                    self.assertEqual(layers, before)

    def test_explicit_types_replace_defaults_and_preserve_traversal_order(self):
        model = self._layer_types()
        expected = {"conv3d": model["conv3d"], "embedding": model["embedding"]}
        for family, finder in _FINDERS.items():
            for layers in ([nn.Embedding, nn.Conv3d], (nn.Embedding, nn.Conv3d)):
                with self.subTest(family=family, layers=layers):
                    before = layers[:]
                    self.assert_layers(finder(model, layers=layers), expected)
                    self.assertEqual(layers, before)

    def test_exact_type_matching_still_visits_subclass_children(self):
        derived = _DerivedLinear(1, 1)
        derived.add_module("inner", nn.Linear(1, 1))
        model = nn.ModuleDict({"derived": derived})
        for family, finder in _FINDERS.items():
            with self.subTest(family=family):
                self.assert_layers(
                    finder(model, layers=[nn.Linear]), {"derived.inner": derived.inner}
                )
                self.assert_layers(
                    finder(model, layers=[_DerivedLinear]), {"derived": derived}
                )

    def test_matching_root_retains_prefix_and_stops_before_children(self):
        model = nn.Linear(1, 1)
        model.add_module("inner", nn.Linear(1, 1))
        for family, finder in _FINDERS.items():
            for prefix in ("", "model.block", "model.block."):
                with self.subTest(family=family, prefix=prefix):
                    self.assert_layers(finder(model, name=prefix), {prefix: model})

    def test_matching_nested_container_stops_before_children(self):
        branch = nn.Sequential(nn.Linear(1, 1))
        model = nn.ModuleDict({"branch": branch, "leaf": nn.Linear(1, 1)})
        for family, finder in _FINDERS.items():
            with self.subTest(family=family):
                self.assert_layers(
                    finder(model, layers=(nn.Sequential, nn.Linear)),
                    {"branch": branch, "leaf": model["leaf"]},
                )

    def test_depth_first_registration_order_module_list_paths_and_prefix(self):
        first = nn.Linear(1, 1)
        nested = nn.Linear(1, 1)
        second = nn.Linear(1, 1)
        last = nn.Linear(1, 1)
        model = nn.ModuleDict(
            {
                "z_first": first,
                "blocks": nn.ModuleList([nn.ModuleDict({"nested": nested}), second]),
                "a_last": last,
            }
        )
        for family, finder in _FINDERS.items():
            for prefix in ("", "tower", "tower."):
                with self.subTest(family=family, prefix=prefix):
                    stem = f"{prefix}." if prefix else ""
                    self.assert_layers(
                        finder(model, name=prefix),
                        {
                            f"{stem}z_first": first,
                            f"{stem}blocks.0.nested": nested,
                            f"{stem}blocks.1": second,
                            f"{stem}a_last": last,
                        },
                    )

    def test_aliases_deduplicate_only_within_each_parent(self):
        shared = nn.Linear(1, 1)
        model = nn.ModuleDict(
            {
                "first": shared,
                "alias": shared,
                "left": nn.ModuleDict({"shared": shared}),
                "right": nn.ModuleDict({"shared": shared}),
            }
        )
        for family, finder in _FINDERS.items():
            with self.subTest(family=family):
                self.assert_layers(
                    finder(model),
                    {"first": shared, "left.shared": shared, "right.shared": shared},
                )

    def test_get_quantizable_layers_keeps_multimodal_defaults(self):
        model = self._layer_types()
        expected = {
            name: model[name]
            for name in ["linear", "conv1d", "conv2d", "conv3d", "transpose2d"]
        }
        for finder in (get_qwen3_vl_layers, get_gemma4_layers):
            with self.subTest(finder=finder.__module__):
                self.assert_layers(finder(model), expected)

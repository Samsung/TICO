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

import json
import tempfile
import unittest
from pathlib import Path

from tico.quantization.recipes.config import (
    apply_overrides,
    Config,
    get_by_path,
    load_recipe_config,
    parse_override,
    parse_scalar,
    save_effective_config,
    set_by_path,
)


class TestRecipeConfig(unittest.TestCase):
    def test_load_recipe_config_applies_nested_list_and_named_overrides(self):
        """Recipe config loading should support dotted paths and list item names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "recipe.yaml"
            config_path.write_text(
                """
model:
  family: llama
  name_or_path: old-model
runtime:
  device: cuda
pipeline:
  - name: gptq
    enabled: true
  - name: ptq
    enabled: true
evaluation:
  enabled: false
""",
                encoding="utf-8",
            )

            cfg = load_recipe_config(
                config_path,
                overrides=[
                    "model.name_or_path=new-model",
                    "runtime.device=cpu",
                    "pipeline.gptq.enabled=false",
                    "pipeline.ptq.linear_weight_bits=4",
                    "evaluation.enabled=true",
                    "created.items.0.name=first",
                    "created.items.1.name=second",
                ],
            )

        self.assertEqual(cfg["model"]["name_or_path"], "new-model")
        self.assertEqual(cfg["runtime"]["device"], "cpu")
        self.assertFalse(cfg["pipeline"][0]["enabled"])
        self.assertEqual(cfg["pipeline"][1]["linear_weight_bits"], 4)
        self.assertTrue(cfg["evaluation"]["enabled"])
        self.assertEqual(cfg["created"]["items"][0]["name"], "first")
        self.assertEqual(cfg["created"]["items"][1]["name"], "second")

    def test_load_recipe_config_supports_json(self):
        """JSON configs should load with the same override semantics as YAML configs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "recipe.json"
            config_path.write_text(
                json.dumps({"pipeline": [{"name": "ptq", "enabled": True}]}),
                encoding="utf-8",
            )

            cfg = load_recipe_config(
                config_path, overrides=["pipeline.0.enabled=false"]
            )

        self.assertFalse(cfg["pipeline"][0]["enabled"])

    def test_parse_scalar(self):
        """Scalar parsing should preserve common YAML-like value types."""
        cases = {
            "true": True,
            "false": False,
            "null": None,
            "none": None,
            "4": 4,
            "4.5": 4.5,
            "[1, 2]": [1, 2],
            '{"a": 1}': {"a": 1},
            "plain-string": "plain-string",
        }

        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_scalar(raw), expected)

    def test_parse_override_rejects_missing_equals(self):
        """Malformed overrides should fail early with a helpful exception."""
        with self.assertRaises(ValueError):
            parse_override("runtime.device")

    def test_set_by_path_supports_named_list_items(self):
        """List traversal should support unique mapping items addressed by name."""
        cfg: Config = {
            "pipeline": [
                {"name": "gptq", "enabled": True},
                {"name": "ptq", "enabled": True},
            ]
        }

        set_by_path(cfg, ["pipeline", "gptq", "mse"], "mse")
        set_by_path(cfg, ["pipeline", "ptq", "linear_weight"], "uint4")

        self.assertEqual(cfg["pipeline"][0]["mse"], "mse")
        self.assertEqual(cfg["pipeline"][1]["linear_weight"], "uint4")

    def test_set_by_path_rejects_missing_named_list_item(self):
        """Named list traversal should fail when no item has the requested name."""
        cfg: Config = {"pipeline": [{"name": "ptq"}]}

        with self.assertRaises(KeyError):
            set_by_path(cfg, ["pipeline", "gptq", "enabled"], False)

    def test_set_by_path_rejects_ambiguous_named_list_item(self):
        """Named list traversal should reject duplicate item names."""
        cfg: Config = {
            "pipeline": [
                {"name": "ptq", "enabled": True},
                {"name": "ptq", "enabled": False},
            ]
        }

        with self.assertRaises(ValueError):
            set_by_path(cfg, ["pipeline", "ptq", "enabled"], False)

    def test_get_by_path_returns_default_for_missing_values(self):
        """Nested path lookup should return the provided default when traversal fails."""
        cfg: Config = {"pipeline": [{"name": "ptq"}]}

        self.assertEqual(get_by_path(cfg, "pipeline.0.name"), "ptq")
        self.assertEqual(get_by_path(cfg, "pipeline.ptq.name"), "ptq")
        self.assertEqual(get_by_path(cfg, "pipeline.1.name", "missing"), "missing")
        self.assertEqual(get_by_path(cfg, "pipeline.name", "missing"), "missing")

    def test_apply_overrides_mutates_existing_config(self):
        """Applying overrides should mutate the target mapping in place."""
        cfg: Config = {"runtime": {"device": "cuda"}}

        apply_overrides(cfg, ["runtime.device=cpu", "pipeline.0.name=ptq"])

        self.assertEqual(cfg["runtime"]["device"], "cpu")
        self.assertEqual(cfg["pipeline"][0]["name"], "ptq")

    def test_apply_overrides_supports_named_list_items(self):
        """Applying overrides should support named list item traversal."""
        cfg: Config = {
            "pipeline": [
                {"name": "gptq", "mse": None},
                {"name": "ptq", "lm_head_weight": "uint8"},
            ]
        }

        apply_overrides(
            cfg,
            [
                "pipeline.gptq.mse=mse",
                "pipeline.ptq.lm_head_weight=uint4",
            ],
        )

        self.assertEqual(cfg["pipeline"][0]["mse"], "mse")
        self.assertEqual(cfg["pipeline"][1]["lm_head_weight"], "uint4")

    def test_save_effective_config_serializes_non_yaml_values(self):
        """Effective config saving should serialize tuples and non-plain values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "effective.yaml"
            save_effective_config(output_path, {"shape": (1, 2), "obj": object()})
            text = output_path.read_text(encoding="utf-8")

        self.assertIn("shape", text)
        self.assertIn("obj", text)

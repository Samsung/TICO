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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.examples.quantize as quantize_cli

import torch


class TestQuantizationExamplesCLI(unittest.TestCase):
    def test_quantize_cli_builds_overrides_and_runs_recipe(self):
        """quantize.py should translate CLI arguments into config overrides."""
        calls: dict[str, Any] = {}

        class FakeRunner:
            """Fake runner that records the loaded config."""

            def run(self, cfg):
                """Record the config passed by the CLI."""
                calls["run"] = cfg

        def fake_load_recipe_config(path, overrides):
            calls["load"] = (path, overrides)
            return {"loaded": True}

        argv = [
            "quantize.py",
            "--config",
            "recipe.yaml",
            "--model",
            "new-model",
            "--device",
            "cpu",
            "--output-dir",
            "out",
            "--set",
            "pipeline.0.enabled=false",
        ]

        with patch.object(
            quantize_cli, "load_recipe_config", fake_load_recipe_config
        ), patch.object(quantize_cli, "QuantizationRunner", FakeRunner), patch.object(
            sys, "argv", argv
        ):
            quantize_cli.main()

        self.assertEqual(
            calls["load"],
            (
                "recipe.yaml",
                [
                    "pipeline.0.enabled=false",
                    "model.name_or_path=new-model",
                    "runtime.device=cpu",
                    "export.output_dir=out",
                ],
            ),
        )
        self.assertEqual(calls["run"], {"loaded": True})

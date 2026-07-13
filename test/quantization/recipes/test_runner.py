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

import contextlib
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.runner as runner_mod
from tico.quantization.recipes.runner import QuantizationRunner


class DummyAdapter:
    """Adapter fake for runner tests."""

    family = "dummy"

    def __init__(self, events):
        self.events = events

    def load_model(self, ctx):
        """Record model loading and attach a dummy model."""
        self.events.append("load")
        ctx.model = SimpleNamespace(name="model")
        return ctx

    def build_calibration_inputs(self, ctx):
        """Record calibration input construction."""
        self.events.append("calibration")
        return ["sample"]

    def evaluate(self, ctx):
        """Record evaluation."""
        self.events.append("evaluate")

    def export(self, ctx):
        """Record export."""
        self.events.append("export")


class DummyStage:
    """Stage fake that records execution order."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def run(self, ctx, stage_cfg):
        """Record the stage run and mutate the model."""
        self.events.append(f"stage:{self.name}")
        ctx.model = f"after-{self.name}"
        return ctx


class TestQuantizationRunner(unittest.TestCase):
    def test_runner_executes_enabled_stages_in_order(self):
        """QuantizationRunner should load, calibrate, run enabled stages, evaluate, and export."""
        events = []

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            runner_mod, "set_seed", lambda seed: events.append(f"seed:{seed}")
        ), patch.object(
            runner_mod, "get_adapter", lambda family: DummyAdapter(events)
        ), patch.object(
            runner_mod, "get_stage", lambda name: DummyStage(name, events)
        ), patch.object(
            runner_mod,
            "save_effective_config",
            lambda path, cfg: events.append(f"save:{path.name}"),
        ):
            cfg = {
                "model": {"family": "dummy", "name_or_path": "model"},
                "runtime": {"seed": 7},
                "pipeline": [
                    {"name": "gptq", "enabled": True},
                    {"name": "skip", "enabled": False},
                    {"name": "ptq", "enabled": True},
                ],
                "export": {"output_dir": tmpdir},
            }

            with contextlib.redirect_stdout(io.StringIO()):
                ctx = QuantizationRunner().run(cfg)

        self.assertEqual(ctx.model, "after-ptq")
        self.assertEqual(
            events,
            [
                "seed:7",
                "load",
                "calibration",
                "stage:gptq",
                "stage:ptq",
                "evaluate",
                "export",
                "save:effective_config.yaml",
            ],
        )

    def test_config_summary_prints_ptq_override_policies(self):
        """Config summary should print PTQ override policy declarations."""
        cfg = {
            "model": {
                "family": "qwen3_vl",
                "name_or_path": "Qwen/Qwen3-VL-4B-Instruct",
            },
            "pipeline": [
                {
                    "name": "ptq",
                    "enabled": True,
                    "specs": {
                        "mx_int8_activation": {
                            "kind": "mx",
                            "elem_format": "int8",
                            "axis": -1,
                            "shared_exp_method": "max",
                            "round": "nearest",
                        }
                    },
                    "override_policies": [
                        {
                            "name": "all_linear_activations_mx",
                            "target": {
                                "component": "all",
                                "layers": "all",
                                "op_type": "linear",
                                "observer_role": "activation",
                            },
                            "spec": "mx_int8_activation",
                        },
                        {
                            "name": "optional_vision_softmax",
                            "enabled": False,
                            "allow_empty": True,
                            "target": {
                                "component": "vision",
                                "layers": "all",
                                "module": "attn",
                                "observers": ["softmax"],
                            },
                            "spec": "int16",
                        },
                    ],
                }
            ],
            "calibration": {"n_samples": 2, "seq_len": 128},
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runner_mod._print_config_summary(cfg)

        output = stdout.getvalue()
        header = "=== PTQ override policies ==="

        self.assertIn(header, output)
        self.assertIn(
            "all_linear_activations_mx -> "
            "spec=mx_int8_activation "
            "[mx(elem_format=int8, axis=-1, "
            "shared_exp_method=max, round=nearest)]",
            output,
        )
        self.assertIn("component=all, layers=all", output)
        self.assertIn("op_type=linear, observer_role=activation", output)
        self.assertIn("optional_vision_softmax (disabled)", output)
        self.assertIn("module=attn, observers=['softmax']", output)
        self.assertIn("allow_empty=True", output)
        self.assertLess(output.index("Calibration samples"), output.index("Profile"))
        self.assertLess(output.index("Profile"), output.index(header))
        self.assertIn("Profile               : not set\n\n" + header, output)

    def test_config_summary_omits_override_policy_section_without_policies(self):
        """Config summary should omit the policy section when no policies exist."""
        cfg = {
            "model": {"family": "dummy", "name_or_path": "model"},
            "pipeline": [{"name": "ptq", "enabled": True}],
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runner_mod._print_config_summary(cfg)

        self.assertNotIn("=== PTQ override policies ===", stdout.getvalue())

    def test_config_summary_respects_print_config_for_override_policies(self):
        """Disabling the config summary should also hide policy declarations."""
        cfg = {
            "runtime": {"print_config": False},
            "pipeline": [
                {
                    "name": "ptq",
                    "override_policies": [
                        {
                            "name": "all_linear_activations",
                            "target": {
                                "op_type": "linear",
                                "observer_role": "activation",
                            },
                            "spec": "int16",
                        }
                    ],
                }
            ],
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runner_mod._print_config_summary(cfg)

        self.assertEqual(stdout.getvalue(), "")

    def test_runner_requires_model_family_and_name(self):
        """QuantizationRunner should reject configs without required model keys."""
        for cfg in [
            {"model": {"name_or_path": "model"}},
            {"model": {"family": "dummy"}},
        ]:
            with self.subTest(cfg=cfg):
                with self.assertRaises(KeyError):
                    QuantizationRunner().run(cfg)

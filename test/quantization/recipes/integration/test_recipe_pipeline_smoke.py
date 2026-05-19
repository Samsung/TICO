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


class SmokeAdapter:
    """Small adapter used to exercise the full recipe runner flow."""

    family = "smoke"

    def __init__(self, events):
        self.events = events

    def load_model(self, ctx):
        """Attach a synthetic model."""
        self.events.append("load")
        ctx.model = SimpleNamespace(name="model")
        return ctx

    def build_calibration_inputs(self, ctx):
        """Return synthetic calibration samples."""
        self.events.append("calibration")
        return ["sample-0"]

    def evaluate(self, ctx):
        """Record evaluation."""
        self.events.append(("evaluate", ctx.model))

    def export(self, ctx):
        """Record export."""
        self.events.append(("export", ctx.model))


class SmokeStage:
    """Small stage used to mutate the context in a deterministic way."""

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def run(self, ctx, stage_cfg):
        """Append the stage name to the model marker."""
        self.events.append(("stage", self.name, list(ctx.calibration_inputs)))
        ctx.model = f"{ctx.model.name if hasattr(ctx.model, 'name') else ctx.model}->{self.name}"
        return ctx


class TestRecipePipelineSmoke(unittest.TestCase):
    def test_runner_pipeline_smoke(self):
        """A synthetic recipe should execute through the same public runner path as examples."""
        events = []

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            runner_mod, "set_seed", lambda seed: events.append(("seed", seed))
        ), patch.object(
            runner_mod, "get_adapter", lambda family: SmokeAdapter(events)
        ), patch.object(
            runner_mod, "get_stage", lambda name: SmokeStage(name, events)
        ), patch.object(
            runner_mod,
            "save_effective_config",
            lambda path, cfg: events.append(("save", path.name)),
        ):
            cfg = {
                "model": {"family": "smoke", "name_or_path": "synthetic"},
                "runtime": {"seed": 123},
                "pipeline": [{"name": "first"}, {"name": "second"}],
                "export": {"output_dir": tmpdir},
            }

            with contextlib.redirect_stdout(io.StringIO()):
                ctx = QuantizationRunner().run(cfg)

        self.assertEqual(ctx.model, "model->first->second")
        self.assertIn(("stage", "first", ["sample-0"]), events)
        self.assertIn(("stage", "second", ["sample-0"]), events)
        self.assertIn(("evaluate", "model->first->second"), events)
        self.assertIn(("export", "model->first->second"), events)
        self.assertIn(("save", "effective_config.yaml"), events)

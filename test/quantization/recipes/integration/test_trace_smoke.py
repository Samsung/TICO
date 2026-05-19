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

import contextlib
import io
import unittest
from unittest.mock import patch

import tico.quantization.recipes.debug.trace as trace_mod

import torch
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import trace_ptq_parity


class TraceAdapter:
    """Fake adapter that provides enough PTQ hooks for trace smoke tests."""

    family = "trace"

    def build_ptq_config(self, ctx, stage_cfg):
        """Return a sentinel PTQ config."""
        return {"ptq": True}

    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg):
        """Run one calibration pass through the model."""
        with torch.no_grad():
            prepared_model(ctx.calibration_inputs[0])


class TestTraceSmoke(unittest.TestCase):
    def test_trace_ptq_parity_smoke(self):
        """Trace parity should run on a tiny model when prepare/convert are patched."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        ctx = RecipeContext(
            cfg={"pipeline": [{"name": "ptq"}]},
            adapter=TraceAdapter(),
            model=model,
            calibration_inputs=[torch.ones(1, 2)],
        )
        ctx.device = torch.device("cpu")

        with patch.object(
            trace_mod, "prepare", lambda model, qcfg: model
        ), patch.object(trace_mod, "convert", lambda model: model):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                trace_ptq_parity(
                    ctx, enable_quantization=True, interesting_modules=["0"]
                )

        output = buffer.getvalue()
        self.assertIn("FP trace", output)
        self.assertIn("PTQ trace", output)
        self.assertIn("Side-by-side diff", output)

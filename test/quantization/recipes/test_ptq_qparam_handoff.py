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
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.qparams as qparams
import tico.quantization.recipes.stages.ptq as ptq_mod

import torch

from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.ptq import PTQStage


class FakeObserver:
    """Fake affine observer that records injected quantization parameters."""

    def __init__(self):
        self.loaded = None

    def load_qparams(self, scale, zero, lock):
        """Record the qparams passed by the injection helper."""
        self.loaded = (scale, zero, lock)


class FakeQuantModule(torch.nn.Module):
    """Fake prepared quant module with a named weight observer."""

    def __init__(self, fp_name, observer):
        super().__init__()
        self.fp_name = fp_name
        self._observer = observer

    def get_observer(self, name):
        """Return the fake observer for the weight role."""
        if name == "weight":
            return self._observer
        return None


class DummyAdapter:
    """Adapter fake used by the GPTQ-to-PTQ handoff regression test."""

    family = "llama"

    def __init__(self):
        self.calibrated_model = None

    def build_ptq_config(self, ctx, stage_cfg):
        """Return a sentinel PTQ config accepted by the no-override path."""
        return {"config": "ptq", "stage": stage_cfg}

    def calibrate_prepared_model(self, ctx, model, stage_cfg):
        """Record the prepared model passed to calibration."""
        self.calibrated_model = model


class TestPTQQParamHandoff(unittest.TestCase):
    def test_stage_injects_nested_owner_qparams_into_prepared_tree(self):
        """PTQStage should traverse prepared observers, not the quantizer owner."""
        scale = torch.tensor([0.5])
        zero = torch.tensor([1])
        quantizers = {
            "linear": SimpleNamespace(scale=scale, zero=zero),
        }

        source_model = torch.nn.Module()
        source_model.quantizers = quantizers

        observer = FakeObserver()
        top_level_wrapper = torch.nn.Module()
        top_level_wrapper.module = source_model
        top_level_wrapper.linear = FakeQuantModule("linear", observer)

        prepared_model = torch.nn.Module()
        prepared_model.wrapped = top_level_wrapper

        owner, found_quantizers = qparams.find_gptq_quantizers(prepared_model)
        self.assertIs(owner, source_model)
        self.assertIs(found_quantizers, quantizers)
        self.assertFalse(hasattr(top_level_wrapper, "quantizers"))

        adapter = DummyAdapter()
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)

        with patch.object(
            ptq_mod, "prepare", lambda model, config: prepared_model
        ), patch.object(ptq_mod, "convert", lambda model: model), patch.object(
            qparams, "QuantModuleBase", FakeQuantModule
        ), patch.object(
            qparams, "AffineObserverBase", FakeObserver
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                result = PTQStage().run(ctx, {"name": "ptq"})

        self.assertIs(result.model, prepared_model)
        self.assertEqual(observer.loaded, (scale, zero, True))
        self.assertFalse(hasattr(source_model, "quantizers"))
        self.assertIs(adapter.calibrated_model, prepared_model)


if __name__ == "__main__":
    unittest.main()

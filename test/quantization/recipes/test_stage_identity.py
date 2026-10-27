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

"""Stage discovery must recognize the same names as execution dispatch."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import contextlib
import copy
import io
import unittest
from typing import Any

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.gemma4_assistant import Gemma4AssistantAdapter
from tico.quantization.recipes.adapters.llama import (
    _find_ptq_profile,
    _is_stage_enabled,
    LlamaAdapter,
)
from tico.quantization.recipes.adapters.qwen3_vl import (
    _find_stage as find_qwen_stage,
    Qwen3VLAdapter,
)
from tico.quantization.recipes.config import get_by_path, set_by_path
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import trace_ptq_parity
from tico.quantization.recipes.runner import _find_stage, _print_config_summary
from tico.quantization.recipes.stages import get_stage
from tico.quantization.recipes.stages.ptq import _find_enabled_stage, PTQStage


class _CpuAdapter:
    """Use real PTQ preparation, calibration, and conversion on a tiny linear."""

    family = "stage_identity_cpu"

    def __init__(self):
        self.selected_stages = []
        self.calibration_calls = 0

    def build_ptq_config(self, ctx, stage_cfg):
        """Record which config block reaches the public PTQ implementation."""
        self.selected_stages.append(stage_cfg)
        return PTQConfig()

    def calibrate_prepared_model(self, ctx, model, stage_cfg):
        """Collect real observer statistics without a model or dataset download."""
        self.calibration_calls += 1
        with torch.no_grad():
            model(ctx.calibration_inputs[0])


def _context(cfg):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 2).eval()
    return RecipeContext(
        cfg=cfg,
        adapter=_CpuAdapter(),
        model=model,
        calibration_inputs=[torch.tensor([[1.0, -1.0]])],
    )


class TestStageIdentity(unittest.TestCase):
    def test_dispatch_keeps_case_and_whitespace_policy(self):
        """Case variants already dispatch; surrounding whitespace stays invalid."""
        for name in ("ptq", "PTQ", "PtQ"):
            self.assertIs(get_stage(name), get_stage("ptq"))
        for name in (" ptq", "ptq ", " PTQ "):
            with self.subTest(name=name), self.assertRaises(KeyError):
                get_stage(name)

    def test_real_ptq_rejects_missing_metadata_for_case_variants(self):
        """An enabled GPTQ stage must never silently fall back to PTQ stats."""
        for name in ("gptq", "GPTQ", "GpTq"):
            with self.subTest(name=name):
                cfg = {"pipeline": [{"name": name}, {"name": "ptq"}]}
                before = copy.deepcopy(cfg)
                ctx = _context(cfg)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "no GPTQ quantizers"):
                        PTQStage().run(ctx, cfg["pipeline"][1])
                self.assertEqual(ctx.adapter.calibration_calls, 0)
                self.assertEqual(cfg, before)

    def test_real_ptq_preserves_non_handoff_workflows(self):
        """Pure PTQ, disabled GPTQ, and explicit non-reuse still calibrate."""
        pipelines: tuple[list[dict[str, Any]], ...] = (
            [{"name": "PTQ"}],
            [{"name": "GPTQ", "enabled": False}, {"name": "PTQ"}],
            [{"name": "GPTQ"}, {"name": "PTQ", "reuse_gptq_qparams": False}],
        )
        for pipeline in pipelines:
            with self.subTest(pipeline=pipeline):
                cfg = {"pipeline": pipeline}
                before = copy.deepcopy(cfg)
                ctx = _context(cfg)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = get_stage(pipeline[-1]["name"]).run(ctx, pipeline[-1])
                self.assertIs(result, ctx)
                self.assertEqual(ctx.adapter.calibration_calls, 1)
                self.assertFalse(hasattr(ctx.model, "tico_quantizer"))
                with torch.no_grad():
                    output = ctx.model(ctx.calibration_inputs[0])
                self.assertEqual(output.shape, (1, 2))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(cfg, before)

    def test_enabled_search_skips_disabled_matching_stage(self):
        """The handoff search still selects the first enabled matching entry."""
        first = {"name": "GPTQ", "enabled": False}
        second = {"name": "GpTq", "enabled": True}
        third = {"name": "gptq", "enabled": True}
        cfg = {"pipeline": [None, first, second, third]}
        self.assertIs(_find_enabled_stage(cfg, "gptq"), second)

    def test_first_match_searches_keep_disabled_entry(self):
        """Summary, export, and adapter searches retain their first-match rules."""
        first = {
            "name": "PTQ",
            "enabled": False,
            "profile": "reference_eval",
            "activation": "int16",
        }
        second = {"name": "ptq", "profile": "npu_export", "activation": "uint8"}
        cfg = {"pipeline": [first, second]}
        before = copy.deepcopy(cfg)
        self.assertIs(_find_stage(cfg, "ptq"), first)
        self.assertIs(find_qwen_stage(cfg, "ptq"), first)
        self.assertFalse(_is_stage_enabled(cfg, "ptq"))
        self.assertEqual(_find_ptq_profile(cfg), "reference_eval")
        self.assertEqual(
            Gemma4AssistantAdapter._quantization_profile(cfg),
            {"activation": "int16"},
        )
        self.assertEqual(cfg, before)

    def test_decode_calibration_uses_first_enabled_ptq_case_variant(self):
        """PTQ-specific decode sampling must agree with the dispatched stage."""
        cfg = {
            "calibration": {"decode_steps": 1},
            "pipeline": [
                {"name": "PTQ", "enabled": False, "decode_calibration_steps": 8},
                {"name": "PtQ", "decode_calibration_steps": 3},
                {"name": "ptq", "decode_calibration_steps": 5},
            ],
        }
        before = copy.deepcopy(cfg)
        self.assertEqual(LlamaAdapter()._ptq_decode_calibration_steps(cfg), 3)
        self.assertEqual(cfg, before)

    def test_spinquant_rejects_text_smoothquant_case_variant(self):
        """Mixed case must not bypass the existing Qwen3-VL combination guard."""
        cfg = {"pipeline": [{"name": "SmoothQuant", "components": "text"}]}
        before = copy.deepcopy(cfg)
        ctx = RecipeContext(cfg=cfg, adapter=Qwen3VLAdapter())
        with self.assertRaisesRegex(ValueError, "does not support text SmoothQuant"):
            ctx.adapter._ensure_spinquant_compatible(ctx, {"name": "SpinQuant"})
        self.assertEqual(cfg, before)

    def test_gemma_decode_calibration_keeps_multi_stage_policy(self):
        """Gemma's active-stage agreement and explicit-zero policy retain identity."""
        for name in ("ptq", "PTQ", "PtQ"):
            with self.subTest(name=name):
                cfg = {
                    "calibration": {"decode_steps": 1},
                    "pipeline": [{"name": name, "decode_calibration_steps": 3}],
                }
                before = copy.deepcopy(cfg)
                ctx = RecipeContext(cfg=cfg, adapter=Gemma4Adapter())
                self.assertEqual(ctx.adapter._ptq_decode_calibration_steps(ctx), 3)
                self.assertEqual(cfg, before)

        cfg = {
            "calibration": {"decode_steps": 9},
            "pipeline": [
                {"name": "PTQ", "enabled": False, "decode_calibration_steps": 8},
                {"name": "PtQ", "decode_calibration_steps": 0},
                {"name": "ptq", "decode_calibration_steps": 0},
            ],
        }
        before = copy.deepcopy(cfg)
        ctx = RecipeContext(cfg=cfg, adapter=Gemma4Adapter())
        self.assertEqual(ctx.adapter._ptq_decode_calibration_steps(ctx), 0)
        self.assertEqual(cfg, before)

        cfg = {
            "pipeline": [
                {"name": "PTQ", "decode_calibration_steps": 3},
                {"name": "ptq", "decode_calibration_steps": 4},
            ]
        }
        before = copy.deepcopy(cfg)
        ctx = RecipeContext(cfg=cfg, adapter=Gemma4Adapter())
        with self.assertRaisesRegex(ValueError, "must use the same decode"):
            ctx.adapter._ptq_decode_calibration_steps(ctx)
        self.assertEqual(cfg, before)

    def test_spinquant_keeps_allowed_smoothquant_combinations(self):
        """Vision-only and disabled text SmoothQuant keep their existing policy."""
        for stage in (
            {"name": "SmoothQuant", "components": "vision"},
            {"name": "SmoothQuant", "components": "text", "enabled": False},
        ):
            with self.subTest(stage=stage):
                cfg = {"pipeline": [stage]}
                before = copy.deepcopy(cfg)
                ctx = RecipeContext(cfg=cfg, adapter=Qwen3VLAdapter())
                ctx.adapter._ensure_spinquant_compatible(ctx, {"name": "SpinQuant"})
                self.assertEqual(cfg, before)

    def test_spinquant_checks_tied_weight_specs_for_ptq_case_variant(self):
        """The existing embedding/head policy also applies to a dispatched PTQ."""
        cfg = {
            "pipeline": [{"name": "PTQ", "embedding_weight": 4, "lm_head_weight": 8}]
        }
        ctx = RecipeContext(cfg=cfg, adapter=Qwen3VLAdapter())
        with self.assertRaisesRegex(ValueError, "must match"):
            ctx.adapter._ensure_spinquant_compatible(ctx, {"name": "spinquant"})

    def test_inspector_uses_case_variant_without_mutating_config(self):
        """Trace runs real PTQ from the same mixed-case stage the runner accepts."""
        first = {"name": "PtQ", "enabled": False}
        cfg = {"pipeline": [first, {"name": "ptq"}]}
        before = copy.deepcopy(cfg)
        ctx = _context(cfg)
        original = ctx.model
        with contextlib.redirect_stdout(io.StringIO()):
            trace_ptq_parity(ctx, enable_quantization=True)
        self.assertIs(ctx.adapter.selected_stages[0], first)
        self.assertEqual(ctx.adapter.calibration_calls, 1)
        self.assertIs(ctx.model, original)
        self.assertFalse(hasattr(original, "tico_quantizer"))
        self.assertEqual(cfg, before)

    def test_summary_uses_case_variants(self):
        """Diagnostics reflect the algorithms that dispatch will actually run."""
        cfg = {"pipeline": [{"name": "GPTQ"}, {"name": "PTQ", "activation": "int16"}]}
        before = copy.deepcopy(cfg)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            _print_config_summary(cfg)
        self.assertIn("GPTQ enabled          : True", stdout.getvalue())
        self.assertIn("PTQ enabled           : True", stdout.getvalue())
        self.assertIn("Activation            : int16", stdout.getvalue())
        self.assertEqual(cfg, before)

    def test_named_overrides_remain_case_sensitive(self):
        """CLI list-item names are a separate, exact-key addressing contract."""
        cfg = {"pipeline": [{"name": "PTQ", "enabled": False}]}
        before = copy.deepcopy(cfg)
        with self.assertRaises(KeyError):
            set_by_path(cfg, ["pipeline", "ptq", "enabled"], True)
        self.assertIsNone(get_by_path(cfg, "pipeline.ptq.enabled"))
        self.assertEqual(cfg, before)
        set_by_path(cfg, ["pipeline", "PTQ", "enabled"], True)
        self.assertTrue(get_by_path(cfg, "pipeline.PTQ.enabled"))

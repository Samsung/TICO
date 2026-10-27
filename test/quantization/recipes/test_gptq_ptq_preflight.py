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

"""Tests for the static GPTQ->PTQ weight-qparam compatibility preflight."""

from test.quantization.recipes.optional_dependency_stubs import (
    install_optional_dependency_stubs,
)

install_optional_dependency_stubs()

import contextlib
import io
import unittest
from unittest.mock import patch

import tico.quantization.recipes.runner as runner_mod

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.cpu_fixtures import (
    fixture_context,
    gemma4_multimodal_model,
)
from tico.quantization.recipes.preflight import (
    plan_gptq_weight_bits,
    PreflightError,
    run_gptq_ptq_preflight,
)
from tico.quantization.recipes.qparams import find_gptq_quantizers


def _gptq_stage(**overrides):
    stage = {
        "name": "gptq",
        "enabled": True,
        "weight_bits": 4,
        "show_progress": False,
    }
    stage.update(overrides)
    return stage


def _ptq_stage(**overrides):
    stage = {
        "name": "ptq",
        "enabled": True,
        "activation": "int16",
        "linear_weight": "uint4",
        "embedding_weight": "uint8",
        "lm_head_weight": "uint8",
        "decode_calibration_steps": 2,
    }
    stage.update(overrides)
    return stage


def _ctx(family, gptq_stage, ptq_stage):
    ctx = fixture_context(family)
    pipeline = []
    if gptq_stage is not None:
        pipeline.append(gptq_stage)
    if ptq_stage is not None:
        pipeline.append(ptq_stage)
    ctx.cfg["pipeline"] = pipeline
    return ctx


def _gemma4_ctx(gptq_stage, ptq_stage) -> RecipeContext:
    cfg = {
        "model": {"family": "gemma4", "name_or_path": "local-random-config"},
        "runtime": {"device": "cpu", "seed": 42, "show_progress": False},
        "model_args": {
            "vision": {"profile": "e2b_66x36_264", "validate_static_layout": False}
        },
        "pipeline": [stage for stage in (gptq_stage, ptq_stage) if stage],
        "evaluation": {"enabled": False},
        "export": {"enabled": False},
    }
    return RecipeContext(
        cfg=cfg, adapter=Gemma4Adapter(), model=gemma4_multimodal_model()
    )


def _run(ctx) -> str:
    """Run the preflight expecting success; return captured stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_gptq_ptq_preflight(ctx)
    return buf.getvalue()


def _run_error(ctx) -> tuple[str, str]:
    """Run the preflight expecting PreflightError; return (message, stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            run_gptq_ptq_preflight(ctx)
        except PreflightError as exc:
            return str(exc), buf.getvalue()
    raise AssertionError("PreflightError was not raised")


class TestGptqPtqPreflight(unittest.TestCase):
    def test_compatible_llama_passes(self):
        ctx = _ctx("llama", _gptq_stage(), _ptq_stage())
        out = _run(ctx)
        self.assertIn("compatible", out)

    def test_mismatch_reports_every_module(self):
        ctx = _ctx("llama", _gptq_stage(), _ptq_stage(linear_weight="uint8"))
        message, _ = _run_error(ctx)
        # Two decoder layers x 7 linear projections.
        self.assertEqual(message.count("  - 'model.layers."), 14)
        self.assertIn("'model.layers.0.self_attn.q_proj'", message)
        self.assertIn("weight bit width differs", message)
        self.assertIn("weight_bits default", message)
        self.assertIn("weight_bits_overrides", message)

    def test_override_bit_mismatch_only_flags_overridden_modules(self):
        gptq = _gptq_stage(weight_bits_overrides={"self_attn.q_proj": 8})
        ctx = _ctx("llama", gptq, _ptq_stage())
        message, out = _run_error(ctx)
        self.assertEqual(message.count("  - 'model.layers."), 2)
        self.assertIn("self_attn.q_proj", message)
        self.assertNotIn("k_proj", message)
        self.assertIn("weight_bits_overrides['self_attn.q_proj']", message)
        # The override matched modules, so no unused-key warning.
        self.assertNotIn("matched no quantizable module", out)

    def test_qwen_patch_embed_scenario(self):
        """Mirror the Gemma4 incident shape on the Qwen3-VL fixture."""
        ptq = _ptq_stage(vision_patch_embed_weight="uint8")

        # 1) 4-bit GPTQ vs uint8 patch embed: hard error naming the module.
        ctx = _ctx("qwen3_vl", _gptq_stage(), ptq)
        message, _ = _run_error(ctx)
        self.assertIn("'model.visual.patch_embed.proj'", message)
        self.assertIn("weight bit width differs", message)
        self.assertIn("patch_embed.proj", message)

        # 2) Correct override aligns the bits: preflight passes.
        gptq = _gptq_stage(weight_bits_overrides={"patch_embed.proj": 8})
        ctx = _ctx("qwen3_vl", gptq, ptq)
        out = _run(ctx)
        self.assertIn("compatible", out)
        self.assertNotIn("matched no quantizable module", out)

        # 3) A key written for another family warns and still errors (strict
        # overrides disabled so the run reaches the C2 mismatch check).
        gptq = _gptq_stage(
            weight_bits_overrides={"patch_embedder.input_proj": 8},
            strict_overrides=False,
        )
        ctx = _ctx("qwen3_vl", gptq, ptq)
        message, out = _run_error(ctx)
        self.assertIn("'model.visual.patch_embed.proj'", message)
        self.assertIn("'patch_embedder.input_proj' matched no quantizable module", out)

    def test_unused_override_key_errors_by_default(self):
        # strict_overrides defaults to true: a dead key aborts the run.
        gptq = _gptq_stage(weight_bits_overrides={"no.such.module": 8})
        ctx = _ctx("llama", gptq, _ptq_stage())
        message, _ = _run_error(ctx)
        self.assertIn("'no.such.module' matched no quantizable module", message)
        self.assertIn("strict_overrides", message)

    def test_unused_override_key_warns_when_strict_disabled(self):
        gptq = _gptq_stage(
            weight_bits_overrides={"no.such.module": 8}, strict_overrides=False
        )
        ctx = _ctx("llama", gptq, _ptq_stage())
        out = _run(ctx)
        self.assertIn("'no.such.module' matched no quantizable module", out)
        self.assertIn("compatible", out)

    def test_gemma4_patch_embedder_scenario(self):
        """Reproduce the original Gemma4 GPTQ->PTQ mismatch end to end."""
        ptq = _ptq_stage(
            weight="int16",
            vision_patch_embed_weight="uint8",
            norm_weight="int16",
            strict_wrap=True,
        )

        # The failing configuration: an override key that matches no Gemma4
        # module, so the patch embedder silently stays at 4-bit. Strict
        # overrides are disabled so the run reaches the C2 mismatch check.
        gptq = _gptq_stage(
            weight_bits_overrides={"patch_embed.proj": 8}, strict_overrides=False
        )
        ctx = _gemma4_ctx(gptq, ptq)
        message, out = _run_error(ctx)
        self.assertIn("'model.vision_tower.patch_embedder.input_proj'", message)
        self.assertIn("weight bit width differs", message)
        self.assertIn("'patch_embed.proj' matched no quantizable module", out)

        # The suggested fix aligns the bits: preflight passes.
        gptq = _gptq_stage(weight_bits_overrides={"patch_embedder.input_proj": 8})
        ctx = _gemma4_ctx(gptq, ptq)
        run_out = _run(ctx)
        self.assertIn("compatible", run_out)
        self.assertNotIn("matched no quantizable module", run_out)

    def test_gemma4_audio_tower_excluded_from_plan(self):
        """The Gemma4 GPTQ quantizer skips the audio tower; the plan must too."""
        model = gemma4_multimodal_model()
        cfg = Gemma4GPTQConfig(weight_bits=4, show_progress=False)
        unfiltered, _ = plan_gptq_weight_bits(model, cfg)
        self.assertTrue(any("audio" in name for name in unfiltered))

        roots = Gemma4Adapter().gptq_target_roots(model, cfg)
        self.assertIsNotNone(roots)
        filtered, _ = plan_gptq_weight_bits(model, cfg, target_roots=roots)
        self.assertFalse(any("audio" in name for name in filtered))
        self.assertLess(len(filtered), len(unfiltered))

        # End-to-end preflight stays silent about audio modules.
        ctx = _gemma4_ctx(
            _gptq_stage(weight_bits_overrides={"patch_embedder.input_proj": 8}),
            _ptq_stage(
                weight="int16",
                vision_patch_embed_weight="uint8",
                norm_weight="int16",
                strict_wrap=True,
            ),
        )
        out = _run(ctx)
        self.assertIn("compatible", out)
        self.assertNotIn("audio", out)

    def test_gemma4_target_roots_skipped_for_non_family_config(self):
        # The universal GPTQ variant is model-agnostic and also processes
        # the audio tower, so the family component filter must not apply.
        roots = Gemma4Adapter().gptq_target_roots(
            gemma4_multimodal_model(), GPTQConfig(weight_bits=4)
        )
        self.assertIsNone(roots)

    def test_skipped_when_gptq_stage_disabled(self):
        gptq = _gptq_stage(enabled=False)
        ctx = _ctx("llama", gptq, _ptq_stage(linear_weight="uint8"))
        self.assertEqual(_run(ctx), "")

    def test_skipped_when_reuse_disabled(self):
        ctx = _ctx(
            "llama",
            _gptq_stage(),
            _ptq_stage(linear_weight="uint8", reuse_gptq_qparams=False),
        )
        self.assertEqual(_run(ctx), "")

    def test_skipped_by_preflight_flag(self):
        ctx = _ctx(
            "llama",
            _gptq_stage(),
            _ptq_stage(linear_weight="uint8", preflight=False),
        )
        self.assertEqual(_run(ctx), "")

    def test_symmetric_gptq_passes_when_bits_match(self):
        # Symmetric GPTQ grids use the full unsigned maxq range with an
        # integral midpoint zero-point, which the PTQ handoff accepts.
        ctx = _ctx("llama", _gptq_stage(symmetric=True), _ptq_stage())
        self.assertIn("compatible", _run(ctx))

    def test_grouped_gptq_is_rejected(self):
        ctx = _ctx("llama", _gptq_stage(groupsize=128), _ptq_stage())
        message, _ = _run_error(ctx)
        self.assertIn("groupsize=128", message)

    def test_build_meta_model_uses_meta_device(self):
        ctx = fixture_context("llama")
        twin = ctx.adapter.build_meta_model(ctx)
        param = next(twin.parameters())
        self.assertTrue(param.is_meta)

    def test_plan_parity_with_real_gptq_llama(self):
        """The static plan must name exactly the modules GPTQ quantizes."""
        ctx = fixture_context("llama")
        model = ctx.require_model()
        gptq_config = GPTQConfig(weight_bits=4, show_progress=False)
        plan, unused = plan_gptq_weight_bits(model, gptq_config)
        self.assertEqual(unused, [])

        q_model = prepare(model, gptq_config)
        for batch in ctx.calibration_inputs:
            q_model(batch)
        convert(q_model)

        _, quantizers = find_gptq_quantizers(q_model)
        if quantizers is None:
            self.fail("find_gptq_quantizers returned no quantizers")
        self.assertEqual(set(plan), set(quantizers))

    def test_runner_invokes_preflight_after_model_load(self):
        fixture = fixture_context("llama")
        cfg = {
            "model": {"family": "llama", "name_or_path": "local-random-config"},
            "runtime": {"device": "cpu", "seed": 42, "show_progress": False},
            "pipeline": [],
            "evaluation": {"enabled": False},
            "export": {"enabled": False},
        }
        seen = {}

        def fake_load_model(self_adapter, ctx):
            ctx.model = fixture.model
            return ctx

        def spy(ctx):
            seen["called"] = True
            seen["model_loaded"] = ctx.model is not None

        with patch.object(LlamaAdapter, "load_model", fake_load_model):
            with patch.object(runner_mod, "run_gptq_ptq_preflight", spy):
                runner_mod.QuantizationRunner().run(cfg)
        self.assertEqual(seen, {"called": True, "model_loaded": True})


if __name__ == "__main__":
    unittest.main()

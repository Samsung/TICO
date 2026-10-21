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

"""Regression tests for independent eager quality and deployment contexts."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.evaluation.eager import (
    prepare_eager_evaluation,
    validate_eager_evaluation_config,
)
from tico.quantization.recipes.sequence_length import (
    model_context_length,
    resolve_evaluation_max_seq_len,
    resolve_export_max_seq_len,
    resolve_ppl_max_seq_len,
)
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class _Model(torch.nn.Module):
    """Tiny model exposing native text and generation configuration."""

    def __init__(self, capacity=8192, multimodal=False):
        super().__init__()
        text_config = SimpleNamespace(
            max_position_embeddings=capacity,
            use_cache=False,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
        )
        self.config = (
            SimpleNamespace(text_config=text_config, use_cache=False)
            if multimodal
            else text_config
        )
        self.generation_config = SimpleNamespace(use_cache=False)
        self.weight = torch.nn.Parameter(torch.ones(1))


class _QuantLeaf(QuantModuleBase):
    """Quantized leaf used to check lifecycle validation without a real model."""

    def _all_observers(self):
        return []


def _context(adapter=None, *, evaluation=None):
    adapter = adapter or LlamaAdapter()
    return RecipeContext(
        cfg={
            "model": {"family": adapter.family, "name_or_path": "local-test-model"},
            "runtime": {"device": "cpu", "seed": 42},
            "calibration": {"seq_len": 4},
            "export": {"max_seq_len": 16},
            "evaluation": {"enabled": True, **(evaluation or {})},
        },
        adapter=adapter,
        model=_Model(multimodal=adapter.family != "llama"),
        processor=SimpleNamespace(tokenizer=object()),
    )


class TestIndependentSequenceLengths(unittest.TestCase):
    def test_native_eager_default_ignores_calibration_and_export(self):
        for adapter in (LlamaAdapter(), Qwen3VLAdapter(), Gemma4Adapter()):
            with self.subTest(family=adapter.family):
                ctx = _context(adapter)
                ctx.model.static_max_seq = 8
                self.assertEqual(resolve_evaluation_max_seq_len(ctx), 8192)
                self.assertEqual(resolve_export_max_seq_len(ctx), 16)
                ctx.cfg["calibration"]["seq_len"] = 128
                ctx.cfg["export"]["max_seq_len"] = 64
                self.assertEqual(resolve_evaluation_max_seq_len(ctx), 8192)
                ctx.cfg["evaluation"]["max_seq_len"] = 4096
                self.assertEqual(resolve_evaluation_max_seq_len(ctx), 4096)
                self.assertEqual(resolve_export_max_seq_len(ctx), 64)

    def test_export_default_is_independent_and_bounded_by_native_capacity(self):
        ctx = _context(evaluation={"max_seq_len": 4096})
        ctx.cfg["export"] = {}
        self.assertEqual(resolve_export_max_seq_len(ctx), 2048)
        ctx.cfg["calibration"]["seq_len"] = 32
        self.assertEqual(resolve_export_max_seq_len(ctx), 2048)
        ctx.model.config.max_position_embeddings = 8
        self.assertEqual(resolve_export_max_seq_len(ctx), 8)

    def test_invalid_lengths_do_not_silently_use_defaults(self):
        for value in (0, -1, True, 2.5, "1024"):
            with self.subTest(value=value):
                ctx = _context(evaluation={"max_seq_len": value})
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resolve_evaluation_max_seq_len(ctx)
                ctx.cfg["export"]["max_seq_len"] = value
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    resolve_export_max_seq_len(ctx)

    def test_legacy_clamped_checkpoint_requires_explicit_common_cap(self):
        ctx = _context()
        ctx.artifacts["pretrained_context_length"] = 8192
        ctx.model.config.max_position_embeddings = 2048
        with self.assertRaisesRegex(ValueError, "regenerate a checkpoint"):
            resolve_evaluation_max_seq_len(ctx)
        ctx.cfg["evaluation"]["max_seq_len"] = 2048
        self.assertEqual(resolve_evaluation_max_seq_len(ctx), 2048)
        self.assertEqual(ctx.model.config.max_position_embeddings, 2048)

    def test_native_capacity_is_not_inferred_from_export_templates(self):
        model = SimpleNamespace(static_max_seq=2048)
        self.assertIsNone(model_context_length(model))
        ctx = _context()
        ctx.model = model
        with self.assertRaisesRegex(ValueError, "Set evaluation.max_seq_len"):
            resolve_evaluation_max_seq_len(ctx)

    def test_ppl_window_can_be_shorter_than_generation_context(self):
        self.assertEqual(
            resolve_ppl_max_seq_len({"max_seq_len": 2048}, evaluation_max_seq_len=8192),
            2048,
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            resolve_ppl_max_seq_len({"max_seq_len": 8192}, evaluation_max_seq_len=2048)


class TestEagerEvaluationProtocol(unittest.TestCase):
    def test_protocol_preserves_native_config_and_uses_common_cache_policy(self):
        for adapter in (LlamaAdapter(), Qwen3VLAdapter(), Gemma4Adapter()):
            for use_cache in (True, False):
                with self.subTest(family=adapter.family, use_cache=use_cache):
                    ctx = _context(adapter, evaluation={"use_cache": use_cache})
                    text_config = getattr(
                        ctx.model.config, "text_config", ctx.model.config
                    )
                    rope_scaling = copy.deepcopy(text_config.rope_scaling)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(prepare_eager_evaluation(ctx), 8192)
                    self.assertFalse(ctx.model.training)
                    self.assertEqual(text_config.max_position_embeddings, 8192)
                    self.assertEqual(text_config.rope_scaling, rope_scaling)
                    self.assertEqual(text_config.use_cache, use_cache)
                    self.assertEqual(ctx.model.generation_config.use_cache, use_cache)
                    protocol = ctx.artifacts["evaluation_protocol"]
                    self.assertEqual(protocol["purpose"], "quantization_quality")
                    self.assertEqual(protocol["context"]["calibration_seq_len"], 4)
                    self.assertEqual(protocol["context"]["export_max_seq_len"], 16)

    def test_metadata_sidecar_records_checkpoint_context_and_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = _context(evaluation={"output_dir": directory, "max_seq_len": 4096})
            ctx.cfg["model"]["hf_token"] = "do-not-persist"
            ctx.artifacts["evaluation_checkpoint"] = "quantized_model.pt"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_eager_evaluation(ctx)
            text = (Path(directory) / "evaluation_protocol.json").read_text()
            protocol = json.loads(text)
            self.assertNotIn("backend", protocol)
            self.assertNotIn("backend", protocol["evaluation"])
            self.assertEqual(protocol["context"]["resolved"], 4096)
            self.assertEqual(protocol["checkpoint"], "quantized_model.pt")
            self.assertEqual(protocol["versions"]["torch"], torch.__version__)
            self.assertNotIn("do-not-persist", text)

    def test_wrapped_checkpoint_uses_the_same_cache_policy_as_fp(self):
        """PTQ checkpoints expose their HF configuration through wrapped."""
        for use_cache in (True, False):
            with self.subTest(use_cache=use_cache):
                ctx = _context(Qwen3VLAdapter(), evaluation={"use_cache": use_cache})
                execution_model = ctx.model
                execution_model.config._commit_hash = "source-revision"
                checkpoint = torch.nn.Module()
                checkpoint.wrapped = execution_model
                ctx.model = checkpoint
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(prepare_eager_evaluation(ctx), 8192)
                self.assertEqual(execution_model.config.use_cache, use_cache)
                self.assertEqual(
                    execution_model.config.text_config.use_cache, use_cache
                )
                self.assertEqual(execution_model.generation_config.use_cache, use_cache)
                self.assertEqual(
                    ctx.artifacts["evaluation_protocol"]["model"]["resolved_revision"],
                    "source-revision",
                )

    def test_calibration_state_is_rejected_without_changing_modes(self):
        ctx = _context()
        ctx.model.quant_leaf = _QuantLeaf()
        ctx.model.quant_leaf.enable_calibration()
        with self.assertRaisesRegex(ValueError, "calibration mode"):
            prepare_eager_evaluation(ctx)
        self.assertIs(ctx.model.quant_leaf._mode, Mode.CALIB)

    def test_converted_quant_state_is_preserved(self):
        ctx = _context()
        ctx.model.quant_leaf = _QuantLeaf()
        ctx.model.quant_leaf.freeze_qparams()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_eager_evaluation(ctx)
        self.assertIs(ctx.model.quant_leaf._mode, Mode.QUANT)
        self.assertEqual(
            ctx.artifacts["evaluation_protocol"]["quantization_modes"], ["quant"]
        )

    def test_forced_export_is_rejected(self):
        ctx = _context()
        ctx.model.force_export = True
        with self.assertRaisesRegex(ValueError, "force_export"):
            prepare_eager_evaluation(ctx)

    def test_default_protocol_needs_no_execution_selector(self):
        """All model families accept the shared eager policy without a selector."""
        validate_eager_evaluation_config({})
        for adapter in (LlamaAdapter(), Qwen3VLAdapter(), Gemma4Adapter()):
            with self.subTest(family=adapter.family):
                adapter.validate_evaluation_config({"evaluation": {"enabled": True}})

    def test_adapter_preflight_rejects_invalid_protocol(self):
        cases = (
            ("max_seq_len", 0, ValueError),
            ("use_cache", "true", TypeError),
            ("output_dir", " ", ValueError),
        )
        for adapter in (LlamaAdapter(), Qwen3VLAdapter(), Gemma4Adapter()):
            for field, value, error_type in cases:
                with self.subTest(family=adapter.family, field=field):
                    with self.assertRaisesRegex(error_type, f"evaluation\\.{field}"):
                        adapter.validate_evaluation_config(
                            {"evaluation": {"enabled": True, field: value}}
                        )

    def test_standard_presets_use_eager_evaluation_without_a_selector(self):
        """Every standard preset retains its protocol without a backend field."""
        import yaml

        root = Path(__file__).resolve().parents[3]
        presets = (
            "llama_eval_suite",
            "qwen3_vl_eval_suite",
            "gemma4_eval_suite",
            "qwen3_vl_eval_suite_mx_override_polices",
        )
        for preset in presets:
            with self.subTest(preset=preset):
                path = root / "tico/quantization/examples/configs" / (preset + ".yaml")
                config = yaml.safe_load(path.read_text())["evaluation"]
                self.assertNotIn("backend", config)
                self.assertIsNone(config["max_seq_len"])
                self.assertTrue(config["use_cache"])
                validate_eager_evaluation_config(config)


class TestNativeModelLoading(unittest.TestCase):
    def test_llama_loading_preserves_positional_configuration(self):
        ctx = _context()
        model = ctx.model
        tokenizer = SimpleNamespace(pad_token_id=0)
        with patch(
            "tico.quantization.recipes.adapters.llama.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ), patch(
            "tico.quantization.recipes.adapters.llama.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ):
            ctx.adapter.load_model(ctx)
        self.assertEqual(model.config.max_position_embeddings, 8192)
        self.assertEqual(model.config.rope_scaling["factor"], 2.0)
        self.assertEqual(ctx.artifacts["pretrained_context_length"], 8192)

    def test_vlm_loading_preserves_positional_configuration(self):
        for adapter in (Qwen3VLAdapter(), Gemma4Adapter()):
            with self.subTest(family=adapter.family):
                ctx = _context(adapter)
                model = ctx.model
                with patch(
                    "transformers.AutoProcessor.from_pretrained",
                    return_value=ctx.processor,
                ), patch(
                    "transformers.AutoModelForImageTextToText.from_pretrained",
                    return_value=model,
                ), patch(
                    "tico.quantization.recipes.adapters.gemma4.assert_gemma4_e2b_no_moe"
                ), patch(
                    "tico.quantization.recipes.adapters.gemma4.fuse_gemma4_ple_embedding_scale",
                    return_value=0,
                ):
                    adapter.load_model(ctx)
                self.assertEqual(model.config.text_config.max_position_embeddings, 8192)
                self.assertEqual(model.config.text_config.rope_scaling["factor"], 2.0)
                self.assertEqual(ctx.artifacts["pretrained_context_length"], 8192)

    def test_adapters_forward_native_context_to_text_and_video_evaluators(self):
        for family, adapter in (
            ("qwen3_vl", Qwen3VLAdapter()),
            ("gemma4", Gemma4Adapter()),
        ):
            with self.subTest(family=family):
                ctx = _context(
                    adapter,
                    evaluation={
                        "selected_tasks": ["mmlu", "videomme"],
                        "use_cache": False,
                    },
                )
                module_path = f"tico.quantization.recipes.adapters.{family}"
                with patch(f"{module_path}.evaluate_and_print_mmlu") as mmlu, patch(
                    f"{module_path}.evaluate_and_print_video_mme"
                ) as video, contextlib.redirect_stdout(io.StringIO()):
                    adapter.evaluate(ctx)
                self.assertEqual(mmlu.call_args.kwargs["max_seq_len"], 8192)
                self.assertEqual(video.call_args.kwargs["max_seq_len"], 8192)
                self.assertIs(video.call_args.kwargs["kv_cache"], False)


if __name__ == "__main__":
    unittest.main()

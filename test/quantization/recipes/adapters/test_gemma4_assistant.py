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

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tico.quantization import convert, prepare
from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.adapters.gemma4_assistant import (
    Gemma4AssistantAdapter,
    TARGET_MODEL_ENV_VAR,
)
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.dtypes import DType


_SKIP_MSG = "required transformers Gemma4 assistant modules are not installed"
_CONFIG_DIR = (
    Path(__file__).resolve().parents[4]
    / "tico"
    / "quantization"
    / "examples"
    / "configs"
)


def _has_gemma4_assistant() -> bool:
    try:
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (  # noqa: F401
            Gemma4AssistantForCausalLM,
        )
    except Exception:
        return False
    return True


def _make_sample(model: torch.nn.Module, kv_len: int = 10) -> dict:
    text_cfg = model.config.get_text_config()
    kv_heads = int(text_cfg.num_key_value_heads)
    return {
        "inputs_embeds": torch.randn(1, 1, 2 * int(model.config.backbone_hidden_size)),
        "position_ids": torch.tensor([[kv_len - 1]]),
        "attention_mask": torch.ones(1, kv_len, dtype=torch.long),
        "shared_kv_states": {
            "full_attention": (
                torch.randn(1, kv_heads, kv_len, int(text_cfg.global_head_dim)),
                torch.randn(1, kv_heads, kv_len, int(text_cfg.global_head_dim)),
            ),
            "sliding_attention": (
                torch.randn(1, kv_heads, kv_len, int(text_cfg.head_dim)),
                torch.randn(1, kv_heads, kv_len, int(text_cfg.head_dim)),
            ),
        },
        "use_cache": False,
    }


class TestGemma4AssistantAdapterRegistry(unittest.TestCase):
    def test_adapter_registry_returns_gemma4_assistant(self):
        self.assertIsInstance(get_adapter("gemma4_assistant"), Gemma4AssistantAdapter)
        self.assertIsInstance(get_adapter("gemma4-assistant"), Gemma4AssistantAdapter)

    def test_target_path_env_overrides_config(self):
        cfg = {"target_model": {"name_or_path": "from-config"}}
        with patch.dict(os.environ, {TARGET_MODEL_ENV_VAR: "from-env"}):
            self.assertEqual(
                Gemma4AssistantAdapter._resolve_target_path(cfg), "from-env"
            )
        env_without_target = {
            key: value
            for key, value in os.environ.items()
            if key != TARGET_MODEL_ENV_VAR
        }
        with patch.dict(os.environ, env_without_target, clear=True):
            self.assertEqual(
                Gemma4AssistantAdapter._resolve_target_path(cfg), "from-config"
            )
            with self.assertRaisesRegex(ValueError, "target_model"):
                Gemma4AssistantAdapter._resolve_target_path({})


class TestGemma4AssistantExampleConfigs(unittest.TestCase):
    def test_quantize_config_parses_with_expected_contract(self):
        cfg = load_recipe_config(_CONFIG_DIR / "gemma4_e2b_assistant_quantize.yaml")
        self.assertEqual(cfg["model"]["family"], "gemma4_assistant")
        self.assertEqual(
            cfg["model"]["name_or_path"], "google/gemma-4-E2B-it-assistant"
        )
        self.assertEqual(cfg["target_model"]["name_or_path"], "google/gemma-4-E2B-it")
        assistant_args = cfg["model_args"]["assistant"]
        self.assertEqual(assistant_args["batch_size"], 1)
        self.assertEqual(assistant_args["query_length"], 1)
        # The bidirectional sliding overlay is inclusive: the capacity must
        # exceed sliding_window (512) by at least one.
        self.assertGreater(assistant_args["sliding_kv_length"], 512)

        ptq_stage = next(stage for stage in cfg["pipeline"] if stage["name"] == "ptq")
        self.assertEqual(ptq_stage["activation"], "int16")
        self.assertEqual(ptq_stage["linear_weight"], "uint8")
        self.assertTrue(ptq_stage["strict_wrap"])
        self.assertEqual(
            set(cfg["export"]["artifacts"]),
            {
                "ptq_checkpoint",
                "assistant_core_circle",
                "assistant_sparse_head",
                "assistant_manifest",
            },
        )

    def test_export_config_parses_with_expected_contract(self):
        cfg = load_recipe_config(_CONFIG_DIR / "gemma4_e2b_assistant_export.yaml")
        self.assertEqual(cfg["model"]["family"], "gemma4_assistant")
        self.assertEqual(cfg["pipeline"], [])
        self.assertIn("assistant_core_circle", cfg["export"]["artifacts"])


@unittest.skipUnless(_has_gemma4_assistant(), _SKIP_MSG)
class TestGemma4AssistantAdapterWithTinyModel(unittest.TestCase):
    def setUp(self):
        from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
            make_tiny_gemma4_assistant_model,
        )

        torch.manual_seed(2032)
        self.adapter = Gemma4AssistantAdapter()
        self.fp_model = make_tiny_gemma4_assistant_model()

    def _converted_ctx(self, cfg: dict) -> RecipeContext:
        ctx = RecipeContext(cfg=cfg, adapter=self.adapter, model=self.fp_model)
        stage_cfg = next(
            stage for stage in cfg.get("pipeline", []) if stage["name"] == "ptq"
        )
        qcfg = self.adapter.build_ptq_config(ctx, stage_cfg)
        prepared = prepare(ctx.model, qcfg)
        with torch.no_grad():
            prepared(**_make_sample(self.fp_model))
        ctx.model = convert(prepared)
        return ctx

    def test_build_ptq_config_maps_stage_fields(self):
        ctx = RecipeContext(cfg={}, adapter=self.adapter, model=self.fp_model)
        qcfg = self.adapter.build_ptq_config(
            ctx,
            {
                "activation": "int16",
                "linear_weight": "uint4",
                "projection_weight": "uint8",
                "centroid_weight": "uint8",
                "lm_head_weight": "uint8",
                "norm_weight": "int16",
                "strict_wrap": True,
            },
        )
        self.assertEqual(qcfg.activation.dtype, DType.int(16))
        self.assertEqual(
            qcfg.overrides["model"]["layers"]["0"]["self_attn"]["q_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            qcfg.overrides["pre_projection"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            sorted(qcfg.overrides["model"]["layers"]),  # type: ignore[arg-type]
            ["0", "1"],
        )

    def test_load_target_model_compares_backbone_hidden_size(self):
        """The target compatibility check must use the assistant's
        ``backbone_hidden_size`` (the target width it consumes), not the
        assistant's own, much smaller, text-config ``hidden_size``."""
        text_cfg = self.fp_model.config.get_text_config()
        backbone_hidden_size = int(self.fp_model.config.backbone_hidden_size)
        draft_hidden_size = int(text_cfg.hidden_size)
        # The tiny fixture keeps the two widths distinct so the wrong
        # comparison is observable.
        self.assertNotEqual(backbone_hidden_size, draft_hidden_size)

        class _FakeTarget(torch.nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.config = types.SimpleNamespace(
                    hidden_size=hidden_size, vocab_size=int(text_cfg.vocab_size)
                )

        def _ctx() -> RecipeContext:
            return RecipeContext(
                cfg={"target_model": {"name_or_path": "fake-target"}},
                adapter=self.adapter,
                model=self.fp_model,
            )

        module = "tico.quantization.recipes.adapters.gemma4_assistant._load_causal_lm"
        matching = _FakeTarget(backbone_hidden_size)
        with patch(module, return_value=matching):
            ctx = _ctx()
            self.assertIs(self.adapter._load_target_model(ctx), matching)
            self.assertIs(ctx.artifacts["gemma4_assistant_target_model"], matching)

        with patch(module, return_value=_FakeTarget(draft_hidden_size)):
            with self.assertRaisesRegex(ValueError, "backbone_hidden_size"):
                self.adapter._load_target_model(_ctx())

    def test_forward_calibration_streams_real_assisted_generation(self):
        """Calibration must route prompts through target.generate with the
        prepared assistant plugged in as the assistant model."""
        qcfg = self.adapter.build_ptq_config(
            RecipeContext(cfg={}, adapter=self.adapter, model=self.fp_model),
            {"activation": "int16", "linear_weight": "uint8"},
        )
        prepared = prepare(self.fp_model, qcfg)
        sample = _make_sample(prepared.wrapped)

        generate_calls = []

        class _FakeTarget:
            def generate(self, *, input_ids, attention_mask, assistant_model, **kw):
                generate_calls.append(kw)
                # Mimic the MTP candidate generator: call the assistant with
                # the exact draft kwargs twice per prompt.
                for _ in range(2):
                    assistant_model(**sample)
                return input_ids

        ctx = RecipeContext(
            cfg={
                "calibration": {"max_new_tokens": 4, "num_assistant_tokens": 2},
                "runtime": {"show_progress": False},
            },
            adapter=self.adapter,
            model=prepared,
        )
        ctx.tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 1})()

        with patch.object(
            Gemma4AssistantAdapter, "_load_target_model", return_value=_FakeTarget()
        ):
            self.adapter.forward_calibration(
                ctx,
                prepared,
                [torch.ones(1, 4, dtype=torch.long)],
                desc="test",
            )

        self.assertEqual(len(generate_calls), 1)
        self.assertEqual(generate_calls[0]["max_new_tokens"], 4)
        converted = convert(prepared)
        missing = [
            name
            for name, obs in converted.wrapped.named_observers()
            if hasattr(obs, "has_qparams") and not obs.has_qparams
        ]
        self.assertEqual(missing, [])

    def test_export_dispatches_sparse_head_and_manifest(self):
        cfg: dict = {
            "model": {
                "family": "gemma4_assistant",
                "name_or_path": "tiny-synthetic-assistant",
            },
            "model_args": {
                "assistant": {
                    "batch_size": 1,
                    "query_length": 1,
                    "full_kv_length": 16,
                    "sliding_kv_length": 8,
                }
            },
            "pipeline": [
                {
                    "name": "ptq",
                    "enabled": True,
                    "activation": "int16",
                    "linear_weight": "uint8",
                }
            ],
            "export": {
                "enabled": True,
                "artifacts": [
                    "ptq_checkpoint",
                    "assistant_sparse_head",
                    "assistant_manifest",
                ],
            },
        }
        ctx = self._converted_ctx(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg["export"]["output_dir"] = tmpdir
            self.adapter.export(ctx)

            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "quantized_model.pt").exists())
            artifact_path = output_dir / "gemma4_assistant_sparse_head.pt"
            manifest_path = output_dir / "gemma4_assistant_manifest.json"
            self.assertTrue(artifact_path.exists())
            self.assertTrue(manifest_path.exists())

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["static_shape"]["full_kv_length"], 16)
            self.assertEqual(manifest["static_shape"]["sliding_kv_length"], 8)
            self.assertEqual(
                [entry["name"] for entry in manifest["outputs"]],
                ["projected_state", "assistant_hidden", "centroid_logits"],
            )
            self.assertEqual(manifest["sparse_head"]["execution_location"], "host")
            self.assertTrue(manifest["sparse_head"]["token_ordering_sha256"])
            self.assertEqual(manifest["assistant_config"]["num_hidden_layers"], 2)

            artifact = torch.load(artifact_path, weights_only=False)
            self.assertEqual(artifact["lm_head_weight_dtype"], "uint8")
            self.assertEqual(artifact["lm_head_weight_channel_axis"], 0)
            self.assertTrue(artifact["tied_to_embedding"])

            # The dequantized artifact weight must equal the fake-quantized
            # tied LM-head weight used by the quantized eager path.
            from tico.quantization.wrapq.wrappers.gemma4_assistant.sparse_head import (
                Gemma4AssistantSparseHead,
            )

            head = Gemma4AssistantSparseHead.from_artifact(artifact)
            expected = ctx.model.wrapped._lm_head_weight()
            torch.testing.assert_close(
                head.lm_head_weight, expected, atol=1e-6, rtol=1e-6
            )

    def test_export_core_circle_requires_converted_model(self):
        """Floating-point models must be rejected by artifact export."""
        cfg = {
            "model": {"family": "gemma4_assistant", "name_or_path": "tiny"},
            "model_args": {"assistant": {"full_kv_length": 16, "sliding_kv_length": 8}},
            "export": {
                "enabled": True,
                "artifacts": ["assistant_manifest"],
            },
        }
        ctx = RecipeContext(cfg=cfg, adapter=self.adapter, model=self.fp_model)
        with self.assertRaises((TypeError, RuntimeError)):
            self.adapter.export(ctx)


if __name__ == "__main__":
    unittest.main()

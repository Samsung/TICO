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

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.adapters.llama as llama_mod
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.context import RecipeContext


def _fake_llama_context(cfg):
    """Build a fake LLaMA context with just enough structure for adapter tests."""
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[object(), object()]),
        config=SimpleNamespace(max_position_embeddings=16),
    )
    return RecipeContext(cfg=cfg, adapter=LlamaAdapter(), model=model)


class TestLlamaAdapter(unittest.TestCase):
    def test_build_ptq_config_forwards_profile_and_weight_options(self):
        """LlamaAdapter should forward recipe PTQ options to the config builder."""
        captured = {}

        def fake_build_llm_ptq_config(**kwargs):
            captured.update(kwargs)
            return {"ptq": "config"}

        ctx = _fake_llama_context({"model_args": {"profile": "reference_eval"}})

        with patch.object(llama_mod, "build_llm_ptq_config", fake_build_llm_ptq_config):
            config = LlamaAdapter().build_ptq_config(
                ctx,
                {
                    "activation_dtype": "int16",
                    "default_qscheme": "per_tensor_symm",
                    "linear_weight_bits": 4,
                    "embedding_weight_bits": 8,
                    "lm_head_weight_bits": 8,
                    "norm_weight_dtype": "int16",
                    "strict_wrap": False,
                },
            )

        self.assertEqual(config, {"ptq": "config"})
        self.assertEqual(captured["model_type"], "llama")
        self.assertEqual(captured["num_hidden_layers"], 2)
        self.assertEqual(captured["linear_weight_bits"], 4)
        self.assertEqual(captured["profile"], "reference_eval")
        self.assertFalse(captured["strict_wrap"])

    def test_build_calibration_inputs_rejects_non_positive_effective_sequence_length(
        self,
    ):
        """LlamaAdapter should validate seq_len after decode calibration steps are subtracted."""
        cfg = {
            "calibration": {"seq_len": 4, "decode_steps": 4},
            "runtime": {"seed": 1},
            "model": {},
        }
        ctx = _fake_llama_context(cfg)
        ctx.tokenizer = object()

        with self.assertRaises(ValueError):
            LlamaAdapter().build_calibration_inputs(ctx)

    def test_export_rejects_circle_export_for_reference_eval_profile(self):
        """Circle export should be blocked for reference_eval LLaMA profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                "pipeline": [{"name": "ptq", "profile": "reference_eval"}],
                "export": {
                    "enabled": True,
                    "output_dir": tmpdir,
                    "artifacts": ["circle_full"],
                },
            }
            ctx = _fake_llama_context(cfg)
            ctx.calibration_inputs = [object()]

            with self.assertRaisesRegex(ValueError, "profile='npu_export'"):
                LlamaAdapter().export(ctx)

    def test_export_checkpoint_only_uses_checkpoint_writer(self):
        """Checkpoint-only export should not require Circle export prerequisites."""
        calls = {}

        def fake_save_checkpoint(model, output_dir):
            calls["checkpoint"] = (model, output_dir)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            llama_mod, "save_checkpoint", fake_save_checkpoint
        ):
            cfg = {
                "export": {
                    "enabled": True,
                    "output_dir": tmpdir,
                    "artifacts": ["ptq_checkpoint"],
                }
            }
            ctx = _fake_llama_context(cfg)
            LlamaAdapter().export(ctx)

        self.assertIs(calls["checkpoint"][0], ctx.model)

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

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tico.quantization.examples.export as export_example
import tico.quantization.recipes.export.llama as llama_export

import torch

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

skip_msg = "required transformers not installed — skipping floating-point LLaMA test"


class FakeLayerWrapper(torch.nn.Module):
    """Fake wrapped decoder layer used for per-layer export tests."""

    def __init__(self, max_seq=4):
        super().__init__()
        self.causal_mask_template = torch.zeros(1, 1, max_seq, max_seq)

    def _slice_rope(self, start, seq_len, device, dtype):
        """Return fake RoPE tensors with the requested sequence length."""
        return (
            torch.ones(1, seq_len, 4, device=device, dtype=dtype),
            torch.zeros(1, seq_len, 4, device=device, dtype=dtype),
        )

    def as_export_module(self, mode, return_kv=False):
        """Return a trivial module representing the export adapter."""
        return torch.nn.Identity()


class FakeWrappedModel(torch.nn.Module):
    """Fake top-level PTQ-wrapped model."""

    def __init__(self, qmodel):
        super().__init__()
        self.wrapped = qmodel


def _make_fake_wrapped_model(max_seq: int = 4) -> FakeWrappedModel:
    """Build a minimal wrapped model for per-layer export tests."""
    layer = SimpleNamespace(wrapped=FakeLayerWrapper(max_seq=max_seq))
    qmodel = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
        model=SimpleNamespace(wrapped=SimpleNamespace(layers=[layer])),
    )
    return FakeWrappedModel(qmodel)


class TestLlamaExport(unittest.TestCase):
    def test_export_llama_per_layer_exports_embedding_layers_and_lm_head(self):
        """Per-layer LLaMA export should preserve quantized artifact names."""
        calls = []
        wrapped_model = _make_fake_wrapped_model()

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            calls.append(save_path.name)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            llama_export,
            "export_token_embedding",
            lambda qmodel, max_seq_len, output_dir, **kwargs: calls.append(
                "token_embedding.q.circle"
            ),
        ), patch.object(
            llama_export,
            "export_lm_head",
            lambda qmodel, output_dir, **kwargs: calls.append("lm_head.q.circle"),
        ), patch.object(
            llama_export, "_convert_and_save", fake_convert_and_save
        ):
            llama_export.export_llama_per_layer(
                q_model=wrapped_model,
                max_seq_len=4,
                output_dir=tmpdir,
                prefill_decode=True,
            )

        self.assertEqual(
            calls,
            [
                "token_embedding.q.circle",
                "decoder_layer_prefill_0.q.circle",
                "decoder_layer_decode_0.q.circle",
                "lm_head.q.circle",
            ],
        )

    def test_export_llama_per_layer_uses_float_artifact_names(self):
        """Floating-point per-layer export should use an explicit f32 tag."""
        calls = []
        wrapped_model = _make_fake_wrapped_model()

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            calls.append(save_path.name)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            llama_export,
            "_prepare_llama_export_model",
            return_value=(wrapped_model, "f32", False),
        ), patch.object(
            llama_export,
            "export_token_embedding",
            lambda qmodel, max_seq_len, output_dir, **kwargs: calls.append(
                "token_embedding.f32.circle"
            ),
        ), patch.object(
            llama_export,
            "export_lm_head",
            lambda qmodel, output_dir, **kwargs: calls.append("lm_head.f32.circle"),
        ), patch.object(
            llama_export, "_convert_and_save", fake_convert_and_save
        ):
            llama_export.export_llama_per_layer(
                q_model=torch.nn.Identity(),
                max_seq_len=4,
                output_dir=tmpdir,
                prefill_decode=True,
            )

        self.assertEqual(
            calls,
            [
                "token_embedding.f32.circle",
                "decoder_layer_prefill_0.f32.circle",
                "decoder_layer_decode_0.f32.circle",
                "lm_head.f32.circle",
            ],
        )

    def test_float_export_rejects_non_float32_model(self):
        """The first float export implementation should reject other dtypes."""
        model = torch.nn.Linear(2, 2).to(dtype=torch.float16)
        with self.assertRaisesRegex(TypeError, "supports float32 only"):
            llama_export._float_artifact_tag(model)

    @unittest.skipUnless(has_transformers_for("llama"), skip_msg)
    def test_float_model_is_wrapped_without_enabling_quantization(self):
        """Structural float wrapping should keep quant modules in NO_QUANT mode."""
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            num_hidden_layers=1,
            max_position_embeddings=4,
            vocab_size=32,
            use_cache=False,
            return_dict=True,
        )
        model = LlamaForCausalLM(config).eval()

        (
            export_model,
            artifact_tag,
            needs_fake_quant_kernels,
        ) = llama_export._prepare_llama_export_model(model)

        self.assertEqual(artifact_tag, "f32")
        self.assertFalse(needs_fake_quant_kernels)
        quant_modules = [
            module
            for module in export_model.modules()
            if isinstance(module, QuantModuleBase)
        ]
        self.assertTrue(quant_modules)
        self.assertTrue(all(module._mode is Mode.NO_QUANT for module in quant_modules))


class TestResolveExportSource(unittest.TestCase):
    def test_checkpoint_is_inferred_for_backward_compatibility(self):
        """Passing only a checkpoint should retain the existing CLI behavior."""
        self.assertEqual(
            export_example._resolve_source(None, "model.pt"),
            "checkpoint",
        )

    def test_model_source_does_not_require_checkpoint(self):
        """Floating-point export should accept an explicit model source."""
        self.assertEqual(export_example._resolve_source("model", None), "model")

    def test_missing_source_is_rejected(self):
        """The CLI should reject an ambiguous export request."""
        with self.assertRaisesRegex(ValueError, "Export source is not specified"):
            export_example._resolve_source(None, None)

    def test_model_source_rejects_checkpoint(self):
        """A model source should not silently ignore a checkpoint argument."""
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            export_example._resolve_source("model", "model.pt")

    def test_checkpoint_source_requires_checkpoint(self):
        """An explicit checkpoint source should require a checkpoint path."""
        with self.assertRaisesRegex(ValueError, "requires --checkpoint"):
            export_example._resolve_source("checkpoint", None)


class TestExportMain(unittest.TestCase):
    def _args(self, **overrides):
        values: dict[str, str | None | bool | list] = {
            "config": "config.yaml",
            "source": "model",
            "checkpoint": None,
            "model": None,
            "device": None,
            "output_dir": None,
            "artifacts": None,
            "no_prefill_decode": False,
            "set": [],
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_model_source_loads_the_adapter_model(self):
        """Model-source export should load the FP model through the adapter."""
        cfg = {
            "model": {"family": "llama", "name_or_path": "model"},
            "runtime": {"seed": 42},
            "export": {"enabled": True},
        }
        adapter = Mock()
        loaded_ctx = Mock()
        adapter.load_model.return_value = loaded_ctx

        with patch.object(
            export_example, "parse_args", return_value=self._args()
        ), patch.object(
            export_example, "load_recipe_config", return_value=cfg
        ), patch.object(
            export_example, "get_adapter", return_value=adapter
        ), patch.object(
            export_example, "set_seed"
        ), patch.object(
            export_example.torch, "load"
        ) as torch_load:
            export_example.main()

        adapter.load_model.assert_called_once()
        adapter.export.assert_called_once_with(loaded_ctx)
        torch_load.assert_not_called()

    def test_checkpoint_argument_preserves_checkpoint_export(self):
        """Checkpoint-only invocation should continue to bypass model loading."""
        cfg = {
            "model": {"family": "llama", "name_or_path": "model"},
            "runtime": {"device": "cpu", "dtype": "float32", "seed": 42},
            "export": {"enabled": True},
        }
        adapter = Mock()
        checkpoint = Mock()
        checkpoint.to.return_value = checkpoint
        checkpoint.eval.return_value = checkpoint

        with patch.object(
            export_example,
            "parse_args",
            return_value=self._args(source="checkpoint", checkpoint="model.pt"),
        ), patch.object(
            export_example, "load_recipe_config", return_value=cfg
        ), patch.object(
            export_example, "get_adapter", return_value=adapter
        ), patch.object(
            export_example, "set_seed"
        ), patch.object(
            export_example.torch, "load", return_value=checkpoint
        ) as torch_load:
            export_example.main()

        adapter.load_model.assert_not_called()
        adapter.export.assert_called_once()
        torch_load.assert_called_once_with(
            "model.pt",
            map_location=export_example.torch.device("cpu"),
            weights_only=False,
        )

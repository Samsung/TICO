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

import tico.quantization.recipes.adapters.gemma4 as gemma_adapter_mod
import tico.quantization.recipes.export.gemma4 as gemma_export

import torch
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4VisionPrefillExportAdapter,
)


class FakePTQWrapper(torch.nn.Module):
    """Minimal PTQWrapper-like container."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        """Forward to the wrapped module."""
        return self.wrapped(*args, **kwargs)


class FakeVisionExport(torch.nn.Module):
    """Return a model-output-like object for the vision stage."""

    def forward(self, pixel_values):
        """Return four tiny visual tokens from a pixel-values-only ABI."""
        del pixel_values
        return SimpleNamespace(last_hidden_state=torch.zeros(4, 8))


class FakeVision(torch.nn.Module):
    """Expose the fixed Gemma4 vision geometry."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=2,
            pooling_kernel_size=2,
            default_output_length=4,
        )

    def as_export_module(self, mode, *, pixel_position_ids):
        """Return the trivial static vision module."""
        del mode, pixel_position_ids
        return FakeVisionExport()


class FakeAttention(torch.nn.Module):
    """Expose one Gemma4 text-attention export contract."""

    def __init__(self, *, layer_idx, is_sliding, is_shared):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(num_attention_heads=2)
        self.num_key_value_groups = 2
        self.head_dim = 4
        self.sliding_window = 2 if is_sliding else None
        self.is_sliding = is_sliding
        self.is_kv_shared_layer = is_shared
        self.max_seq = 4


class FakeDecoderLayer(torch.nn.Module):
    """Expose prefill and decode export modules."""

    def __init__(self, *, layer_idx, is_sliding, is_shared):
        super().__init__()
        self.self_attn = FakePTQWrapper(
            FakeAttention(
                layer_idx=layer_idx,
                is_sliding=is_sliding,
                is_shared=is_shared,
            )
        )
        self.export_calls = []

    def as_export_module(
        self,
        mode,
        *,
        return_kv=True,
        per_layer_input_observer=None,
    ):
        """Return a placeholder module for the requested export mode."""
        self.export_calls.append((mode, bool(return_kv), per_layer_input_observer))
        return torch.nn.Identity()


class FakeText(torch.nn.Module):
    """Minimal Gemma4 text wrapper hierarchy."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            hidden_size_per_layer_input=2,
            max_position_embeddings=4,
            num_hidden_layers=2,
            vocab_size=32,
            enable_moe_block=False,
        )
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.obs_per_layer_inputs = torch.nn.Identity()
        self.layers = torch.nn.ModuleList(
            [
                FakePTQWrapper(
                    FakeDecoderLayer(
                        layer_idx=0,
                        is_sliding=True,
                        is_shared=False,
                    )
                ),
                FakePTQWrapper(
                    FakeDecoderLayer(
                        layer_idx=1,
                        is_sliding=False,
                        is_shared=True,
                    )
                ),
            ]
        )
        self.norm = torch.nn.Identity()


class FakeGemmaModel(torch.nn.Module):
    """Minimal multimodal Gemma4 wrapper hierarchy."""

    def __init__(self):
        super().__init__()
        self.vision_tower = FakePTQWrapper(FakeVision())
        self.language_model = FakePTQWrapper(FakeText())
        self.embed_vision = torch.nn.Identity()
        self.visual_start_idx = 0
        self.num_visual_tokens = 4


class FakeTopLevelGemma(torch.nn.Module):
    """Minimal conditional-generation wrapper hierarchy."""

    def __init__(self):
        super().__init__()
        self.model = FakePTQWrapper(FakeGemmaModel())
        self.lm_head = torch.nn.Linear(8, 32, bias=False)


class FakeExportModel(torch.nn.Module):
    """Outer PTQWrapper-like model returned by prepare/convert."""

    def __init__(self):
        super().__init__()
        self.wrapped = FakeTopLevelGemma()


def _model_args():
    """Return the fixed tiny vision contract used by exporter tests."""
    return {
        "vision": {
            "visual_start_idx": 0,
            "num_visual_tokens": 4,
            "max_soft_tokens": 4,
            "patch_grid_height": 4,
            "patch_grid_width": 4,
            "image_height": 8,
            "image_width": 8,
        }
    }


class TestGemma4PerLayerExport(unittest.TestCase):
    def test_exports_all_static_runtime_stages(self):
        """Gemma4 staged export should emit vision, prefill, and decode graphs."""
        calls = []
        vision_modules = []
        export_model = FakeExportModel()
        dynamic_shapes = {"input_ids": {1: "S"}}

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            if save_path.name == "vision_prefill.q.circle":
                vision_modules.append(module)
                self.assertEqual(len(example_inputs), 1)
            calls.append((save_path.name, kwargs.get("dynamic_shapes")))

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            gemma_export,
            "_prepare_gemma4_export_model",
            return_value=(export_model, "q"),
        ), patch.object(
            gemma_export,
            "make_token_embedding_dynamic_shapes",
            return_value=dynamic_shapes,
        ), patch.object(
            gemma_export,
            "_convert_and_save",
            fake_convert_and_save,
        ):
            gemma_export.export_gemma4_per_layer(
                q_model=torch.nn.Identity(),
                max_seq_len=4,
                output_dir=tmpdir,
                model_args=_model_args(),
                prefill_decode=True,
            )

        self.assertEqual(
            [name for name, _ in calls],
            [
                "vision_prefill.q.circle",
                "token_embedding.q.circle",
                "multimodal_fusion_prefill.q.circle",
                "decoder_layer_prefill_0.q.circle",
                "decoder_layer_decode_0.q.circle",
                "decoder_layer_prefill_1.q.circle",
                "decoder_layer_decode_1.q.circle",
                "lm_head.q.circle",
            ],
        )
        token_embedding_shapes = [
            shapes for name, shapes in calls if name == "token_embedding.q.circle"
        ]
        self.assertEqual(token_embedding_shapes, [dynamic_shapes])
        self.assertEqual(len(vision_modules), 1)
        self.assertIsInstance(vision_modules[0], Gemma4VisionPrefillExportAdapter)

        qtext = export_model.wrapped.model.wrapped.language_model.wrapped
        for layer in qtext.layers:
            self.assertEqual(
                [call[0] for call in layer.wrapped.export_calls],
                ["prefill", "decode"],
            )
            self.assertTrue(
                all(
                    call[2] is qtext.obs_per_layer_inputs
                    for call in layer.wrapped.export_calls
                )
            )

    def test_prefill_only_export_uses_unsuffixed_stage_names(self):
        """Disabling decode export should omit all decode artifacts."""
        names = []
        export_model = FakeExportModel()

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            del module, example_inputs, kwargs
            names.append(save_path.name)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            gemma_export,
            "_prepare_gemma4_export_model",
            return_value=(export_model, "f32"),
        ), patch.object(
            gemma_export,
            "_convert_and_save",
            fake_convert_and_save,
        ):
            gemma_export.export_gemma4_per_layer(
                q_model=torch.nn.Identity(),
                max_seq_len=4,
                output_dir=tmpdir,
                model_args=_model_args(),
                prefill_decode=False,
            )

        self.assertEqual(
            names,
            [
                "vision_prefill.f32.circle",
                "token_embedding.f32.circle",
                "multimodal_fusion.f32.circle",
                "decoder_layer_0.f32.circle",
                "decoder_layer_1.f32.circle",
                "lm_head.f32.circle",
            ],
        )

    def test_adapter_routes_circle_per_layer_artifact(self):
        """The Gemma4 adapter should dispatch the generic Circle artifact key."""
        model = torch.nn.Identity()
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 2048},
                "model_args": _model_args(),
                "export": {
                    "enabled": True,
                    "output_dir": "./out/gemma4",
                    "max_seq_len": 1024,
                    "prefill_decode": True,
                    "strict": True,
                    "artifacts": ["circle_per_layer"],
                },
            },
            adapter=Gemma4Adapter(),
            model=model,
        )

        with patch.object(
            gemma_adapter_mod,
            "export_gemma4_per_layer",
        ) as export_per_layer:
            Gemma4Adapter().export(ctx)

        export_per_layer.assert_called_once_with(
            q_model=model,
            max_seq_len=1024,
            output_dir=gemma_adapter_mod.Path("./out/gemma4"),
            model_args=_model_args(),
            prefill_decode=True,
            strict=True,
            vision_granularity="monolithic",
        )


if __name__ == "__main__":
    unittest.main()

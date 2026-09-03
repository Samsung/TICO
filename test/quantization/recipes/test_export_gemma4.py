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
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma_adapter_mod
import tico.quantization.recipes.export.gemma4 as gemma_export

import torch
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4PLEEmbeddingExportAdapter,
    Gemma4PLEProjectionExportAdapter,
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

    def __init__(self, *, ple_dim: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            hidden_size_per_layer_input=ple_dim,
            max_position_embeddings=4,
            num_hidden_layers=2,
            vocab_size=32,
            vocab_size_per_layer_input=16,
            enable_moe_block=False,
        )
        self._mode = Mode.NO_QUANT
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.hidden_size_per_layer_input = ple_dim
        self.embed_tokens_per_layer = None
        self.per_layer_model_projection = None
        self.per_layer_projection_norm = None
        self.per_layer_input_scale = 1.0
        self.per_layer_model_projection_scale = 1.0
        self.obs_per_layer_token_inputs = None
        self.obs_per_layer_projection = None
        self.obs_per_layer_inputs = None
        if ple_dim:
            packed = self.config.num_hidden_layers * ple_dim
            self.embed_tokens_per_layer = torch.nn.Embedding(16, packed)
            self.per_layer_model_projection = torch.nn.Linear(8, packed, bias=False)
            self.per_layer_projection_norm = torch.nn.Identity()
            self.per_layer_input_scale = 0.5
            self.per_layer_model_projection_scale = 0.25
            self.obs_per_layer_token_inputs = torch.nn.Identity()
            self.obs_per_layer_projection = torch.nn.Identity()
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

    def __init__(self, *, ple_dim: int = 2):
        super().__init__()
        self.vision_tower = FakePTQWrapper(FakeVision())
        self.language_model = FakePTQWrapper(FakeText(ple_dim=ple_dim))
        self.embed_vision = torch.nn.Identity()
        self.visual_start_idx = 0
        self.num_visual_tokens = 4


class FakeTopLevelGemma(torch.nn.Module):
    """Minimal conditional-generation wrapper hierarchy."""

    def __init__(self, *, ple_dim: int = 2):
        super().__init__()
        self.model = FakePTQWrapper(FakeGemmaModel(ple_dim=ple_dim))
        self.lm_head = torch.nn.Linear(8, 32, bias=False)


class FakeExportModel(torch.nn.Module):
    """Outer PTQWrapper-like model returned by prepare/convert."""

    def __init__(self, *, ple_dim: int = 2):
        super().__init__()
        self.wrapped = FakeTopLevelGemma(ple_dim=ple_dim)


def _patch_small_ple_table(estimated_bytes: int = 1024):
    """Bypass the wrapper-based size estimate for the plain fake PLE table."""
    return patch.object(
        gemma_export,
        "estimate_gemma4_ple_embedding_circle_bytes",
        return_value=estimated_bytes,
    )


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
        """Gemma4 staged export should emit vision, PLE, prefill, and decode graphs."""
        calls = []
        modules = {}
        example_shapes = {}
        export_model = FakeExportModel()
        dynamic_shapes = {"input_ids": {1: "S"}}

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            if save_path.name == "vision_prefill.q.circle":
                self.assertEqual(len(example_inputs), 1)
            modules[save_path.name] = module
            example_shapes[save_path.name] = [
                (tuple(tensor.shape), tensor.dtype) for tensor in example_inputs
            ]
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
            "register_fake_quant_meta_kernels_for_dynamic_export",
        ) as register_fake_kernels, _patch_small_ple_table(), patch.object(
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
                "ple_embedding.q.circle",
                "multimodal_fusion_prefill.q.circle",
                "ple_projection_prefill.q.circle",
                "ple_projection_decode.q.circle",
                "decoder_layer_prefill_0.q.circle",
                "decoder_layer_decode_0.q.circle",
                "decoder_layer_prefill_1.q.circle",
                "decoder_layer_decode_1.q.circle",
                "lm_head.q.circle",
            ],
        )
        register_fake_kernels.assert_called_once_with()

        # Dynamic ``(1, S)`` contract is shared by token and PLE embeddings.
        token_embedding_shapes = [
            shapes for name, shapes in calls if name == "token_embedding.q.circle"
        ]
        self.assertEqual(token_embedding_shapes, [dynamic_shapes])
        ple_embedding_shapes = [
            shapes for name, shapes in calls if name == "ple_embedding.q.circle"
        ]
        self.assertEqual(ple_embedding_shapes, [dynamic_shapes])
        self.assertEqual(
            example_shapes["ple_embedding.q.circle"],
            [((1, 4), torch.long)],
        )

        # Static projection ABI: prefill uses max_seq_len, decode uses one token.
        self.assertEqual(
            example_shapes["ple_projection_prefill.q.circle"],
            [((1, 4, 8), torch.float32), ((1, 4, 2, 2), torch.float32)],
        )
        self.assertEqual(
            example_shapes["ple_projection_decode.q.circle"],
            [((1, 1, 8), torch.float32), ((1, 1, 2, 2), torch.float32)],
        )
        # Static projection artifacts must not carry dynamic shapes.
        for name, shapes in calls:
            if name.startswith("ple_projection"):
                self.assertIsNone(shapes)

        self.assertIsInstance(
            modules["vision_prefill.q.circle"], Gemma4VisionPrefillExportAdapter
        )
        self.assertIsInstance(
            modules["ple_embedding.q.circle"], Gemma4PLEEmbeddingExportAdapter
        )
        self.assertIsInstance(
            modules["ple_projection_prefill.q.circle"],
            Gemma4PLEProjectionExportAdapter,
        )
        self.assertIs(
            modules["ple_projection_prefill.q.circle"],
            modules["ple_projection_decode.q.circle"],
        )

        qtext = export_model.wrapped.model.wrapped.language_model.wrapped
        # The lookup table exists only in the shared embedding stage.
        ple_embedding_params = dict(
            modules["ple_embedding.q.circle"].named_parameters()
        )
        self.assertEqual(
            list(ple_embedding_params),
            ["embed_tokens_per_layer.weight"],
        )
        self.assertIs(
            ple_embedding_params["embed_tokens_per_layer.weight"],
            qtext.embed_tokens_per_layer.weight,
        )
        projection_state = list(
            modules["ple_projection_prefill.q.circle"].state_dict().keys()
        )
        self.assertEqual(projection_state, ["per_layer_model_projection.weight"])
        for name in projection_state:
            self.assertNotIn("embed_tokens", name)

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

    def test_ple_disabled_model_skips_ple_artifacts(self):
        """A model without PLE must produce exactly the previous artifact set."""
        names = []
        export_model = FakeExportModel(ple_dim=0)

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            del module, example_inputs, kwargs
            names.append(save_path.name)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            gemma_export,
            "_prepare_gemma4_export_model",
            return_value=(export_model, "q"),
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
            names,
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
        qtext = export_model.wrapped.model.wrapped.language_model.wrapped
        for layer in qtext.layers:
            self.assertTrue(all(call[2] is None for call in layer.wrapped.export_calls))

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
        ), _patch_small_ple_table(), patch.object(
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
                "ple_embedding.f32.circle",
                "multimodal_fusion.f32.circle",
                "ple_projection.f32.circle",
                "decoder_layer_0.f32.circle",
                "decoder_layer_1.f32.circle",
                "lm_head.f32.circle",
            ],
        )

    def _run_export_with_ple_format(
        self, *, artifact_tag, ple_embedding_format, estimated_bytes=None
    ):
        """Run the exporter and return (circle names, saved .pt paths, manifest)."""
        circle_names = []
        pt_saves = []
        export_model = FakeExportModel()

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            del module, example_inputs, kwargs
            circle_names.append(save_path.name)

        def fake_save_pt(module, path):
            pt_saves.append((module, gemma_export.Path(path).name))
            return path

        if estimated_bytes is None:
            estimated_bytes = 1024

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            gemma_export,
            "_prepare_gemma4_export_model",
            return_value=(export_model, artifact_tag),
        ), patch.object(
            gemma_export, "_convert_and_save", fake_convert_and_save
        ), patch.object(
            gemma_export, "save_gemma4_ple_embedding_artifact", fake_save_pt
        ), _patch_small_ple_table(
            estimated_bytes
        ):
            gemma_export.export_gemma4_per_layer(
                q_model=torch.nn.Identity(),
                max_seq_len=4,
                output_dir=tmpdir,
                model_args=_model_args(),
                prefill_decode=True,
                ple_embedding_format=ple_embedding_format,
            )
            manifest = json.loads(
                (gemma_export.Path(tmpdir) / "ple_pipeline.json").read_text()
            )
        return circle_names, pt_saves, manifest

    def test_auto_format_keeps_circle_when_table_fits(self):
        """A small table stays a dynamic Circle graph under ``auto``."""
        circle_names, pt_saves, manifest = self._run_export_with_ple_format(
            artifact_tag="q", ple_embedding_format="auto"
        )

        self.assertIn("ple_embedding.q.circle", circle_names)
        self.assertEqual(pt_saves, [])
        self.assertEqual(manifest["embedding"]["format"], "circle")
        self.assertEqual(manifest["embedding"]["artifact"], "ple_embedding.q.circle")
        self.assertEqual(manifest["embedding"]["inputs"][0]["sequence_range"], [1, 4])
        self.assertEqual(
            [stage["id"] for stage in manifest["projections"]],
            ["ple_projection_prefill", "ple_projection_decode"],
        )
        self.assertEqual(manifest["projections"][0]["inputs"][1]["shape"], [1, 4, 2, 2])
        self.assertEqual(
            [boundary["name"] for boundary in manifest["boundaries"]],
            ["per_layer_token_inputs", "per_layer_projection", "per_layer_inputs"],
        )

    def test_auto_format_falls_back_to_pt_above_circle_limit(self):
        """Tables at or above the flatbuffer limit are saved as host ``.pt``."""
        for artifact_tag in ("f32", "q"):
            with self.subTest(artifact_tag=artifact_tag):
                circle_names, pt_saves, manifest = self._run_export_with_ple_format(
                    artifact_tag=artifact_tag,
                    ple_embedding_format="auto",
                    estimated_bytes=gemma_export.CIRCLE_FLATBUFFER_LIMIT_BYTES,
                )

                self.assertNotIn(f"ple_embedding.{artifact_tag}.circle", circle_names)
                self.assertEqual(len(pt_saves), 1)
                module, name = pt_saves[0]
                self.assertIsInstance(module, Gemma4PLEEmbeddingExportAdapter)
                self.assertEqual(name, f"ple_embedding.{artifact_tag}.pt")
                self.assertEqual(manifest["embedding"]["format"], "pt")
                self.assertEqual(manifest["embedding"]["requested_format"], "auto")
                self.assertEqual(
                    manifest["embedding"]["estimated_circle_bytes"],
                    gemma_export.CIRCLE_FLATBUFFER_LIMIT_BYTES,
                )
                # Projection graphs are unaffected by the lookup format.
                self.assertIn(
                    f"ple_projection_prefill.{artifact_tag}.circle", circle_names
                )
                self.assertIn(
                    f"ple_projection_decode.{artifact_tag}.circle", circle_names
                )

    def test_explicit_pt_and_circle_formats_are_honored(self):
        """Explicit formats override the size heuristic in both directions."""
        circle_names, pt_saves, manifest = self._run_export_with_ple_format(
            artifact_tag="q", ple_embedding_format="pt"
        )
        self.assertNotIn("ple_embedding.q.circle", circle_names)
        self.assertEqual([name for _, name in pt_saves], ["ple_embedding.q.pt"])
        self.assertEqual(manifest["embedding"]["requested_format"], "pt")

        circle_names, pt_saves, manifest = self._run_export_with_ple_format(
            artifact_tag="q",
            ple_embedding_format="circle",
            estimated_bytes=gemma_export.CIRCLE_FLATBUFFER_LIMIT_BYTES * 4,
        )
        self.assertIn("ple_embedding.q.circle", circle_names)
        self.assertEqual(pt_saves, [])
        self.assertEqual(manifest["embedding"]["format"], "circle")

    def test_invalid_ple_embedding_format_is_rejected(self):
        """Unknown formats fail before any artifact is written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "ple_embedding format"):
                gemma_export.export_gemma4_per_layer(
                    q_model=torch.nn.Identity(),
                    max_seq_len=4,
                    output_dir=tmpdir,
                    model_args=_model_args(),
                    ple_embedding_format="safetensors",
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
            ple_embedding_format="auto",
        )


if __name__ == "__main__":
    unittest.main()

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

import tico.quantization.recipes.adapters.qwen3_vl as qwen_adapter_mod
import tico.quantization.recipes.export.qwen3_vl as qwen_export

import torch
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.wrappers.qwen_vl.vision_profile import Qwen3VLVisionProfile


class FakePTQWrapper(torch.nn.Module):
    """Minimal PTQWrapper-like container."""

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        """Forward to the wrapped module when a test executes the adapter."""
        return self.wrapped(*args, **kwargs)


class FakePatchEmbed(torch.nn.Module):
    """Expose the patch geometry used to build vision export inputs."""

    in_channels = 3
    temporal_patch_size = 2
    patch_size = 2

    def forward(self, value):
        """Return a placeholder tensor."""
        return value


class FakeDecoderLayer(torch.nn.Module):
    """Expose prefill and decode export modules."""

    def as_export_module(self, mode, *, return_kv=True):
        """Return a trivial module for the requested export mode."""
        del mode, return_kv
        return torch.nn.Identity()


class FakeVisionExport(torch.nn.Module):
    """Represent one profile-specialized vision graph."""

    def __init__(self, grid_thw):
        super().__init__()
        self.profile = Qwen3VLVisionProfile.from_grid_thw(grid_thw)

    def forward(self, pixel_values):
        """Return placeholder image and DeepStack outputs."""
        return pixel_values, ()


class FakeVision(torch.nn.Module):
    """Minimal profile-independent Qwen3-VL vision wrapper."""

    def __init__(self):
        super().__init__()
        self.spatial_merge_size = 2
        self.patch_embed = FakePTQWrapper(FakePatchEmbed())
        self.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Identity()])
        self.last_export_grid_thw = None

    def as_export_module(self, mode="prefill", *, grid_thw=None):
        """Return the explicit static vision module used by the exporter."""
        if mode != "prefill":
            raise ValueError(f"Unsupported fake vision export mode: {mode!r}")
        if grid_thw is None:
            raise ValueError("Fake vision export requires grid_thw.")
        self.last_export_grid_thw = Qwen3VLVisionProfile.from_grid_thw(
            grid_thw
        ).grid_thw
        return FakeVisionExport(grid_thw)


class FakeText(torch.nn.Module):
    """Minimal Qwen3-VL text wrapper hierarchy."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            head_dim=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=1,
            max_position_embeddings=4,
            vocab_size=32,
        )
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.layers = torch.nn.ModuleList([FakePTQWrapper(FakeDecoderLayer())])
        self.norm = torch.nn.Identity()
        self.rotate_embedding = None


class FakeQwenModel(torch.nn.Module):
    """Minimal multimodal wrapper hierarchy."""

    def __init__(self):
        super().__init__()
        self.visual = FakePTQWrapper(FakeVision())
        self.language_model = FakePTQWrapper(FakeText())
        self.visual_start_idx = 0


class FakeTopLevelQwen(torch.nn.Module):
    """Minimal conditional-generation wrapper hierarchy."""

    def __init__(self):
        super().__init__()
        self.model = FakePTQWrapper(FakeQwenModel())
        self.lm_head = torch.nn.Linear(8, 32, bias=False)
        self.rotate_lm_head = None


class FakeExportModel(torch.nn.Module):
    """Outer PTQWrapper-like model returned by prepare or convert."""

    def __init__(self):
        super().__init__()
        self.wrapped = FakeTopLevelQwen()


def _model_args():
    """Return the fixed vision contract used by exporter tests."""
    return {
        "vision": {
            "grid_thw": [1, 4, 4],
            "visual_start_idx": 0,
            "spatial_merge_size": 2,
        }
    }


class TestQwen3VLPerLayerExport(unittest.TestCase):
    """Validate the staged exporter and profile-specific vision contract."""

    def test_exports_all_static_runtime_stages(self):
        """The exporter should emit prefill, decode, and DeepStack graphs."""
        calls = []
        vision_inputs = []
        export_model = FakeExportModel()
        dynamic_shapes = {"input_ids": {1: "S"}}

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            calls.append((save_path.name, kwargs.get("dynamic_shapes")))
            if save_path.name.startswith("vision_prefill_"):
                vision_inputs.append((module, example_inputs))

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            qwen_export,
            "_prepare_qwen3_vl_export_model",
            return_value=(export_model, "q"),
        ), patch.object(
            qwen_export,
            "make_token_embedding_dynamic_shapes",
            return_value=dynamic_shapes,
        ), patch.object(
            qwen_export,
            "register_fake_quant_meta_kernels_for_dynamic_export",
        ) as register_meta, patch.object(
            qwen_export, "_convert_and_save", fake_convert_and_save
        ):
            qwen_export.export_qwen3_vl_per_layer(
                q_model=torch.nn.Identity(),
                max_seq_len=4,
                output_dir=tmpdir,
                model_args=_model_args(),
                prefill_decode=True,
            )

        self.assertEqual(
            [name for name, _ in calls],
            [
                "vision_prefill_t1_h4_w4.q.circle",
                "token_embedding.q.circle",
                "multimodal_embedding_prefill_t1_h4_w4.q.circle",
                "decoder_layer_prefill_0.q.circle",
                "deepstack_fusion_0_t1_h4_w4.q.circle",
                "decoder_layer_decode_0.q.circle",
                "lm_head.q.circle",
            ],
        )
        token_embedding_calls = [
            shapes for name, shapes in calls if name == "token_embedding.q.circle"
        ]
        self.assertEqual(token_embedding_calls, [dynamic_shapes])
        register_meta.assert_called_once_with()
        self.assertEqual(len(vision_inputs), 1)
        vision_module, example_inputs = vision_inputs[0]
        self.assertEqual(vision_module.profile.grid_thw, (1, 4, 4))
        self.assertEqual(len(example_inputs), 1)
        self.assertEqual(
            export_model.wrapped.model.wrapped.visual.wrapped.last_export_grid_thw,
            (1, 4, 4),
        )

    def test_prefill_only_export_omits_decode_artifacts(self):
        """Disabling decode should omit decode artifacts and retain profile naming."""
        names = []
        export_model = FakeExportModel()

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            del module, example_inputs, kwargs
            names.append(save_path.name)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            qwen_export,
            "_prepare_qwen3_vl_export_model",
            return_value=(export_model, "f32"),
        ), patch.object(qwen_export, "_convert_and_save", fake_convert_and_save):
            qwen_export.export_qwen3_vl_per_layer(
                q_model=torch.nn.Identity(),
                max_seq_len=4,
                output_dir=tmpdir,
                model_args=_model_args(),
                prefill_decode=False,
            )

        self.assertEqual(
            names,
            [
                "vision_prefill_t1_h4_w4.f32.circle",
                "token_embedding.f32.circle",
                "multimodal_embedding_t1_h4_w4.f32.circle",
                "decoder_layer_0.f32.circle",
                "deepstack_fusion_0_t1_h4_w4.f32.circle",
                "lm_head.f32.circle",
            ],
        )

    def test_rejects_non_integer_vision_profile(self):
        """Export profile normalization should not truncate floating values."""
        args = _model_args()
        args["vision"]["grid_thw"] = [1.5, 4, 4]

        with self.assertRaisesRegex(TypeError, "integers"):
            qwen_export._normalize_model_args(args)

    def test_rejects_visual_span_larger_than_static_sequence(self):
        """The fixed visual span must fit inside max_seq_len."""
        export_model = FakeExportModel()
        args = _model_args()
        args["vision"]["visual_start_idx"] = 1
        export_model.wrapped.model.wrapped.visual_start_idx = 1

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            qwen_export,
            "_prepare_qwen3_vl_export_model",
            return_value=(export_model, "q"),
        ):
            with self.assertRaisesRegex(ValueError, "visual-token span"):
                qwen_export.export_qwen3_vl_per_layer(
                    q_model=torch.nn.Identity(),
                    max_seq_len=4,
                    output_dir=tmpdir,
                    model_args=args,
                )

    def test_adapter_routes_circle_per_layer_artifact(self):
        """The Qwen adapter should dispatch the generic Circle artifact key."""
        model = torch.nn.Identity()
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 2048},
                "model_args": _model_args(),
                "export": {
                    "enabled": True,
                    "output_dir": "./out/qwen",
                    "max_seq_len": 1024,
                    "prefill_decode": True,
                    "strict": True,
                    "artifacts": ["circle_per_layer"],
                },
            },
            adapter=Qwen3VLAdapter(),
            model=model,
        )

        with patch.object(
            qwen_adapter_mod,
            "export_qwen3_vl_per_layer",
        ) as export_per_layer:
            Qwen3VLAdapter().export(ctx)

        export_per_layer.assert_called_once_with(
            q_model=model,
            max_seq_len=1024,
            output_dir=qwen_adapter_mod.Path("./out/qwen"),
            model_args=_model_args(),
            prefill_decode=True,
            strict=True,
        )


if __name__ == "__main__":
    unittest.main()

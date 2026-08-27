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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma_adapter_mod
import tico.quantization.recipes.export.gemma4 as gemma_export

import torch
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.vision_split_export import (
    build_gemma4_vision_split_export_bundle,
    Gemma4VisionEncoderLayerExportAdapter,
    Gemma4VisionPatchStageExportAdapter,
)


class FakeObserver(torch.nn.Module):
    """Apply a deterministic idempotent fake-quantization transform."""

    def __init__(self, name: str, step: float = 0.125) -> None:
        super().__init__()
        self.name = name
        self.dtype = "int16"
        self.qscheme = "per_tensor_symmetric"
        self.channel_axis = None
        self.register_buffer("_cached_scale", torch.tensor(step))
        self.register_buffer("_cached_zp", torch.tensor(0, dtype=torch.int))

    def fake_quant(self, tensor: torch.Tensor) -> torch.Tensor:
        """Round a tensor to the configured fake-quantization step."""
        scale = self._cached_scale.to(device=tensor.device, dtype=tensor.dtype)
        return torch.round(tensor / scale) * scale


class FakePTQWrapper(torch.nn.Module):
    """Provide the transparent wrapper shape used by production PTQ modules."""

    def __init__(self, wrapped: torch.nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        """Forward all arguments to the wrapped module."""
        return self.wrapped(*args, **kwargs)


class FakePatchExport(torch.nn.Module):
    """Project three pixel channels to a two-channel vision hidden state."""

    def __init__(self) -> None:
        super().__init__()
        self.obs_output = FakeObserver("patch_output", 0.25)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return a deterministic patch representation."""
        hidden_states = torch.stack(
            (
                pixel_values[..., 0] + 0.5 * pixel_values[..., 2],
                pixel_values[..., 1] - 0.25 * pixel_values[..., 2],
            ),
            dim=-1,
        )
        return self.obs_output.fake_quant(hidden_states)


class FakeIdentityVisionLayer(torch.nn.Module):
    """Return hidden states while accepting the vision-layer keyword contract."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Return hidden states without consuming quantization observers."""
        del attention_mask, position_embeddings
        return hidden_states


class FakeVisionLayer(torch.nn.Module):
    """Model one quantized vision layer with observable boundaries."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.obs_act_in = FakeObserver(f"layer_{layer_idx}_act_in", 0.125)
        self.obs_output = FakeObserver(f"layer_{layer_idx}_output", 0.125)
        self.bias = torch.nn.Parameter(
            torch.tensor([0.125 * (layer_idx + 1), -0.0625 * (layer_idx + 1)])
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Consume all shared context inputs and update hidden states."""
        hidden_states = self.obs_act_in.fake_quant(hidden_states)
        cos, sin = position_embeddings
        context = attention_mask.mean() + cos.mean() - sin.mean()
        hidden_states = hidden_states + self.bias + 0.03125 * context
        return self.obs_output.fake_quant(hidden_states)


class FakeEncoder(torch.nn.Module):
    """Provide the static Gemma4 encoder contract used by split export."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self._mode = Mode.QUANT
        self.layers = torch.nn.ModuleList(
            [FakePTQWrapper(FakeVisionLayer(index)) for index in range(num_layers)]
        )
        self.obs_act_in = FakeObserver("encoder_act_in", 0.25)
        self.obs_attention_mask = FakeObserver("encoder_attention_mask", 0.5)
        self.obs_position_cos = FakeObserver("encoder_position_cos", 0.125)
        self.obs_position_sin = FakeObserver("encoder_position_sin", 0.125)
        self.obs_encoder_out = FakeObserver("encoder_out", 0.25)

    def materialize_templates(self, pixel_position_ids: torch.Tensor) -> None:
        """Create deterministic mask and RoPE templates from fixed coordinates."""
        valid = (pixel_position_ids[..., 0] >= 0).float()
        attention_mask = (1.0 - valid.unsqueeze(2) * valid.unsqueeze(1)) * -4.0
        clamped = pixel_position_ids.clamp(min=0).float()
        self.register_buffer(
            "attention_mask_template",
            attention_mask.unsqueeze(1),
            persistent=False,
        )
        self.register_buffer(
            "position_embeddings_cos_template",
            clamped / 8.0,
            persistent=False,
        )
        self.register_buffer(
            "position_embeddings_sin_template",
            (clamped + 1.0) / 16.0,
            persistent=False,
        )

    def forward_export(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the monolithic encoder path with baked shared context."""
        attention_mask = self.obs_attention_mask.fake_quant(
            self.attention_mask_template
        )
        cos = self.obs_position_cos.fake_quant(self.position_embeddings_cos_template)
        sin = self.obs_position_sin.fake_quant(self.position_embeddings_sin_template)
        hidden_states = self.obs_act_in.fake_quant(inputs_embeds)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )
        return self.obs_encoder_out.fake_quant(hidden_states)


class FakeEncoderExport(torch.nn.Module):
    """Delegate to a prepared fake encoder export path."""

    def __init__(self, wrapped: FakeEncoder) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the prepared encoder export path."""
        return self.wrapped.forward_export(inputs_embeds)


class FakePooler(torch.nn.Module):
    """Provide a fixed-output pooler with explicit input and output observers."""

    def __init__(self, output_length: int) -> None:
        super().__init__()
        self._mode = Mode.QUANT
        self.output_length = output_length
        self.obs_act_in = FakeObserver("pooler_act_in", 0.25)
        self.obs_pool_out = FakeObserver("pooler_out", 0.25)

    def forward_export(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Select the fixed output prefix and fake-quantize both boundaries."""
        hidden_states = self.obs_act_in.fake_quant(hidden_states)
        hidden_states = hidden_states[:, : self.output_length, :]
        return self.obs_pool_out.fake_quant(hidden_states)


class FakePoolerExport(torch.nn.Module):
    """Delegate to a prepared fake pooler export path."""

    def __init__(self, wrapped: FakePooler, num_valid_outputs: int) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.num_valid_outputs = num_valid_outputs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the prepared pooler export path."""
        return self.wrapped.forward_export(hidden_states)


class FakeVisionModelExport(torch.nn.Module):
    """Delegate to the prepared monolithic fake vision path."""

    def __init__(self, wrapped: "FakeVision") -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, pixel_values: torch.Tensor):
        """Return an object containing monolithic visual hidden states."""
        return self.wrapped.forward_export(pixel_values)


class FakeVision(torch.nn.Module):
    """Expose the production vision-wrapper attributes needed by the builder."""

    def __init__(self, *, num_layers: int = 3, output_length: int = 4) -> None:
        super().__init__()
        self._mode = Mode.QUANT
        self.config = SimpleNamespace(
            hidden_size=2,
            standardize=True,
            patch_size=1,
            pooling_kernel_size=1,
            default_output_length=output_length,
        )
        self.patch_embedder = FakePatchExport()
        self.encoder = FakeEncoder(num_layers)
        self.pooler = FakePooler(output_length)
        self.obs_strip_padding = FakeObserver("strip_padding", 0.25)
        self.obs_minus_bias = FakeObserver("minus_bias", 0.25)
        self.obs_last_hidden_state = FakeObserver("last_hidden_state", 0.25)
        self.obs_std_bias = FakeObserver("std_bias", 0.125)
        self.obs_std_scale = FakeObserver("std_scale", 0.125)
        self.register_buffer("std_bias", torch.tensor([0.125, -0.25]))
        self.register_buffer("std_scale", torch.tensor([1.0, 0.5]))

    def as_export_module(
        self,
        mode: str,
        *,
        pixel_position_ids: torch.Tensor,
    ) -> torch.nn.Module:
        """Prepare static patch, encoder, and pooler export components."""
        if mode != "prefill":
            raise ValueError(f"Unsupported fake vision mode: {mode!r}.")
        self.encoder.materialize_templates(pixel_position_ids)
        self.patch_embedder_export = self.patch_embedder
        self.encoder_export = FakeEncoderExport(self.encoder)
        self.pooler_export = FakePoolerExport(
            self.pooler,
            num_valid_outputs=self.config.default_output_length,
        )
        self.num_valid_pool_outputs = self.config.default_output_length
        return FakeVisionModelExport(self)

    def forward_export(self, pixel_values: torch.Tensor):
        """Run the fake monolithic path using the production operation order."""
        inputs_embeds = self.patch_embedder_export(pixel_values)
        hidden_states = self.encoder_export(inputs_embeds)
        hidden_states = self.pooler_export(hidden_states)
        hidden_states = hidden_states[:, : self.num_valid_pool_outputs, :]
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        hidden_states = self.obs_strip_padding.fake_quant(hidden_states)
        std_bias = self.obs_std_bias.fake_quant(self.std_bias)
        std_scale = self.obs_std_scale.fake_quant(self.std_scale)
        hidden_states = hidden_states - std_bias.float()
        hidden_states = self.obs_minus_bias.fake_quant(hidden_states)
        hidden_states = hidden_states * std_scale.float()
        hidden_states = hidden_states.to(inputs_embeds.dtype)
        hidden_states = self.obs_last_hidden_state.fake_quant(hidden_states)
        return SimpleNamespace(last_hidden_state=hidden_states)


class FakeAttention(torch.nn.Module):
    """Expose one text-attention contract for staged exporter tests."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(num_attention_heads=2)
        self.num_key_value_groups = 2
        self.head_dim = 2
        self.sliding_window = None
        self.is_sliding = False
        self.is_kv_shared_layer = False
        self.max_seq = 4


class FakeDecoderLayer(torch.nn.Module):
    """Expose placeholder text prefill and decode modules."""

    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = FakePTQWrapper(FakeAttention(layer_idx))

    def as_export_module(self, mode: str, *, return_kv: bool = True):
        """Return a placeholder module for the requested text export mode."""
        del mode, return_kv
        return torch.nn.Identity()


class FakeText(torch.nn.Module):
    """Provide the text-wrapper hierarchy consumed by the staged exporter."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            hidden_size_per_layer_input=0,
            max_position_embeddings=4,
            num_hidden_layers=1,
            vocab_size=16,
            enable_moe_block=False,
        )
        self.embed_tokens = torch.nn.Embedding(16, 4)
        self.layers = torch.nn.ModuleList(
            [FakePTQWrapper(FakeDecoderLayer(layer_idx=0))]
        )
        self.norm = torch.nn.Identity()


class FakeGemmaModel(torch.nn.Module):
    """Provide the multimodal Gemma4 wrapper hierarchy."""

    def __init__(self, *, num_vision_layers: int = 3) -> None:
        super().__init__()
        self.vision_tower = FakePTQWrapper(
            FakeVision(num_layers=num_vision_layers, output_length=4)
        )
        self.language_model = FakePTQWrapper(FakeText())
        self.embed_vision = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.embed_vision.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.5, 0.5],
                        [1.0, -1.0],
                    ]
                )
            )
        self.visual_start_idx = 0
        self.num_visual_tokens = 4


class FakeTopLevelGemma(torch.nn.Module):
    """Provide the conditional-generation wrapper hierarchy."""

    def __init__(self, *, num_vision_layers: int = 3) -> None:
        super().__init__()
        self.model = FakePTQWrapper(FakeGemmaModel(num_vision_layers=num_vision_layers))
        self.lm_head = torch.nn.Linear(4, 16, bias=False)


class FakeExportModel(torch.nn.Module):
    """Provide the outer wrapper returned by checkpoint preparation."""

    def __init__(self, *, num_vision_layers: int = 3) -> None:
        super().__init__()
        self.wrapped = FakeTopLevelGemma(num_vision_layers=num_vision_layers)


def _position_ids() -> torch.Tensor:
    """Return a row-major two-by-two patch profile."""
    return torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]], dtype=torch.long)


def _model_args() -> dict:
    """Return the tiny fixed vision contract used by exporter tests."""
    return {
        "vision": {
            "visual_start_idx": 0,
            "num_visual_tokens": 4,
            "max_soft_tokens": 4,
            "patch_grid_height": 2,
            "patch_grid_width": 2,
            "image_height": 2,
            "image_width": 2,
        }
    }


class TestGemma4VisionSplitBundle(unittest.TestCase):
    """Validate split-stage numerical and export contracts."""

    def _build_bundle(self):
        """Build a deterministic split bundle and its example pixel input."""
        gemma_model = FakeGemmaModel(num_vision_layers=3)
        pixel_values = torch.tensor(
            [
                [
                    [0.0, 0.25, 0.5],
                    [0.75, 1.0, 0.25],
                    [0.5, 0.0, 1.0],
                    [1.0, 0.5, 0.0],
                ]
            ]
        )
        bundle = build_gemma4_vision_split_export_bundle(
            gemma_model,
            pixel_position_ids=_position_ids(),
            output_dtype=pixel_values.dtype,
        )
        return gemma_model, bundle, pixel_values

    @staticmethod
    def _run_split(bundle, pixel_values: torch.Tensor) -> torch.Tensor:
        """Execute all split stages in manifest order."""
        hidden_states = bundle.patch_embedder(pixel_values)
        for layer in bundle.encoder_layers:
            hidden_states = layer(
                hidden_states,
                bundle.attention_mask,
                bundle.position_embeddings_cos,
                bundle.position_embeddings_sin,
            )
        pooled = bundle.pooler(hidden_states)
        return bundle.post_projection(pooled)

    def test_one_layer_artifacts_match_monolithic_output(self) -> None:
        """One-layer split execution should match the whole vision graph."""
        _, bundle, pixel_values = self._build_bundle()

        monolithic = bundle.monolithic(pixel_values)
        split = self._run_split(bundle, pixel_values)

        torch.testing.assert_close(split, monolithic)
        self.assertEqual(len(bundle.encoder_layers), 3)
        self.assertEqual(len(bundle.boundary_contracts), 5)

    def test_no_quant_split_boundaries_bypass_fake_quantization(self) -> None:
        """Floating-point split adapters should not apply boundary observers."""
        hidden_states = torch.tensor([[[0.3, -0.3]]])
        observer = FakeObserver("boundary", step=1.0)
        patch_stage = Gemma4VisionPatchStageExportAdapter(
            torch.nn.Identity(),
            mode=Mode.NO_QUANT,
            output_observer=observer,
        )
        layer = Gemma4VisionEncoderLayerExportAdapter(
            FakeIdentityVisionLayer(),
            mode=Mode.NO_QUANT,
            attention_mask_observer=observer,
            position_cos_observer=observer,
            position_sin_observer=observer,
            input_observer=observer,
            output_observers=(observer,),
        )

        patch_output = patch_stage(hidden_states)
        layer_output = layer(
            patch_output,
            torch.tensor([[[[0.3]]]]),
            torch.tensor([[[0.3, -0.3]]]),
            torch.tensor([[[0.2, -0.2]]]),
        )

        torch.testing.assert_close(patch_output, hidden_states)
        torch.testing.assert_close(layer_output, hidden_states)

    def test_encoder_context_is_external_and_torch_exportable(self) -> None:
        """Each encoder artifact should consume four runtime tensors."""
        _, bundle, pixel_values = self._build_bundle()
        hidden_states = bundle.patch_embedder(pixel_values)

        for layer in bundle.encoder_layers:
            large_internal_buffers = [
                name
                for name, tensor in layer.named_buffers()
                if tensor.numel() >= bundle.attention_mask.numel()
            ]
            self.assertEqual(large_internal_buffers, [])
            exported = torch.export.export(
                layer,
                (
                    hidden_states,
                    bundle.attention_mask,
                    bundle.position_embeddings_cos,
                    bundle.position_embeddings_sin,
                ),
                strict=False,
            )
            self.assertEqual(len(exported.graph_signature.user_inputs), 4)
            hidden_states = layer(
                hidden_states,
                bundle.attention_mask,
                bundle.position_embeddings_cos,
                bundle.position_embeddings_sin,
            )

        torch.export.export(bundle.patch_embedder, (pixel_values,), strict=False)
        torch.export.export(bundle.pooler, (hidden_states,), strict=False)
        pooled = bundle.pooler(hidden_states)
        torch.export.export(bundle.post_projection, (pooled,), strict=False)


class TestGemma4VisionSplitRecipe(unittest.TestCase):
    """Validate staged artifact routing, naming, and manifests."""

    def test_both_mode_exports_monolithic_and_split_stages(self) -> None:
        """Both mode should retain the old graph and add all split artifacts."""
        calls: list[str] = []
        export_model = FakeExportModel(num_vision_layers=3)

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            del module, example_inputs, kwargs
            calls.append(save_path.name)

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
                vision_granularity="both",
            )

            output_dir = Path(tmpdir)
            context = torch.load(
                output_dir / "vision_context.pt",
                map_location="cpu",
                weights_only=True,
            )
            manifest = json.loads(
                (output_dir / "vision_pipeline.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            calls,
            [
                "vision_prefill.q.circle",
                "vision_patch_embedder.q.circle",
                "vision_encoder_layer_00.q.circle",
                "vision_encoder_layer_01.q.circle",
                "vision_encoder_layer_02.q.circle",
                "vision_pooler.q.circle",
                "vision_post_projection.q.circle",
                "token_embedding.q.circle",
                "multimodal_fusion_prefill.q.circle",
                "decoder_layer_prefill_0.q.circle",
                "decoder_layer_decode_0.q.circle",
                "lm_head.q.circle",
            ],
        )
        self.assertEqual(
            set(context),
            {
                "attention_mask",
                "position_embeddings_cos",
                "position_embeddings_sin",
            },
        )
        self.assertEqual(manifest["granularity"], "both")
        self.assertEqual(manifest["monolithic_artifact"], "vision_prefill.q.circle")
        self.assertEqual(
            manifest["shared_encoder_inputs_artifact"],
            "vision_context.pt",
        )
        self.assertEqual(len(manifest["stages"]), 6)
        self.assertEqual(len(manifest["boundaries"]), 5)
        self.assertIn("scale", manifest["boundaries"][0]["observer"])

    def test_invalid_vision_options_fail_before_model_preparation(self) -> None:
        """Invalid split options should fail before loading or wrapping a model."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            gemma_export,
            "_prepare_gemma4_export_model",
        ) as prepare_model:
            with self.assertRaisesRegex(ValueError, "granularity"):
                gemma_export.export_gemma4_per_layer(
                    q_model=torch.nn.Identity(),
                    max_seq_len=4,
                    output_dir=tmpdir,
                    model_args=_model_args(),
                    vision_granularity="operator",
                )
        prepare_model.assert_not_called()

    def test_adapter_forwards_nested_vision_export_options(self) -> None:
        """The model adapter should route nested vision options to the exporter."""
        model = torch.nn.Identity()
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 4},
                "model_args": _model_args(),
                "export": {
                    "enabled": True,
                    "output_dir": "./out/gemma4",
                    "max_seq_len": 4,
                    "prefill_decode": True,
                    "strict": False,
                    "vision": {"granularity": "both"},
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
            max_seq_len=4,
            output_dir=Path("./out/gemma4"),
            model_args=_model_args(),
            prefill_decode=True,
            strict=False,
            vision_granularity="both",
        )

    def test_adapter_rejects_non_mapping_vision_options(self) -> None:
        """A malformed nested vision config should fail with a clear error."""
        ctx = RecipeContext(
            cfg={
                "model_args": _model_args(),
                "export": {
                    "enabled": True,
                    "vision": "layer",
                    "artifacts": ["circle_per_layer"],
                },
            },
            adapter=Gemma4Adapter(),
            model=torch.nn.Identity(),
        )

        with self.assertRaisesRegex(TypeError, "export.vision"):
            Gemma4Adapter().export(ctx)


if __name__ == "__main__":
    unittest.main()

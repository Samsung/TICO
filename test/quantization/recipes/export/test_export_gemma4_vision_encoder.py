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

"""Real, checkpoint-free Gemma4 encoder-stack export and boundary tests."""

import json
import tempfile
import unittest
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tico
import torch

from circle_schema import circle as circle_schema
from tico.circle import CircleDocument
from tico.circle.runtime import ExecutionMode, run_circle
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.export.gemma4 import _export_gemma4_vision_stages
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
    QuantGemma4VisionModel,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    Gemma4StaticVisionProfile,
)
from tico.quantization.wrapq.wrappers.gemma4.vision_split_export import (
    build_gemma4_vision_split_export_bundle,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.utils.model import CircleModel
from torch.utils._pytree import tree_flatten


def _has_gemma4_vision() -> bool:
    """Use the same optional-transformers condition as nearby wrapper tests."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionModel,
        )
    except ImportError:
        return False
    return True


class TinyVisionOwner(QuantModuleBase):
    """Own production vision wrappers without allocating a language model."""

    def __init__(self):
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        qcfg = PTQConfig(activation=affine(DType.int(16)), weight=affine(DType.int(16)))
        super().__init__(qcfg)
        config = Gemma4VisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            patch_size=1,
            position_embedding_size=8,
            pooling_kernel_size=2,
            attention_dropout=0.0,
            use_clipped_linears=False,
            rope_parameters={"rope_type": "default", "rope_theta": 100.0},
            standardize=True,
        )
        config._attn_implementation = "eager"
        fp_vision = Gemma4VisionModel(config).eval()
        with torch.no_grad():
            fp_vision.std_bias.copy_(torch.linspace(-0.2, 0.2, config.hidden_size))
            fp_vision.std_scale.copy_(torch.linspace(0.8, 1.2, config.hidden_size))
        self.vision_tower = QuantGemma4VisionModel(fp_vision, qcfg=qcfg)
        self.embed_vision = PTQWrapper(
            torch.nn.Linear(config.hidden_size, 4, bias=False), qcfg=qcfg
        )

    def _all_observers(self):
        return ()

    def _conditional_quant_subtrees(self):
        return {"vision": ("vision_tower", "embed_vision")}

    def forward(self, pixel_values, pixel_position_ids):
        """Calibrate the real vision path and the projection together."""
        output = self.vision_tower(pixel_values, pixel_position_ids)
        return self.embed_vision(output.last_hidden_state)


def _context(bundle):
    return (
        bundle.attention_mask,
        bundle.position_embeddings_cos,
        bundle.position_embeddings_sin,
    )


def _observer_state(model):
    """Snapshot frozen grids and lifecycle flags without including new templates."""
    return deepcopy(
        {
            name: observer.state_dict()
            for name, observer in model.named_modules()
            if isinstance(observer, ObserverBase)
        }
    )


@unittest.skipUnless(_has_gemma4_vision(), "transformers Gemma4 vision is unavailable")
class TestGemma4VisionEncoderExport(unittest.TestCase):
    """Exercise original multi-layer vision math, not mocked converter modules."""

    def _build(self, *, quantized: bool, padded: bool = False):
        torch.manual_seed(17)
        model = TinyVisionOwner().eval()
        pixel_values = torch.rand(1, 4, 3)
        positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        if padded:
            pixel_values = torch.cat(
                (pixel_values, torch.zeros_like(pixel_values)), dim=1
            )
            positions = torch.cat((positions, torch.full_like(positions, -1)), dim=1)
        if quantized:
            model.enable_calibration()
            with torch.no_grad():
                model(pixel_values, positions)
            model.freeze_qparams()
        before = _observer_state(model)
        bundle = build_gemma4_vision_split_export_bundle(
            model, pixel_position_ids=positions, output_dtype=pixel_values.dtype
        )
        for stage in (
            bundle.monolithic,
            bundle.patch_embedder,
            bundle.encoder,
            bundle.pooler,
            bundle.post_projection,
        ):
            stage.eval()
        return model, bundle, pixel_values, before

    def _assert_observer_state(self, before, model):
        after = _observer_state(model)
        # as_export_module registers aliases of existing observers; compare each
        # original observer by its original path, including its frozen snapshot.
        for name, state in before.items():
            expected, expected_spec = tree_flatten(state)
            actual, actual_spec = tree_flatten(after[name])
            self.assertEqual(actual_spec, expected_spec)
            for lhs, rhs in zip(actual, expected):
                if isinstance(lhs, torch.Tensor):
                    torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
                else:
                    self.assertEqual(lhs, rhs)

    def test_full_encoder_and_pipeline_match_monolithic(self):
        """Float and frozen INT16 paths retain every layer, in its original order."""
        for quantized in (False, True):
            with self.subTest(quantized=quantized), torch.no_grad():
                model, bundle, pixels, before = self._build(quantized=quantized)
                encoder = model.vision_tower.encoder.wrapped
                pooler = model.vision_tower.pooler.wrapped
                self.assertEqual(len(bundle.encoder.layers), 2)
                self.assertIs(bundle.encoder.layers[0], encoder.layers[0])
                self.assertIs(bundle.encoder.layers[1], encoder.layers[1])
                hidden = bundle.patch_embedder(pixels)
                expected = encoder.forward_export(hidden)
                if quantized:
                    expected = pooler.obs_act_in.fake_quant(expected)
                encoded = bundle.encoder(hidden, *_context(bundle))
                torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
                pooled = bundle.pooler(encoded)
                actual = bundle.post_projection(pooled)
                torch.testing.assert_close(
                    actual, bundle.monolithic(pixels), rtol=0, atol=0
                )
                layer_output = hidden
                for layer in bundle.encoder_layers:
                    layer_output = layer(layer_output, *_context(bundle))
                torch.testing.assert_close(layer_output, encoded, rtol=0, atol=0)

                # Distinct random attention/MLP weights must make order observable.
                first, second = bundle.encoder.layers
                bundle.encoder.layers[0], bundle.encoder.layers[1] = second, first
                reversed_output = bundle.encoder(hidden, *_context(bundle))
                bundle.encoder.layers[0], bundle.encoder.layers[1] = first, second
                self.assertFalse(torch.allclose(encoded, reversed_output))
                self._assert_observer_state(before, model)

    def test_padded_profile_preserves_valid_token_selection(self):
        """Padded patches and pooled slots stay in their established stages."""
        for quantized in (False, True):
            with self.subTest(quantized=quantized), torch.no_grad():
                model, bundle, pixels, before = self._build(
                    quantized=quantized, padded=True
                )
                hidden = bundle.patch_embedder(pixels)
                encoded = bundle.encoder(hidden, *_context(bundle))
                pooled = bundle.pooler(encoded)
                self.assertEqual(encoded.shape, (1, 8, 8))
                self.assertEqual(pooled.shape, (1, 2, 8))
                actual = bundle.post_projection(pooled)
                self.assertEqual(actual.shape, (1, 4))
                torch.testing.assert_close(
                    actual, bundle.monolithic(pixels), rtol=0, atol=0
                )
                self._assert_observer_state(before, model)

    def test_context_observed_once_and_no_split_only_inner_boundary(self):
        """The full stack has one context observation and one input per layer."""
        model, bundle, pixels, before = self._build(quantized=True)
        encoder = model.vision_tower.encoder.wrapped
        observers = (
            encoder.obs_position_cos,
            encoder.obs_position_sin,
            encoder.obs_attention_mask,
            encoder.obs_act_in,
            encoder.layers[1].wrapped.obs_act_in,
            encoder.obs_encoder_out,
        )
        hidden = bundle.patch_embedder(pixels)
        with ExitStack() as stack, torch.no_grad():
            spies = [
                stack.enter_context(
                    patch.object(obs, "fake_quant", wraps=obs.fake_quant)
                )
                for obs in observers
            ]
            bundle.encoder(hidden, *_context(bundle))
            for spy in spies:
                self.assertEqual(spy.call_count, 1)
        self._assert_observer_state(before, model)

    def test_no_quant_does_not_call_boundary_observers(self):
        """Frozen-style boundary logic is absent in the floating-point mode."""
        _, bundle, pixels, _ = self._build(quantized=False)
        self.assertIs(bundle.encoder._capability_guard.owner._mode, Mode.NO_QUANT)
        hidden = bundle.patch_embedder(pixels)
        observers = (
            bundle.encoder.input_observer,
            bundle.encoder.attention_mask_observer,
            bundle.encoder.position_cos_observer,
            bundle.encoder.position_sin_observer,
            bundle.encoder.encoder_output_observer,
            bundle.encoder.output_observer,
        )
        with ExitStack() as stack:
            for observer in observers:
                stack.enter_context(
                    patch.object(observer, "fake_quant", side_effect=AssertionError)
                )
            bundle.encoder(hidden, *_context(bundle))

    def test_real_torch_export_preserves_context_and_all_layers(self):
        """Mask and both RoPE tensors remain used runtime inputs in one graph."""
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                model, bundle, pixels, before = self._build(quantized=quantized)
                hidden = bundle.patch_embedder(pixels)
                args = (hidden, *_context(bundle))
                exported = torch.export.export(bundle.encoder, args, strict=False)
                self.assertEqual(
                    tuple(exported.graph_signature.user_inputs),
                    (
                        "hidden_states",
                        "attention_mask",
                        "position_embeddings_cos",
                        "position_embeddings_sin",
                    ),
                )
                placeholders = {
                    str(node.target): node
                    for node in exported.graph.nodes
                    if node.op == "placeholder"
                }
                for name in exported.graph_signature.user_inputs:
                    self.assertTrue(placeholders[name].users)
                parameters = tuple(exported.graph_signature.parameters)
                self.assertTrue(
                    any(name.startswith("layers.0.") for name in parameters)
                )
                self.assertTrue(
                    any(name.startswith("layers.1.") for name in parameters)
                )
                self.assertTrue(all(name.startswith("layers.") for name in parameters))
                self.assertFalse(
                    any("template" in name for name in exported.graph_signature.buffers)
                )
                with torch.no_grad():
                    torch.testing.assert_close(
                        exported.module()(*args), bundle.encoder(*args), rtol=0, atol=0
                    )
                self._assert_observer_state(before, model)

    def test_float_circle_pipeline_matches_monolithic(self):
        """Export four actual files and execute their manifest order on reference."""
        model, bundle, pixels, before = self._build(quantized=False)
        profile = Gemma4StaticVisionProfile(
            name="tiny",
            visual_start_idx=0,
            num_visual_tokens=1,
            max_soft_tokens=1,
            patch_grid_height=2,
            patch_grid_width=2,
            patch_size=1,
            pooling_kernel_size=2,
        )
        with tempfile.TemporaryDirectory() as tmpdir, torch.no_grad():
            output_dir = Path(tmpdir)
            _export_gemma4_vision_stages(
                gemma_model=model,
                qvision=model.vision_tower,
                profile=profile,
                pixel_values=pixels,
                pixel_position_ids=profile.build_image_position_ids(),
                text_hidden_size=4,
                artifact_tag="f32",
                output_dir=output_dir,
                granularity="encoder",
                spatial_pooling=True,
                strict=False,
            )
            manifest = json.loads((output_dir / "vision_pipeline.json").read_text())
            self.assertEqual(
                [stage["id"] for stage in manifest["stages"]],
                ["patch_embedder", "encoder", "pooler", "post_projection"],
            )
            self.assertEqual(
                {path.name for path in output_dir.glob("*.circle")},
                {
                    "vision_patch_embedder.f32.circle",
                    "vision_encoder.f32.circle",
                    "vision_pooler.f32.circle",
                    "vision_post_projection.f32.circle",
                },
            )
            context = torch.load(output_dir / "vision_context.pt", weights_only=True)
            context_args = tuple(
                context[name]
                for name in (
                    "attention_mask",
                    "position_embeddings_cos",
                    "position_embeddings_sin",
                )
            )
            actual = pixels
            for stage in manifest["stages"]:
                path = output_dir / stage["artifact"]
                document = CircleDocument.load(path)
                document.verify()
                circle = CircleModel.load(str(path))
                args = (
                    (actual, *context_args) if stage["id"] == "encoder" else (actual,)
                )
                actual = circle(*args)
            np.testing.assert_allclose(
                actual, bundle.monolithic(pixels).numpy(), rtol=1e-5, atol=1e-5
            )
        self._assert_observer_state(before, model)

    def test_quantized_circle_boundaries_are_directly_chainable(self):
        """Serialized INT16 producer/consumer domains agree without recalibration."""
        model, bundle, pixels, before = self._build(quantized=True)
        context = _context(bundle)
        with torch.no_grad():
            hidden = bundle.patch_embedder(pixels)
            encoded = bundle.encoder(hidden, *context)
            pooled = bundle.pooler(encoded)
            modules_and_inputs = (
                (bundle.patch_embedder, (pixels,)),
                (bundle.encoder, (hidden, *context)),
                (bundle.pooler, (encoded,)),
                (bundle.post_projection, (pooled,)),
            )
            circles = [
                tico.convert(module, args, strict=False)
                for module, args in modules_and_inputs
            ]
            graphs = []
            for circle in circles:
                document = CircleDocument.from_bytes(circle.circle_binary)
                document.verify()
                graphs.append(document.model.subgraphs[0])
            self.assertEqual(len(graphs[1].inputs), 4)
            encoder = model.vision_tower.encoder.wrapped
            for index, observer in enumerate(
                (
                    encoder.obs_act_in,
                    encoder.obs_attention_mask,
                    encoder.obs_position_cos,
                    encoder.obs_position_sin,
                )
            ):
                tensor = graphs[1].tensors[graphs[1].inputs[index]]
                self.assertEqual(tensor.type, circle_schema.TensorType.TensorType.INT16)
                np.testing.assert_array_equal(
                    tensor.quantization.scale,
                    observer.frozen_state.scale.reshape(-1).numpy(),
                )
                np.testing.assert_array_equal(
                    tensor.quantization.zeroPoint,
                    observer.frozen_state.zero_point.reshape(-1).numpy(),
                )
            for producer, consumer in zip(graphs, graphs[1:]):
                output = producer.tensors[producer.outputs[0]]
                input_ = consumer.tensors[consumer.inputs[0]]
                self.assertEqual(output.type, circle_schema.TensorType.TensorType.INT16)
                self.assertEqual(output.type, input_.type)
                np.testing.assert_array_equal(output.shape, input_.shape)
                self.assertIsNotNone(output.quantization)
                self.assertIsNotNone(input_.quantization)
                np.testing.assert_array_equal(
                    output.quantization.scale, input_.quantization.scale
                )
                np.testing.assert_array_equal(
                    output.quantization.zeroPoint, input_.quantization.zeroPoint
                )
                self.assertEqual(
                    output.quantization.quantizedDimension,
                    input_.quantization.quantizedDimension,
                )
            # FAKE_QUANTIZE is a serialized semantic reference. It exchanges
            # FLOAT32 values and does not claim native integer-backend parity.
            actual = pixels.numpy()
            for index, circle in enumerate(circles):
                args = (actual, *context) if index == 1 else (actual,)
                actual = run_circle(
                    circle.circle_binary,
                    args,
                    mode=ExecutionMode.FAKE_QUANTIZE,
                ).outputs[0]
                self.assertTrue(np.isfinite(actual).all())
            monolithic_circle = tico.convert(bundle.monolithic, (pixels,), strict=False)
            monolithic = run_circle(
                monolithic_circle.circle_binary,
                (pixels,),
                mode=ExecutionMode.FAKE_QUANTIZE,
            ).outputs[0]
            np.testing.assert_array_equal(actual, monolithic)
            np.testing.assert_array_equal(actual, bundle.monolithic(pixels).numpy())
        self._assert_observer_state(before, model)


if __name__ == "__main__":
    unittest.main()

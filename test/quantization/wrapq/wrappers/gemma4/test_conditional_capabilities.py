# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Unavailable Gemma routes reject before any conditional child or export work."""

import unittest
from unittest import mock

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.recipes.debug.export_regression import _state_contract

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    build_gemma4_vision_prefill_export_module,
    Gemma4PLEEmbeddingExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_model import QuantGemma4Model

from test.quantization.wrapq.wrappers.gemma4.test_ple_export_adapters import (
    _make_ple_text_config,
    _make_quant_text_model,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_model import _make_vision_config


def _make_model():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

    config = Gemma4Config(
        text_config=_make_ple_text_config(),
        vision_config=_make_vision_config(),
        audio_config=None,
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
    )
    return QuantGemma4Model(Gemma4Model(config).eval()).eval()


def _calibrate_external(model):
    model.enable_calibration()
    model(
        inputs_embeds=torch.randn(1, 3, 16),
        per_layer_inputs=torch.randn(1, 3, 2, 8),
        use_cache=False,
    )
    model.freeze_qparams()


class TestGemma4ConditionalCapabilities(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1097)
        self.ids = torch.tensor([[1, 2, 3]])

    def _frozen_and_restored(self):
        model = _make_model()
        _calibrate_external(model)
        restored = _make_model()
        restored.load_state_dict(model.state_dict())
        return model, restored

    def test_top_level_token_path_requires_text_owner_before_lookup(self):
        for model in self._frozen_and_restored():
            text = model.language_model.wrapped
            self.assertTrue(text.has_capability("external_embeddings"))
            self.assertFalse(text.has_capability("token_embeddings"))
            with mock.patch.object(
                text.embed_tokens, "forward", wraps=text.embed_tokens.forward
            ) as lookup:
                with self.assertRaisesRegex(
                    RuntimeError, "capability.*token_embeddings"
                ):
                    model(input_ids=self.ids, use_cache=False)
                lookup.assert_not_called()

    def test_unavailable_token_and_ple_export_reject_before_lookup(self):
        for model in self._frozen_and_restored():
            text = model.language_model.wrapped
            for adapter_type, capability, child_name in (
                (Gemma4TokenEmbeddingExportAdapter, "token_embeddings", "embed_tokens"),
                (
                    Gemma4PLEEmbeddingExportAdapter,
                    "token_ple",
                    "embed_tokens_per_layer",
                ),
            ):
                with self.subTest(adapter=adapter_type.__name__):
                    self.assertFalse(text.has_capability(capability))
                    child = getattr(text, child_name)
                    with mock.patch.object(
                        child, "forward", wraps=child.forward
                    ) as lookup:
                        with self.assertRaisesRegex(
                            RuntimeError, f"capability.*{capability}"
                        ):
                            adapter_type(text)(self.ids)
                        lookup.assert_not_called()

    def test_adapters_created_before_freeze_keep_live_owner_guard(self):
        model = _make_model()
        text = model.language_model.wrapped
        token = Gemma4TokenEmbeddingExportAdapter(text)
        ple = Gemma4PLEEmbeddingExportAdapter(text)
        self.assertNotIn(text, tuple(token.modules()))
        self.assertNotIn(text, tuple(ple.modules()))
        _calibrate_external(model)
        for adapter, capability, child in (
            (token, "token_embeddings", text.embed_tokens),
            (ple, "token_ple", text.embed_tokens_per_layer),
        ):
            with mock.patch.object(child, "forward", wraps=child.forward) as lookup:
                with self.assertRaisesRegex(RuntimeError, f"capability.*{capability}"):
                    adapter(self.ids)
                lookup.assert_not_called()

    def test_ready_ple_stages_reject_mode_change_instead_of_skipping_quantization(self):
        _, text = _make_quant_text_model(_make_ple_text_config(), quantized=False)
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEProjectionExportAdapter,
        )

        embedding = Gemma4PLEEmbeddingExportAdapter(text)
        projection = Gemma4PLEProjectionExportAdapter(text)
        text.enable_calibration()
        text(input_ids=self.ids, use_cache=False)
        text.obs_per_layer_token_inputs.load_qparams(
            torch.tensor(0.125), torch.tensor(113), lock=True
        )
        text.freeze_qparams()
        self.assertTrue(text.has_capability("token_ple"))
        for adapter, args, child in (
            (embedding, (self.ids,), text.embed_tokens_per_layer),
            (
                projection,
                (torch.randn(1, 3, 16), torch.randn(1, 3, 2, 8)),
                text.per_layer_model_projection,
            ),
        ):
            with mock.patch.object(child, "forward", wraps=child.forward) as compute:
                with self.assertRaisesRegex(RuntimeError, "mode"):
                    adapter(*args)
                compute.assert_not_called()

    def test_explicit_float_ple_mode_cannot_bypass_quantized_owner(self):
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEProjectionExportAdapter,
        )

        _, text = _make_quant_text_model(_make_ple_text_config(), quantized=True)
        for adapter_type in (
            Gemma4PLEEmbeddingExportAdapter,
            Gemma4PLEProjectionExportAdapter,
        ):
            with self.assertRaisesRegex(RuntimeError, "mode"):
                adapter_type(text, mode=Mode.NO_QUANT)

    def test_unavailable_ple_host_artifact_rejects_before_reading_weight(self):
        from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
            build_gemma4_ple_embedding_artifact,
        )

        model = _make_model()
        adapter = Gemma4PLEEmbeddingExportAdapter(model.language_model.wrapped)
        _calibrate_external(model)
        with mock.patch(
            "tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host._quantized_weight_payload"
        ) as payload:
            with self.assertRaisesRegex(RuntimeError, "capability.*token_ple"):
                build_gemma4_ple_embedding_artifact(adapter)
            payload.assert_not_called()

    def test_unavailable_vision_export_rejects_before_profile_specialization(self):
        positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]])
        for model in self._frozen_and_restored():
            vision = model.vision_tower.wrapped
            self.assertFalse(model.has_capability("vision"))
            with mock.patch.object(
                vision, "as_export_module", wraps=vision.as_export_module
            ) as specialize:
                with self.assertRaisesRegex(RuntimeError, "capability.*vision"):
                    build_gemma4_vision_prefill_export_module(
                        model, pixel_position_ids=positions
                    )
                specialize.assert_not_called()
            with mock.patch.object(vision, "_fq", wraps=vision._fq) as fake_quant:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    vision.forward_export(torch.zeros(1, 4, 48))
                fake_quant.assert_not_called()

    def test_split_vision_entry_rejects_before_borrowed_context_observers(self):
        from tico.quantization.wrapq.wrappers.gemma4.vision_split_export import (
            Gemma4VisionEncoderLayerExportAdapter,
        )

        for model in self._frozen_and_restored():
            encoder = model.vision_tower.wrapped.encoder.wrapped
            with mock.patch.object(
                encoder.obs_attention_mask,
                "fake_quant",
                wraps=encoder.obs_attention_mask.fake_quant,
            ) as quantize_context:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    adapter = Gemma4VisionEncoderLayerExportAdapter(
                        encoder.layers[0],
                        mode=Mode.QUANT,
                        attention_mask_observer=encoder.obs_attention_mask,
                        position_cos_observer=encoder.obs_position_cos,
                        position_sin_observer=encoder.obs_position_sin,
                    )
                    adapter(
                        torch.zeros(1, 4, 32),
                        torch.zeros(1, 1, 4, 4),
                        torch.ones(1, 4, 8),
                        torch.zeros(1, 4, 8),
                    )
                quantize_context.assert_not_called()

    def test_ready_token_and_ple_circle_contracts_survive_owner_restore(self):
        config = _make_ple_text_config()
        _, source = _make_quant_text_model(config, quantized=True)
        _, restored = _make_quant_text_model(config, quantized=False)
        restored.load_state_dict(source.state_dict())
        initial = _state_contract(source)
        for adapter_type, output_shape, observer in (
            (
                Gemma4TokenEmbeddingExportAdapter,
                (1, 3, 16),
                source.embed_tokens.wrapped.obs_act_out,
            ),
            (
                Gemma4PLEEmbeddingExportAdapter,
                (1, 3, 2, 8),
                source.obs_per_layer_token_inputs,
            ),
        ):
            runtimes = []
            outputs = []
            for owner in (source, restored):
                adapter = adapter_type(owner).eval()
                with torch.no_grad():
                    converted = tico.convert(adapter, (self.ids,), strict=False)
                document = CircleDocument.from_bytes(converted.circle_binary)
                self.assertFalse(document.verify().issues)
                runtime = CircleReferenceRuntime(document)
                runtimes.append(runtime)
                input_tensor = runtime.input_tensors[0]
                output_tensor = runtime.output_tensors[0]
                self.assertEqual(input_tensor.shape, (1, 3))
                self.assertEqual(input_tensor.dtype, np.dtype(np.int64))
                self.assertIsNone(input_tensor.quantization)
                self.assertEqual(output_tensor.shape, output_shape)
                self.assertEqual(output_tensor.dtype, np.dtype(np.uint8))
                self.assertIsNotNone(output_tensor.quantization)
                snapshot = observer.require_frozen()
                assert output_tensor.quantization is not None
                assert snapshot.scale is not None and snapshot.zero_point is not None
                self.assertEqual(
                    output_tensor.quantization.scale, (float(snapshot.scale),)
                )
                self.assertEqual(
                    output_tensor.quantization.zero_point, (int(snapshot.zero_point),)
                )
                self.assertFalse(
                    {"QUANTIZE", "DEQUANTIZE"}
                    & {op.name for op in runtime.program.operators}
                )
                actual = runtime.run(
                    [self.ids.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs[0]
                with torch.no_grad():
                    expected = adapter(self.ids).numpy()
                np.testing.assert_array_equal(actual, expected)
                outputs.append(actual)
            before, after = runtimes
            self.assertEqual(
                [
                    (tensor.shape, tensor.dtype, tensor.quantization)
                    for tensor in before.program.tensors
                ],
                [
                    (tensor.shape, tensor.dtype, tensor.quantization)
                    for tensor in after.program.tensors
                ],
            )
            self.assertEqual(
                [(op.name, op.inputs, op.outputs) for op in before.program.operators],
                [(op.name, op.inputs, op.outputs) for op in after.program.operators],
            )
            for before_tensor, after_tensor in zip(
                before.program.tensors, after.program.tensors
            ):
                if before_tensor.is_constant:
                    np.testing.assert_array_equal(
                        before_tensor.constant, after_tensor.constant
                    )
            np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertEqual(initial, _state_contract(source))

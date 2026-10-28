# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Staged exports must preserve the admission policy of their source owner."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tico
import torch

from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaTokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLMultimodalEmbeddingExportAdapter,
    Qwen3VLTextEmbeddingExportAdapter,
    Qwen3VLVisionPrefillExportAdapter,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


@unittest.skipUnless(has_transformers_for("llama"), "Llama requires transformers")
class TestLlamaConditionalExportEntry(unittest.TestCase):
    @staticmethod
    def _model():
        from tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm import (
            QuantLlamaForCausalLM,
        )
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        torch.manual_seed(1097)
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=8,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        return QuantLlamaForCausalLM(
            LlamaForCausalLM(config).eval(),
            qcfg=make_affine_ptq_config(model_args={"profile": "reference_eval"}),
        ).eval()

    @staticmethod
    def _calibrate_external(model):
        model.enable_calibration()
        hidden = torch.linspace(-1, 1, 24).reshape(1, 3, 8)
        model(inputs_embeds=hidden, use_cache=False, return_dict=False)

    def test_unavailable_token_adapter_rejects_before_lookup_after_restore(self):
        source = self._model()
        self._calibrate_external(source)
        source.freeze_qparams()
        restored = self._model()
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
        for model in (source, restored):
            with self.subTest(restored=model is restored):
                owner = model.model.wrapped
                self.assertFalse(owner.has_capability("token_embeddings"))
                with patch.object(
                    owner.embed_tokens, "forward", wraps=owner.embed_tokens.forward
                ) as lookup:
                    with self.assertRaisesRegex(
                        RuntimeError, "token_embeddings.*unavailable"
                    ):
                        adapter = LlamaTokenEmbeddingExportAdapter(model)
                        adapter(torch.tensor([[1, 2, 3]]))
                    lookup.assert_not_called()

    def test_adapter_created_before_freeze_checks_owner_at_execution(self):
        model = self._model()
        self._calibrate_external(model)
        adapter = LlamaTokenEmbeddingExportAdapter(model)
        model.freeze_qparams()
        owner = model.model.wrapped
        with patch.object(
            owner.embed_tokens, "forward", wraps=owner.embed_tokens.forward
        ) as lookup:
            with self.assertRaisesRegex(RuntimeError, "token_embeddings.*unavailable"):
                adapter(torch.tensor([[1, 2, 3]]))
            lookup.assert_not_called()

    def test_available_token_adapter_preserves_real_circle_contract_after_restore(self):
        model = self._model()
        with torch.no_grad():
            values = (torch.arange(128).reshape(16, 8).remainder(16) - 8) * 0.125
            model.model.wrapped.embed_tokens.wrapped.module.weight.copy_(values)
        model.enable_calibration()
        tokens = torch.tensor([[1, 2, 3]])
        model(input_ids=tokens, use_cache=False, return_dict=False)
        embedding = model.model.wrapped.embed_tokens.wrapped
        # Matching explicit imported domains make GATHER's reference semantics
        # directly comparable with eager lookup without changing export policy.
        embedding.weight_obs.load_qparams(
            torch.full((16,), 0.125), torch.full((16,), 113, dtype=torch.int32)
        )
        embedding.act_out_obs.load_qparams(torch.tensor(0.125), torch.tensor(113))
        model.freeze_qparams()
        restored = self._model()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        artifacts = []
        for source in (model, restored):
            adapter = LlamaTokenEmbeddingExportAdapter(source).eval()
            with torch.no_grad():
                exported = tico.convert(adapter, (tokens,), strict=False)
            document = CircleDocument.from_bytes(exported.circle_binary)
            self.assertTrue(document.verify().ok)
            runtime = CircleReferenceRuntime(document)
            self.assertEqual([op.name for op in runtime.program.operators], ["GATHER"])
            gather = runtime.program.operators[0]
            weight = runtime.program.tensor(gather.inputs[0])
            self.assertTrue(weight.is_constant)
            assert weight.quantization is not None
            self.assertEqual((weight.dtype, weight.shape), (np.dtype("uint8"), (16, 8)))
            self.assertEqual(weight.quantization.scale, (0.125,) * 16)
            self.assertEqual(weight.quantization.zero_point, (113,) * 16)
            self.assertEqual(weight.quantization.quantized_dimension, 0)
            (input_tensor,) = runtime.input_tensors
            (output_tensor,) = runtime.output_tensors
            assert output_tensor.quantization is not None
            self.assertEqual(
                (input_tensor.dtype, input_tensor.shape), (np.dtype("int64"), (1, 3))
            )
            self.assertIsNone(input_tensor.quantization)
            self.assertEqual(
                (output_tensor.dtype, output_tensor.shape),
                (np.dtype("uint8"), (1, 3, 8)),
            )
            self.assertEqual(output_tensor.quantization.scale, (0.125,))
            self.assertEqual(output_tensor.quantization.zero_point, (113,))
            actual = runtime.run(
                [tokens.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
            ).outputs[0]
            np.testing.assert_array_equal(actual, adapter(tokens).detach().numpy())
            artifacts.append((runtime.program, actual))
        left, right = artifacts
        self.assertEqual(
            [(t.shape, t.dtype, t.quantization) for t in left[0].tensors],
            [(t.shape, t.dtype, t.quantization) for t in right[0].tensors],
        )
        for original, reloaded in zip(left[0].tensors, right[0].tensors):
            if original.is_constant:
                np.testing.assert_array_equal(original.constant, reloaded.constant)
        np.testing.assert_array_equal(left[1], right[1])


class _TextContainer(torch.nn.Module):
    def __init__(self, text):
        super().__init__()
        self.wrapped: QuantModuleBase = text


class _OptionalVisionOwner(QuantModuleBase):
    language_model: _TextContainer

    def __init__(self, vision, text=None):
        super().__init__()
        self.visual = vision
        if text is not None:
            self.language_model = _TextContainer(text)

    def _all_observers(self):
        return ()

    def _conditional_quant_subtrees(self):
        return {"vision": ("visual",)}


@unittest.skipUnless(has_transformers_for("qwen3-vl"), "Qwen3-VL requires transformers")
class TestQwenConditionalExportEntry(unittest.TestCase):
    @staticmethod
    def _text():
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_model import (
            QuantQwen3VLTextModel,
        )
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

        torch.manual_seed(1097)
        config = Qwen3VLTextConfig.from_dict(
            dict(
                vocab_size=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=8,
                rope_scaling={"rope_type": "default", "mrope_section": [1, 0, 1]},
                use_cache=False,
            )
        )
        config._attn_implementation = "eager"
        return QuantQwen3VLTextModel(
            Qwen3VLTextModel(config).eval(), qcfg=make_affine_ptq_config()
        ).eval()

    @staticmethod
    def _vision():
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_model import (
            QuantQwen3VLVisionModel,
        )
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        config = Qwen3VLVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            depth=1,
            num_heads=2,
            out_hidden_size=8,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
            num_position_embeddings=16,
        )
        config._attn_implementation = "eager"
        return QuantQwen3VLVisionModel(
            Qwen3VLVisionModel(config).eval(), qcfg=make_affine_ptq_config()
        ).eval()

    @staticmethod
    def _top(owner):
        return SimpleNamespace(model=SimpleNamespace(wrapped=owner))

    def test_token_adapter_rejects_unavailable_owner_before_lookup_after_restore(self):
        model = self._text()
        model.enable_calibration()
        model(inputs_embeds=torch.linspace(-1, 1, 24).reshape(1, 3, 8), use_cache=False)
        model.freeze_qparams()
        restored = self._text()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        for source in (model, restored):
            owner = SimpleNamespace(language_model=SimpleNamespace(wrapped=source))
            with patch.object(
                source.embed_tokens, "forward", wraps=source.embed_tokens.forward
            ) as lookup:
                with self.assertRaisesRegex(
                    RuntimeError, "token_embeddings.*unavailable"
                ):
                    adapter = Qwen3VLTextEmbeddingExportAdapter(self._top(owner))
                    adapter(torch.tensor([[1, 2, 3]]))
                lookup.assert_not_called()

    def test_multimodal_adapter_requires_vision_before_token_lookup(self):
        owner = _OptionalVisionOwner(self._vision(), self._text())
        owner.enable_calibration()
        owner.language_model.wrapped(
            input_ids=torch.tensor([[1, 2, 3]]), use_cache=False
        )
        adapter_before_freeze = Qwen3VLMultimodalEmbeddingExportAdapter(
            self._top(owner), visual_start_idx=0
        )
        owner.freeze_qparams()
        restored = _OptionalVisionOwner(self._vision(), self._text())
        restored.load_state_dict(copy.deepcopy(owner.state_dict()))
        self.assertFalse(owner.has_capability("vision"))
        self.assertTrue(owner.language_model.wrapped.has_capability("token_embeddings"))
        for source in (owner, restored):
            with patch.object(
                source.language_model.wrapped.embed_tokens, "forward"
            ) as lookup:
                with self.assertRaisesRegex(RuntimeError, "vision.*unavailable"):
                    Qwen3VLMultimodalEmbeddingExportAdapter(
                        self._top(source), visual_start_idx=0
                    )
                lookup.assert_not_called()
        with patch.object(
            owner.language_model.wrapped.embed_tokens, "forward"
        ) as lookup:
            with self.assertRaisesRegex(RuntimeError, "vision.*unavailable"):
                adapter_before_freeze(torch.tensor([[1, 2, 3]]), torch.zeros(1, 8))
            lookup.assert_not_called()

    def test_direct_vision_export_rejects_unavailable_subtree_before_profile_work(self):
        owner = _OptionalVisionOwner(self._vision())
        owner.enable_calibration()
        owner.freeze_qparams()
        restored = _OptionalVisionOwner(self._vision())
        restored.load_state_dict(copy.deepcopy(owner.state_dict()))
        for source in (owner, restored):
            with patch.object(
                source.visual, "_fast_pos_embed_interpolate"
            ) as positions:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    Qwen3VLVisionPrefillExportAdapter(source.visual, grid_thw=[1, 4, 4])
                positions.assert_not_called()
            with patch.object(source.visual, "_forward_impl") as compute:
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    source.visual.forward_export(
                        torch.zeros(16, 12),
                        pos_embeds=torch.zeros(16, 8),
                        position_embeddings=(torch.zeros(16, 4), torch.zeros(16, 4)),
                        cu_seqlens=torch.tensor([0, 16]),
                        attention_split_sizes=(16,),
                    )
                compute.assert_not_called()

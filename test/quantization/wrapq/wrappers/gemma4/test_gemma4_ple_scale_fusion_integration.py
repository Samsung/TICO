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

"""Run with the normal TICO test environment; no remote models are needed."""

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import tico

import torch
import torch.nn as nn
from circle_schema import circle
from tico.quantization import convert, prepare
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.embedding_scale_fusion import (
    fuse_gemma4_ple_embedding_scale,
    FusedGemma4PLEEmbedding,
)
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4PLEEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
    build_gemma4_ple_embedding_artifact,
    Gemma4PLEEmbeddingHostTable,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_scaled_word_embedding import (
    QuantGemma4TextScaledWordEmbedding,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.quantization.wrapq.wrappers.gemma4.test_gemma4_ple_scale_fusion import (
    make_model,
)


def make_adapter(fused=True, quantized=True):
    """Use the real wrapper, observers and PLE adapter around a tiny FP table."""
    torch.manual_seed(17)
    model = make_model()
    if fused:
        fuse_gemma4_ple_embedding_scale(model)
    qcfg = make_affine_ptq_config(
        dtype=DType.int(16), overrides={"weight": {"dtype": DType.uint(4)}}
    )
    wrapper = QuantGemma4TextScaledWordEmbedding(
        model.embed_tokens_per_layer, qcfg=qcfg
    ).eval()
    if quantized:
        wrapper.enable_calibration()
        with torch.no_grad():
            wrapper(torch.arange(19).reshape(1, 19))
        wrapper.freeze_qparams()
    text = nn.Module()
    text.embed_tokens_per_layer = wrapper
    text.hidden_size_per_layer_input = 4
    text.config = SimpleNamespace(num_hidden_layers=3)
    # A separate frozen observer with the same per-tensor contract is suitable
    # for this shape-only boundary in the synthetic text model.
    text.obs_per_layer_token_inputs = copy.deepcopy(wrapper.obs_act_out)
    text._mode = Mode.QUANT if quantized else Mode.NO_QUANT
    return Gemma4PLEEmbeddingExportAdapter(text).eval()


class TestGemma4PLEScaleFusionIntegration(unittest.TestCase):
    def test_public_prepare_convert_selects_scaled_wrapper(self):
        torch.manual_seed(17)
        model = make_model()
        fuse_gemma4_ple_embedding_scale(model)
        qcfg = make_affine_ptq_config(dtype=DType.uint(8))
        prepared = prepare(model.embed_tokens_per_layer, qcfg)
        self.assertIsInstance(prepared.wrapped, QuantGemma4TextScaledWordEmbedding)
        self.assertIsInstance(prepared.wrapped.module, FusedGemma4PLEEmbedding)
        with torch.no_grad():
            prepared(torch.arange(19).reshape(1, 19))
        quantized = convert(prepared)
        self.assertIs(quantized.wrapped._mode, Mode.QUANT)
        with self.assertRaisesRegex(RuntimeError, "fresh FP model"):
            fuse_gemma4_ple_embedding_scale(quantized)

    def test_calibration_observes_already_folded_weights(self):
        model = make_model()
        expected = model.embed_tokens_per_layer.weight.detach().clone() * 16
        fuse_gemma4_ple_embedding_scale(model)
        wrapper = QuantGemma4TextScaledWordEmbedding(model.embed_tokens_per_layer)
        with mock.patch.object(
            wrapper.obs_weight, "collect", wraps=wrapper.obs_weight.collect
        ) as collect:
            wrapper.enable_calibration()
        collect.assert_called_once()
        torch.testing.assert_close(collect.call_args.args[0], expected, atol=0, rtol=0)

    def test_adapter_host_parity_for_both_formats_and_modes(self):
        for fused in (True, False):
            for quantized in (True, False):
                with self.subTest(fused=fused, quantized=quantized):
                    adapter = make_adapter(fused, quantized)
                    artifact = build_gemma4_ple_embedding_artifact(adapter)
                    self.assertEqual(artifact["schema_version"], 2 if fused else 1)
                    host = Gemma4PLEEmbeddingHostTable(artifact)
                    for seq in (1, 4, 16):
                        ids = torch.arange(seq).reshape(1, seq)
                        with torch.no_grad():
                            torch.testing.assert_close(
                                host(ids), adapter(ids), atol=0, rtol=0
                            )

    def test_quantized_wrapper_never_calls_scale_fake_quant(self):
        adapter = make_adapter()
        wrapper = adapter.embed_tokens_per_layer
        with mock.patch.object(
            wrapper.obs_embed_scale,
            "fake_quant",
            side_effect=AssertionError("The folded scale must not be executed."),
        ):
            with torch.no_grad():
                adapter(torch.tensor([[1, 2, 3]]))

    def test_dynamic_quantized_export_has_no_mul(self):
        adapter = make_adapter()
        ids = torch.tensor([[1, 2, 3, 4]])
        ep = torch.export.export(
            adapter,
            (ids,),
            dynamic_shapes={"input_ids": {1: torch.export.Dim("seq", min=1, max=16)}},
        )
        targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        self.assertIn("aten.embedding.default", targets)
        self.assertFalse(any(t.startswith("aten.mul.") for t in targets))
        for seq in (1, 4, 16):
            tokens = torch.arange(seq).reshape(1, seq)
            with torch.no_grad():
                torch.testing.assert_close(
                    ep.module()(tokens), adapter(tokens), atol=0, rtol=0
                )

    def test_circle_contains_gather_but_no_mul(self):
        # Inspect the final serialized operators, not merely the Torch graph.
        for quantized in (False, True):
            adapter = make_adapter(quantized=quantized)
            ep = torch.export.export(adapter, (torch.tensor([[1, 2, 3, 4]]),))
            converted = tico.convert_from_exported_program(ep)
            model = circle.Model.Model.GetRootAsModel(converted.circle_binary, 0)
            graph = model.Subgraphs(0)
            codes = []
            for index in range(graph.OperatorsLength()):
                opcode = model.OperatorCodes(graph.Operators(index).OpcodeIndex())
                codes.append(max(opcode.BuiltinCode(), opcode.DeprecatedBuiltinCode()))
            self.assertIn(circle.BuiltinOperator.BuiltinOperator.GATHER, codes)
            self.assertNotIn(circle.BuiltinOperator.BuiltinOperator.MUL, codes)

    def test_prepared_state_dict_roundtrip(self):
        adapter = make_adapter()
        restored = make_adapter()
        with torch.no_grad():
            restored.embed_tokens_per_layer.module.weight.zero_()
        restored.load_state_dict(adapter.state_dict(), strict=True)
        ids = torch.tensor([[1, 5, 18]])
        with torch.no_grad():
            torch.testing.assert_close(restored(ids), adapter(ids), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

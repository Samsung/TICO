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

"""Selective lookup parity against the original complete producer contract."""

import unittest
from typing import Any
from unittest import mock

import torch

from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.wrappers.gemma4.embedding_scale_fusion import (
    fuse_gemma4_ple_embedding_scale,
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
from test.quantization.wrapq.wrappers.gemma4.test_gemma4_ple_scale_fusion_integration import (
    _PLETextModel,
)


def make_text(*, fused, dtype, identity=False, weight_axis=0):
    torch.manual_seed(29)
    model = make_model()
    if fused:
        fuse_gemma4_ple_embedding_scale(model)
    from tico.quantization.config.specs import no_quant

    overrides: dict[str, Any] = (
        {name: no_quant() for name in ("embedding", "embed_scale", "act_out")}
        if identity
        else {}
    )
    overrides["weight"] = {"channel_axis": weight_axis}
    wrapper = QuantGemma4TextScaledWordEmbedding(
        model.embed_tokens_per_layer,
        qcfg=make_affine_ptq_config(dtype=dtype or DType.uint(8), overrides=overrides),
    ).eval()
    if dtype is not None:
        wrapper.enable_calibration()
        with torch.no_grad():
            wrapper(torch.arange(19).reshape(1, 19))
        wrapper.freeze_qparams()
    return _PLETextModel(wrapper).eval()


class TestGemma4PLEEmbeddingSelection(unittest.TestCase):
    def test_selected_lookup_preserves_full_result_and_shared_storage(self):
        ids = torch.tensor([[0, 3, 3, 17, 0, 0]])
        for fused in (False, True):
            for dtype in (None, DType.uint(8), DType.int(16)):
                with self.subTest(fused=fused, dtype=dtype):
                    text = make_text(fused=fused, dtype=dtype)
                    full = Gemma4PLEEmbeddingExportAdapter(text)
                    state = _state_contract(text) if dtype is not None else None
                    prefix = Gemma4PLEEmbeddingExportAdapter(text, layer_end=1)
                    suffix = Gemma4PLEEmbeddingExportAdapter(text, layer_start=1)
                    decode = Gemma4PLEEmbeddingExportAdapter(text, selective=True)
                    self.assertIs(
                        full.embed_tokens_per_layer, prefix.embed_tokens_per_layer
                    )
                    self.assertIs(
                        full.embed_tokens_per_layer, suffix.embed_tokens_per_layer
                    )
                    table = Gemma4PLEEmbeddingHostTable(
                        build_gemma4_ple_embedding_artifact(full)
                    )
                    with torch.no_grad():
                        expected = full(ids)
                        # Native fake-quant must see only S rows and K*P columns.
                        with mock.patch.object(
                            text.embed_tokens_per_layer.obs_weight,
                            "fake_quant",
                            side_effect=AssertionError("full weight quantization"),
                        ):
                            torch.testing.assert_close(
                                prefix(ids), expected[:, :, :1], atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                suffix(ids[:, 3:4]),
                                expected[:, 3:4, 1:],
                                atol=0,
                                rtol=0,
                            )
                            torch.testing.assert_close(
                                decode(ids[:, :1]), expected[:, :1], atol=0, rtol=0
                            )
                        with mock.patch.object(
                            table,
                            "_dequantized_weight",
                            side_effect=AssertionError("full dequantization"),
                        ):
                            torch.testing.assert_close(
                                table(ids, layer_end=1),
                                expected[:, :, :1],
                                atol=0,
                                rtol=0,
                            )
                            torch.testing.assert_close(
                                table(ids[:, 3:4], layer_start=1),
                                expected[:, 3:4, 1:],
                                atol=0,
                                rtol=0,
                            )
                            torch.testing.assert_close(
                                table(ids[:, :1]), expected[:, :1], atol=0, rtol=0
                            )
                    if state is not None:
                        self.assertEqual(state, _state_contract(text))

    def test_weight_fake_quant_kernel_only_sees_selected_rows_and_columns(self):
        text = make_text(fused=False, dtype=DType.uint(8))
        ids = torch.tensor([[0, 3, 3, 17, 0, 0]])
        prefix = Gemma4PLEEmbeddingExportAdapter(text, layer_end=1)
        suffix = Gemma4PLEEmbeddingExportAdapter(text, layer_start=1)
        native = torch.fake_quantize_per_channel_affine
        with (
            torch.no_grad(),
            mock.patch.object(
                torch, "fake_quantize_per_channel_affine", wraps=native
            ) as kernel,
        ):
            prefix(ids)
            suffix(ids[:, 3:4])
        self.assertEqual(
            [tuple(call.args[0].shape) for call in kernel.call_args_list],
            [(6, 4), (1, 8)],
        )
        self.assertEqual(
            [tuple(call.kwargs["scale"].shape) for call in kernel.call_args_list],
            [(6,), (1,)],
        )

    def test_identity_boundaries_are_shared_without_inventing_qparams(self):
        for fused in (False, True):
            text = make_text(fused=fused, dtype=DType.uint(8), identity=True)
            state = _state_contract(text)
            full = Gemma4PLEEmbeddingExportAdapter(text)
            selected = Gemma4PLEEmbeddingExportAdapter(text, layer_start=1)
            artifact = build_gemma4_ple_embedding_artifact(full)
            for payload in artifact["observers"].values():
                self.assertEqual(payload["kind"], "identity")
                self.assertFalse(payload["fake_quant_enabled"])
                self.assertEqual(payload["scale"].numel(), 0)
            host = Gemma4PLEEmbeddingHostTable(artifact)
            ids = torch.tensor([[2, 2, 0]])
            with torch.no_grad():
                expected = full(ids)[:, :, 1:]
                torch.testing.assert_close(selected(ids), expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    host(ids, layer_start=1), expected, atol=0, rtol=0
                )
            self.assertEqual(state, _state_contract(text))

    def test_column_weight_grids_are_sliced_without_reestimation(self):
        for axis in (1, -1, -2):
            text = make_text(fused=False, dtype=DType.int(16), weight_axis=axis)
            full = Gemma4PLEEmbeddingExportAdapter(text)
            suffix = Gemma4PLEEmbeddingExportAdapter(text, layer_start=1)
            host = Gemma4PLEEmbeddingHostTable(
                build_gemma4_ple_embedding_artifact(full)
            )
            ids = torch.tensor([[0, 2, 2, 18]])
            with torch.no_grad():
                expected = full(ids)[:, :, 1:]
                torch.testing.assert_close(suffix(ids), expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    host(ids, layer_start=1), expected, atol=0, rtol=0
                )

    def test_host_gathers_row_qparams_before_dequantization(self):
        from test.quantization.wrapq.wrappers.gemma4.test_gemma4_ple_scale_fusion_host import (
            payload,
        )

        artifact = payload()
        artifact["weight_scale"] = torch.arange(1, 9).float() / 16
        artifact["weight_zero_point"] = torch.arange(8, dtype=torch.int)
        host = Gemma4PLEEmbeddingHostTable(artifact)
        self.assertEqual(host.weight_int.data_ptr(), artifact["weight_int"].data_ptr())
        ids = torch.tensor([[7, 1, 7, 0]])
        # Independent element-wise formula from original integer storage.
        expected = torch.empty(1, 4, 3)
        for token_index, token in enumerate(ids[0].tolist()):
            expected[0, token_index] = (
                artifact["weight_int"][token, 3:].float() - token
            ) * ((token + 1) / 16)
        torch.testing.assert_close(
            host(ids, layer_start=1).squeeze(2),
            expected.div(0.125).round().mul(0.125),
            atol=0,
            rtol=0,
        )

    def test_invalid_range_and_mx_activation_reject_without_source_mutation(self):
        text = make_text(fused=True, dtype=DType.uint(8))
        before = _state_contract(text)
        with self.assertRaisesRegex(ValueError, "layer range"):
            Gemma4PLEEmbeddingExportAdapter(text, layer_start=3)
        self.assertEqual(before, _state_contract(text))
        embedding = text.embed_tokens_per_layer
        original = embedding.obs_embedding
        embedding.obs_embedding = MXObserver(name="embedding")
        with self.assertRaisesRegex(ValueError, "per-tensor affine/identity"):
            Gemma4PLEEmbeddingExportAdapter(text, layer_end=1)
        embedding.obs_embedding = original
        embedding.obs_weight = MXObserver(name="weight")
        with self.assertRaisesRegex(ValueError, "affine weight policy"):
            Gemma4PLEEmbeddingExportAdapter(text, layer_end=1)

    def test_range_adapter_cannot_duplicate_table_artifact(self):
        text = make_text(fused=True, dtype=DType.uint(8))
        adapter = Gemma4PLEEmbeddingExportAdapter(text, layer_end=1)
        with self.assertRaisesRegex(ValueError, "one shared full PLE table"):
            build_gemma4_ple_embedding_artifact(adapter)


if __name__ == "__main__":
    unittest.main()

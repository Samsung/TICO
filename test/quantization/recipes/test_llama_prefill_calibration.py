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

"""Prefill alone must calibrate both persistent KV and attention domains."""

import unittest

import tico

import torch
from tico.circle import CircleDocument
from tico.quantization import convert, prepare
from tico.quantization.config import affine
from tico.quantization.config.builders import build_llm_ptq_config
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)


@unittest.skipUnless(has_transformers_for("llama"), "Required Transformers unavailable")
class TestLlamaPrefillCalibration(unittest.TestCase):
    def test_prefill_only_freezes_and_exports_both_cache_modes(self):
        """No decode calls or injected statistics are needed, even with use_cache off."""
        from transformers import LlamaConfig, LlamaForCausalLM

        for dtype in (DType.uint(8), DType.int(16)):
            for observer_type in (MinMaxObserver, PercentileObserver):
                with self.subTest(dtype=dtype, observer=observer_type.__name__):
                    torch.manual_seed(1083)
                    config = LlamaConfig(
                        hidden_size=16,
                        intermediate_size=32,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=8,
                        max_position_embeddings=8,
                        vocab_size=32,
                        attention_bias=False,
                        attention_dropout=0.0,
                        use_cache=False,
                    )
                    config._attn_implementation = "eager"
                    model = LlamaForCausalLM(config).eval()
                    adapter = LlamaAdapter()
                    inputs = torch.tensor([[1, 4, 9, 2]])
                    ctx = RecipeContext(
                        cfg={
                            "calibration": {"decode_steps": 0},
                            "runtime": {"show_progress": False},
                        },
                        adapter=adapter,
                        model=model,
                        calibration_inputs=[inputs],
                    )
                    activation = affine(dtype, observer=observer_type)
                    qcfg = build_llm_ptq_config(
                        model_type="llama",
                        num_hidden_layers=1,
                        activation=activation,
                        profile="npu_export",
                    )
                    prepared = prepare(model, qcfg)
                    calls = []

                    def record(_module, args, kwargs):
                        ids = kwargs.get("input_ids", args[0] if args else None)
                        calls.append((ids.clone(), dict(kwargs)))

                    handle = prepared.register_forward_pre_hook(
                        record, with_kwargs=True
                    )
                    try:
                        adapter.calibrate_prepared_model(ctx, prepared, {})
                    finally:
                        handle.remove()
                    self.assertEqual(len(calls), 1)
                    torch.testing.assert_close(calls[0][0], inputs, rtol=0, atol=0)
                    self.assertTrue(calls[0][1]["use_cache"])
                    self.assertNotIn("past_key_values", calls[0][1])

                    # Inspect the functional lifecycle's observers rather than
                    # dormant registered attributes that the profile never uses.
                    active = list(prepared.wrapped.named_observers())
                    self.assertTrue(active)
                    for name, observer in active:
                        with self.subTest(observer=name):
                            if isinstance(observer, AffineObserverBase):
                                self.assertTrue(torch.isfinite(observer.min_val).all())
                                self.assertTrue(torch.isfinite(observer.max_val).all())

                    quantized = convert(prepared)
                    layer = quantized.wrapped.model.wrapped.layers[0].wrapped
                    attention = layer.self_attn.wrapped
                    self.assertIs(attention._mode, Mode.QUANT)
                    for observer in (
                        attention.obs_new_k,
                        attention.obs_new_v,
                        attention.obs_present_key,
                        attention.obs_present_value,
                    ):
                        self.assertIsInstance(observer, observer_type)
                        self.assertEqual(observer.dtype, dtype)
                        self.assertTrue(observer.has_qparams)
                        self.assertFalse(observer.enabled)

                    hidden = torch.linspace(-0.4, 0.4, 64).reshape(1, 4, 16)
                    rope = (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
                    mask = torch.triu(torch.full((1, 4, 4), -120.0), diagonal=1)
                    prefill = layer.as_export_module("prefill", return_kv=True).eval()
                    with torch.no_grad():
                        _, past_k, past_v = prefill(hidden, mask, rope)
                    prefill_args = (hidden, mask, rope)
                    decode_args = (
                        hidden[:, :1],
                        torch.zeros(1, 1, 5),
                        (rope[0][:, :1], rope[1][:, :1]),
                        (past_k, past_v),
                    )
                    register_fake_quant_meta_kernels_for_dynamic_export()
                    for mode, args in (
                        ("prefill", prefill_args),
                        ("decode", decode_args),
                    ):
                        with self.subTest(mode=mode):
                            exported = tico.convert(
                                layer.as_export_module(mode, return_kv=True).eval(),
                                args,
                                strict=False,
                            )
                            document = CircleDocument.from_bytes(exported.circle_binary)
                            document.verify()
                            self.assertTrue(exported.circle_binary)

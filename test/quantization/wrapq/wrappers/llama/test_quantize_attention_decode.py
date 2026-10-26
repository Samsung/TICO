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

"""Smoke tests migrated from the Llama attention decode quantization example."""

import unittest

import torch
import yaml

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.utils import quant_spec_from_config
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

skip_msg = "required transformers not installed — skipping Llama attention decode example tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestLlamaAttentionDecodeExample(unittest.TestCase):
    """Exercise the prepare-calibrate-convert flow for one decode attention step."""

    def setUp(self):
        """Create a tiny deterministic LlamaAttention module."""
        torch.manual_seed(123)
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        self.max_seq = 16
        self.cfg = LlamaConfig.from_dict(
            {
                "hidden_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "intermediate_size": 32,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "attn_implementation": "eager",
                "max_position_embeddings": self.max_seq,
            }
        )
        self.fp_attn = LlamaAttention(self.cfg, layer_idx=0).eval()

    def _make_decode_batch(self, qattn=None):
        """Create static decode inputs matching the old example contract."""
        batch_size = 1
        hidden = torch.randn(batch_size, 1, self.cfg.hidden_size)
        assert self.cfg.head_dim is not None
        cos = torch.randn(batch_size, 1, self.cfg.head_dim)
        assert self.cfg.head_dim is not None
        sin = torch.randn(batch_size, 1, self.cfg.head_dim)
        if qattn is not None and qattn.wrapped.attn_options.rope == "pre_negated_sin":
            sin = sin.clone()
            assert self.cfg.head_dim is not None
            sin[..., : self.cfg.head_dim // 2] = -sin[..., : self.cfg.head_dim // 2]
        mask = torch.zeros(batch_size, 1, self.max_seq)
        assert self.cfg.num_key_value_heads is not None
        assert self.cfg.head_dim is not None
        past = (
            torch.randn(
                batch_size,
                self.cfg.num_key_value_heads,
                self.max_seq - 1,
                self.cfg.head_dim,
            ),
            torch.randn(
                batch_size,
                self.cfg.num_key_value_heads,
                self.max_seq - 1,
                self.cfg.head_dim,
            ),
        )
        return hidden, (cos, sin), mask, past

    def test_prepare_convert_decode_flow_matches_example(self):
        """Quantize one Llama attention block and run a decode-step sanity check."""
        from tico.quantization.wrapq.wrappers.llama.quant_attention import (
            QuantLlamaAttention,
        )

        prepared = prepare(self.fp_attn, PTQConfig())
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantLlamaAttention)
        self.assertIs(prepared._mode, Mode.CALIB)

        with torch.no_grad():
            for _ in range(4):
                hidden, pos, mask, past = self._make_decode_batch(prepared)
                float_like_out = prepared(
                    hidden,
                    pos,
                    attention_mask=mask,
                    past_key_value=past,
                    use_cache=True,
                )

        quantized = convert(prepared)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            quant_out = quantized(
                hidden,
                pos,
                attention_mask=mask,
                past_key_value=past,
                use_cache=True,
            )

        self.assertEqual(quant_out[0].shape, float_like_out[0].shape)
        self.assertTrue(torch.isfinite(quant_out[0]).all())
        self.assertGreater((quant_out[0] - float_like_out[0]).abs().mean().item(), 0.0)

    def _prepare_percentile(self, model_args):
        """Parse a YAML activation policy and calibrate every active cache site."""
        config = yaml.safe_load("""
activation:
  kind: affine
  dtype: int16
  observer: percentile
  percentile: 99.9
  max_samples: 256
  samples_per_batch: 256
""")
        activation = quant_spec_from_config(config["activation"])
        assert activation is not None
        prepared = prepare(
            self.fp_attn, PTQConfig(activation=activation, model_args=model_args)
        )
        qattn = prepared.wrapped
        self.assertIsInstance(qattn.obs_logits, PercentileObserver)
        self.assertEqual(qattn.obs_logits.percentile, 99.9)
        self.assertIsInstance(qattn.k_proj.wrapped.obs_weight, MinMaxObserver)
        inputs = self._make_decode_batch(prepared)
        with torch.no_grad():
            prepared(*inputs, use_cache=True)
        return prepared, inputs

    def test_percentile_calibration_respects_attention_scale_fusion(self):
        """Only graph-active observers participate in conversion for each profile."""
        for model_args in (
            {},
            {"profile": "npu_export"},
            {"attention": {"scale_fusion": "q_proj"}},
            {"profile": "reference_eval"},
            {"profile": "npu_export", "attention": {"scale_fusion": "none"}},
        ):
            with self.subTest(model_args=model_args):
                prepared, inputs = self._prepare_percentile(model_args)
                qattn = prepared.wrapped
                state_keys = set(qattn.state_dict())
                quantized = convert(prepared)
                self.assertIs(quantized._mode, Mode.QUANT)
                self.assertIs(qattn._mode, Mode.QUANT)
                active = dict(qattn.named_observers(recurse=False))
                fused = qattn.attn_options.scale_fusion != "none"
                for name in ("logits_raw", "scale"):
                    observer = getattr(qattn, f"obs_{name}")
                    self.assertEqual(name in active, not fused)
                    self.assertEqual(observer.has_qparams, not fused)
                    self.assertEqual(observer.sampled_value_count == 0, fused)
                    self.assertIn(f"obs_{name}.min_val", state_keys)
                    self.assertIn(f"obs_{name}.max_val", state_keys)
                self.assertEqual(set(qattn.state_dict()), state_keys)
                for name, observer in qattn.named_observers():
                    self.assertTrue(observer.has_qparams, name)
                    scale, zero_point = observer.compute_qparams()
                    self.assertTrue(torch.isfinite(scale).all(), name)
                    self.assertTrue((scale > 0).all(), name)
                    self.assertTrue(torch.isfinite(zero_point).all(), name)
                with torch.no_grad():
                    output = quantized(*inputs, use_cache=True)
                self.assertEqual(output[0].shape, inputs[0].shape)
                self.assertTrue(torch.isfinite(output[0]).all())

    def test_percentile_conversion_rejects_missing_active_calibration(self):
        """Skipping fused sites must not hide missing statistics on active sites."""
        for model_args, name in (
            ({}, "logits"),
            ({"profile": "reference_eval"}, "logits_raw"),
            ({"profile": "reference_eval"}, "scale"),
        ):
            with self.subTest(model_args=model_args, observer=name):
                prepared, _ = self._prepare_percentile(model_args)
                getattr(prepared.wrapped, f"obs_{name}").reset()
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"No calibration values were collected for observer '{name}'",
                ):
                    convert(prepared)

    def test_percentile_npu_profile_exports_prefill_and_decode(self):
        """Frozen Percentile qparams support both NPU-profile export adapters."""
        prepared, decode_inputs = self._prepare_percentile({})
        quantized = convert(prepared)
        hidden, rope, _, _ = decode_inputs
        prefill_inputs = (
            hidden.repeat(1, 2, 1),
            tuple(value.repeat(1, 2, 1) for value in rope),
            torch.zeros(1, 2, 2),
        )
        register_fake_quant_meta_kernels_for_dynamic_export()
        for mode, inputs in (("prefill", prefill_inputs), ("decode", decode_inputs)):
            with self.subTest(mode=mode), torch.no_grad():
                adapter = quantized.wrapped.as_export_module(
                    mode, return_kv=True, require_npu_profile=True
                )
                expected = adapter(*inputs)
                exported = torch.export.export(adapter, inputs, strict=False)
                actual = exported.module()(*inputs)
                self.assertEqual(len(actual), 3)
                self.assertEqual(actual[0].shape, inputs[0].shape)
                assert self.cfg.num_key_value_heads is not None
                assert self.cfg.head_dim is not None
                self.assertEqual(
                    actual[1].shape,
                    (
                        1,
                        self.cfg.num_key_value_heads,
                        inputs[0].shape[1],
                        self.cfg.head_dim,
                    ),
                )
                for got, want in zip(actual, expected):
                    self.assertTrue(torch.isfinite(got).all())
                    torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

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

"""Synthetic regression tests for Gemma4 RoPE and identity-scale handling."""

import copy
import unittest
from types import SimpleNamespace
from typing import Any, cast, Dict, Tuple
from unittest.mock import patch

import torch

from tico.quantization.config.gemma4_attention import (
    AttentionLayout,
    Gemma4TextAttentionOptions,
    get_gemma4_text_attention_options,
    is_npu_export_text_attention_options,
    RopeConvention,
)
from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
    QuantGemma4TextDecoderLayer,
)
from tico.quantization.wrapq.wrappers.gemma4.rope import prepare_gemma4_rope_sin

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _AttentionExportAdapter,
    _has_gemma4,
    _make_text_config,
)


class TestGemma4RoPEOptions(unittest.TestCase):
    def test_defaults_and_layout_rope_combinations(self):
        expected_default = Gemma4TextAttentionOptions(
            layout="unrolled", rope="pre_negated_sin"
        )
        self.assertEqual(
            Gemma4TextAttentionOptions(),
            Gemma4TextAttentionOptions(layout="unrolled", rope="hf"),
        )
        self.assertEqual(get_gemma4_text_attention_options(None), expected_default)
        self.assertEqual(
            get_gemma4_text_attention_options(PTQConfig()), expected_default
        )
        for profile, rope in (
            ("reference_eval", "hf"),
            ("npu_export", "pre_negated_sin"),
        ):
            cfg = PTQConfig(model_args={"profile": profile})
            self.assertEqual(get_gemma4_text_attention_options(cfg).rope, rope)
        layouts: Tuple[AttentionLayout, ...] = ("batched", "unrolled")
        ropes: Tuple[RopeConvention, ...] = ("hf", "pre_negated_sin")
        for layout in layouts:
            for rope in ropes:
                with self.subTest(layout=layout, rope=rope):
                    cfg = PTQConfig(
                        model_args={"attention": {"layout": layout, "rope": rope}}
                    )
                    options = get_gemma4_text_attention_options(cfg)
                    self.assertEqual(options, Gemma4TextAttentionOptions(layout, rope))
                    self.assertEqual(
                        is_npu_export_text_attention_options(options),
                        layout == "unrolled" and rope == "pre_negated_sin",
                    )

    def test_profile_aliases_and_explicit_override_precedence(self):
        cases: Tuple[Tuple[Dict[str, Any], str, str], ...] = (
            ({"attention": "npu_export"}, "unrolled", "pre_negated_sin"),
            ({"attention": {"profile": "npu_export"}}, "unrolled", "pre_negated_sin"),
            (
                {"profile": "reference_eval", "attention": "npu_export"},
                "unrolled",
                "pre_negated_sin",
            ),
            ({"profile": "npu_export", "attention": "reference_eval"}, "batched", "hf"),
            ({"profile": "npu_export", "attention": {"rope": "hf"}}, "unrolled", "hf"),
            (
                {"profile": "reference_eval", "attention": {"rope": "pre_negated_sin"}},
                "batched",
                "pre_negated_sin",
            ),
            (
                {"profile": "npu_export", "attention": {"layout": "batched"}},
                "batched",
                "pre_negated_sin",
            ),
            (
                {"profile": "reference_eval", "attention": {"layout": "unrolled"}},
                "unrolled",
                "hf",
            ),
            ({"attention": None}, "unrolled", "pre_negated_sin"),
        )
        for model_args, layout, rope in cases:
            with self.subTest(model_args=model_args):
                options = get_gemma4_text_attention_options(
                    PTQConfig(model_args=model_args)
                )
                self.assertEqual((options.layout, options.rope), (layout, rope))

    def test_dataclass_defaults_do_not_select_an_execution_profile(self):
        options = Gemma4TextAttentionOptions()
        self.assertEqual((options.layout, options.rope), ("unrolled", "hf"))
        self.assertFalse(is_npu_export_text_attention_options(options))
        self.assertNotIn("__setstate__", Gemma4TextAttentionOptions.__dict__)
        resolved = get_gemma4_text_attention_options(PTQConfig())
        self.assertTrue(is_npu_export_text_attention_options(resolved))

    @staticmethod
    def _decoder_shell(options) -> QuantGemma4TextDecoderLayer:
        shell = SimpleNamespace(
            self_attn=SimpleNamespace(wrapped=SimpleNamespace(attn_options=options)),
            _mode=None,
        )
        # Only the attributes read by ``as_export_module`` are needed here.
        return cast(QuantGemma4TextDecoderLayer, shell)

    def test_decoder_export_rejects_noncanonical_options(self):
        noncanonical: Tuple[Tuple[AttentionLayout, RopeConvention], ...] = (
            ("batched", "hf"),
            ("batched", "pre_negated_sin"),
            ("unrolled", "hf"),
        )
        modes: Tuple[ExportMode, ...] = ("prefill", "decode")
        for layout, rope in noncanonical:
            for mode in modes:
                with self.subTest(layout=layout, rope=rope, mode=mode):
                    shell = self._decoder_shell(
                        Gemma4TextAttentionOptions(layout, rope)
                    )
                    with self.assertRaisesRegex(ValueError, "rope='pre_negated_sin'"):
                        QuantGemma4TextDecoderLayer.as_export_module(shell, mode)

    def test_decoder_export_accepts_npu_and_explicit_reference_bypass(self):
        module = "tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer"
        adapters: Tuple[Tuple[ExportMode, str], ...] = (
            ("prefill", "Gemma4TextDecoderLayerPrefillExportAdapter"),
            ("decode", "Gemma4TextDecoderLayerDecodeExportAdapter"),
        )
        for mode, adapter in adapters:
            for options, require_npu in (
                (get_gemma4_text_attention_options(PTQConfig()), True),
                (Gemma4TextAttentionOptions("unrolled", "hf"), False),
            ):
                with self.subTest(mode=mode, require_npu=require_npu):
                    shell = self._decoder_shell(options)
                    # The observer is forwarded verbatim, so a sentinel suffices.
                    observer = cast(Any, object())
                    with patch(f"{module}.{adapter}") as factory:
                        result = QuantGemma4TextDecoderLayer.as_export_module(
                            shell,
                            mode,
                            return_kv=False,
                            require_npu_profile=require_npu,
                            per_layer_input_observer=observer,
                        )
                    self.assertIs(result, factory.return_value)
                    factory.assert_called_once_with(
                        shell,
                        return_kv=False,
                        mode=None,
                        per_layer_input_observer=observer,
                    )

    def test_invalid_options_are_rejected(self):
        for invalid in ("unknown", None, 1):
            with self.subTest(rope=invalid), self.assertRaises(ValueError):
                get_gemma4_text_attention_options(
                    PTQConfig(model_args={"attention": {"rope": invalid}})
                )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            get_gemma4_text_attention_options(
                PTQConfig(model_args={"attention": {"scale_fusion": "q_proj"}})
            )

    def test_sine_transform_preserves_input_and_partial_rope(self):
        torch.manual_seed(41)
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            for head_dim in (8, 32):
                for fraction in (1.0, 0.25):
                    with self.subTest(
                        dtype=dtype, head_dim=head_dim, fraction=fraction
                    ):
                        phase = torch.randn(1, 5, 1, head_dim // 2, dtype=dtype)
                        phase[..., int(head_dim // 2 * fraction) :] = 0
                        phase = torch.cat((phase, phase), -1)
                        cos, sin = phase.cos(), phase.sin()
                        original = sin.clone()
                        self.assertIs(prepare_gemma4_rope_sin(sin, "hf"), sin)
                        prepared = prepare_gemma4_rope_sin(sin, "pre_negated_sin")
                        torch.testing.assert_close(sin, original, rtol=0, atol=0)
                        self.assertEqual(
                            (prepared.dtype, prepared.device, prepared.shape),
                            (sin.dtype, sin.device, sin.shape),
                        )
                        x = torch.randn(1, 5, 2, head_dim, dtype=dtype)
                        a, b = x.chunk(2, -1)
                        expected = x * cos + torch.cat((-b, a), -1) * sin
                        actual = x * cos + torch.cat((b, a), -1) * prepared
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sine_transform_rejects_invalid_inputs(self):
        for sin in (torch.tensor(1.0), torch.empty(1, 0), torch.zeros(1, 3)):
            with self.assertRaises(ValueError):
                prepare_gemma4_rope_sin(sin, "pre_negated_sin")
        with self.assertRaises(TypeError):
            prepare_gemma4_rope_sin(torch.zeros(2, 4, dtype=torch.int64))
        with self.assertRaises(ValueError):
            prepare_gemma4_rope_sin(torch.zeros(2, 4), cast(RopeConvention, "other"))

    @staticmethod
    def _scale_shell(scaling, fq) -> QuantGemma4TextAttention:
        shell = SimpleNamespace(
            scaling=scaling,
            _fq=fq,
            obs_logits_raw="raw",
            obs_logits="output",
            obs_scale="scale",
        )
        # Only the attributes read by ``_apply_attention_scale`` are needed here.
        return cast(QuantGemma4TextAttention, shell)

    def test_identity_scale_retains_both_logits_observers(self):
        calls = []

        def fq(value, observer):
            calls.append(observer)
            if observer == "scale":
                self.fail("The identity scale observer must never be visited")
            return value + 1

        shell = self._scale_shell(1.0, fq)
        logits = torch.randn(1, 3, 4)
        actual = QuantGemma4TextAttention._apply_attention_scale(shell, logits)
        torch.testing.assert_close(actual, logits + 2)
        self.assertEqual(calls, ["raw", "output"])

    def test_nonunit_scale_is_not_fused(self):
        for scale in (0.25, 0.0, -0.5, 1.000001):
            calls = []

            def fq(value, observer):
                calls.append(observer)
                return value

            shell = self._scale_shell(scale, fq)
            logits = torch.randn(1, 3, 4)
            actual = QuantGemma4TextAttention._apply_attention_scale(shell, logits)
            torch.testing.assert_close(actual, logits * scale, rtol=0, atol=0)
            self.assertEqual(calls, ["raw", "scale", "output"])


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are not installed")
class TestGemma4RoPEIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(431)

    @staticmethod
    def _config(shared=0, k_eq_v=False):
        return _make_text_config(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=16,
            global_head_dim=32,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
            num_kv_shared_layers=shared,
            attention_k_eq_v=k_eq_v,
            rope_parameters={
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            },
        )

    @staticmethod
    def _qcfg(layout, rope):
        return PTQConfig(
            model_args={
                "attention": {"layout": layout, "rope": rope},
                "text": {"max_seq": 32},
            }
        )

    def _assert_nested(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        elif expected is None:
            self.assertIsNone(actual)
        else:
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self._assert_nested(left, right)

    def test_prefill_decode_and_shared_kv_parity(self):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextAttention,
            Gemma4TextRotaryEmbedding,
        )

        for layout in ("batched", "unrolled"):
            for k_eq_v in (False, True):
                cfg = self._config(shared=2, k_eq_v=k_eq_v)
                rotary = Gemma4TextRotaryEmbedding(cfg)
                for layer_idx in range(4):
                    with self.subTest(
                        layout=layout, k_eq_v=k_eq_v, layer=layer_idx
                    ), torch.no_grad():
                        fp = Gemma4TextAttention(cfg, layer_idx=layer_idx).eval()
                        hf = QuantGemma4TextAttention(
                            copy.deepcopy(fp), qcfg=self._qcfg(layout, "hf")
                        ).eval()
                        pre = QuantGemma4TextAttention(
                            copy.deepcopy(fp),
                            qcfg=self._qcfg(layout, "pre_negated_sin"),
                        ).eval()
                        hidden = torch.randn(1, 5, cfg.hidden_size)
                        cos, sin = rotary(
                            hidden,
                            torch.arange(5).unsqueeze(0),
                            cfg.layer_types[layer_idx],
                        )
                        kwargs = dict(
                            attention_mask=torch.zeros(1, 1, 5, 5),
                            use_cache=True,
                            cache_output_mode="present",
                        )
                        if fp.is_kv_shared_layer:
                            kwargs["shared_key_value"] = tuple(
                                torch.randn(1, 1, 5, fp.head_dim) for _ in range(2)
                            )
                        expected = hf(hidden, (cos, sin), **kwargs)
                        actual = pre(
                            hidden,
                            (cos, prepare_gemma4_rope_sin(sin, "pre_negated_sin")),
                            **kwargs,
                        )
                        self._assert_nested(actual, expected)
                        if not fp.is_kv_shared_layer:
                            decode = torch.randn(1, 1, cfg.hidden_size)
                            cos, sin = rotary(
                                decode,
                                torch.tensor([[5]]),
                                cfg.layer_types[layer_idx],
                            )
                            mask = torch.zeros(1, 1, 1, 6)
                            expected_decode = hf(
                                decode,
                                (cos, sin),
                                attention_mask=mask,
                                past_key_value=expected[2],
                                use_cache=True,
                            )
                            pre_sin = prepare_gemma4_rope_sin(sin, "pre_negated_sin")
                            actual_decode = pre(
                                decode,
                                (cos, pre_sin),
                                attention_mask=mask,
                                past_key_value=actual[2],
                                use_cache=True,
                            )
                            self._assert_nested(actual_decode, expected_decode)

    def test_active_observers_calibrate_freeze_and_run(self):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextAttention,
            Gemma4TextRotaryEmbedding,
        )

        ropes: Tuple[RopeConvention, ...] = ("hf", "pre_negated_sin")
        for rope in ropes:
            cfg = self._config()
            fp = Gemma4TextAttention(cfg, layer_idx=1).eval()
            wrapped = QuantGemma4TextAttention(
                fp, qcfg=self._qcfg("unrolled", rope)
            ).eval()
            active_ids = {id(obs) for obs in wrapped._all_observers()}
            self.assertNotIn(id(wrapped.obs_scale), active_ids)
            self.assertEqual(id(wrapped.obs_q_neg) in active_ids, rope == "hf")
            self.assertEqual(id(wrapped.obs_k_neg) in active_ids, rope == "hf")
            hidden = torch.randn(1, 5, cfg.hidden_size)
            cos, sin = Gemma4TextRotaryEmbedding(cfg)(
                hidden, torch.arange(5).unsqueeze(0), "full_attention"
            )
            tables = (cos, prepare_gemma4_rope_sin(sin, rope))
            wrapped.enable_calibration()
            with torch.no_grad():
                wrapped(hidden, tables, attention_mask=torch.zeros(1, 1, 5, 5))
            wrapped.freeze_qparams()
            with torch.no_grad():
                output = wrapped(
                    hidden, tables, attention_mask=torch.zeros(1, 1, 5, 5)
                )[0]
            self.assertTrue(torch.isfinite(output).all())

    def test_pre_negated_export_has_no_activation_negation(self):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextAttention,
            Gemma4TextRotaryEmbedding,
        )

        cfg = self._config()
        fp = Gemma4TextAttention(cfg, layer_idx=1).eval()
        wrapped = QuantGemma4TextAttention(
            fp, qcfg=PTQConfig(model_args={"profile": "npu_export"})
        ).eval()
        hidden = torch.randn(1, 5, cfg.hidden_size)
        cos, sin = Gemma4TextRotaryEmbedding(cfg)(
            hidden, torch.arange(5).unsqueeze(0), "full_attention"
        )
        args = (
            hidden,
            cos,
            prepare_gemma4_rope_sin(sin, "pre_negated_sin"),
            torch.zeros(1, 1, 5, 5),
        )
        exported = torch.export.export(
            _AttentionExportAdapter(wrapped), args, strict=False
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten.neg.default
                for node in exported.graph.nodes
            )
        )
        for node in exported.graph.nodes:
            if node.target in (
                torch.ops.aten.matmul.default,
                torch.ops.aten.bmm.default,
            ):
                for arg in node.args[:2]:
                    if isinstance(arg, torch.fx.Node) and "val" in arg.meta:
                        self.assertLessEqual(arg.meta["val"].dim(), 3)
        self._assert_nested(
            exported.module()(*args), _AttentionExportAdapter(wrapped)(*args)
        )

    def test_model_dynamic_and_static_tables_match_once(self):
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        profiles: Tuple[Tuple[str, RopeConvention], ...] = (
            ("reference_eval", "hf"),
            ("npu_export", "pre_negated_sin"),
        )
        for profile, rope in profiles:
            cfg = self._config()
            fp = Gemma4TextModel(cfg).eval()
            model = QuantGemma4TextModel(
                fp,
                qcfg=PTQConfig(
                    model_args={"profile": profile, "text": {"max_seq": 16}}
                ),
            ).eval()
            self.assertEqual(model.rope_convention, rope)
            for layer in model.layers:
                # ``nn.Module.__getattr__`` types submodules as ``Tensor | Module``.
                decoder = cast(Any, layer.wrapped)
                self.assertEqual(decoder.self_attn.wrapped.attn_options.rope, rope)
            hidden = torch.randn(1, 5, cfg.hidden_size)
            positions = torch.arange(5).unsqueeze(0)
            dynamic = model._make_position_embeddings(
                hidden_states=hidden,
                position_ids=positions,
                past_key_values=None,
                use_static_templates=False,
            )
            static = model._make_position_embeddings(
                hidden_states=hidden,
                position_ids=positions,
                past_key_values=None,
                use_static_templates=True,
            )
            for layer_type in model.unique_layer_types:
                self._assert_nested(static[layer_type], dynamic[layer_type])
                cos, sin = fp.rotary_emb(hidden, positions, layer_type)
                self._assert_nested(
                    dynamic[layer_type], (cos, prepare_gemma4_rope_sin(sin, rope))
                )


if __name__ == "__main__":
    unittest.main()

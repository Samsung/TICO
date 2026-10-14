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

"""Regression tests for GPTQ grids after Llama projection scale folding."""

import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from tico.quantization.config import affine, PTQConfig
from tico.quantization.recipes.qparams import inject_gptq_qparams
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.utils.linear_folding import (
    _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR,
    reused_weight_qparam_scale_multiplier,
)
from tico.quantization.wrapq.wrappers.llama.quant_attention import (
    _clone_projection_with_scale,
    QuantLlamaAttention,
)
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


class TestLlamaProjectionScaleMetadata(unittest.TestCase):
    """Keep the scalar metadata aligned with the copied projection's weight."""

    def test_clone_preserves_source_parameters(self):
        for bias in (False, True):
            with self.subTest(bias=bias):
                source = nn.Linear(4, 3, bias=bias).eval()
                source.weight.requires_grad_(False)
                weight = source.weight.detach().clone()
                original_bias = (
                    source.bias.detach().clone() if source.bias is not None else None
                )
                folded = _clone_projection_with_scale(source, torch.tensor(0.125))

                self.assertIsNot(folded, source)
                self.assertIsNot(folded.weight, source.weight)
                self.assertFalse(folded.training)
                self.assertFalse(folded.weight.requires_grad)
                self.assertEqual(folded.weight.device, source.weight.device)
                torch.testing.assert_close(source.weight, weight, rtol=0, atol=0)
                torch.testing.assert_close(
                    folded.weight, weight * 0.125, rtol=0, atol=0
                )
                if original_bias is None:
                    self.assertIsNone(folded.bias)
                else:
                    torch.testing.assert_close(
                        source.bias, original_bias, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        folded.bias, original_bias * 0.125, rtol=0, atol=0
                    )
                self.assertFalse(
                    hasattr(source, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR)
                )
                self.assertEqual(reused_weight_qparam_scale_multiplier(folded), 0.125)

    def test_multiplier_uses_the_effective_weight_dtype(self):
        scale = torch.tensor(0.1, dtype=torch.float64)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                source = nn.Linear(4, 3, bias=False, dtype=dtype)
                folded = _clone_projection_with_scale(source, scale)
                effective_scale = scale.to(dtype=dtype)
                self.assertEqual(folded.weight.dtype, dtype)
                self.assertEqual(
                    reused_weight_qparam_scale_multiplier(folded),
                    float(effective_scale.item()),
                )
                torch.testing.assert_close(
                    folded.weight, source.weight * effective_scale, rtol=0, atol=0
                )

    def test_repeated_folds_compose_the_multiplier(self):
        source = nn.Linear(4, 3, bias=False)
        first = _clone_projection_with_scale(source, torch.tensor(2.0))
        second = _clone_projection_with_scale(first, torch.tensor(0.125))
        self.assertEqual(reused_weight_qparam_scale_multiplier(first), 2.0)
        self.assertEqual(reused_weight_qparam_scale_multiplier(second), 0.25)
        torch.testing.assert_close(second.weight, source.weight * 0.25, rtol=0, atol=0)
        self.assertFalse(hasattr(source, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR))

    def test_unsupported_fold_marker_is_preserved(self):
        source = nn.Linear(4, 3, bias=False)
        setattr(source, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR, None)
        folded = _clone_projection_with_scale(source, torch.tensor(0.125))
        self.assertIsNone(getattr(source, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR))
        self.assertIsNone(getattr(folded, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR))
        with self.assertRaisesRegex(RuntimeError, "Recompute"):
            reused_weight_qparam_scale_multiplier(folded)

    def test_weightless_module_does_not_gain_fold_metadata(self):
        source = nn.Identity()
        folded = _clone_projection_with_scale(source, torch.tensor(0.125))
        self.assertIsNot(folded, source)
        self.assertFalse(hasattr(folded, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR))
        self.assertEqual(reused_weight_qparam_scale_multiplier(folded), 1.0)


class TestLlamaAttentionQParamHandoff(unittest.TestCase):
    """Exercise real PTQ wrappers and injection without downloading a model."""

    _PREFIX = "model.layers.0.self_attn"
    _PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

    @classmethod
    def _source(cls, *, head_dim=4, scaling=None, bias=True):
        """Use a tiny GQA-shaped module whose weights lie on known U4 grids."""
        hidden_size = 2 * head_dim
        source = nn.Module()
        source.config = SimpleNamespace(  # type: ignore[assignment]
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=head_dim,
            max_position_embeddings=16,
        )
        source.scaling = head_dim**-0.5 if scaling is None else scaling
        quantizers = {}
        codes = {}
        for name in cls._PROJECTIONS:
            out_features = head_dim if name in ("k_proj", "v_proj") else hidden_size
            linear = nn.Linear(hidden_size, out_features, bias=bias)
            row = torch.arange(out_features)
            # Nonuniform per-output-channel scales and nonzero zero-points.
            scale = torch.pow(2.0, -(row % 3 + 4).float()).reshape(-1, 1)
            zero = (row % 5 + 4).float().reshape(-1, 1)
            code = (torch.arange(linear.weight.numel()) % 16).reshape_as(linear.weight)
            with torch.no_grad():
                linear.weight.copy_((code - zero) * scale)
                if linear.bias is not None:
                    linear.bias.copy_((row.float() - 2) / 32)
            setattr(source, name, linear)
            quantizers[f"{cls._PREFIX}.{name}"] = SimpleNamespace(
                maxq=15, perchannel=True, scale=scale, zero=zero
            )
            codes[name] = code
        source.eval()
        return source, quantizers, codes

    def _check_handoff(self, fusion, *, default_profile=False):
        for head_dim, scaling in ((4, 0.125), (8, None)):
            for bias in (False, True):
                with self.subTest(head_dim=head_dim, scaling=scaling, bias=bias):
                    source, quantizers, codes = self._source(
                        head_dim=head_dim, scaling=scaling, bias=bias
                    )
                    original = copy.deepcopy(source)
                    grids = {
                        name: (quantizer.scale.clone(), quantizer.zero.clone())
                        for name, quantizer in quantizers.items()
                    }
                    model_args = (
                        {}
                        if default_profile
                        else {
                            "profile": "reference_eval",
                            "attention": {"scale_fusion": fusion},
                        }
                    )
                    qattn = QuantLlamaAttention(
                        source,
                        qcfg=PTQConfig(
                            weight=affine(DType.uint(4)), model_args=model_args
                        ),
                        fp_name=self._PREFIX,
                        layer_idx=0,
                    )
                    self.assertEqual(qattn.attn_options.scale_fusion, fusion)
                    result = inject_gptq_qparams(qattn, quantizers)
                    self.assertEqual(result, {"matched": 4, "missed": 0, "unused": 0})

                    # Re-enter calibration: injected weight grids must stay locked.
                    qattn.enable_calibration()
                    for name in self._PROJECTIONS:
                        projection = getattr(qattn, name).wrapped
                        self.assertIsInstance(projection, QuantLinear)
                        fp = getattr(original, name)
                        quantizer = quantizers[f"{self._PREFIX}.{name}"]
                        factor = (
                            float(qattn.attn_scale.item()) if name == fusion else 1.0
                        )
                        self.assertEqual(
                            reused_weight_qparam_scale_multiplier(projection), factor
                        )
                        x = (
                            torch.arange(
                                2 * fp.in_features, dtype=torch.float32
                            ).reshape(2, fp.in_features)
                            / 32
                        )
                        projection(x)
                        projection.freeze_qparams()
                        self.assertTrue(projection.obs_act_in.has_qparams)
                        self.assertTrue(projection.obs_act_out.has_qparams)
                        observer = projection.obs_weight
                        self.assertTrue(observer._qparams_locked)
                        qparams = observer.compute_qparams()
                        self.assertIsNotNone(qparams)
                        scale, zero = qparams
                        self.assertEqual(scale.dtype, torch.float32)
                        self.assertEqual(zero.dtype, torch.int32)
                        self.assertEqual(scale.shape, (fp.out_features,))
                        self.assertEqual(zero.shape, (fp.out_features,))
                        torch.testing.assert_close(
                            scale, quantizer.scale.flatten() * factor, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            zero,
                            quantizer.zero.flatten().to(torch.int32),
                            rtol=0,
                            atol=0,
                        )
                        weight = projection.module.weight
                        torch.testing.assert_close(
                            weight, fp.weight * factor, rtol=0, atol=0
                        )
                        if fp.bias is not None:
                            torch.testing.assert_close(
                                projection.module.bias,
                                fp.bias * factor,
                                rtol=0,
                                atol=0,
                            )
                        got_codes = torch.round(weight / scale[:, None] + zero[:, None])
                        torch.testing.assert_close(
                            got_codes.to(torch.int64), codes[name], rtol=0, atol=0
                        )
                        fake_weight = observer.fake_quant(weight)
                        torch.testing.assert_close(
                            fake_weight, weight, rtol=1e-6, atol=1e-7
                        )
                        torch.testing.assert_close(
                            nn.functional.linear(
                                x, fake_weight, projection.module.bias
                            ),
                            fp(x) * factor,
                            rtol=1e-5,
                            atol=1e-6,
                        )
                        # Neither source parameters nor GPTQ grids may change.
                        torch.testing.assert_close(
                            getattr(source, name).weight, fp.weight, rtol=0, atol=0
                        )
                        if fp.bias is not None:
                            torch.testing.assert_close(
                                getattr(source, name).bias,
                                fp.bias,
                                rtol=0,
                                atol=0,
                            )
                        self.assertFalse(
                            hasattr(
                                getattr(source, name),
                                _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR,
                            )
                        )
                        saved_scale, saved_zero = grids[f"{self._PREFIX}.{name}"]
                        torch.testing.assert_close(
                            quantizer.scale, saved_scale, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            quantizer.zero, saved_zero, rtol=0, atol=0
                        )

    def test_k_proj_fold_preserves_gptq_grid(self):
        self._check_handoff("k_proj")

    def test_q_proj_fold_preserves_gptq_grid(self):
        self._check_handoff("q_proj")

    def test_no_fold_preserves_all_projection_grids(self):
        self._check_handoff("none")

    def test_default_npu_profile_preserves_k_proj_gptq_grid(self):
        self._check_handoff("k_proj", default_profile=True)

    def test_plain_ptq_ignores_fold_metadata(self):
        source, _, _ = self._source()
        qcfg = PTQConfig(weight=affine(DType.uint(4)))
        qattn = QuantLlamaAttention(source, qcfg=qcfg, layer_idx=0)
        projection = qattn.k_proj.wrapped
        assert isinstance(projection, QuantLinear)
        reference_linear = copy.deepcopy(projection.module)
        assert isinstance(reference_linear, nn.Linear)
        delattr(reference_linear, _REUSED_WEIGHT_QPARAM_SCALE_MULTIPLIER_ATTR)
        reference = QuantLinear(reference_linear, qcfg=qcfg)
        # No inject_gptq_qparams call: emulate explicit reuse_gptq_qparams=false.
        for linear in (projection, reference):
            linear.enable_calibration()
            linear(torch.zeros(2, source.config.hidden_size))
            linear.freeze_qparams()
            self.assertFalse(linear.obs_weight._qparams_locked)
        actual = projection.obs_weight.compute_qparams()
        expected = reference.obs_weight.compute_qparams()
        assert actual is not None
        assert expected is not None
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

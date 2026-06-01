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

"""
The tests run only if *transformers* is available (they depend on the genuine
`transformers.models.llama.modeling_llama.LlamaModel`).
"""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_model import QuantLlamaModel

from test.quantization.quant_spec_helpers import make_affine_ptq_config

skip_msg = "required transformers not installed — skipping LlamaModel tests"


@unittest.skipUnless(has_transformers_for("llama"), skip_msg)
class TestQuantLlamaModel(unittest.TestCase):
    seq_len: int
    vocab_size: int
    fp_model: torch.nn.Module
    head_dim: int

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel

        cls.seq_len = 16
        cls.vocab_size = 10000
        cls.head_dim = 4

        cfg = LlamaConfig(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=cls.head_dim,
            attention_bias=False,
            attention_dropout=0.0,
            attn_implementation="eager",
            num_hidden_layers=2,
            max_position_embeddings=cls.seq_len,
            use_cache=False,
            return_dict=False,
        )

        cls.fp_model = LlamaModel(cfg)

    def test_mode_transitions(self):
        qmodel = QuantLlamaModel(self.fp_model, qcfg=make_affine_ptq_config())
        self.assertIs(qmodel._mode, Mode.NO_QUANT)

        qmodel.enable_calibration()
        self.assertIs(qmodel._mode, Mode.CALIB)

        x = torch.randint(0, self.vocab_size, (1, self.seq_len))
        _ = qmodel(x, use_cache=False, return_dict=False)

        qmodel.freeze_qparams()
        self.assertIs(qmodel._mode, Mode.QUANT)

    def test_forward_diff(self):
        qmodel = QuantLlamaModel(self.fp_model, qcfg=make_affine_ptq_config())
        qmodel.enable_calibration()

        calib_set = []
        for _ in range(4):
            inp = torch.randint(0, self.vocab_size, (1, self.seq_len))
            _ = qmodel(inp, use_cache=False, return_dict=False)
            calib_set.append(inp)

        qmodel.freeze_qparams()

        with torch.no_grad():
            q_out = qmodel(
                calib_set[0],
                use_cache=False,
                return_dict=False,
            )[0]

            fp_out = self.fp_model(
                calib_set[0],
                use_cache=False,
                return_dict=False,
            )[0]

        diff = (fp_out - q_out).abs().mean().item()

        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.4)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_reference_eval_profile_propagates_to_layers_and_attention(self):
        qcfg = make_affine_ptq_config(model_args={"profile": "reference_eval"})
        qmodel = QuantLlamaModel(self.fp_model, qcfg=qcfg)

        first_layer = qmodel.layers[0].wrapped
        first_attn = first_layer.self_attn.wrapped

        self.assertEqual(qmodel.attn_options.scale_fusion, "none")
        self.assertEqual(qmodel.attn_options.rope, "hf")
        self.assertEqual(qmodel.attn_options.layout, "batched")

        self.assertEqual(first_layer.attn_options.scale_fusion, "none")
        self.assertEqual(first_layer.attn_options.rope, "hf")
        self.assertEqual(first_layer.attn_options.layout, "batched")

        self.assertEqual(first_attn.attn_options.scale_fusion, "none")
        self.assertEqual(first_attn.attn_options.rope, "hf")
        self.assertEqual(first_attn.attn_options.layout, "batched")

    def test_attention_specific_profile_override_propagates(self):
        qcfg = make_affine_ptq_config(
            model_args={
                "profile": "reference_eval",
                "attention": {
                    "profile": "npu_export",
                },
            }
        )
        qmodel = QuantLlamaModel(self.fp_model, qcfg=qcfg)

        first_attn = qmodel.layers[0].wrapped.self_attn.wrapped

        self.assertEqual(qmodel.attn_options.scale_fusion, "k_proj")
        self.assertEqual(qmodel.attn_options.rope, "pre_negated_sin")
        self.assertEqual(qmodel.attn_options.layout, "unrolled")
        self.assertEqual(first_attn.attn_options.scale_fusion, "k_proj")
        self.assertEqual(first_attn.attn_options.rope, "pre_negated_sin")
        self.assertEqual(first_attn.attn_options.layout, "unrolled")

    def test_rope_sin_template_convention_depends_on_profile(self):
        qmodel_ref = QuantLlamaModel(
            self.fp_model,
            qcfg=make_affine_ptq_config(model_args={"profile": "reference_eval"}),
        )
        qmodel_npu = QuantLlamaModel(
            self.fp_model,
            qcfg=make_affine_ptq_config(model_args={"profile": "npu_export"}),
        )

        half_dim = self.head_dim // 2

        torch.testing.assert_close(
            qmodel_ref.rope_cos_template,
            qmodel_npu.rope_cos_template,
        )
        torch.testing.assert_close(
            qmodel_npu.rope_sin_template[..., :half_dim],
            -qmodel_ref.rope_sin_template[..., :half_dim],
        )
        torch.testing.assert_close(
            qmodel_npu.rope_sin_template[..., half_dim:],
            qmodel_ref.rope_sin_template[..., half_dim:],
        )

    def test_layer_qscheme_override_propagates_to_projection_weight_observer(self):
        """
        Ensure layer-local qscheme overrides are propagated through
        QuantLlamaModel -> QuantLlamaDecoderLayer -> QuantLlamaAttention -> q_proj.

        This catches naming mismatches such as looking up `model.layers.0`
        after the config has already been narrowed to `layers`.
        """
        qcfg = make_affine_ptq_config(
            dtype=DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
            overrides={
                "layers": {
                    "0": {
                        "self_attn": {
                            "q_proj": {
                                "weight": {
                                    "dtype": DType.uint(4),
                                    "qscheme": QScheme.PER_TENSOR_ASYMM,
                                }
                            }
                        }
                    }
                }
            },
        )

        qmodel = QuantLlamaModel(self.fp_model, qcfg=qcfg, fp_name="model")

        obs = qmodel.layers[0].wrapped.self_attn.wrapped.q_proj.get_observer("weight")

        self.assertEqual(obs.dtype, DType.uint(4))
        self.assertEqual(obs.qscheme, QScheme.PER_TENSOR_ASYMM)

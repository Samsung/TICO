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

"""Unit tests for the Gemma4 text-model PTQ wrapper."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode


_SKIP_MSG = "required transformers Gemma4 modules are not installed"

_GEMMA4_FULL_ROPE_PARAMETERS = {
    "rope_type": "proportional",
    "partial_rotary_factor": 0.25,
    "rope_theta": 1_000_000.0,
}
_GEMMA4_SLIDING_ROPE_PARAMETERS = {
    "rope_type": "default",
    "rope_theta": 10_000.0,
}


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextModel,
        )
    except Exception:
        return False
    return True


def _rope_parameters_for_layer_types(layer_types: list[str]) -> dict[str, dict]:
    """Return Gemma4 RoPE parameters matching the requested layer types."""
    params = {}
    if "sliding_attention" in layer_types:
        params["sliding_attention"] = dict(_GEMMA4_SLIDING_ROPE_PARAMETERS)
    if "full_attention" in layer_types:
        params["full_attention"] = dict(_GEMMA4_FULL_ROPE_PARAMETERS)
    return params


def _make_text_config(**overrides):
    """Create a tiny dense Gemma4 text config for synthetic text-model tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    layer_types = list(
        overrides.pop("layer_types", ["full_attention", "full_attention"])
    )
    kwargs = dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=8,
        layer_types=layer_types,
        rope_parameters=_rope_parameters_for_layer_types(layer_types),
        hidden_size_per_layer_input=0,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_cache=False,
    )
    kwargs.update(overrides)
    cfg = Gemma4TextConfig(**kwargs)
    if not hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")
    else:
        cfg._attn_implementation = "eager"
    return cfg


def _make_text_model(cfg=None):
    """Create a floating-point Gemma4 text model in eval mode."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    cfg = cfg if cfg is not None else _make_text_config()
    return Gemma4TextModel(cfg).eval()


def _assert_close(
    testcase: unittest.TestCase, actual: torch.Tensor, expected: torch.Tensor
):
    """Assert that two tensors are numerically close for no-quant wrapper parity."""
    testcase.assertEqual(actual.shape, expected.shape)
    testcase.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4TextModel(unittest.TestCase):
    """Validate dense Gemma4 text-model wrapper behavior."""

    def setUp(self):
        """Create deterministic test inputs."""
        torch.manual_seed(2026)

    def test_00_prepare_wraps_text_model_when_registered(self):
        """Check that registry-based prepare wraps Gemma4TextModel."""
        from tico.quantization import prepare
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )
        from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

        prepared = prepare(_make_text_model(_make_text_config()), PTQConfig())

        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantGemma4TextModel)

    def test_no_quant_forward_matches_hf_text_model_with_input_ids(self):
        """Check that the wrapper matches Hugging Face text-model output."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(layer_types=["sliding_attention", "full_attention"])
        fp_model = _make_text_model(cfg)
        qmodel = QuantGemma4TextModel(fp_model).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            quant_out = qmodel(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            fp_out = fp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        _assert_close(self, quant_out.last_hidden_state, fp_out.last_hidden_state)

    def test_static_attention_mask_mapping_matches_hf_text_model(self):
        """Check the static CPU-provided mask mapping path."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        cfg = _make_text_config(layer_types=["sliding_attention", "full_attention"])
        fp_model = _make_text_model(cfg)
        qmodel = QuantGemma4TextModel(fp_model).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
        inputs_embeds = fp_model.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask_kwargs = {
            "config": cfg,
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.ones_like(input_ids),
            "past_key_values": None,
            "position_ids": position_ids,
        }
        mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        with torch.no_grad():
            quant_out = qmodel(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                return_shared_kv_states=True,
                return_dict=True,
            )
            fp_out = fp_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                return_shared_kv_states=True,
                return_dict=True,
            )

        _assert_close(self, quant_out.last_hidden_state, fp_out.last_hidden_state)
        self.assertIsNotNone(quant_out.shared_kv_states)

    def test_ple_path_matches_hf_with_input_ids(self):
        """Check Hugging Face parity when Gemma4 PLE is enabled."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(
            layer_types=["full_attention", "full_attention"],
            hidden_size_per_layer_input=8,
        )
        fp_model = _make_text_model(cfg)
        qmodel = QuantGemma4TextModel(fp_model).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))

        with torch.no_grad():
            quant_out = qmodel(input_ids=input_ids, return_dict=True)
            fp_out = fp_model(input_ids=input_ids, return_dict=True)

        _assert_close(self, quant_out.last_hidden_state, fp_out.last_hidden_state)

    def test_ple_path_matches_hf_with_inputs_embeds_and_explicit_per_layer_inputs(self):
        """Check multimodal-style PLE entry with explicit token-identity inputs."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(
            layer_types=["full_attention", "full_attention"],
            hidden_size_per_layer_input=8,
        )
        fp_model = _make_text_model(cfg)
        qmodel = QuantGemma4TextModel(fp_model).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        inputs_embeds = fp_model.embed_tokens(input_ids)
        per_layer_inputs = fp_model.get_per_layer_inputs(input_ids, inputs_embeds)

        with torch.no_grad():
            quant_out = qmodel(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                return_dict=True,
            )
            fp_out = fp_model(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                return_dict=True,
            )

        _assert_close(self, quant_out.last_hidden_state, fp_out.last_hidden_state)

    def test_shared_kv_text_model_returns_shared_state_when_requested(self):
        """Check full text-model execution with one shared-KV consumer layer."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention", "full_attention"],
            num_kv_shared_layers=1,
        )
        fp_model = _make_text_model(cfg)
        qmodel = QuantGemma4TextModel(fp_model).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 5))

        with torch.no_grad():
            quant_out = qmodel(
                input_ids=input_ids,
                return_shared_kv_states=True,
                return_dict=True,
            )
            fp_out = fp_model(
                input_ids=input_ids,
                return_shared_kv_states=True,
                return_dict=True,
            )

        _assert_close(self, quant_out.last_hidden_state, fp_out.last_hidden_state)
        self.assertIsNotNone(quant_out.shared_kv_states)
        self.assertIn("full_attention", quant_out.shared_kv_states)

    def test_validation_errors_match_expected_contract(self):
        """Check user-facing validation errors for invalid input combinations."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(hidden_size_per_layer_input=8)
        qmodel = QuantGemma4TextModel(_make_text_model(cfg)).eval()
        input_ids = torch.randint(0, cfg.vocab_size, (1, 3))
        inputs_embeds = qmodel.embed_tokens(input_ids)
        per_layer_inputs = torch.randn(
            1,
            3,
            cfg.num_hidden_layers,
            cfg.hidden_size_per_layer_input,
        )

        with self.assertRaisesRegex(ValueError, "exactly one"):
            qmodel(input_ids=input_ids, inputs_embeds=inputs_embeds)

        with self.assertRaisesRegex(ValueError, "per_layer_inputs"):
            qmodel(input_ids=input_ids, per_layer_inputs=per_layer_inputs)

        qmodel.force_export = True
        with self.assertRaisesRegex(NotImplementedError, "static masks"):
            qmodel(input_ids=input_ids)

    def test_text_model_wrapper_does_not_own_export_adapter_hook(self):
        """Check that TextModel export subgraphs are created outside this wrapper."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        qmodel = QuantGemma4TextModel(_make_text_model(_make_text_config())).eval()

        self.assertFalse(hasattr(qmodel, "as_export_module"))

    def test_mode_transitions_and_observer_collection(self):
        """Check calibration and quantization lifecycle for the text model."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(layer_types=["full_attention", "full_attention"])
        qmodel = QuantGemma4TextModel(_make_text_model(cfg), qcfg=PTQConfig()).eval()
        self.assertIs(qmodel._mode, Mode.NO_QUANT)

        qmodel.enable_calibration()
        self.assertIs(qmodel._mode, Mode.CALIB)
        qmodel(input_ids=torch.randint(0, cfg.vocab_size, (1, 4)))
        qmodel.freeze_qparams()

        self.assertIs(qmodel._mode, Mode.QUANT)
        self.assertIsNotNone(qmodel.get_observer("inputs_embeds"))
        self.assertIsNotNone(qmodel.get_observer("layers.0.self_attn.q_proj.act_in"))

    def test_moe_text_model_is_rejected_for_e2b_scope(self):
        """Check that the E2B text-model wrapper rejects MoE configs explicitly."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(
            enable_moe_block=True,
            num_experts=2,
            top_k_experts=1,
            moe_intermediate_size=16,
        )
        fp_model = _make_text_model(cfg)

        with self.assertRaisesRegex(ValueError, "dense decoder layers only"):
            QuantGemma4TextModel(fp_model)

    def test_quant_mode_requires_explicit_per_layer_inputs_with_inputs_embeds(self):
        """Require explicit PLE token inputs for inputs_embeds in QUANT mode."""
        from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
            QuantGemma4TextModel,
        )

        cfg = _make_text_config(
            layer_types=["full_attention", "full_attention"],
            hidden_size_per_layer_input=8,
        )
        fp_model = _make_text_model(cfg)
        qmodel = QuantGemma4TextModel(fp_model, qcfg=PTQConfig()).eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        inputs_embeds = fp_model.embed_tokens(input_ids)
        per_layer_inputs = fp_model.get_per_layer_inputs(input_ids, inputs_embeds)

        # Calibrate the same inputs_embeds + explicit PLE path.
        qmodel.enable_calibration()
        with torch.no_grad():
            qmodel(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                return_dict=True,
            )
        qmodel.freeze_qparams()

        with self.assertRaisesRegex(ValueError, "explicit per_layer_inputs"):
            qmodel(
                inputs_embeds=inputs_embeds,
                return_dict=True,
            )

        # The supported explicit path should still work.
        with torch.no_grad():
            output = qmodel(
                inputs_embeds=inputs_embeds,
                per_layer_inputs=per_layer_inputs,
                return_dict=True,
            )

        self.assertTrue(torch.isfinite(output.last_hidden_state).all())


if __name__ == "__main__":
    unittest.main()

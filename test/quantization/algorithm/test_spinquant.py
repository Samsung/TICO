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

import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.algorithm.spinquant.quantizer import SpinQuantQuantizer
from tico.quantization.algorithm.spinquant.spin_llama import SpinLlamaForCausalLM
from tico.quantization.config.spinquant import SpinQuantConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class SpinQuantTest(unittest.TestCase):
    def _build_llama_model(
        self,
        *,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        vocab_size: int = 64,
        tie_word_embeddings: bool = True,
    ) -> LlamaForCausalLM:
        """
        Build a small LLaMA model for unit tests.

        Parameters:
            hidden_size: Hidden dimension.
            intermediate_size: MLP intermediate dimension.
            num_hidden_layers: Number of decoder layers.
            num_attention_heads: Number of attention heads.
            vocab_size: Vocabulary size.
            tie_word_embeddings: Whether to tie embedding and lm_head weights.

        Returns:
            A small LlamaForCausalLM instance.
        """
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            max_position_embeddings=64,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = LlamaForCausalLM(config)
        model.eval()
        return model

    def _clone_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """
        Clone a model state_dict into detached tensors.

        Parameters:
            model: Source model.

        Returns:
            A copied state_dict.
        """
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _assert_identity_linear(self, layer: torch.nn.Linear) -> None:
        """
        Assert that a linear layer is initialized as identity.

        Parameters:
            layer: Target linear layer.
        """
        self.assertEqual(layer.in_features, layer.out_features)
        expected = torch.eye(
            layer.in_features,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        self.assertTrue(torch.allclose(layer.weight, expected))
        if layer.bias is not None:
            self.assertTrue(torch.allclose(layer.bias, torch.zeros_like(layer.bias)))

    def test_spinquant_config_validate_accepts_random(self):
        cfg = SpinQuantConfig(init_method="random")
        self.assertEqual(cfg.init_method, "random")
        self.assertEqual(cfg.name, "spinquant")

    def test_spinquant_config_validate_accepts_hadamard(self):
        cfg = SpinQuantConfig(init_method="hadamard")
        self.assertEqual(cfg.init_method, "hadamard")

    def test_spinquant_config_validate_requires_r1_for_external(self):
        with self.assertRaises(ValueError):
            SpinQuantConfig(init_method="external")

    def test_spinquant_config_validate_rejects_non_tensor_r1(self):
        with self.assertRaises(ValueError):
            SpinQuantConfig(init_method="random", r1="invalid")  # type: ignore[arg-type]

    def test_spinquant_config_validate_rejects_non_tensor_r2(self):
        with self.assertRaises(ValueError):
            SpinQuantConfig(
                init_method="random",
                r2_map={"model.layers.0.self_attn.R2": "invalid"},  # type: ignore[dict-item]
            )

    @torch.inference_mode()
    def test_prepare_converts_llama_to_spin_llama(self):
        model = self._build_llama_model()
        q_m = prepare(model, SpinQuantConfig())

        self.assertIsInstance(q_m, SpinLlamaForCausalLM)
        self.assertTrue(hasattr(q_m.model, "rotate_embedding"))
        self.assertTrue(hasattr(q_m, "rotate_lm_head"))
        self._assert_identity_linear(q_m.model.rotate_embedding)
        self._assert_identity_linear(q_m.rotate_lm_head)

    @torch.inference_mode()
    def test_prepare_preserves_generation_related_attributes(self):
        model = self._build_llama_model()
        model.name_or_path = "dummy-llama"
        model._keep_in_fp32_modules = {"lm_head"}

        q_m = prepare(model, SpinQuantConfig())

        self.assertEqual(q_m.name_or_path, "dummy-llama")
        self.assertEqual(q_m._keep_in_fp32_modules, {"lm_head"})
        self.assertIs(q_m.config, model.config)

    @torch.inference_mode()
    def test_prepare_preserves_original_weights_before_convert(self):
        model = self._build_llama_model()
        original_state = self._clone_state_dict(model)

        q_m = prepare(model, SpinQuantConfig())

        # Check that original model weights are copied into the converted model.
        self.assertTrue(
            torch.allclose(
                q_m.model.embed_tokens.weight,
                original_state["model.embed_tokens.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                q_m.lm_head.weight,
                original_state["lm_head.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                q_m.model.layers[0].self_attn.q_proj.weight,
                original_state["model.layers.0.self_attn.q_proj.weight"],
            )
        )

    @torch.inference_mode()
    def test_prepare_preserves_tied_embedding_sharing(self):
        model = self._build_llama_model(tie_word_embeddings=True)
        self.assertIs(model.model.embed_tokens.weight, model.lm_head.weight)

        q_m = prepare(model, SpinQuantConfig())

        # Check that the converted model still uses tied weights.
        self.assertIs(q_m.model.embed_tokens.weight, q_m.lm_head.weight)

    @torch.inference_mode()
    def test_convert_changes_decoder_weights(self):
        model = self._build_llama_model()
        q_m = prepare(model, SpinQuantConfig(init_method="random"))

        before_q = q_m.model.layers[0].self_attn.q_proj.weight.detach().clone()
        before_o = q_m.model.layers[0].self_attn.o_proj.weight.detach().clone()
        before_gate = q_m.model.layers[0].mlp.gate_proj.weight.detach().clone()
        before_down = q_m.model.layers[0].mlp.down_proj.weight.detach().clone()

        q_m = convert(q_m)

        after_q = q_m.model.layers[0].self_attn.q_proj.weight
        after_o = q_m.model.layers[0].self_attn.o_proj.weight
        after_gate = q_m.model.layers[0].mlp.gate_proj.weight
        after_down = q_m.model.layers[0].mlp.down_proj.weight

        self.assertFalse(torch.allclose(before_q, after_q))
        self.assertFalse(torch.allclose(before_o, after_o))
        self.assertFalse(torch.allclose(before_gate, after_gate))
        self.assertFalse(torch.allclose(before_down, after_down))

    @torch.inference_mode()
    def test_convert_updates_rotation_side_layers(self):
        model = self._build_llama_model()
        q_m = prepare(model, SpinQuantConfig(init_method="random"))

        before_embed_rot = q_m.model.rotate_embedding.weight.detach().clone()
        before_lm_head_rot = q_m.rotate_lm_head.weight.detach().clone()

        q_m = convert(q_m)

        after_embed_rot = q_m.model.rotate_embedding.weight
        after_lm_head_rot = q_m.rotate_lm_head.weight

        self.assertFalse(torch.allclose(before_embed_rot, after_embed_rot))
        self.assertFalse(torch.allclose(before_lm_head_rot, after_lm_head_rot))

    @torch.inference_mode()
    def test_convert_resets_folded_layer_norms_to_identity(self):
        model = self._build_llama_model()
        q_m = prepare(model, SpinQuantConfig(init_method="random"))
        q_m = convert(q_m)

        for layer in q_m.model.layers:
            self.assertTrue(
                torch.allclose(
                    layer.input_layernorm.weight,
                    torch.ones_like(layer.input_layernorm.weight),
                )
            )
            self.assertTrue(
                torch.allclose(
                    layer.post_attention_layernorm.weight,
                    torch.ones_like(layer.post_attention_layernorm.weight),
                )
            )

        self.assertTrue(
            torch.allclose(
                q_m.model.norm.weight,
                torch.ones_like(q_m.model.norm.weight),
            )
        )

    @torch.inference_mode()
    def test_quantizer_convert_with_external_identity_r1(self):
        model = self._build_llama_model(hidden_size=32, num_attention_heads=4)

        hidden_size = model.config.hidden_size
        head_dim = hidden_size // model.config.num_attention_heads

        r1 = torch.eye(hidden_size, dtype=torch.float64)
        r2_map = {
            f"model.layers.{idx}.self_attn.R2": torch.eye(head_dim, dtype=torch.float64)
            for idx in range(model.config.num_hidden_layers)
        }

        quantizer = SpinQuantQuantizer(
            SpinQuantConfig(
                init_method="external",
                r1=r1,
                r2_map=r2_map,
            )
        )

        q_m = quantizer.prepare(model)
        q_m = quantizer.convert(q_m)

        expected_embed_rot = torch.eye(
            hidden_size,
            device=q_m.model.rotate_embedding.weight.device,
            dtype=q_m.model.rotate_embedding.weight.dtype,
        )
        self.assertTrue(
            torch.allclose(q_m.model.rotate_embedding.weight, expected_embed_rot)
        )

        # The final norm scale is folded into rotate_lm_head, so this layer should
        # become a diagonal matrix equal to the original final norm weights.
        expected_lm_rot = torch.diag(
            model.model.norm.weight.detach().to(
                device=q_m.rotate_lm_head.weight.device,
                dtype=q_m.rotate_lm_head.weight.dtype,
            )
        )
        self.assertTrue(torch.allclose(q_m.rotate_lm_head.weight, expected_lm_rot))

    def test_quantizer_prepare_rejects_non_module(self):
        quantizer = SpinQuantQuantizer(SpinQuantConfig())
        with self.assertRaises(TypeError):
            quantizer.prepare("not a module")  # type: ignore[arg-type]

    def test_quantizer_prepare_rejects_non_llama_model_type(self):
        class DummyConfig:
            model_type = "not_llama"

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = DummyConfig()
                self.model = torch.nn.Module()
                self.lm_head = torch.nn.Linear(4, 4, bias=False)

        quantizer = SpinQuantQuantizer(SpinQuantConfig())
        with self.assertRaises(ValueError):
            quantizer.prepare(DummyModel())

    @torch.inference_mode()
    def test_forward_runs_after_spinquant_prepare_and_convert(self):
        model = self._build_llama_model()
        q_m = prepare(model, SpinQuantConfig(init_method="random"))
        q_m = convert(q_m)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        outputs = q_m(input_ids=input_ids)

        self.assertEqual(outputs.logits.shape[0], 1)
        self.assertEqual(outputs.logits.shape[1], 4)
        self.assertEqual(outputs.logits.shape[2], q_m.config.vocab_size)

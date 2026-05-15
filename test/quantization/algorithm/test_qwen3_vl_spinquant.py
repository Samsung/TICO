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

from __future__ import annotations

import unittest

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig

try:
    from tico.quantization.algorithm.spinquant.qwen3_vl_quantizer import (
        Qwen3VLSpinQuantQuantizer,
    )
    from tico.quantization.algorithm.spinquant.spin_qwen3_vl import (
        SpinQwen3VLForConditionalGeneration,
    )
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    QWEN3_VL_AVAILABLE = True
    QWEN3_VL_SKIP_REASON = ""
except (AttributeError, ImportError, ModuleNotFoundError) as exc:
    Qwen3VLConfig = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]
    Qwen3VLSpinQuantQuantizer = None  # type: ignore[assignment, arg-type, misc]
    SpinQwen3VLForConditionalGeneration = None  # type: ignore[assignment, arg-type, misc]
    QWEN3_VL_AVAILABLE = False
    QWEN3_VL_SKIP_REASON = f"Qwen3-VL transformers integration is unavailable: {exc}"


class Qwen3VLSpinQuantConfigTest(unittest.TestCase):
    def test_config_default_is_random(self):
        cfg = Qwen3VLSpinQuantConfig()
        self.assertEqual(cfg.init_method, "random")
        self.assertEqual(cfg.name, "spinquant")

    def test_config_accepts_hadamard(self):
        cfg = Qwen3VLSpinQuantConfig(init_method="hadamard")
        self.assertEqual(cfg.init_method, "hadamard")

    def test_config_requires_r1_for_external(self):
        with self.assertRaises(ValueError):
            Qwen3VLSpinQuantConfig(init_method="external")

    def test_config_rejects_non_tensor_r1(self):
        with self.assertRaises(TypeError):
            Qwen3VLSpinQuantConfig(
                init_method="random",
                r1="invalid",  # type: ignore[arg-type]
            )

    def test_config_rejects_non_dict_r2_map(self):
        with self.assertRaises(TypeError):
            Qwen3VLSpinQuantConfig(
                init_method="random",
                r2_map="invalid",  # type: ignore[arg-type]
            )

    def test_config_rejects_non_tensor_r2_value(self):
        with self.assertRaises(TypeError):
            Qwen3VLSpinQuantConfig(
                init_method="random",
                r2_map={"model.language_model.layers.0.self_attn.R2": "invalid"},  # type: ignore[dict-item]
            )


@unittest.skipUnless(QWEN3_VL_AVAILABLE, QWEN3_VL_SKIP_REASON)
class Qwen3VLSpinQuantTest(unittest.TestCase):
    def _build_qwen3_vl_model(
        self,
        *,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 2,
        head_dim: int = 8,
        vocab_size: int = 80,
        tie_word_embeddings: bool = True,
        deepstack_visual_indexes: list[int] | None = None,
    ) -> torch.nn.Module:
        """
        Build a tiny Qwen3-VL model for SpinQuant unit tests.

        Parameters:
            hidden_size: Text hidden dimension.
            intermediate_size: Text MLP intermediate dimension.
            num_hidden_layers: Number of text decoder layers.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value attention heads.
            head_dim: Attention head dimension.
            vocab_size: Text vocabulary size.
            tie_word_embeddings: Whether input embedding and LM head share storage.
            deepstack_visual_indexes: Vision layer indexes used for DeepStack outputs.

        Returns:
            A tiny Qwen3VLForConditionalGeneration instance.
        """
        assert Qwen3VLConfig is not None
        assert Qwen3VLForConditionalGeneration is not None

        if deepstack_visual_indexes is None:
            deepstack_visual_indexes = [0]

        text_config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "use_cache": False,
            "rope_theta": 10000.0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 1, 1],
            },
            "attention_bias": False,
            "attention_dropout": 0.0,
            "pad_token_id": 0,
        }
        vision_config = {
            "depth": 1,
            "hidden_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "out_hidden_size": hidden_size,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": deepstack_visual_indexes,
            "initializer_range": 0.02,
        }
        config = Qwen3VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=vocab_size - 4,
            video_token_id=vocab_size - 3,
            vision_start_token_id=vocab_size - 2,
            vision_end_token_id=vocab_size - 1,
            tie_word_embeddings=tie_word_embeddings,
        )
        self._patch_qwen3_vl_rope_config(config)
        model = Qwen3VLForConditionalGeneration(config)
        self._force_qwen3_vl_embedding_tie(
            model, tie_word_embeddings=tie_word_embeddings
        )
        model.eval()
        return model

    def _force_qwen3_vl_embedding_tie(
        self,
        model: torch.nn.Module,
        *,
        tie_word_embeddings: bool,
    ) -> None:
        """
        Force the synthetic Qwen3-VL model to match the requested tie setting.

        Some transformers versions do not automatically tie ``lm_head`` to
        ``model.language_model.embed_tokens`` when a model is constructed from a
        tiny synthetic config. Real checkpoints can still be tied after loading,
        but unit tests need the in-memory test model to expose the same storage
        alias before SpinQuant validation runs.

        Parameters:
            model: Qwen3-VL model to update in-place.
            tie_word_embeddings: Whether to tie or explicitly untie the weights.
        """
        model.config.tie_word_embeddings = tie_word_embeddings

        if tie_word_embeddings:
            try:
                model.tie_weights()
            except Exception:
                pass

            if (
                model.model.language_model.embed_tokens.weight.data_ptr()
                != model.lm_head.weight.data_ptr()
            ):
                model.lm_head.weight = model.model.language_model.embed_tokens.weight
            return

        if (
            model.model.language_model.embed_tokens.weight.data_ptr()
            == model.lm_head.weight.data_ptr()
        ):
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone()
            )

    def _patch_qwen3_vl_rope_config(self, config) -> None:
        """
        Patch tiny Qwen3-VL RoPE config fields for transformers version drift.

        Some Qwen3-VL releases read ``text_config.rope_parameters`` while other
        releases read ``text_config.rope_scaling`` directly. Unit tests construct
        a tiny synthetic config, so both fields are populated to avoid depending
        on one specific transformers minor version.

        Parameters:
            config: Qwen3-VL top-level configuration to patch in-place.
        """
        text_config = config.text_config
        mrope_section = [2, 1, 1]

        rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": mrope_section,
        }
        rope_scaling = {
            "type": "default",
            "rope_type": "default",
            "mrope_section": mrope_section,
        }

        if getattr(text_config, "rope_parameters", None) is None:
            text_config.rope_parameters = dict(rope_parameters)

        if getattr(text_config, "rope_scaling", None) is None:
            text_config.rope_scaling = dict(rope_scaling)

        if not hasattr(text_config, "rope_theta"):
            text_config.rope_theta = 10000.0

    def _force_tied_qwen3_vl_word_embeddings(self, model: torch.nn.Module) -> None:
        """
        Force Qwen3-VL input embeddings and LM head to share storage.

        Some transformers releases do not tie synthetic Qwen3-VL models even
        when ``tie_word_embeddings=True`` is passed to the tiny test config.
        The production SpinQuant path assumes tied embeddings, so the test
        helper explicitly creates the same storage alias used by real tied
        checkpoints.

        Parameters:
            model: Target Qwen3-VL model.
        """
        model.config.tie_word_embeddings = True
        model.lm_head.weight = model.model.language_model.embed_tokens.weight

    def _force_untied_qwen3_vl_word_embeddings(self, model: torch.nn.Module) -> None:
        """
        Force Qwen3-VL input embeddings and LM head to use separate storage.

        Parameters:
            model: Target Qwen3-VL model.
        """
        model.config.tie_word_embeddings = False
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

    def _clone_state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """
        Clone a model state_dict into detached tensors.

        Parameters:
            model: Source model.

        Returns:
            A copied state_dict.
        """
        return {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    def _assert_identity_linear(self, layer: torch.nn.Linear) -> None:
        """
        Assert that a Linear layer is initialized as identity.

        Parameters:
            layer: Target Linear layer.
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

    def _assert_tied_word_embedding(self, model: torch.nn.Module) -> None:
        """
        Assert that a Qwen3-VL model has tied input and output embeddings.

        Parameters:
            model: Target Qwen3-VL model.
        """
        embed_weight = model.model.language_model.embed_tokens.weight
        lm_head_weight = model.lm_head.weight
        self.assertEqual(embed_weight.data_ptr(), lm_head_weight.data_ptr())

    def _make_permutation_rotation(
        self,
        size: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Create a deterministic orthogonal permutation rotation.

        Parameters:
            size: Matrix size.
            device: Optional target device.

        Returns:
            A square permutation matrix with dtype float64.
        """
        return torch.eye(size, device=device, dtype=torch.float64).roll(
            shifts=1, dims=1
        )

    def _make_identity_r2_map(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """
        Build an identity R2 map for all Qwen3-VL text layers.

        Parameters:
            model: Target Qwen3-VL model.

        Returns:
            Mapping from supported R2 keys to identity matrices.
        """
        text_config = model.config.text_config
        head_dim = int(text_config.head_dim)
        return {
            f"model.language_model.layers.{idx}.self_attn.R2": torch.eye(
                head_dim,
                dtype=torch.float64,
            )
            for idx in range(int(text_config.num_hidden_layers))
        }

    def _make_external_config(
        self,
        model: torch.nn.Module,
        r1: torch.Tensor,
        *,
        apply_r2: bool = True,
    ) -> Qwen3VLSpinQuantConfig:
        """
        Build an external-rotation Qwen3-VL SpinQuant configuration.

        Parameters:
            model: Target Qwen3-VL model.
            r1: Global hidden-dimension rotation.
            apply_r2: Whether to apply identity R2 rotations.

        Returns:
            A Qwen3VLSpinQuantConfig instance.
        """
        return Qwen3VLSpinQuantConfig(
            init_method="external",
            r1=r1,
            r2_map=self._make_identity_r2_map(model) if apply_r2 else None,
            apply_r2=apply_r2,
            show_progress=False,
        )

    @torch.inference_mode()
    def test_prepare_converts_qwen3_vl_to_spin_qwen3_vl(self):
        model = self._build_qwen3_vl_model()
        cfg = Qwen3VLSpinQuantConfig(show_progress=False)

        q_m = prepare(model, cfg)

        self.assertIsInstance(q_m, SpinQwen3VLForConditionalGeneration)
        self.assertTrue(hasattr(q_m.model.language_model, "rotate_embedding"))
        self.assertTrue(hasattr(q_m, "rotate_lm_head"))
        self._assert_identity_linear(q_m.model.language_model.rotate_embedding)
        self._assert_identity_linear(q_m.rotate_lm_head)

    @torch.inference_mode()
    def test_prepare_preserves_generation_related_attributes(self):
        model = self._build_qwen3_vl_model()
        model.name_or_path = "dummy-qwen3-vl"
        model._keep_in_fp32_modules = {"lm_head"}
        model.hf_device_map = {"": "cpu"}

        q_m = prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))

        self.assertEqual(q_m.name_or_path, "dummy-qwen3-vl")
        self.assertEqual(q_m._keep_in_fp32_modules, {"lm_head"})
        self.assertEqual(q_m.hf_device_map, {"": "cpu"})
        self.assertIs(q_m.config, model.config)

    @torch.inference_mode()
    def test_prepare_preserves_original_weights_before_convert(self):
        model = self._build_qwen3_vl_model()
        original_state = self._clone_state_dict(model)

        q_m = prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))

        self.assertTrue(
            torch.allclose(
                q_m.model.language_model.embed_tokens.weight,
                original_state["model.language_model.embed_tokens.weight"],
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
                q_m.model.language_model.layers[0].self_attn.q_proj.weight,
                original_state["model.language_model.layers.0.self_attn.q_proj.weight"],
            )
        )

    @torch.inference_mode()
    def test_prepare_preserves_tied_embedding_sharing(self):
        model = self._build_qwen3_vl_model(tie_word_embeddings=True)
        self._assert_tied_word_embedding(model)

        q_m = prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))

        self._assert_tied_word_embedding(q_m)

    @torch.inference_mode()
    def test_prepare_rejects_untied_embedding(self):
        model = self._build_qwen3_vl_model(tie_word_embeddings=False)
        self.assertNotEqual(
            model.model.language_model.embed_tokens.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )

        with self.assertRaises(ValueError):
            prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))

    @torch.inference_mode()
    def test_prepare_identity_forward_matches_original(self):
        torch.manual_seed(0)
        model = self._build_qwen3_vl_model()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        original_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        q_m = prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))
        prepared_logits = q_m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        self.assertTrue(
            torch.allclose(original_logits, prepared_logits, atol=1e-5, rtol=1e-5)
        )

    @torch.inference_mode()
    def test_rotate_embedding_is_on_forward_path(self):
        model = self._build_qwen3_vl_model()
        q_m = prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        q_m.model.language_model.rotate_embedding.weight.zero_()
        logits = q_m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        self.assertTrue(
            torch.allclose(logits, torch.zeros_like(logits), atol=1e-6, rtol=0.0)
        )

    @torch.inference_mode()
    def test_rotate_lm_head_is_on_forward_path(self):
        model = self._build_qwen3_vl_model()
        q_m = prepare(model, Qwen3VLSpinQuantConfig(show_progress=False))

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        q_m.rotate_lm_head.weight.zero_()
        logits = q_m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        self.assertTrue(
            torch.allclose(logits, torch.zeros_like(logits), atol=1e-6, rtol=0.0)
        )

    @torch.inference_mode()
    def test_convert_with_external_r1_preserves_text_logits(self):
        torch.manual_seed(0)
        model = self._build_qwen3_vl_model()
        hidden_size = int(model.config.text_config.hidden_size)
        r1 = self._make_permutation_rotation(hidden_size)
        cfg = self._make_external_config(model, r1, apply_r2=True)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        original_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        q_m = prepare(model, cfg)
        q_m = convert(q_m)
        converted_logits = q_m(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

        self.assertTrue(
            torch.allclose(
                original_logits,
                converted_logits,
                atol=1e-4,
                rtol=1e-4,
            )
        )

    @torch.inference_mode()
    def test_convert_updates_runtime_boundary_layers(self):
        model = self._build_qwen3_vl_model()
        hidden_size = int(model.config.text_config.hidden_size)
        r1 = self._make_permutation_rotation(hidden_size)
        cfg = self._make_external_config(model, r1, apply_r2=False)

        q_m = prepare(model, cfg)
        final_norm_scale = q_m.model.language_model.norm.weight.detach().clone()

        q_m = convert(q_m)

        expected_rotate_embedding = r1.T.to(
            device=q_m.model.language_model.rotate_embedding.weight.device,
            dtype=q_m.model.language_model.rotate_embedding.weight.dtype,
        )
        expected_rotate_lm_head = (
            final_norm_scale.to(device=r1.device, dtype=torch.float64).view(-1, 1) * r1
        ).to(
            device=q_m.rotate_lm_head.weight.device,
            dtype=q_m.rotate_lm_head.weight.dtype,
        )

        self.assertTrue(
            torch.allclose(
                q_m.model.language_model.rotate_embedding.weight,
                expected_rotate_embedding,
            )
        )
        self.assertTrue(
            torch.allclose(q_m.rotate_lm_head.weight, expected_rotate_lm_head)
        )

    @torch.inference_mode()
    def test_convert_resets_folded_text_norms_to_identity(self):
        model = self._build_qwen3_vl_model()
        hidden_size = int(model.config.text_config.hidden_size)
        r1 = self._make_permutation_rotation(hidden_size)
        cfg = self._make_external_config(model, r1, apply_r2=False)

        q_m = prepare(model, cfg)
        q_m = convert(q_m)

        for layer in q_m.model.language_model.layers:
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
                q_m.model.language_model.norm.weight,
                torch.ones_like(q_m.model.language_model.norm.weight),
            )
        )

    @torch.inference_mode()
    def test_convert_changes_decoder_weights_but_keeps_tied_embedding(self):
        model = self._build_qwen3_vl_model()
        hidden_size = int(model.config.text_config.hidden_size)
        r1 = self._make_permutation_rotation(hidden_size)
        cfg = self._make_external_config(model, r1, apply_r2=False)

        q_m = prepare(model, cfg)
        before_q = (
            q_m.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()
        )
        before_o = (
            q_m.model.language_model.layers[0].self_attn.o_proj.weight.detach().clone()
        )
        before_gate = (
            q_m.model.language_model.layers[0].mlp.gate_proj.weight.detach().clone()
        )
        before_down = (
            q_m.model.language_model.layers[0].mlp.down_proj.weight.detach().clone()
        )

        q_m = convert(q_m)

        self.assertFalse(
            torch.allclose(
                before_q, q_m.model.language_model.layers[0].self_attn.q_proj.weight
            )
        )
        self.assertFalse(
            torch.allclose(
                before_o, q_m.model.language_model.layers[0].self_attn.o_proj.weight
            )
        )
        self.assertFalse(
            torch.allclose(
                before_gate, q_m.model.language_model.layers[0].mlp.gate_proj.weight
            )
        )
        self.assertFalse(
            torch.allclose(
                before_down, q_m.model.language_model.layers[0].mlp.down_proj.weight
            )
        )
        self._assert_tied_word_embedding(q_m)

    @torch.inference_mode()
    def test_convert_fuses_deepstack_outputs_but_not_main_visual_merger(self):
        model = self._build_qwen3_vl_model(deepstack_visual_indexes=[0])
        hidden_size = int(model.config.text_config.hidden_size)
        r1 = self._make_permutation_rotation(hidden_size)
        cfg = self._make_external_config(model, r1, apply_r2=False)

        q_m = prepare(model, cfg)
        main_before = q_m.model.visual.merger.linear_fc2.weight.detach().clone()
        deep_before = (
            q_m.model.visual.deepstack_merger_list[0].linear_fc2.weight.detach().clone()
        )

        q_m = convert(q_m)

        main_after = q_m.model.visual.merger.linear_fc2.weight
        deep_after = q_m.model.visual.deepstack_merger_list[0].linear_fc2.weight
        expected_deep = (
            r1.T.to(device=deep_before.device, dtype=deep_before.dtype) @ deep_before
        )

        self.assertTrue(torch.allclose(main_after, main_before))
        self.assertTrue(torch.allclose(deep_after, expected_deep))

    @torch.inference_mode()
    def test_quantizer_direct_prepare_and_convert_with_identity_external_rotation(self):
        assert Qwen3VLSpinQuantQuantizer is not None

        model = self._build_qwen3_vl_model()
        hidden_size = int(model.config.text_config.hidden_size)
        r1 = torch.eye(hidden_size, dtype=torch.float64)
        cfg = self._make_external_config(model, r1, apply_r2=True)
        quantizer = Qwen3VLSpinQuantQuantizer(cfg)

        q_m = quantizer.prepare(model)
        q_m = quantizer.convert(q_m)

        expected_identity = torch.eye(
            hidden_size,
            device=q_m.model.language_model.rotate_embedding.weight.device,
            dtype=q_m.model.language_model.rotate_embedding.weight.dtype,
        )
        self.assertTrue(
            torch.allclose(
                q_m.model.language_model.rotate_embedding.weight, expected_identity
            )
        )
        self._assert_tied_word_embedding(q_m)


if __name__ == "__main__":
    unittest.main()

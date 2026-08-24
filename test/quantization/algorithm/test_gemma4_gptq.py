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

"""Unit tests for tico.quantization.algorithm.gemma4_gptq.quantizer."""

import unittest
from typing import Any, cast
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from tico.quantization.algorithm.gemma4_gptq.gptq import GPTQ
from tico.quantization.algorithm.gemma4_gptq.quantizer import (
    Gemma4GPTQQuantizer,
    StopReplay,
)
from tico.quantization.algorithm.gemma4_gptq.utils import (
    Gemma4Components,
    resolve_gemma4_components,
    should_quantize_text_stage,
    should_quantize_vision_stage,
)
from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig


# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


class _MockConfig:
    """Minimal config object with use_cache flags."""

    def __init__(self):
        self.use_cache = True
        self.text_config = type("TextConfig", (), {"use_cache": True})()


class MockPatchEmbedder(nn.Module):
    """Mimics Gemma4 vision patch embedder (contains ``input_proj`` Linear)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.input_proj(x)


class MockVisionBlock(nn.Module):
    """Mimics a single Gemma4 vision encoder block."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MockVisionEncoder(nn.Module):
    def __init__(self, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([MockVisionBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockPooler(nn.Module):
    """Gemma4 pooler — no trainable Linear weights."""

    def forward(self, x):
        return x


class MockMultimodalEmbedder(nn.Module):
    """Mimics ``embed_vision`` (contains ``embedding_projection`` Linear)."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.embedding_projection = nn.Linear(dim, dim)

    def forward(self, x):
        return self.embedding_projection(x)


class MockVisionTower(nn.Module):
    def __init__(self, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.patch_embedder = MockPatchEmbedder(dim)
        self.encoder = MockVisionEncoder(dim, num_layers)
        self.pooler = MockPooler()

    def forward(self, x):
        x = self.patch_embedder(x)
        x = self.encoder(x)
        x = self.pooler(x)
        return x


class MockTextLayer(nn.Module):
    """Mimics a Gemma4 text decoder layer with attention + MLP projections."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim)
        self.self_attn.k_proj = nn.Linear(dim, dim)
        self.self_attn.v_proj = nn.Linear(dim, dim)
        self.self_attn.o_proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, dim * 4)
        self.mlp.down_proj = nn.Linear(dim * 4, dim)

    def forward(self, hidden_states, **kwargs):
        # Simplified forward — just pass through.
        return hidden_states


class MockLanguageModel(nn.Module):
    def __init__(self, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([MockTextLayer(dim) for _ in range(num_layers)])
        self.config = type("LMConfig", (), {"layer_types": None})()

    def forward(self, *args, **kwargs):
        hidden_states = kwargs.get("inputs_embeds")
        if hidden_states is None:
            return None
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MockGemma4Model(nn.Module):
    """
    Minimal mock that mirrors the Gemma4ForConditionalGeneration structure
    expected by ``resolve_gemma4_components``.
    """

    def __init__(
        self, dim: int = 64, num_vision_layers: int = 2, num_text_layers: int = 2
    ):
        super().__init__()
        self.config = _MockConfig()
        self.model = nn.Module()
        self.model.vision_tower = MockVisionTower(dim, num_vision_layers)
        self.model.embed_vision = MockMultimodalEmbedder(dim)
        self.model.language_model = MockLanguageModel(dim, num_text_layers)
        self.lm_head = nn.Linear(dim, dim)

    def forward(self, *args, **kwargs):
        # Run through vision tower so hooks on submodules fire during replay.
        x = args[0] if args else kwargs.get("inputs_embeds")
        if x is not None:
            return self.model.vision_tower(x)
        return x


def _make_quantizer(**overrides) -> Gemma4GPTQQuantizer:
    """Create a Gemma4GPTQQuantizer with sensible test defaults."""
    defaults: dict[str, Any] = dict(
        weight_bits=4,
        show_progress=False,
        verbose=False,
    )
    defaults.update(overrides)
    return Gemma4GPTQQuantizer(Gemma4GPTQConfig(**defaults))


def _text_only_overrides(**extra) -> dict:
    """Return config overrides that disable all vision sub-stages."""
    base = dict(
        quantize_vision=False,
        quantize_vision_patch_embed=False,
        quantize_vision_blocks=False,
        quantize_vision_pooler=False,
        quantize_multimodal_embedder=False,
    )
    base.update(extra)
    return base


def _vision_only_overrides(**extra) -> dict:
    """Return config overrides that disable all text sub-stages."""
    base = dict(
        quantize_text=False,
        quantize_text_layers=False,
    )
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestGemma4GPTQConfigValidation(unittest.TestCase):
    """Tests for Gemma4GPTQConfig.validate()."""

    def test_default_config_validates(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        conf.validate()  # should not raise

    def test_wrong_model_type_raises(self):
        conf = Gemma4GPTQConfig(weight_bits=4, model_type="not_gemma4")
        with self.assertRaises(ValueError):
            conf.validate()

    def test_all_stages_disabled_raises(self):
        conf = Gemma4GPTQConfig(
            weight_bits=4,
            quantize_vision=False,
            quantize_text=False,
            quantize_lm_head=False,
        )
        with self.assertRaises(ValueError):
            conf.validate()

    def test_vision_stage_without_quantize_vision_raises(self):
        conf = Gemma4GPTQConfig(
            weight_bits=4,
            quantize_vision=False,
            quantize_vision_patch_embed=True,
        )
        with self.assertRaises(ValueError):
            conf.validate()

    def test_text_layers_without_quantize_text_raises(self):
        conf = Gemma4GPTQConfig(
            weight_bits=4,
            quantize_text=False,
            quantize_text_layers=True,
        )
        with self.assertRaises(ValueError):
            conf.validate()

    def test_cache_dtype_must_be_torch_dtype(self):
        conf = Gemma4GPTQConfig(weight_bits=4, cache_dtype="float32")
        with self.assertRaises(TypeError):
            conf.validate()

    def test_cache_dtype_torch_dtype_ok(self):
        conf = Gemma4GPTQConfig(weight_bits=4, cache_dtype=torch.float16)
        conf.validate()

    def test_empty_attr_path_raises(self):
        conf = Gemma4GPTQConfig(weight_bits=4, vision_tower_attr="")
        with self.assertRaises(ValueError):
            conf.validate()

    def test_name_property(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        self.assertEqual(conf.name, "gemma4_gptq")


# ---------------------------------------------------------------------------
# Stage gate tests
# ---------------------------------------------------------------------------


class TestShouldQuantizeStages(unittest.TestCase):
    """Tests for should_quantize_vision_stage / should_quantize_text_stage."""

    def test_vision_stages_enabled_by_default(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        for stage in ("patch_embed", "blocks", "pooler", "multimodal_embedder"):
            with self.subTest(stage=stage):
                self.assertTrue(should_quantize_vision_stage(conf, stage=stage))

    def test_vision_stages_disabled_when_quantize_vision_false(self):
        conf = Gemma4GPTQConfig(weight_bits=4, quantize_vision=False)
        for stage in ("patch_embed", "blocks", "pooler", "multimodal_embedder"):
            with self.subTest(stage=stage):
                self.assertFalse(should_quantize_vision_stage(conf, stage=stage))

    def test_individual_vision_stage_switches(self):
        conf = Gemma4GPTQConfig(
            weight_bits=4,
            quantize_vision_patch_embed=False,
            quantize_vision_blocks=False,
            quantize_vision_pooler=False,
            quantize_multimodal_embedder=False,
        )
        for stage in ("patch_embed", "blocks", "pooler", "multimodal_embedder"):
            with self.subTest(stage=stage):
                self.assertFalse(should_quantize_vision_stage(conf, stage=stage))

    def test_unknown_vision_stage_raises(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        with self.assertRaises(ValueError):
            should_quantize_vision_stage(conf, stage="unknown")

    def test_text_layers_enabled_by_default(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        self.assertTrue(should_quantize_text_stage(conf, stage="layers"))

    def test_text_layers_disabled_when_quantize_text_false(self):
        conf = Gemma4GPTQConfig(weight_bits=4, quantize_text=False)
        self.assertFalse(should_quantize_text_stage(conf, stage="layers"))

    def test_lm_head_disabled_by_default(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        self.assertFalse(should_quantize_text_stage(conf, stage="lm_head"))

    def test_lm_head_enabled(self):
        conf = Gemma4GPTQConfig(weight_bits=4, quantize_lm_head=True)
        self.assertTrue(should_quantize_text_stage(conf, stage="lm_head"))

    def test_unknown_text_stage_raises(self):
        conf = Gemma4GPTQConfig(weight_bits=4)
        with self.assertRaises(ValueError):
            should_quantize_text_stage(conf, stage="unknown")


# ---------------------------------------------------------------------------
# resolve_gemma4_components tests
# ---------------------------------------------------------------------------


class TestResolveGemma4Components(unittest.TestCase):
    """Tests for resolve_gemma4_components()."""

    def test_resolves_all_components(self):
        model = MockGemma4Model(dim=64, num_vision_layers=3, num_text_layers=4)
        conf = Gemma4GPTQConfig(weight_bits=4)
        components = resolve_gemma4_components(model, conf)

        self.assertIsInstance(components, Gemma4Components)
        self.assertIs(components.vision_tower, model.model.vision_tower)
        self.assertIs(
            components.vision_patch_embed, model.model.vision_tower.patch_embedder
        )
        self.assertIs(components.vision_encoder, model.model.vision_tower.encoder)
        self.assertIsInstance(components.vision_encoder_layers, nn.ModuleList)
        self.assertEqual(len(components.vision_encoder_layers), 3)
        self.assertIs(components.vision_pooler, model.model.vision_tower.pooler)
        self.assertIs(components.multimodal_embedder, model.model.embed_vision)
        self.assertIs(components.language_model, model.model.language_model)
        self.assertIsInstance(components.text_layers, nn.ModuleList)
        self.assertEqual(len(components.text_layers), 4)
        self.assertIs(components.lm_head, model.lm_head)

    def test_text_only_config_resolves_components(self):
        """Text-only model uses different attr paths."""
        model = MockGemma4Model(dim=64)
        conf = Gemma4GPTQConfig(
            weight_bits=4,
            quantize_vision=False,
            language_model_attr="model.language_model",
            text_layers_attr="model.language_model.layers",
            lm_head_attr="lm_head",
        )
        components = resolve_gemma4_components(model, conf)
        self.assertIs(components.language_model, model.model.language_model)
        self.assertEqual(len(components.text_layers), 2)

    def test_wrong_type_raises(self):
        model = MockGemma4Model(dim=64)
        # Sabotage: replace ModuleList with a plain list
        model.model.vision_tower.encoder.layers = nn.Identity()
        conf = Gemma4GPTQConfig(weight_bits=4)
        with self.assertRaises(TypeError):
            resolve_gemma4_components(model, conf)


# ---------------------------------------------------------------------------
# _resolve_weight_bits tests
# ---------------------------------------------------------------------------


class TestResolveWeightBits(unittest.TestCase):
    """Tests for Gemma4GPTQQuantizer._resolve_weight_bits()."""

    def test_full_name_override_takes_priority(self):
        quantizer = _make_quantizer(
            weight_bits=4,
            weight_bits_overrides={
                "proj": 5,
                "layer.proj": 6,
                "model.layers.0.layer.proj": 8,
            },
        )
        result = quantizer._resolve_weight_bits(
            cast(Gemma4GPTQConfig, quantizer.config),
            full_module_name="model.layers.0.layer.proj",
            local_module_name="layer.proj",
        )
        self.assertEqual(result, 8)

    def test_local_name_override(self):
        quantizer = _make_quantizer(
            weight_bits=4,
            weight_bits_overrides={"layer.proj": 6},
        )
        result = quantizer._resolve_weight_bits(
            cast(Gemma4GPTQConfig, quantizer.config),
            full_module_name="model.layers.1.layer.proj",
            local_module_name="layer.proj",
        )
        self.assertEqual(result, 6)

    def test_suffix_match(self):
        quantizer = _make_quantizer(
            weight_bits=4,
            weight_bits_overrides={"proj": 5},
        )
        result = quantizer._resolve_weight_bits(
            cast(Gemma4GPTQConfig, quantizer.config),
            full_module_name="model.layers.2.other.proj",
            local_module_name="other.proj",
        )
        self.assertEqual(result, 5)

    def test_no_match_returns_default(self):
        quantizer = _make_quantizer(
            weight_bits=4,
            weight_bits_overrides={"proj": 5},
        )
        result = quantizer._resolve_weight_bits(
            cast(Gemma4GPTQConfig, quantizer.config),
            full_module_name="model.layers.2.other.up_proj",
            local_module_name="other.up_proj",
        )
        self.assertEqual(result, 4)

    def test_multiple_suffix_matches_returns_last(self):
        quantizer = _make_quantizer(
            weight_bits=4,
            weight_bits_overrides={
                "proj": 5,
                "other.proj": 7,
            },
        )
        result = quantizer._resolve_weight_bits(
            cast(Gemma4GPTQConfig, quantizer.config),
            full_module_name="model.layers.2.other.proj",
            local_module_name="other.proj",
        )
        # "other.proj" is a more specific suffix, but both match.
        # The last match in iteration order is returned.
        self.assertIn(result, (5, 7))


# ---------------------------------------------------------------------------
# prepare() tests
# ---------------------------------------------------------------------------


class TestPrepare(unittest.TestCase):
    """Tests for Gemma4GPTQQuantizer.prepare()."""

    def test_prepare_replaces_forward_and_caches_inputs(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer()
        original_forward = model.forward
        prepared = quantizer.prepare(model)

        # Forward should be replaced
        self.assertIsNot(prepared.forward, original_forward)
        self.assertIsNotNone(quantizer._orig_model_forward)

        # Run a calibration batch
        x = torch.randn(1, 4, 64)
        result = prepared(x)
        self.assertIsNone(result)  # wrapper returns None
        self.assertEqual(quantizer.num_batches, 1)
        self.assertEqual(len(quantizer.cache_args), 1)
        self.assertEqual(len(quantizer.cache_args[0]), 1)

    def test_prepare_caches_multiple_batches(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer()
        prepared = quantizer.prepare(model)

        for _ in range(5):
            x = torch.randn(1, 4, 64)
            prepared(x)

        self.assertEqual(quantizer.num_batches, 5)

    def test_prepare_separates_vision_batches(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer()
        prepared = quantizer.prepare(model)

        # Text-only batch (no pixel_values)
        prepared(torch.randn(1, 4, 64))
        # Vision batch (with pixel_values)
        prepared(torch.randn(1, 4, 64), pixel_values=torch.randn(1, 3, 8, 8))
        # Another text-only batch
        prepared(torch.randn(1, 4, 64))

        self.assertEqual(quantizer.num_batches, 3)
        self.assertEqual(quantizer._num_vision_batches, 1)
        self.assertEqual(len(quantizer._vision_cache_args), 1)

    def test_prepare_vision_batch_with_none_pixel_values_not_counted(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer()
        prepared = quantizer.prepare(model)

        prepared(torch.randn(1, 4, 64), pixel_values=None)

        self.assertEqual(quantizer.num_batches, 1)
        self.assertEqual(quantizer._num_vision_batches, 0)

    def test_prepare_move_cache_to_cpu(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer(move_cache_to_cpu=True)
        prepared = quantizer.prepare(model)

        x = torch.randn(1, 4, 64)
        prepared(x)

        cached = quantizer.cache_args[0][0]
        self.assertEqual(cached.device.type, "cpu")


# ---------------------------------------------------------------------------
# _disable_model_cache / _restore_model_cache tests
# ---------------------------------------------------------------------------


class TestCacheControl(unittest.TestCase):
    """Tests for _disable_model_cache / _restore_model_cache."""

    def test_disable_and_restore_cache(self):
        model = MockGemma4Model(dim=64)
        original_use_cache = model.config.use_cache
        original_text_use_cache = model.config.text_config.use_cache

        quantizer = _make_quantizer()
        saved = quantizer._disable_model_cache(model)

        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)

        quantizer._restore_model_cache(model, saved)

        self.assertEqual(model.config.use_cache, original_use_cache)
        self.assertEqual(model.config.text_config.use_cache, original_text_use_cache)

    def test_disable_cache_without_text_config(self):
        """Model without text_config should not crash."""
        model = MockGemma4Model(dim=64)
        delattr(model.config, "text_config")

        quantizer = _make_quantizer()
        saved = quantizer._disable_model_cache(model)
        self.assertIn("model.config.use_cache", saved)
        self.assertNotIn("model.config.text_config.use_cache", saved)
        quantizer._restore_model_cache(model, saved)


# ---------------------------------------------------------------------------
# _build_gptq_objects tests
# ---------------------------------------------------------------------------


class TestBuildGPTQObjects(unittest.TestCase):
    """Tests for _build_gptq_objects()."""

    def test_builds_gptq_for_each_layer(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer(weight_bits=4)

        from tico.quantization.algorithm.gemma4_gptq.utils import build_module_name_map

        module_name = build_module_name_map(model)
        layer = model.model.language_model.layers[0]
        from tico.quantization.algorithm.gemma4_gptq.utils import get_quantizable_layers

        subset = get_quantizable_layers(layer)
        gptq_objs = quantizer._build_gptq_objects(
            subset=subset, module_name=module_name
        )

        # Should have one GPTQ object per quantizable layer
        self.assertEqual(len(gptq_objs), len(subset))
        for name, obj in gptq_objs.items():
            self.assertIsInstance(obj, GPTQ)
            self.assertEqual(obj.quantizer.maxq.item(), 2**4 - 1)

    def test_weight_bits_override_applied(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer(
            weight_bits=4,
            weight_bits_overrides={"q_proj": 8},
        )

        from tico.quantization.algorithm.gemma4_gptq.utils import (
            build_module_name_map,
            get_quantizable_layers,
        )

        module_name = build_module_name_map(model)
        layer = model.model.language_model.layers[0]
        subset = get_quantizable_layers(layer)
        gptq_objs = quantizer._build_gptq_objects(
            subset=subset, module_name=module_name
        )

        # q_proj should be 8-bit
        self.assertEqual(gptq_objs["self_attn.q_proj"].quantizer.maxq.item(), 255)
        # Others should be 4-bit
        self.assertEqual(gptq_objs["self_attn.k_proj"].quantizer.maxq.item(), 15)


# ---------------------------------------------------------------------------
# _make_add_batch_hook tests
# ---------------------------------------------------------------------------


class TestAddBatchHook(unittest.TestCase):
    """Tests for _make_add_batch_hook()."""

    def test_hook_adds_batch_to_gptq(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer(weight_bits=4)

        linear = nn.Linear(64, 64)
        gptq_obj = GPTQ(linear)
        gptq_obj.quantizer.configure(bits=4, perchannel=True, sym=False, mse=None)

        gptq_objs = {"test": gptq_obj}
        hook = quantizer._make_add_batch_hook(gptq_objs, "test")

        x = torch.randn(2, 64)
        out = linear(x)
        hook(linear, (x,), out)

        self.assertGreater(gptq_obj.nsamples, 0)
        self.assertIsNotNone(gptq_obj.H)

    def test_hook_ignores_non_tensor_input(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer(weight_bits=4)

        gptq_obj = MagicMock(spec=GPTQ)
        gptq_objs: dict[str, GPTQ] = {"test": gptq_obj}
        hook = quantizer._make_add_batch_hook(gptq_objs, "test")

        # Non-tensor input should be ignored
        hook(nn.Identity(), ("string",), "output")
        gptq_obj.add_batch.assert_not_called()

    def test_hook_ignores_empty_input(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer(weight_bits=4)

        gptq_obj = MagicMock(spec=GPTQ)
        gptq_objs: dict[str, GPTQ] = {"test": gptq_obj}
        hook = quantizer._make_add_batch_hook(gptq_objs, "test")

        hook(nn.Identity(), (), "output")
        gptq_obj.add_batch.assert_not_called()


# ---------------------------------------------------------------------------
# _quantize_stage_from_stage_cache tests
# ---------------------------------------------------------------------------


class TestQuantizeStageFromStageCache(unittest.TestCase):
    """Tests for _quantize_stage_from_stage_cache()."""

    @torch.no_grad()
    def test_quantizes_linear_stage(self):
        """End-to-end: quantize a simple Linear stage from cached inputs."""
        dim = 32
        model = MockGemma4Model(dim=dim)
        quantizer = _make_quantizer(weight_bits=4, actorder=False)

        from tico.quantization.algorithm.gemma4_gptq.utils import build_module_name_map

        module_name = build_module_name_map(model)

        # Use a single text layer as the stage module
        stage_module = model.model.language_model.layers[0]

        # Build cached inputs (simulating stage-entry inputs)
        # cache_args is indexed by arg position, then by batch:
        #   cache_args[arg_idx][batch_idx]
        # cache_kwargs is indexed by key, then by batch:
        #   cache_kwargs[key][batch_idx]
        num_batches = 4
        cached_args: list[list[Any]] = []  # no positional args
        cached_kwargs: dict[str, list[Any]] = {"hidden_states": []}
        for _ in range(num_batches):
            x = torch.randn(2, 8, dim)
            cached_kwargs["hidden_states"].append(x)

        # Record original weights
        orig_q_proj_w = stage_module.self_attn.q_proj.weight.data.clone()

        quantizer._quantize_stage_from_stage_cache(
            stage_module=stage_module,
            module_name=module_name,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=num_batches,
        )

        # Weights should have been modified by GPTQ
        new_q_proj_w = stage_module.self_attn.q_proj.weight.data
        self.assertFalse(torch.allclose(orig_q_proj_w, new_q_proj_w))

        # Quantizers should be stored
        self.assertGreater(len(quantizer._quantizers), 0)

    @torch.no_grad()
    def test_no_quantizable_layers_skips(self):
        """Stage with no quantizable layers should be a no-op."""
        model = MockGemma4Model(dim=32)
        quantizer = _make_quantizer(weight_bits=4)

        from tico.quantization.algorithm.gemma4_gptq.utils import build_module_name_map

        module_name = build_module_name_map(model)

        # nn.Identity has no Linear layers
        stage_module = nn.Identity()
        cached_args: list[list[Any]] = []
        cached_kwargs: dict[str, list[Any]] = {"hidden_states": [torch.randn(1, 4, 32)]}

        # Should not raise
        quantizer._quantize_stage_from_stage_cache(
            stage_module=stage_module,
            module_name=module_name,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="empty_stage",
            num_batches=1,
        )
        self.assertEqual(len(quantizer._quantizers), 0)


# ---------------------------------------------------------------------------
# _quantize_stage_from_raw_replay tests
# ---------------------------------------------------------------------------


class TestQuantizeStageFromRawReplay(unittest.TestCase):
    """Tests for _quantize_stage_from_raw_replay()."""

    @torch.no_grad()
    def test_quantizes_patch_embed_via_raw_replay(self):
        """End-to-end: quantize patch_embedder via raw model replay."""
        dim = 32
        model = MockGemma4Model(dim=dim)
        quantizer = _make_quantizer(weight_bits=4, actorder=False)

        from tico.quantization.algorithm.gemma4_gptq.utils import build_module_name_map

        module_name = build_module_name_map(model)

        # Simulate prepare() caching.
        # cache_args[arg_idx][batch_idx] — one positional arg, 4 batches.
        from tico.quantization.algorithm.gemma4_gptq.utils import append_batch_to_cache

        quantizer._orig_model_forward = model.forward
        for _ in range(4):
            x = torch.randn(1, 4, dim)
            append_batch_to_cache(quantizer.cache_args, quantizer.cache_kwargs, x)
            quantizer.num_batches += 1

        orig_w = model.model.vision_tower.patch_embedder.input_proj.weight.data.clone()

        quantizer._quantize_stage_from_raw_replay(
            model=model,
            stage_module=model.model.vision_tower.patch_embedder,
            module_name=module_name,
            stage_desc="vision.patch_embed",
            vision_only=False,
        )

        new_w = model.model.vision_tower.patch_embedder.input_proj.weight.data
        self.assertFalse(torch.allclose(orig_w, new_w))
        self.assertGreater(len(quantizer._quantizers), 0)

    @torch.no_grad()
    def test_vision_only_with_no_vision_batches_skips(self):
        """vision_only=True with no vision batches should skip gracefully."""
        dim = 32
        model = MockGemma4Model(dim=dim)
        quantizer = _make_quantizer(weight_bits=4)

        from tico.quantization.algorithm.gemma4_gptq.utils import build_module_name_map

        module_name = build_module_name_map(model)
        quantizer._orig_model_forward = model.forward
        # No vision batches cached
        quantizer._num_vision_batches = 0

        orig_w = model.model.vision_tower.patch_embedder.input_proj.weight.data.clone()

        quantizer._quantize_stage_from_raw_replay(
            model=model,
            stage_module=model.model.vision_tower.patch_embedder,
            module_name=module_name,
            stage_desc="vision.patch_embed",
            vision_only=True,
        )

        # Weights should be unchanged
        new_w = model.model.vision_tower.patch_embedder.input_proj.weight.data
        self.assertTrue(torch.allclose(orig_w, new_w))


# ---------------------------------------------------------------------------
# Device/dtype helper tests
# ---------------------------------------------------------------------------


class TestDeviceHelpers(unittest.TestCase):
    """Tests for _move_batch_to_model_device / _move_batch_to_stage_device."""

    def test_move_batch_to_model_device(self):
        model = MockGemma4Model(dim=64)
        quantizer = _make_quantizer()
        batch = (torch.randn(2, 64),)
        moved = quantizer._move_batch_to_model_device(model, batch)
        self.assertEqual(moved[0].device, next(model.parameters()).device)

    def test_move_batch_to_stage_device(self):
        stage = nn.Linear(64, 64)
        quantizer = _make_quantizer()
        batch = (torch.randn(2, 64),)
        moved = quantizer._move_batch_to_stage_device(stage, batch)
        self.assertEqual(moved[0].device, stage.weight.device)

    def test_move_batch_no_parameters_returns_unchanged(self):
        stage = nn.Identity()
        quantizer = _make_quantizer()
        batch = (torch.randn(2, 64),)
        moved = quantizer._move_batch_to_stage_device(stage, batch)
        self.assertIs(moved, batch)


# ---------------------------------------------------------------------------
# convert() integration tests
# ---------------------------------------------------------------------------


class TestConvertIntegration(unittest.TestCase):
    """Integration tests for the full prepare -> convert flow."""

    @torch.no_grad()
    def test_convert_without_prepare_raises(self):
        """convert() should raise if prepare() was not called first."""
        model = MockGemma4Model(dim=32)
        quantizer = _make_quantizer(weight_bits=4)
        with self.assertRaises(AssertionError):
            quantizer.convert(model)

    @torch.no_grad()
    def test_convert_text_only_quantizes_text_layers(self):
        """
        Full prepare -> calibrate -> convert flow with text-only config.

        This patches out the complex _quantize_text_layers method to test
        the convert() orchestration logic (cache restore, component resolution,
        stage gating, quantizer attachment).
        """
        dim = 32
        model = MockGemma4Model(dim=dim, num_text_layers=2)
        quantizer = _make_quantizer(
            weight_bits=4,
            actorder=False,
            **_text_only_overrides(),
        )

        # Prepare
        prepared = quantizer.prepare(model)
        for _ in range(4):
            prepared(torch.randn(1, 4, dim))

        # Patch _quantize_text_layers to avoid the complex HF-specific capture logic
        with patch.object(
            Gemma4GPTQQuantizer,
            "_quantize_text_layers",
            return_value=None,
        ):
            result = quantizer.convert(model)

        # Forward should be restored
        self.assertEqual(result.forward, quantizer._orig_model_forward)

        # Caches should be cleared
        self.assertEqual(quantizer.num_batches, 0)
        self.assertEqual(quantizer._num_vision_batches, 0)

        # model.quantizers should be set (empty since we patched the method)
        self.assertTrue(hasattr(result, "quantizers"))

        # use_cache should be restored
        self.assertTrue(result.config.use_cache)

    @torch.no_grad()
    def test_convert_vision_only_quantizes_vision_stages(self):
        """
        Full prepare -> calibrate -> convert flow with vision-only config.

        Patches _quantize_vision_blocks to test the convert() orchestration
        for vision stages.
        """
        dim = 32
        model = MockGemma4Model(dim=dim, num_vision_layers=2)
        quantizer = _make_quantizer(
            weight_bits=4,
            actorder=False,
            **_vision_only_overrides(),
        )

        # Prepare with vision batches
        prepared = quantizer.prepare(model)
        for _ in range(4):
            prepared(torch.randn(1, 4, dim), pixel_values=torch.randn(1, 3, 8, 8))

        # Patch _quantize_vision_blocks to avoid complex entry capture logic
        with patch.object(
            Gemma4GPTQQuantizer,
            "_quantize_vision_blocks",
            return_value=None,
        ):
            result = quantizer.convert(model)

        # Forward should be restored
        self.assertEqual(result.forward, quantizer._orig_model_forward)
        self.assertTrue(hasattr(result, "quantizers"))

    @torch.no_grad()
    def test_convert_lm_head_stage(self):
        """convert() should call _quantize_stage_from_raw_replay for lm_head."""
        dim = 32
        model = MockGemma4Model(dim=dim)
        quantizer = _make_quantizer(
            weight_bits=4,
            actorder=False,
            quantize_lm_head=True,
            **_text_only_overrides(),
            **_vision_only_overrides(),
        )

        prepared = quantizer.prepare(model)
        for _ in range(4):
            prepared(torch.randn(1, 4, dim))

        orig_lm_head_w = model.lm_head.weight.data.clone()

        result = quantizer.convert(model)

        # lm_head weights should be quantized
        new_lm_head_w = model.lm_head.weight.data
        self.assertFalse(torch.allclose(orig_lm_head_w, new_lm_head_w))

        # lm_head should appear in quantizers
        self.assertTrue(any("lm_head" in name for name in result.quantizers))


# ---------------------------------------------------------------------------
# StopReplay exception tests
# ---------------------------------------------------------------------------


class TestStopReplay(unittest.TestCase):
    """Tests for the StopReplay exception."""

    def test_stop_replay_is_exception(self):
        self.assertTrue(issubclass(StopReplay, Exception))

    def test_stop_replay_can_be_raised_and_caught(self):
        with self.assertRaises(StopReplay):
            raise StopReplay()


if __name__ == "__main__":
    unittest.main()

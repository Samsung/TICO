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

"""Initial-prefill PLE range computation through real Gemma4 decoder stages."""

import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.config.specs import no_quant
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.debug.static_gemma4_runtime import (
    StaticGemma4RuntimeConfig,
)

from tico.quantization.wrapq.wrappers.gemma4.utils import gemma4_layer_type_head_dim

from test.quantization.recipes.debug import test_static_gemma4_finalize as fixture
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _has_gemma4


class _ExplicitPLEStage(torch.nn.Module):
    """A caller-owned stage that forwards one lookup or two projection inputs."""

    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def forward(self, *args):
        return self.stage(*args)


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are required")
class TestStaticGemma4SplitPLE(unittest.TestCase):
    def assert_caches_close(self, runtime, expected):
        for cache, (key, value) in zip(runtime.layer_caches, expected):
            # The producer prefix always has the original S-token execution.
            torch.testing.assert_close(cache.past_k, key, atol=0, rtol=0)
            torch.testing.assert_close(cache.past_v, value, atol=0, rtol=0)

    def _compare_three_paths(self, *, length, quantized=False, visual_tokens=0):
        runtime = fixture.make_runtime(quantized=quantized, visual_tokens=visual_tokens)
        batch = fixture.make_batch(length, visual_tokens=visual_tokens)
        before = _state_contract(runtime.text_model) if quantized else None
        last_hidden = []
        handle = runtime.lm_head.register_forward_pre_hook(
            lambda module, args: last_hidden.append(args[0].clone())
        )
        with torch.no_grad():
            full_logits = runtime.prefill(batch, last_token_only=False)
            full_hidden = last_hidden[-1]
            expected_cache = fixture.cache_snapshot(runtime)
            self.assertEqual(runtime.past_len, length)
            continuation = []
            for token in (17, 8, 29)[: min(3, 9 - length)]:
                result = runtime.decode_one(torch.tensor([[token]]))
                continuation.append(
                    (
                        token,
                        result.clone(),
                        fixture.cache_snapshot(runtime),
                        runtime.past_len,
                    )
                )
            for split in (False, True):
                logits = runtime.prefill(
                    batch, last_token_only=True, ple_prefill_split=split
                )
                # The float S-token versus one-token GEMM reduction can differ
                # by roundoff. This is the existing finalize regression bound.
                torch.testing.assert_close(logits, full_logits, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    last_hidden[-1], full_hidden, atol=1e-5, rtol=1e-5
                )
                self.assert_caches_close(runtime, expected_cache)
                self.assertEqual(runtime.past_len, length)
                for token, expected_logits, cache, past_len in continuation:
                    actual = runtime.decode_one(torch.tensor([[token]]))
                    torch.testing.assert_close(
                        actual, expected_logits, atol=1e-5, rtol=1e-5
                    )
                    self.assert_caches_close(runtime, cache)
                    self.assertEqual(runtime.past_len, past_len)
        handle.remove()
        if quantized:
            self.assertEqual(_state_contract(runtime.text_model), before)

    def test_fp_and_frozen_quantized_three_paths_with_continuation(self):
        for quantized in (False, True):
            for length in (1, 4, 9):
                with self.subTest(quantized=quantized, length=length):
                    self._compare_three_paths(length=length, quantized=quantized)

    def test_multimodal_fused_h0_three_path_parity(self):
        # p=2 falls inside the visual span, so replacing H0 with the text lookup
        # or the post-prefix decoder state changes the suffix projection input.
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                self._compare_three_paths(
                    length=3, quantized=quantized, visual_tokens=2
                )

    def test_selected_inputs_shapes_storage_and_normal_decode(self):
        runtime = fixture.make_runtime(visual_tokens=2)
        batch = fixture.make_batch(3, visual_tokens=2)
        stages = runtime._get_split_ple_stages()
        prefix_lookup, prefix_projection, suffix_lookup, suffix_projection = stages
        h0 = []
        inputs: dict[str, tuple[torch.Tensor, ...]] = {}
        outputs: dict[str, tuple[int, ...]] = {}
        calls = []
        handles = [
            runtime.mm_fusion.register_forward_hook(
                lambda module, args, result: h0.append(result.clone())
            )
        ]
        for name, module in zip(
            (
                "prefix_lookup",
                "prefix_projection",
                "suffix_lookup",
                "suffix_projection",
            ),
            stages,
        ):
            handles.append(
                module.register_forward_pre_hook(
                    lambda module, args, name=name: inputs.setdefault(
                        name, tuple(arg.clone() for arg in args)
                    )
                    and None
                )
            )
            handles.append(
                module.register_forward_hook(
                    lambda module, args, result, name=name: outputs.setdefault(
                        name, tuple(result.shape)
                    )
                    and None
                )
            )
        for phase, layers in (
            ("prefill", runtime.prefill_layers),
            ("decode", runtime.decode_layers),
        ):
            for index, layer in enumerate(layers):
                handles.append(
                    layer.register_forward_pre_hook(
                        lambda module, args, kwargs, phase=phase, index=index: calls.append(
                            (phase, index, kwargs["hidden_states"].shape[1])
                        ),
                        with_kwargs=True,
                    )
                )
        full_projection_shapes = []
        handles.append(
            runtime.ple_projection.register_forward_hook(
                lambda module, args, result: full_projection_shapes.append(
                    tuple(result.shape)
                )
            )
        )
        with torch.no_grad():
            runtime.prefill(batch, last_token_only=True, ple_prefill_split=True)
            self.assertEqual(full_projection_shapes, [])
            self.assertEqual(
                outputs,
                {
                    "prefix_lookup": (1, 10, 2, 8),
                    "prefix_projection": (1, 10, 2, 8),
                    "suffix_lookup": (1, 1, 2, 8),
                    "suffix_projection": (1, 1, 2, 8),
                },
            )
            torch.testing.assert_close(
                inputs["prefix_projection"][0], h0[0], atol=0, rtol=0
            )
            torch.testing.assert_close(
                inputs["suffix_projection"][0], h0[0][:, 2:3], atol=0, rtol=0
            )
            torch.testing.assert_close(
                inputs["suffix_lookup"][0],
                batch["llm_input_ids"][:, 2:3],
                atol=0,
                rtol=0,
            )
            self.assertFalse(torch.equal(inputs["suffix_projection"][0], h0[0][:, -1:]))
            text = runtime.token_embedding(batch["llm_input_ids"])
            self.assertFalse(torch.equal(inputs["suffix_projection"][0], text[:, 2:3]))
            self.assertEqual(
                calls,
                [
                    ("prefill", 0, 10),
                    ("prefill", 1, 10),
                    ("decode", 2, 1),
                    ("decode", 3, 1),
                ],
            )
            self.assertIs(
                prefix_lookup.embed_tokens_per_layer,
                runtime.ple_embedding.embed_tokens_per_layer,
            )
            self.assertIs(
                suffix_lookup.embed_tokens_per_layer,
                runtime.ple_embedding.embed_tokens_per_layer,
            )
            self.assertFalse(
                any(
                    "embed_tokens" in name
                    for name, _ in prefix_projection.named_modules()
                )
            )
            self.assertFalse(
                any(
                    "embed_tokens" in name
                    for name, _ in suffix_projection.named_modules()
                )
            )
            calls.clear()
            runtime.decode_one(torch.tensor([[17]]))
            self.assertEqual(calls, [("decode", i, 1) for i in range(4)])
            self.assertEqual(full_projection_shapes, [(1, 1, 4, 8)])
        for handle in handles:
            handle.remove()

    def test_prefix_buffer_released_before_suffix_decoder(self):
        runtime = fixture.make_runtime()
        stages = runtime._get_split_ple_stages()
        prefix = []
        lifetime = []
        handles = [
            stages[1].register_forward_hook(
                lambda module, args, result: prefix.append(weakref.ref(result))
            ),
            runtime.decode_layers[2].register_forward_pre_hook(
                lambda module, args, kwargs: lifetime.append(prefix[0]() is None),
                with_kwargs=True,
            ),
        ]
        with torch.no_grad():
            runtime.prefill(
                fixture.make_batch(4), last_token_only=True, ple_prefill_split=True
            )
        self.assertEqual(lifetime, [True])
        for handle in handles:
            handle.remove()

    def test_no_ple_and_no_suffix_are_noops(self):
        for options in ({"ple": False}, {"shared": False}):
            with self.subTest(options=options):
                runtime = fixture.make_runtime(**options)
                batch = fixture.make_batch(4)
                with torch.no_grad():
                    expected = runtime.prefill(batch, last_token_only=False)
                    cache = fixture.cache_snapshot(runtime)
                    actual = runtime.prefill(
                        batch, last_token_only=True, ple_prefill_split=True
                    )
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                self.assert_caches_close(runtime, cache)
                self.assertIsNone(getattr(runtime, "_split_ple_stages", None))

    def test_non_e2b_topology_uses_resolved_layer_mapping(self):
        make_config = fixture._make_text_config

        def alternate_config(**kwargs):
            kwargs.update(
                layer_types=["sliding_attention", "full_attention"] * 3,
                num_kv_shared_layers=2,
            )
            return make_config(**kwargs)

        with mock.patch.object(
            fixture, "_make_text_config", side_effect=alternate_config
        ):
            runtime = fixture.make_runtime()
        self.assertEqual(runtime.shared_kv_suffix_start, 4)
        batch = fixture.make_batch(4)
        with torch.no_grad():
            expected = runtime.prefill(batch, last_token_only=False)
            cache = fixture.cache_snapshot(runtime)
            actual = runtime.prefill(
                batch, last_token_only=True, ple_prefill_split=True
            )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assert_caches_close(runtime, cache)
        prefix_lookup, _, suffix_lookup, _ = runtime._get_split_ple_stages()
        self.assertEqual((prefix_lookup.layer_start, prefix_lookup.layer_end), (0, 4))
        self.assertEqual((suffix_lookup.layer_start, suffix_lookup.layer_end), (4, 6))

    @staticmethod
    def _replace_frontend(runtime, replacement, *, decoder=True):
        """Reproduce the documented field-based calibrated frontend install."""
        names = ["text_model", "token_embedding", "ple_embedding", "ple_projection"]
        if decoder:
            names += ["prefill_layers", "decode_layers", "lm_head"]
        for name in names:
            setattr(runtime, name, getattr(replacement, name))

    def _replacement_parity(self, *, split, saved_pipeline=False):
        runtime = fixture.make_runtime()
        # Construction with ple_prefill_split=true performs this eager setup;
        # it must not pin the original FP lookup or projection after replacement.
        old_stages = runtime._get_split_ple_stages()
        replacement = fixture.make_runtime(quantized=True)
        frozen_before = _state_contract(replacement.text_model)
        batch = fixture.make_batch(4)
        expected_ple = []
        actual_ple = []
        expected_handle = replacement.ple_projection.register_forward_hook(
            lambda module, args, result: expected_ple.append(result.clone())
        )
        with torch.no_grad():
            expected_prefill = replacement.prefill(batch, last_token_only=False)
            expected_prefill_cache = fixture.cache_snapshot(replacement)
            expected_decode = []
            for token in (17, 8):
                logits = replacement.decode_one(torch.tensor([[token]]))
                expected_decode.append(
                    (
                        logits.clone(),
                        fixture.cache_snapshot(replacement),
                        replacement.past_len,
                    )
                )
        expected_handle.remove()
        self._replace_frontend(runtime, replacement, decoder=not saved_pipeline)
        if saved_pipeline:
            specs = []
            for kind, layer in zip(
                replacement.text_config.layer_types, replacement.text_model.layers
            ):
                attention = layer.wrapped.self_attn.wrapped
                specs.append(
                    {
                        "layer_type": kind,
                        "head_dim": gemma4_layer_type_head_dim(
                            replacement.text_config, kind
                        ),
                        "num_kv_heads": attention.config.num_attention_heads
                        // attention.num_key_value_groups,
                        "sliding_window": attention.sliding_window,
                    }
                )
            pipeline = SimpleNamespace(
                suffix_start=replacement.shared_kv_suffix_start,
                producer_indices=replacement.shared_kv_producers,
                hidden_size=replacement.text_config.hidden_size,
                ple_dim=replacement.text_model.hidden_size_per_layer_input,
                rope_convention=replacement.rope_convention,
                layers=specs,
                prefill_layers=replacement.prefill_layers,
                decode_layers=replacement.decode_layers,
                lm_head=replacement.lm_head,
                finalize_supported=True,
                decode_kv_policy="compact_sliding_v1",
                decode_capacities=(3, 10, 3, 10),
            )
            with mock.patch(
                "tico.quantization.recipes.export.gemma4_text_pipeline.load_gemma4_text_pipeline",
                return_value=pipeline,
            ) as loader:
                runtime.use_text_pipeline("calibrated-artifacts")
            loader.assert_called_once_with(
                "calibrated-artifacts",
                expected_max_seq_len=10,
                expected_num_layers=4,
                expected_decode_kv_policy=None,
            )
            self.assertEqual(runtime.past_len, 0)
            self.assertEqual(runtime.layer_caches, [])
        handles = [
            runtime.ple_projection.register_forward_hook(
                lambda module, args, result: actual_ple.append(result.clone())
            )
        ]
        split_ple = []
        if split:
            stages = runtime._get_split_ple_stages()
            for module in (stages[1], stages[3]):
                handles.append(
                    module.register_forward_hook(
                        lambda module, args, result: split_ple.append(result.clone())
                    )
                )
        with torch.no_grad():
            actual = runtime.prefill(
                batch, last_token_only=split, ple_prefill_split=split
            )
            torch.testing.assert_close(actual, expected_prefill, atol=1e-5, rtol=1e-5)
            self.assert_caches_close(runtime, expected_prefill_cache)
            self.assertEqual(runtime.past_len, 4)
            if split:
                self.assertEqual(len(split_ple), 2)
                torch.testing.assert_close(
                    split_ple[0], expected_ple[0][:, :, :2], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    split_ple[1], expected_ple[0][:, 3:4, 2:], atol=0, rtol=0
                )
            else:
                torch.testing.assert_close(
                    actual_ple[0], expected_ple[0], atol=0, rtol=0
                )
            for token, (expected, cache, past_len) in zip((17, 8), expected_decode):
                result = runtime.decode_one(torch.tensor([[token]]))
                torch.testing.assert_close(
                    actual_ple[-1], expected_ple[past_len - 4], atol=0, rtol=0
                )
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
                self.assert_caches_close(runtime, cache)
                self.assertEqual(runtime.past_len, past_len)
        for handle in handles:
            handle.remove()
        refreshed = runtime._get_split_ple_stages()
        self.assertIsNot(refreshed, old_stages)
        self.assertIs(runtime._get_split_ple_stages(), refreshed)
        self.assertEqual(_state_contract(replacement.text_model), frozen_before)

    def test_replaced_calibrated_frontend_full_prefill_and_continuation(self):
        self._replacement_parity(split=False)

    def test_replaced_calibrated_frontend_split_prefill_and_continuation(self):
        self._replacement_parity(split=True)

    def test_legacy_frontend_install_with_saved_text_pipeline(self):
        # Only file loading is mocked: real frozen frontend and decoder stages
        # execute through the public use_text_pipeline/prefill/decode route.
        self._replacement_parity(split=True, saved_pipeline=True)

    def test_same_frontend_reuses_cached_range_and_decode_lookup_stages(self):
        runtime = fixture.make_runtime(quantized=True)
        stages = runtime._get_split_ple_stages()
        selected = runtime._selective_ple_embedding
        with torch.no_grad():
            for _ in range(2):
                runtime.prefill(
                    fixture.make_batch(4), last_token_only=True, ple_prefill_split=True
                )
                runtime.decode_one(torch.tensor([[17]]))
                self.assertIs(runtime._get_split_ple_stages(), stages)
                self.assertIs(runtime._selective_ple_embedding, selected)
        runtime.reset_cache()
        self.assertIs(runtime._get_split_ple_stages(), stages)

    def test_explicit_decode_lookup_survives_owned_split_cache_refresh(self):
        runtime = fixture.make_runtime()
        old_stages = runtime._get_split_ple_stages()
        replacement = fixture.make_runtime(quantized=True)
        replacement._get_split_ple_stages()
        custom_lookup = torch.nn.Sequential(replacement._selective_ple_embedding)
        self._replace_frontend(runtime, replacement)
        runtime._selective_ple_embedding = custom_lookup
        refreshed = runtime._get_split_ple_stages()
        self.assertIsNot(refreshed, old_stages)
        self.assertIs(runtime._selective_ple_embedding, custom_lookup)
        batch = fixture.make_batch(4)
        with torch.no_grad():
            expected = replacement.prefill(
                batch, last_token_only=True, ple_prefill_split=True
            )
            expected_decode = replacement.decode_one(torch.tensor([[17]]))
            actual = runtime.prefill(
                batch, last_token_only=True, ple_prefill_split=True
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            actual_decode = runtime.decode_one(torch.tensor([[17]]))
            torch.testing.assert_close(
                actual_decode, expected_decode, atol=1e-5, rtol=1e-5
            )
        self.assertIs(runtime._get_split_ple_stages(), refreshed)
        self.assertIs(runtime._selective_ple_embedding, custom_lookup)

    def test_explicit_split_stages_survive_owned_decode_lookup_refresh(self):
        runtime = fixture.make_runtime()
        runtime._get_split_ple_stages()
        old_lookup = runtime._selective_ple_embedding
        replacement = fixture.make_runtime(quantized=True)
        custom_stages = tuple(
            _ExplicitPLEStage(stage) for stage in replacement._get_split_ple_stages()
        )
        self._replace_frontend(runtime, replacement)
        runtime._split_ple_stages = custom_stages
        self.assertIs(runtime._get_split_ple_stages(), custom_stages)
        batch = fixture.make_batch(4)
        with torch.no_grad():
            expected = replacement.prefill(
                batch, last_token_only=True, ple_prefill_split=True
            )
            expected_decode = replacement.decode_one(torch.tensor([[17]]))
            actual = runtime.prefill(
                batch, last_token_only=True, ple_prefill_split=True
            )
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            actual_decode = runtime.decode_one(torch.tensor([[17]]))
            torch.testing.assert_close(
                actual_decode, expected_decode, atol=1e-5, rtol=1e-5
            )
        self.assertIs(runtime._get_split_ple_stages(), custom_stages)
        self.assertIsNot(runtime._selective_ple_embedding, old_lookup)

    def test_custom_full_frontend_requires_explicit_split_stages(self):
        for name in ("ple_embedding", "ple_projection"):
            with self.subTest(stage=name):
                runtime = fixture.make_runtime()
                runtime._get_split_ple_stages()
                batch = fixture.make_batch(4)
                with torch.no_grad():
                    expected = runtime.prefill(batch, last_token_only=False)
                custom = _ExplicitPLEStage(getattr(runtime, name))
                setattr(runtime, name, custom)
                with self.assertRaisesRegex(
                    ValueError, "explicitly installed _split_ple_stages"
                ):
                    runtime.prefill(batch, last_token_only=True, ple_prefill_split=True)
                self.assertIs(getattr(runtime, name), custom)
                self.assertIsNone(runtime._split_ple_stages)
                with torch.no_grad():
                    actual = runtime.prefill(batch, last_token_only=False)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertIs(getattr(runtime, name), custom)

    def test_split_history_does_not_enable_selective_lookup_for_replaced_source(self):
        runtime = fixture.make_runtime()
        batch = fixture.make_batch(4)
        with torch.no_grad():
            # A per-call opt-in must belong to this source, not later installs.
            runtime.prefill(batch, last_token_only=True, ple_prefill_split=True)
        make_config = fixture.PTQConfig
        with mock.patch.object(
            fixture,
            "PTQConfig",
            side_effect=lambda **kwargs: make_config(weight=no_quant(), **kwargs),
        ):
            replacement = fixture.make_runtime(quantized=True)
        frozen_before = _state_contract(replacement.text_model)

        def reference_trace(frontend):
            ple = []
            outputs = []
            handle = frontend.ple_projection.register_forward_hook(
                lambda module, args, result: ple.append(result.clone())
            )
            with torch.no_grad():
                logits = frontend.prefill(
                    batch, last_token_only=False, ple_prefill_split=False
                )
                outputs.append(
                    (
                        logits.clone(),
                        fixture.cache_snapshot(frontend),
                        frontend.past_len,
                    )
                )
                for token in (17, 8):
                    logits = frontend.decode_one(torch.tensor([[token]]))
                    outputs.append(
                        (
                            logits.clone(),
                            fixture.cache_snapshot(frontend),
                            frontend.past_len,
                        )
                    )
            handle.remove()
            return outputs, ple

        def compare_generation(expected, *, split):
            with torch.no_grad():
                for index, (logits, cache, past_len) in enumerate(expected):
                    actual = (
                        runtime.prefill(
                            batch, last_token_only=split, ple_prefill_split=split
                        )
                        if index == 0
                        else runtime.decode_one(torch.tensor([[(17, 8)[index - 1]]]))
                    )
                    tolerance = 1e-5 if split else 0.0
                    torch.testing.assert_close(
                        actual, logits, atol=tolerance, rtol=tolerance
                    )
                    self.assert_caches_close(runtime, cache)
                    self.assertEqual(runtime.past_len, past_len)

        expected, expected_ple = reference_trace(replacement)
        self._replace_frontend(runtime, replacement)
        actual_ple = []
        handle = runtime.ple_projection.register_forward_hook(
            lambda module, args, result: actual_ple.append(result.clone())
        )
        # Identity weights are valid for the full producer and ordinary decode;
        # a stale opt-in used to reject the first decode at the lookup validator.
        compare_generation(expected, split=False)
        handle.remove()
        for actual, expected_value in zip(actual_ple, expected_ple):
            torch.testing.assert_close(actual, expected_value, atol=0, rtol=0)
        self.assertEqual(len(actual_ple), 3)
        self.assertFalse(getattr(runtime, "_model_backed_selective_ple_enabled", False))
        self.assertEqual(_state_contract(replacement.text_model), frozen_before)
        with self.assertRaisesRegex(ValueError, "affine weight policy only"):
            runtime.prefill(batch, last_token_only=True, ple_prefill_split=True)
        self.assertEqual(_state_contract(replacement.text_model), frozen_before)

        # A later source can opt into supported slicing again. Its derived
        # stages are reused within that source after the marker was reset.
        supported = fixture.make_runtime(quantized=True)
        supported_before = _state_contract(supported.text_model)
        expected, _ = reference_trace(supported)
        self._replace_frontend(runtime, supported)
        compare_generation(expected, split=True)
        stages = runtime._get_split_ple_stages()
        lookup = runtime._selective_ple_embedding
        self.assertTrue(runtime._model_backed_selective_ple_enabled)
        compare_generation(expected, split=True)
        self.assertIs(runtime._get_split_ple_stages(), stages)
        self.assertIs(runtime._selective_ple_embedding, lookup)
        self.assertEqual(_state_contract(supported.text_model), supported_before)

    def test_config_default_reference_override_and_type_diagnostics(self):
        self.assertFalse(StaticGemma4RuntimeConfig().ple_prefill_split)
        self.assertTrue(
            StaticGemma4RuntimeConfig(
                last_token_only_prefill=True, ple_prefill_split=True
            ).ple_prefill_split
        )
        with self.assertRaisesRegex(ValueError, "requires last_token_only_prefill"):
            StaticGemma4RuntimeConfig(ple_prefill_split=True)
        with self.assertRaisesRegex(ValueError, "ple_prefill_split must be a boolean"):
            StaticGemma4RuntimeConfig(ple_prefill_split="true")  # type: ignore[arg-type]
        runtime = fixture.make_runtime()
        runtime.last_token_only_prefill = True
        runtime.ple_prefill_split = True
        with (
            torch.no_grad(),
            mock.patch.object(
                runtime,
                "_build_split_per_layer_inputs",
                side_effect=AssertionError("split called"),
            ),
        ):
            runtime.prefill(fixture.make_batch(3), last_token_only=False)
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            runtime.prefill(fixture.make_batch(3), ple_prefill_split="true")


if __name__ == "__main__":
    unittest.main()

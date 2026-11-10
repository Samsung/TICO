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

"""Real, download-free CPU coverage of the Qwen adapter/PTQ cache lifecycle."""

import unittest

import tico

import torch

from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.quantization import prepare
from tico.quantization.recipes.debug.cpu_fixtures import fixture_context, qwen_batch
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
    QuantQwen3VLTextAttention,
)
from transformers.cache_utils import DynamicCache, StaticCache


def _attentions(model: torch.nn.Module) -> list[QuantQwen3VLTextAttention]:
    return [m for m in model.modules() if isinstance(m, QuantQwen3VLTextAttention)]


def _affine_storage_key(layer: QuantQwen3VLTextAttention) -> AffineObserverBase:
    """Narrow cache statistics to the affine policy used by these fixtures."""
    observer = layer.obs_new_k
    assert isinstance(observer, AffineObserverBase)
    return observer


def _frozen(*, override=False, steps=2):
    ctx = fixture_context("qwen3_vl")
    stage = ctx.cfg["pipeline"][0]
    stage["decode_calibration_steps"] = steps
    if override:
        stage["raw_overrides"] = {
            f"model.language_model.layers.{layer}.self_attn.{observer}": {
                "kind": "affine",
                "dtype": "uint8",
                "observer": "ema",
                "momentum": 0.8,
            }
            for layer in range(2)
            for observer in ("new_k", "new_v")
        }
    PTQStage().run(ctx, stage)
    return ctx


class TestQwenCacheCoverageCPU(unittest.TestCase):
    def test_recipe_rejects_obsolete_past_override(self):
        ctx = fixture_context("qwen3_vl")
        stage = ctx.cfg["pipeline"][0]
        stage["raw_overrides"] = {
            "model.language_model.layers.0.self_attn.past_key": "uint8"
        }
        with self.assertRaisesRegex(ValueError, "past_key.*new_.*present_"):
            PTQStage().run(ctx, stage)

    def test_prefill_observes_both_domains_without_filtering_padding(self):
        ctx = fixture_context("qwen3_vl")
        ctx.calibration_inputs = [ctx.calibration_inputs[0]]
        ctx.calibration_inputs[0]["attention_mask"][:, :2] = 0
        stage = ctx.cfg["pipeline"][0]
        stage["decode_calibration_steps"] = 0
        prepared = prepare(ctx.model, ctx.adapter.build_ptq_config(ctx, stage))
        records: dict[tuple[int | None, str], list[torch.Tensor]] = {}
        originals = []
        for layer in _attentions(prepared):
            self.assertFalse(hasattr(layer, "obs_past_key"))
            self.assertFalse(hasattr(layer, "obs_past_value"))
            for kind in ("new_k", "new_v", "present_key", "present_value"):
                obs = getattr(layer, "obs_" + kind)
                key = (layer.layer_idx, kind)
                records[key] = []
                original = obs.collect

                def collect(value, *, saved=original, target=key):
                    records[target].append(value.detach().clone())
                    saved(value)

                originals.append((obs, original))
                setattr(obs, "collect", collect)
        try:
            ctx.adapter.calibrate_prepared_model(ctx, prepared, stage)
        finally:
            for obs, original in originals:
                setattr(obs, "collect", original)
        for values in records.values():
            self.assertEqual([tuple(value.shape) for value in values], [(1, 1, 8, 32)])
            self.assertTrue(
                all(value.numel() and torch.isfinite(value).all() for value in values)
            )
        self.assertFalse(
            torch.equal(records[(0, "new_k")][0], records[(1, "new_k")][0])
        )
        for index in range(2):
            for storage, compute in (
                ("new_k", "present_key"),
                ("new_v", "present_value"),
            ):
                torch.testing.assert_close(
                    records[(index, storage)][0],
                    records[(index, compute)][0],
                    atol=0,
                    rtol=0,
                )

    def test_standard_ptq_observes_both_layers_and_override(self):
        for override in (False, True):
            with self.subTest(override=override):
                ctx = _frozen(override=override)
                layers = _attentions(ctx.model)
                self.assertEqual(len(layers), 2)
                for layer in layers:
                    for kind in (
                        "new_k",
                        "new_v",
                        "present_key",
                        "present_value",
                    ):
                        obs = getattr(layer, "obs_" + kind)
                        self.assertTrue(torch.isfinite(obs.min_val).all())
                        self.assertTrue(torch.isfinite(obs.max_val).all())
                        self.assertTrue(torch.isfinite(obs._cached_scale).all())
                        self.assertTrue((obs._cached_scale > 0).all())
                    self.assertIs(layer._mode, Mode.QUANT)
                    self.assertEqual(layer.obs_new_k.dtype.bits, 8 if override else 16)
                    if override:
                        self.assertEqual(type(layer.obs_new_k).__name__, "EMAObserver")
                        self.assertEqual(layer.obs_new_k.momentum, 0.8)
                    self.assertEqual(layer.obs_present_key.dtype.bits, 16)
                    layer.require_calibrated_cache()
                self.assertFalse(
                    torch.equal(
                        _affine_storage_key(layers[0]).min_val,
                        _affine_storage_key(layers[1]).min_val,
                    )
                )

    def test_prefill_only_calibration_exports_decode_and_preserves_qparams(self):
        ctx = _frozen(steps=0)
        layer = _attentions(ctx.model)[0]
        saved = {name: value.clone() for name, value in layer.named_buffers()}
        self.assertTrue(torch.isfinite(_affine_storage_key(layer).min_val).all())
        layer.as_export_module("prefill")
        decode = layer.as_export_module("decode")
        cache = DynamicCache(config=layer.config)
        hidden = torch.randn(1, 3, layer.config.hidden_size)
        rope = (torch.ones(1, 3, layer.head_dim), torch.zeros(1, 3, layer.head_dim))
        with torch.no_grad():
            layer(hidden, rope, past_key_values=cache, use_cache=True)
        cache_values = layer._extract_layer_kv_from_cache(cache)
        assert cache_values is not None
        decode_args = (
            hidden[:, :1],
            (rope[0][:, :1], rope[1][:, :1]),
            torch.zeros(1, 1, 1, 4),
            cache_values,
        )
        with torch.no_grad():
            exported = torch.export.export(decode, decode_args)
            expected = decode(*decode_args)
            actual = exported.module()(*decode_args)
            cached = layer(
                hidden[:, :1],
                (rope[0][:, :1], rope[1][:, :1]),
                past_key_values=cache,
                use_cache=True,
                cache_output_mode="delta",
            )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
        for result, reference in zip((cached[0], *cached[2]), expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
        self.assertEqual(cache.get_seq_length(), 4)
        converted = tico.convert_from_exported_program(exported)
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify()
        program = CircleReferenceRuntime(document).program
        for input_index, output_index, observer in zip(
            program.inputs[-2:],
            program.outputs[-2:],
            (layer.obs_new_k, layer.obs_new_v),
        ):
            assert isinstance(observer, AffineObserverBase)
            for index in (input_index, output_index):
                tensor = program.tensor(index)
                self.assertEqual(tensor.dtype.name, "int16")
                self.assertIsNotNone(tensor.quantization)
                assert tensor.quantization is not None
                self.assertEqual(
                    list(tensor.quantization.scale), [observer._cached_scale.item()]
                )
                self.assertEqual(
                    list(tensor.quantization.zero_point), [observer._cached_zp.item()]
                )
        for name, value in layer.named_buffers():
            self.assertTrue(torch.equal(value, saved[name]), name)

    def test_prefill_only_calibration_supports_generate(self):
        ctx = _frozen(steps=0)
        model = ctx.model.wrapped
        saved = {name: value.clone() for name, value in model.named_buffers()}
        with torch.no_grad():
            result = model.generate(
                **qwen_batch(image=True),
                do_sample=False,
                use_cache=True,
                min_new_tokens=3,
                max_new_tokens=3,
                eos_token_id=None,
                pad_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        self.assertEqual(tuple(result.sequences.shape), (1, 11))
        self.assertEqual(len(result.scores), 3)
        self.assertTrue(all(torch.isfinite(score).all() for score in result.scores))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, saved[name]), name)

    def test_observer_provenance_handles_constants_sentinels_and_stateless_types(self):
        layer = _attentions(_frozen().model)[0]
        for obs in (layer.obs_new_k, layer.obs_new_v):
            obs.reset()
            obs.enabled = True
            obs.collect(torch.zeros(1, 1, 3, layer.head_dim))
            obs.compute_qparams()
        layer.require_calibrated_cache()
        storage_key = _affine_storage_key(layer)
        storage_key.reset()
        storage_key.load_qparams(torch.tensor(1e-6), torch.tensor(0))
        # Imported compatible grids carry their own provenance; finite min/max
        # is no longer a model-specific readiness requirement.
        self.assertEqual(storage_key.require_frozen().provenance, "imported")
        layer.require_calibrated_cache()
        storage_key.reset()
        layer.freeze_qparams()
        with self.assertRaisesRegex(RuntimeError, "capability 'cache'"):
            layer.require_calibrated_cache()
        layer.obs_new_k.disable_fake_quant()
        layer.freeze_qparams()
        layer.require_calibrated_cache()
        self.assertEqual(layer.obs_new_k.require_frozen().kind, "bypass")
        layer.obs_new_k = IdentityObserver(name="new_k")
        layer.freeze_qparams()
        layer.require_calibrated_cache()
        self.assertEqual(layer.obs_new_k.require_frozen().kind, "identity")
        layer._mode = Mode.NO_QUANT
        layer.obs_new_k = MinMaxObserver(name="new_k")
        layer.require_calibrated_cache()

    def test_static_cache_layout_validation_and_empty_tuple(self):
        layer = _attentions(_frozen().model)[0]
        hidden = torch.randn(1, 1, layer.config.hidden_size)
        rope = (torch.ones(1, 1, layer.head_dim), torch.zeros(1, 1, layer.head_dim))
        empty = torch.empty(1, layer.num_kv_heads, 0, layer.head_dim)
        for cache in ((None, empty), (empty, None), (empty, empty, empty)):
            with self.subTest(cache_len=len(cache)):
                with self.assertRaises(ValueError):
                    layer(hidden, rope, past_key_values=cache)
        with torch.no_grad():
            absent = layer(hidden, rope, use_cache=True, cache_output_mode="delta")
            explicit = layer(
                hidden,
                rope,
                past_key_values=(empty, empty),
                use_cache=True,
                cache_output_mode="delta",
            )
        torch.testing.assert_close(absent[0], explicit[0], atol=0, rtol=0)

    def test_hf_and_static_delta_cache_are_identical_over_multiple_steps(self):
        ctx = _frozen(override=True)
        torch.manual_seed(34)
        for layer in _attentions(ctx.model):
            with self.subTest(layer=layer.layer_idx):
                hf_cache = DynamicCache(config=layer.config)
                tuple_cache: tuple[torch.Tensor, torch.Tensor] | None = None
                total_length = 0
                snapshot = {
                    name: value.clone() for name, value in layer.named_buffers()
                }
                with torch.no_grad():
                    # Reach fifteen past positions plus the current token,
                    # retaining two masked padding slots throughout decode.
                    for length in (5, *([1] * 11)):
                        total_length += length
                        mask = torch.ones(1, total_length, dtype=torch.long)
                        mask[:, :2] = 0
                        hidden = torch.randn(1, length, layer.config.hidden_size)
                        position = torch.randn(1, length, layer.head_dim)
                        rope = (position.cos(), position.sin())
                        hf = layer(
                            hidden,
                            rope,
                            attention_mask=mask,
                            past_key_values=hf_cache,
                            use_cache=True,
                            cache_output_mode="delta",
                        )
                        static = layer(
                            hidden,
                            rope,
                            attention_mask=mask,
                            past_key_values=tuple_cache,
                            use_cache=True,
                            cache_output_mode="delta",
                        )
                        for actual, expected in zip(
                            (hf[0], *hf[2]), (static[0], *static[2])
                        ):
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        delta = static[2]
                        if tuple_cache is None:
                            tuple_cache = delta
                        else:
                            tuple_cache = (
                                torch.cat((tuple_cache[0], delta[0]), dim=2),
                                torch.cat((tuple_cache[1], delta[1]), dim=2),
                            )
                        raw = layer._extract_layer_kv_from_cache(hf_cache)
                        self.assertIsNotNone(raw)
                        assert raw is not None
                        assert tuple_cache is not None
                        for actual, expected in zip(raw, tuple_cache):
                            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertEqual(total_length, 16)
                for name, value in layer.named_buffers():
                    self.assertTrue(torch.equal(value, snapshot[name]), name)

    def test_static_hf_cache_is_explicitly_unsupported(self):
        ctx = _frozen()
        layer = _attentions(ctx.model)[0]
        cache = StaticCache(config=layer.config, max_cache_len=16)
        hidden = torch.randn(1, 1, layer.config.hidden_size)
        rope = (torch.ones(1, 1, layer.head_dim), torch.zeros(1, 1, layer.head_dim))
        with self.assertRaisesRegex(RuntimeError, "append-only.*static KV tuple"):
            layer(hidden, rope, past_key_values=cache, use_cache=True)

    def test_default_decode_budget_is_teacher_forced_inside_sample(self):
        ctx = fixture_context("qwen3_vl")
        ctx.model.config.use_cache = False
        ctx.model.config.text_config.use_cache = False
        stage = ctx.cfg["pipeline"][0]
        stage.pop("decode_calibration_steps")
        config = ctx.adapter.build_ptq_config(ctx, stage)
        prepared = prepare(ctx.model, config)
        lengths = []
        handle = prepared.register_forward_pre_hook(
            lambda _model, _args, kwargs: lengths.append(kwargs["input_ids"].shape[-1]),
            with_kwargs=True,
        )
        try:
            ctx.adapter.calibrate_prepared_model(ctx, prepared, stage)
        finally:
            handle.remove()
        self.assertEqual(lengths, [7, 1, 7, 1])
        self.assertEqual(sum(lengths), 16)
        self.assertFalse(prepared.wrapped.config.use_cache)
        self.assertFalse(prepared.wrapped.config.text_config.use_cache)

    def test_short_sample_clamps_and_negative_steps_reject(self):
        ctx = fixture_context("qwen3_vl")
        ctx.calibration_inputs = [{"input_ids": torch.tensor([[5]])}]
        stage = ctx.cfg["pipeline"][0]
        prepared = prepare(ctx.model, ctx.adapter.build_ptq_config(ctx, stage))
        ctx.adapter.calibrate_prepared_model(ctx, prepared, stage)
        for layer in _attentions(prepared):
            for kind in ("new_k", "new_v", "present_key", "present_value"):
                self.assertTrue(
                    torch.isfinite(getattr(layer, "obs_" + kind).min_val).all()
                )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ctx.adapter.calibrate_prepared_model(
                ctx, prepared, {"decode_calibration_steps": -1}
            )
        ctx.calibration_inputs = [{"input_ids": torch.tensor([[5, 120, 121, 124]])}]
        with self.assertRaisesRegex(ValueError, "trailing text"):
            ctx.adapter.calibrate_prepared_model(
                ctx, prepared, {"decode_calibration_steps": 1}
            )
        ctx.calibration_inputs = [{"input_ids": torch.arange(9).unsqueeze(0)}]
        with self.assertRaisesRegex(ValueError, "token budget"):
            ctx.adapter.calibrate_prepared_model(ctx, prepared, stage)

    def test_calibration_restores_request_state_on_error(self):
        ctx = fixture_context("qwen3_vl")
        ctx.model.config.use_cache = False
        ctx.model.config.text_config.use_cache = False
        stage = ctx.cfg["pipeline"][0]
        prepared = prepare(ctx.model, ctx.adapter.build_ptq_config(ctx, stage))
        owners = [m for m in prepared.modules() if hasattr(m, "rope_deltas")]
        for index, module in enumerate(owners):
            module.rope_deltas = torch.tensor([[index + 13]])
        before = [(m, m.rope_deltas, m.rope_deltas.clone()) for m in owners]
        calls = 0

        def fail_decode(_module, _args, kwargs):
            nonlocal calls
            calls += 1
            if kwargs["input_ids"].shape[-1] == 1:
                raise RuntimeError("deliberate calibration failure")

        handle = prepared.register_forward_pre_hook(fail_decode, with_kwargs=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "deliberate calibration failure"):
                ctx.adapter.calibrate_prepared_model(ctx, prepared, stage)
        finally:
            handle.remove()
        self.assertEqual(calls, 2)
        self.assertFalse(prepared.wrapped.config.use_cache)
        self.assertFalse(prepared.wrapped.config.text_config.use_cache)
        for module, state, value in before:
            self.assertIs(module.rope_deltas, state)
            torch.testing.assert_close(module.rope_deltas, value, atol=0, rtol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

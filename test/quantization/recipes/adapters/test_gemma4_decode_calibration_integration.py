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

"""CPU integration for adapter calibration with real, synthetic Gemma4 wrappers."""

import io
import unittest
from contextlib import ExitStack
from typing import Any
from unittest import mock

import torch

from test.quantization.recipes.optional_dependency_stubs import (
    install_optional_dependency_stubs,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_for_conditional_generation import (
    _make_vision_config,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import (
    _has_gemma4,
    _make_text_config,
)

install_optional_dependency_stubs()

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import (
    QuantGemma4ForConditionalGeneration,
)


@unittest.skipUnless(
    _has_gemma4(), "required transformers Gemma4 modules are not installed"
)
class TestGemma4DecodeCalibrationIntegration(unittest.TestCase):
    """Exercise prepared forward, PLE, shared KV and vision without model downloads."""

    def _make_prepared_model(
        self, *, max_seq=16, mixed_attention=False, shared_kv=True
    ):
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )

        torch.manual_seed(2026)
        layer_types = (
            ["sliding_attention", "full_attention"] * 2
            if mixed_attention
            else ["full_attention"] * 2
        )
        text_config = _make_text_config(
            layer_types=layer_types,
            max_position_embeddings=16,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=128,
            num_kv_shared_layers=(2 if mixed_attention else 1) if shared_kv else 0,
            rope_parameters={
                layer_type: {"rope_type": "default", "rope_theta": 10000.0}
                for layer_type in set(layer_types)
            },
        )
        config = Gemma4Config.from_dict(
            {
                "text_config": text_config,
                "vision_config": _make_vision_config(),
                "audio_config": None,
                # Greedy text outputs cannot produce multimodal placeholders.
                "image_token_id": 128,
                "video_token_id": 129,
                "audio_token_id": 130,
            }
        )
        model = Gemma4ForConditionalGeneration(config).eval()
        qcfg = PTQConfig(
            model_args={
                "text": {"max_seq": max_seq},
                "vision": {"num_visual_tokens": 4},
            }
        )
        prepared = prepare(model, qcfg)
        self.assertIsInstance(prepared.wrapped, QuantGemma4ForConditionalGeneration)
        return model, prepared

    def _calibrate_and_record(self, *, with_image, padded=False):
        model, prepared = self._make_prepared_model()
        adapter = Gemma4Adapter()
        input_ids = torch.arange(1, 15).reshape(1, 14)
        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": False,
        }
        if padded:
            input_ids = torch.tensor(
                [list(range(1, 13)) + [0, 0], [0] + list(range(1, 14))]
            )
            batch["input_ids"] = input_ids
            batch["attention_mask"] = (input_ids != 0).long()
        if with_image:
            batch["input_ids"] = input_ids.clone()
            batch["input_ids"][:, :4] = model.config.image_token_id
            # The existing tiny vision fixture pools sixteen patches to four tokens.
            batch["pixel_values"] = torch.randn(1, 16, 3 * 4**2)
            coords = torch.arange(16)
            batch["image_position_ids"] = torch.stack(
                (coords % 4, coords // 4), dim=-1
            ).unsqueeze(0)
        original_batch = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 16, "decode_steps": 2},
                "pipeline": [{"name": "ptq"}],
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=model,
            calibration_inputs=[batch, batch],
        )
        wrapped = prepared.wrapped
        text = wrapped.model.wrapped.language_model.wrapped
        owner = text.layers[0].wrapped.self_attn.wrapped
        consumer = text.layers[1].wrapped.self_attn.wrapped
        self.assertTrue(owner.store_full_length_kv)
        self.assertTrue(consumer.is_kv_shared_layer)
        self.assertEqual(text.hidden_size_per_layer_input, 8)
        calls = []
        positions = []
        ple_lengths = []
        shared_states = []
        vision_calls = []
        handles = []
        observed: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "new_k",
                "new_v",
                "past_key",
                "past_value",
                "shared_past_key",
                "shared_past_value",
            )
        }

        def record_observer(observer, name):
            collect = observer.collect

            def record(tensor):
                observed[name].append(tensor.detach().clone())
                collect(tensor)

            return mock.patch.object(observer, "collect", side_effect=record)

        def before_forward(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            calls.append(
                {
                    "query_length": kwargs["input_ids"].shape[1],
                    "cache_in": cache,
                    "length_in": 0 if cache is None else int(cache.get_seq_length()),
                    "use_cache": kwargs.get("use_cache"),
                    "has_image": kwargs.get("pixel_values") is not None,
                    "mode": module._mode,
                    "grad_enabled": torch.is_grad_enabled(),
                    "training": module.training,
                    "input_ids": kwargs["input_ids"].detach().clone(),
                    "attention_mask": kwargs["attention_mask"].detach().clone(),
                }
            )

        def after_forward(module, args, kwargs, output):
            calls[-1]["cache_out"] = output.past_key_values
            calls[-1]["length_out"] = int(output.past_key_values.get_seq_length())
            calls[-1]["logits_shape"] = tuple(output.logits.shape)
            if calls[-1]["query_length"] == 1:
                next_logits = output.logits[:, -1, :]
            else:
                mask = calls[-1]["attention_mask"]
                last = (mask * torch.arange(mask.shape[1])).max(dim=1).values
                next_logits = output.logits[torch.arange(mask.shape[0]), last, :]
            calls[-1]["greedy_token"] = next_logits.argmax(dim=-1, keepdim=True)

        def record_positions(module, args, kwargs):
            positions.append(kwargs["position_ids"].detach().clone())

        def record_ple(module, args):
            ple_lengths.append(args[0].shape[1])

        def record_shared_state(module, args, kwargs):
            state = kwargs["shared_kv_states"]
            shared_states.append((state, int(state["full_attention"][0].shape[-2])))

        handles.append(
            prepared.register_forward_pre_hook(before_forward, with_kwargs=True)
        )
        handles.append(prepared.register_forward_hook(after_forward, with_kwargs=True))
        handles.append(
            text.layers[0].register_forward_pre_hook(record_positions, with_kwargs=True)
        )
        handles.append(
            text.embed_tokens_per_layer.register_forward_pre_hook(record_ple)
        )
        handles.append(
            consumer.register_forward_pre_hook(record_shared_state, with_kwargs=True)
        )
        handles.append(
            wrapped.model.wrapped.vision_tower.register_forward_pre_hook(
                lambda module, args: vision_calls.append(True)
            )
        )
        try:
            with (
                ExitStack() as observer_spies,
                mock.patch.object(
                    model,
                    "forward",
                    side_effect=AssertionError("unwrapped top forward"),
                ),
                mock.patch.object(
                    model.model,
                    "forward",
                    side_effect=AssertionError("unwrapped model forward"),
                ),
                mock.patch.object(
                    model.model.language_model,
                    "forward",
                    side_effect=AssertionError("unwrapped text forward"),
                ),
            ):
                for name in ("new_k", "new_v", "past_key", "past_value"):
                    observer_spies.enter_context(
                        record_observer(getattr(owner, f"obs_{name}"), name)
                    )
                for name in ("past_key", "past_value"):
                    observer_spies.enter_context(
                        record_observer(
                            getattr(consumer, f"obs_{name}"), f"shared_{name}"
                        )
                    )
                adapter.calibrate_prepared_model(ctx, prepared, {"name": "ptq"})
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual([call["query_length"] for call in calls], [14, 1, 1] * 2)
        self.assertEqual([call["length_in"] for call in calls], [0, 14, 15] * 2)
        self.assertEqual([call["length_out"] for call in calls], [14, 15, 16] * 2)
        self.assertEqual([length for _, length in shared_states], [14, 15, 16] * 2)
        for suffix, new_name in (("key", "new_k"), ("value", "new_v")):
            past = observed[f"past_{suffix}"]
            shared = observed[f"shared_past_{suffix}"]
            new = observed[new_name]
            self.assertEqual([tensor.shape[-2] for tensor in past], [14, 15] * 2)
            self.assertEqual([tensor.shape[-2] for tensor in shared], [14, 15, 16] * 2)
            # The owner reads only the preceding cache. In particular, the
            # current decode token must not enter its own past statistics.
            for sample in range(2):
                current = new[sample * 3 : sample * 3 + 3]
                for step in range(3):
                    expected = torch.cat(current[: step + 1], dim=-2)
                    torch.testing.assert_close(
                        shared[sample * 3 + step], expected, atol=0, rtol=0
                    )
                    if step < 2:
                        torch.testing.assert_close(
                            past[sample * 2 + step], expected, atol=0, rtol=0
                        )
        self.assertEqual(ple_lengths, [14, 1, 1] * 2)
        self.assertEqual(len(vision_calls), 2 if with_image else 0)
        self.assertEqual(
            [call["has_image"] for call in calls],
            ([True, False, False] if with_image else [False] * 3) * 2,
        )
        for sample_index in (0, 3):
            self.assertIsNone(calls[sample_index]["cache_in"])
            for offset in (1, 2):
                self.assertIs(
                    calls[sample_index + offset]["cache_in"],
                    calls[sample_index + offset - 1]["cache_out"],
                )
                torch.testing.assert_close(
                    calls[sample_index + offset]["input_ids"],
                    calls[sample_index + offset - 1]["greedy_token"],
                    atol=0,
                    rtol=0,
                )
        self.assertIsNot(calls[0]["cache_out"], calls[3]["cache_out"])
        self.assertEqual(len({id(state) for state, _ in shared_states}), 6)
        for call in calls:
            self.assertIs(call["use_cache"], True)
            self.assertIs(call["mode"], Mode.CALIB)
            self.assertFalse(call["grad_enabled"])
            self.assertFalse(call["training"])
            self.assertEqual(
                call["logits_shape"], (input_ids.shape[0], call["query_length"], 128)
            )
        for sample_index in (0, 3):
            lengths = batch["attention_mask"].sum(dim=-1, keepdim=True)
            expected_positions = batch["attention_mask"].cumsum(-1) - 1
            expected_positions.masked_fill_(batch["attention_mask"] == 0, 0)
            torch.testing.assert_close(positions[sample_index], expected_positions)
            torch.testing.assert_close(positions[sample_index + 1], lengths)
            torch.testing.assert_close(positions[sample_index + 2], lengths + 1)
            for offset in (1, 2):
                expected_mask = torch.cat(
                    (
                        batch["attention_mask"],
                        torch.ones(input_ids.shape[0], offset, dtype=torch.long),
                    ),
                    dim=-1,
                )
                torch.testing.assert_close(
                    calls[sample_index + offset]["attention_mask"], expected_mask
                )
        self.assertFalse(model.config.get_text_config().use_cache)
        for key, value in original_batch.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(batch[key], value, atol=0, rtol=0)
            else:
                self.assertEqual(batch[key], value)

        # Existing conversion freezes the statistics collected by all three calls.
        quantized = convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)
        self.assertIs(text._mode, Mode.QUANT)
        self.assertTrue(text.obs_per_layer_token_inputs.has_qparams)
        self.assertFalse(text.obs_per_layer_token_inputs.enabled)
        self.assertTrue(owner.obs_present_key.has_qparams)
        self.assertFalse(owner.obs_present_key.enabled)
        for attention in (owner, consumer):
            for observer in (attention.obs_past_key, attention.obs_past_value):
                self.assertTrue(observer.has_qparams)
                self.assertFalse(observer.enabled)
                scale, zero_point = observer.compute_qparams()
                self.assertTrue(bool(torch.isfinite(scale).all()))
                self.assertTrue(bool((scale > 0).all()))
                self.assertTrue(bool(torch.isfinite(zero_point).all()))

    def test_zero_decode_steps_keeps_past_unobserved_until_cached_inference(self):
        """Prefill-only conversion works; an uncalibrated cache read explains recovery."""
        model, prepared = self._make_prepared_model(shared_kv=False)
        adapter = Gemma4Adapter()
        input_ids = torch.arange(1, 5).reshape(1, 4)
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 16, "decode_steps": 0},
                "pipeline": [{"name": "ptq"}],
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=model,
            calibration_inputs=[{"input_ids": input_ids, "use_cache": False}],
        )
        text = prepared.wrapped.model.wrapped.language_model.wrapped
        owner = text.layers[0].wrapped.self_attn.wrapped
        with (
            mock.patch.object(owner.obs_past_key, "collect") as key_collect,
            mock.patch.object(owner.obs_past_value, "collect") as value_collect,
        ):
            adapter.calibrate_prepared_model(ctx, prepared, {"name": "ptq"})
        key_collect.assert_not_called()
        value_collect.assert_not_called()
        quantized = convert(prepared, inplace=True)
        self.assertFalse(owner.obs_past_key.has_qparams)
        self.assertFalse(owner.obs_past_value.has_qparams)
        self.assertTrue(owner.obs_new_k.has_qparams)
        self.assertTrue(owner.obs_present_key.has_qparams)
        with (
            torch.no_grad(),
            mock.patch.object(
                owner.obs_past_key,
                "fake_quant",
                side_effect=AssertionError("Prefill invoked past fake quantization"),
            ),
            mock.patch.object(
                owner.obs_past_value,
                "fake_quant",
                side_effect=AssertionError("Prefill invoked past fake quantization"),
            ),
        ):
            output = quantized(input_ids=input_ids, use_cache=True, return_dict=True)
        self.assertEqual(output.past_key_values.get_seq_length(), 4)
        with (
            torch.no_grad(),
            self.assertRaisesRegex(
                RuntimeError, "(?i)(decode.*calibrat|calibrat.*decode)"
            ),
        ):
            quantized(
                input_ids=torch.tensor([[5]]),
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )

    def test_loaded_frozen_read_qparams_allow_cached_inference(self):
        """Loading a calibrated checkpoint does not require another decode stage."""
        model, prepared = self._make_prepared_model()
        adapter = Gemma4Adapter()
        input_ids = torch.arange(1, 5).reshape(1, 4)
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 16, "decode_steps": 2},
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=model,
            calibration_inputs=[{"input_ids": input_ids}],
        )
        adapter.calibrate_prepared_model(ctx, prepared, {})
        quantized = convert(prepared, inplace=True)
        stream = io.BytesIO()
        torch.save(quantized, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        text = restored.wrapped.model.wrapped.language_model.wrapped
        source = quantized.wrapped.model.wrapped.language_model.wrapped
        for actual_layer, expected_layer in zip(text.layers, source.layers):
            actual = actual_layer.wrapped.self_attn.wrapped
            expected = expected_layer.wrapped.self_attn.wrapped
            for name in ("past_key", "past_value"):
                observer = getattr(actual, f"obs_{name}")
                expected_observer = getattr(expected, f"obs_{name}")
                self.assertTrue(observer.has_qparams)
                for actual_qparam, expected_qparam in zip(
                    observer.compute_qparams(), expected_observer.compute_qparams()
                ):
                    torch.testing.assert_close(
                        actual_qparam, expected_qparam, atol=0, rtol=0
                    )
        with torch.no_grad():
            output = restored(input_ids=input_ids, use_cache=True, return_dict=True)
            output = restored(
                input_ids=torch.tensor([[5]]),
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        self.assertEqual(output.past_key_values.get_seq_length(), 5)
        self.assertTrue(bool(torch.isfinite(output.logits).all()))

    def test_text_prefill_and_two_cached_decodes_with_ple_and_shared_kv(self):
        """The actual prepared text cache grows from fourteen to sixteen tokens."""
        self._calibrate_and_record(with_image=False)

    def test_image_vision_runs_only_during_each_prefill(self):
        """The actual tiny vision tower runs once per sample across three forwards."""
        self._calibrate_and_record(with_image=True)

    def test_padding_uses_valid_logits_and_per_row_decode_positions(self):
        """Left and right padding preserve masks, greedy selection and positions."""
        self._calibrate_and_record(with_image=False, padded=True)

    def test_actual_prepared_native_capacity_is_checked_before_forward(self):
        """The native context budget still rejects overflow before calibration."""
        model, prepared = self._make_prepared_model(max_seq=8)
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 32, "decode_steps": 2},
                "model_args": {"text": {"max_seq": 16}},
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=model,
            calibration_inputs=[{"input_ids": torch.arange(1, 16).reshape(1, 15)}],
        )
        with mock.patch.object(prepared, "forward") as forward:
            with self.assertRaisesRegex(ValueError, "capacity=16"):
                adapter.calibrate_prepared_model(ctx, prepared, {})
        forward.assert_not_called()

    def test_sliding_right_padding_is_rejected_on_actual_prepared_wrapper(self):
        """Right-padded prompts fail before inserting gaps in the sliding cache."""
        _, prepared = self._make_prepared_model(mixed_attention=True)
        adapter = Gemma4Adapter()
        input_ids = torch.tensor([list(range(1, 14)) + [0]])
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 16, "decode_steps": 2},
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=prepared,
            calibration_inputs=[
                {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}
            ],
        )
        with mock.patch.object(prepared, "forward") as forward:
            with self.assertRaisesRegex(ValueError, "right padding"):
                adapter.calibrate_prepared_model(ctx, prepared, {})
        forward.assert_not_called()

    def test_sliding_and_full_shared_kv_cached_decode_with_ple(self):
        """Cached decode crosses the sliding window with the actual hybrid model."""
        model, prepared = self._make_prepared_model(mixed_attention=True)
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 16, "decode_steps": 2},
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=model,
            calibration_inputs=[{"input_ids": torch.arange(1, 15).reshape(1, 14)}],
        )
        text = prepared.wrapped.model.wrapped.language_model.wrapped
        sliding_owner = text.layers[0].wrapped.self_attn.wrapped
        sliding_consumer = text.layers[2].wrapped.self_attn.wrapped
        full_consumer = text.layers[3].wrapped.self_attn.wrapped
        self.assertEqual(text.config.sliding_window, 8)
        self.assertTrue(sliding_owner.store_full_length_kv)
        self.assertTrue(sliding_consumer.is_kv_shared_layer)
        self.assertTrue(full_consumer.is_kv_shared_layer)
        lengths = []
        positions = []
        shared = []
        ple_lengths = []
        sliding_masks = []
        cache_lengths = []
        handles = []

        def record_call(module, args, kwargs):
            lengths.append(kwargs["input_ids"].shape[1])
            self.assertIs(module._mode, Mode.CALIB)
            self.assertTrue(kwargs["use_cache"])
            self.assertFalse(torch.is_grad_enabled())

        def record_output(module, args, output):
            cache = output.past_key_values
            cache_lengths.append(
                (int(cache.get_seq_length()), int(cache.layers[1].keys.shape[-2]))
            )

        def record_layer(module, args, kwargs):
            positions.append(kwargs["position_ids"].detach().clone())

        def record_shared(module, args, kwargs):
            states = kwargs["shared_kv_states"]
            shared.append(
                (
                    states,
                    int(states["sliding_attention"][0].shape[-2]),
                    int(states["full_attention"][0].shape[-2]),
                )
            )

        handles.append(
            prepared.register_forward_pre_hook(record_call, with_kwargs=True)
        )
        handles.append(prepared.register_forward_hook(record_output))
        handles.append(
            text.layers[0].register_forward_pre_hook(record_layer, with_kwargs=True)
        )
        handles.append(
            text.embed_tokens_per_layer.register_forward_pre_hook(
                lambda module, args: ple_lengths.append(args[0].shape[1])
            )
        )
        handles.append(
            full_consumer.register_forward_pre_hook(record_shared, with_kwargs=True)
        )
        # Inspect the sliding consumer mask independently from the full consumer.
        handles.append(
            sliding_consumer.register_forward_pre_hook(
                lambda module, args, kwargs: sliding_masks.append(
                    kwargs["attention_mask"].detach().clone()
                ),
                with_kwargs=True,
            )
        )
        try:
            adapter.calibrate_prepared_model(ctx, prepared, {})
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(lengths, [14, 1, 1])
        self.assertEqual(ple_lengths, [14, 1, 1])
        self.assertEqual(cache_lengths, [(14, 14), (15, 15), (16, 16)])
        self.assertEqual(
            [(sliding, full) for _, sliding, full in shared],
            [(14, 14), (8, 15), (8, 16)],
        )
        self.assertEqual(len({id(state) for state, _, _ in shared}), 3)
        torch.testing.assert_close(positions[1], torch.tensor([[14]]))
        torch.testing.assert_close(positions[2], torch.tensor([[15]]))
        for mask in sliding_masks[1:]:
            self.assertEqual(tuple(mask.shape), (1, 1, 1, 8))
            self.assertEqual(torch.count_nonzero(mask).item(), 0)
        quantized = convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)
        self.assertIs(text._mode, Mode.QUANT)
        self.assertTrue(text.obs_per_layer_token_inputs.has_qparams)
        self.assertFalse(sliding_owner.obs_present_key.enabled)


if __name__ == "__main__":
    unittest.main()

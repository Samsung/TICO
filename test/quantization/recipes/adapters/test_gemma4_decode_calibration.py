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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod

import torch
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.mode import Mode


class _Cache:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def get_seq_length(self, layer_idx=0):
        del layer_idx
        return self.seq_len


class _OriginalModel(torch.nn.Module):
    def forward(self, **kwargs):
        raise AssertionError("Calibration bypassed the prepared wrapper")


class _RecordingModel(torch.nn.Module):
    """Record routing independently of the real wrapper integration tests."""

    def __init__(
        self, *, capacity=16, return_cache=True, eos=False, missing_cache_call=None
    ):
        super().__init__()
        self.module = _OriginalModel()
        self.config = SimpleNamespace(
            use_cache=False,
            eos_token_id=1,
            text_config=SimpleNamespace(
                max_position_embeddings=capacity,
                use_cache=False,
                eos_token_id=1,
            ),
        )
        self.config.get_text_config = lambda: self.config.text_config
        self.mode = Mode.CALIB
        self.calls = []
        self.caches = []
        self.return_cache = return_cache
        self.missing_cache_call = missing_cache_call
        self.eos = eos
        self.reset_observers = Mock(side_effect=AssertionError("observer reset"))
        self.freeze_observers = Mock(side_effect=AssertionError("observer freeze"))

    def forward(self, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "training": self.training,
                "grad_enabled": torch.is_grad_enabled(),
                "mode": self.mode,
            }
        )
        input_ids = kwargs["input_ids"]
        old_cache = kwargs.get("past_key_values")
        seq_len = input_ids.shape[1]
        if old_cache is not None:
            seq_len += old_cache.get_seq_length()
        cache = (
            _Cache(seq_len)
            if self.return_cache and len(self.calls) != self.missing_cache_call
            else None
        )
        self.caches.append(cache)
        targets = torch.ones_like(input_ids) if self.eos else (input_ids + 1) % 32
        logits = torch.zeros((*input_ids.shape, 32))
        logits.scatter_(-1, targets.unsqueeze(-1), 1)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _context(*, global_steps=None, stage=None, capacity=16, model=None):
    adapter = Gemma4Adapter()
    calibration = {"seq_len": capacity, "n_samples": 1}
    if global_steps is not None:
        calibration["decode_steps"] = global_steps
    stage = {"name": "ptq"} if stage is None else stage
    model = _RecordingModel(capacity=capacity) if model is None else model
    return RecipeContext(
        cfg={
            "calibration": calibration,
            "pipeline": [stage],
            "runtime": {"show_progress": False},
        },
        adapter=adapter,
        model=model,
        processor=SimpleNamespace(),
    )


def _batch(length=4):
    return {
        "input_ids": torch.arange(2, length + 2).unsqueeze(0),
        "attention_mask": torch.ones((1, length), dtype=torch.long),
        "use_cache": False,
    }


class TestGemma4DecodeCalibrationConfig(unittest.TestCase):
    def test_effective_steps_reserve_the_same_prompt_budget_as_execution(self):
        cases = [
            (None, {}, 0),
            (2, {}, 2),
            (4, {"decode_calibration_steps": 2}, 2),
            (4, {"decode_calibration_steps": 0}, 0),
            ("2", {}, 2),
        ]
        for global_steps, override, expected in cases:
            with self.subTest(global_steps=global_steps, stage=override):
                stage = {"name": "ptq", **override}
                ctx = _context(global_steps=global_steps, stage=stage)
                cfg_before = copy.deepcopy(ctx.cfg)
                sample = _batch(16 - expected)
                with patch.object(
                    gemma4_mod,
                    "build_vlm_calibration_inputs",
                    return_value=[sample],
                ) as build:
                    ctx.calibration_inputs = ctx.adapter.build_calibration_inputs(ctx)
                self.assertEqual(build.call_args.kwargs["max_seq_len"], 16 - expected)
                ctx.adapter.calibrate_prepared_model(ctx, ctx.model, stage)
                self.assertEqual(len(ctx.model.calls), 1 + expected)
                self.assertEqual(ctx.cfg, cfg_before)

    def test_disabled_ptq_and_gptq_only_do_not_reserve_or_execute_decode(self):
        for pipeline in (
            [{"name": "ptq", "enabled": False, "decode_calibration_steps": 8}],
            [{"name": "gptq"}],
            [],
        ):
            with self.subTest(pipeline=pipeline):
                ctx = _context(global_steps=2)
                ctx.cfg["pipeline"] = pipeline
                sample = _batch(16)
                with patch.object(
                    gemma4_mod,
                    "build_vlm_calibration_inputs",
                    return_value=[sample],
                ) as build:
                    inputs = ctx.adapter.build_calibration_inputs(ctx)
                self.assertEqual(build.call_args.kwargs["max_seq_len"], 16)
                ctx.adapter.forward_calibration(
                    ctx, ctx.model, inputs, desc="GPTQ statistics collection"
                )
                self.assertEqual(len(ctx.model.calls), 1)
                self.assertFalse(ctx.model.calls[0]["use_cache"])

    def test_invalid_decode_steps_fail_before_loading_data_or_forward(self):
        for invalid in (-1, "bad", 1.5, True, None):
            for field in ("global", "stage"):
                with self.subTest(value=invalid, field=field):
                    ctx = _context()
                    if field == "global":
                        ctx.cfg["calibration"]["decode_steps"] = invalid
                    else:
                        ctx.cfg["pipeline"][0]["decode_calibration_steps"] = invalid
                    with patch.object(
                        gemma4_mod, "build_vlm_calibration_inputs"
                    ) as build:
                        with self.assertRaisesRegex(ValueError, "decode"):
                            ctx.adapter.build_calibration_inputs(ctx)
                    build.assert_not_called()
                    self.assertEqual(ctx.model.calls, [])
                    ctx.calibration_inputs = [_batch()]
                    with self.assertRaisesRegex(ValueError, "decode"):
                        ctx.adapter.calibrate_prepared_model(
                            ctx, ctx.model, ctx.cfg["pipeline"][0]
                        )
                    self.assertEqual(ctx.model.calls, [])

    def test_repeated_ptq_stages_must_reserve_the_same_decode_budget(self):
        cases: tuple[tuple[list[dict[str, int]], int], ...] = (
            ([{}, {}], 0),
            ([{"decode_calibration_steps": 2}, {"decode_calibration_steps": 2}], 2),
            ([{"decode_calibration_steps": 0}, {"decode_calibration_steps": 0}], 0),
        )
        for stages, expected in cases:
            with self.subTest(stages=stages):
                ctx = _context()
                ctx.cfg["pipeline"] = [{"name": "ptq", **stage} for stage in stages]
                with patch.object(
                    gemma4_mod, "build_vlm_calibration_inputs", return_value=[]
                ) as build:
                    ctx.adapter.build_calibration_inputs(ctx)
                self.assertEqual(build.call_args.kwargs["max_seq_len"], 16 - expected)
        ctx = _context(global_steps=2)
        ctx.cfg["pipeline"] = [
            {"name": "ptq"},
            {"name": "ptq", "decode_calibration_steps": 0},
        ]
        with patch.object(gemma4_mod, "build_vlm_calibration_inputs") as build:
            with self.assertRaisesRegex(ValueError, "(PTQ|decode)"):
                ctx.adapter.build_calibration_inputs(ctx)
        build.assert_not_called()

    def test_seq_len_must_leave_prefill_space(self):
        for capacity in (1, 2):
            with self.subTest(capacity=capacity):
                ctx = _context(global_steps=2, capacity=capacity)
                with patch.object(gemma4_mod, "build_vlm_calibration_inputs") as build:
                    with self.assertRaisesRegex(
                        ValueError, "(seq_len|capacity|context)"
                    ):
                        ctx.adapter.build_calibration_inputs(ctx)
                build.assert_not_called()

    def test_unspecified_recipe_capacity_uses_model_capacity(self):
        ctx = _context(global_steps=2)
        del ctx.cfg["calibration"]["seq_len"]
        with patch.object(
            gemma4_mod, "build_vlm_calibration_inputs", return_value=[]
        ) as build:
            ctx.adapter.build_calibration_inputs(ctx)
        self.assertEqual(build.call_args.kwargs["max_seq_len"], 14)

    def test_model_context_limit_is_respected(self):
        ctx = _context(global_steps=2)
        ctx.cfg["calibration"]["seq_len"] = 32
        with patch.object(
            gemma4_mod, "build_vlm_calibration_inputs", return_value=[]
        ) as build:
            ctx.adapter.build_calibration_inputs(ctx)
        self.assertEqual(build.call_args.kwargs["max_seq_len"], 14)

    def test_model_args_capacity_is_used_for_budget_and_runtime_validation(self):
        ctx = _context(global_steps=2)
        ctx.cfg["model_args"] = {"text": {"max_seq": 8}}
        with patch.object(
            gemma4_mod, "build_vlm_calibration_inputs", return_value=[]
        ) as build:
            ctx.adapter.build_calibration_inputs(ctx)
        self.assertEqual(build.call_args.kwargs["max_seq_len"], 6)
        ctx.calibration_inputs = [_batch(7)]
        with self.assertRaisesRegex(ValueError, "(context|capacity|budget)"):
            ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertEqual(ctx.model.calls, [])

    def test_builder_processed_image_sequence_is_checked_after_preprocessing(self):
        ctx = _context(global_steps=2)
        sample = _batch(15)
        sample["pixel_values"] = torch.ones(1, 3, 2, 2)
        snapshot = copy.deepcopy(sample)
        with patch.object(
            gemma4_mod, "build_vlm_calibration_inputs", return_value=[sample]
        ) as build:
            with self.assertRaisesRegex(ValueError, "(context|capacity|budget)"):
                ctx.adapter.build_calibration_inputs(ctx)
        self.assertEqual(build.call_args.kwargs["max_seq_len"], 14)
        self.assertEqual(ctx.model.calls, [])
        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                torch.testing.assert_close(sample[key], snapshot[key])
            else:
                self.assertEqual(sample[key], snapshot[key])

    def test_external_processed_sample_overflow_fails_without_truncating_images(self):
        ctx = _context(global_steps=2)
        sample = _batch(15)
        sample["pixel_values"] = torch.ones(1, 3, 2, 2)
        sample["image_position_ids"] = torch.tensor([[[0, 0]]])
        input_ids = sample["input_ids"].clone()
        image_positions = sample["image_position_ids"].clone()
        ctx.calibration_inputs = [sample]
        with self.assertRaisesRegex(ValueError, "(seq_len|capacity|context|length)"):
            ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertEqual(ctx.model.calls, [])
        torch.testing.assert_close(sample["input_ids"], input_ids)
        torch.testing.assert_close(sample["image_position_ids"], image_positions)


class TestGemma4DecodeCalibrationControlFlow(unittest.TestCase):
    def test_zero_steps_preserves_prefill_only_inputs(self):
        ctx = _context()
        sample = _batch()
        sample["pixel_values"] = torch.ones(1, 3, 2, 2)
        ctx.calibration_inputs = [sample]
        ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertEqual(len(ctx.model.calls), 1)
        self.assertFalse(ctx.model.calls[0]["use_cache"])
        self.assertNotIn("past_key_values", ctx.model.calls[0])
        self.assertIs(ctx.model.calls[0]["pixel_values"], sample["pixel_values"])

    def test_two_decodes_are_single_token_and_cache_state_is_sample_local(self):
        ctx = _context(global_steps=2)
        samples = [_batch(), _batch(3)]
        snapshots = copy.deepcopy(samples)
        config_before = copy.deepcopy(ctx.cfg)
        ctx.calibration_inputs = samples
        ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        calls = ctx.model.calls
        self.assertEqual(len(calls), 6)
        for offset, prompt_length in ((0, 4), (3, 3)):
            self.assertIsNone(calls[offset].get("past_key_values"))
            self.assertEqual(calls[offset]["input_ids"].shape, (1, prompt_length))
            for step in (1, 2):
                call = calls[offset + step]
                self.assertEqual(call["input_ids"].shape, (1, 1))
                self.assertIs(
                    call["past_key_values"], ctx.model.caches[offset + step - 1]
                )
                self.assertEqual(
                    ctx.model.caches[offset + step].get_seq_length(),
                    prompt_length + step,
                )
                torch.testing.assert_close(
                    call["input_ids"],
                    (calls[offset + step - 1]["input_ids"][:, -1:] + 1) % 32,
                )
                self.assertEqual(
                    call["attention_mask"].shape, (1, prompt_length + step)
                )
                torch.testing.assert_close(
                    call["position_ids"], torch.tensor([[prompt_length + step - 1]])
                )
        self.assertIsNot(ctx.model.caches[0], ctx.model.caches[3])
        for call in calls:
            self.assertTrue(call["use_cache"])
            self.assertFalse(call["training"])
            self.assertFalse(call["grad_enabled"])
            self.assertEqual(call["mode"], Mode.CALIB)
        ctx.model.reset_observers.assert_not_called()
        ctx.model.freeze_observers.assert_not_called()
        self.assertFalse(ctx.model.config.use_cache)
        self.assertFalse(ctx.model.config.text_config.use_cache)
        self.assertEqual(ctx.cfg, config_before)
        for sample, snapshot in zip(samples, snapshots):
            self.assertEqual(set(sample), set(snapshot))
            for key in sample:
                if isinstance(sample[key], torch.Tensor):
                    torch.testing.assert_close(sample[key], snapshot[key])
                else:
                    self.assertEqual(sample[key], snapshot[key])

    def test_images_and_prefill_metadata_are_not_replayed_on_decode(self):
        ctx = _context(global_steps=2)
        sample = _batch()
        sample.update(
            pixel_values=torch.ones(1, 3, 2, 2),
            image_position_ids=torch.tensor([[[0, 0]]]),
            mm_token_type_ids=torch.tensor([[0, 1, 1, 0]]),
            token_type_ids=torch.tensor([[0, 1, 1, 0]]),
        )
        ctx.calibration_inputs = [sample]
        ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertIs(ctx.model.calls[0]["pixel_values"], sample["pixel_values"])
        for call in ctx.model.calls[1:]:
            self.assertIsNone(call.get("pixel_values"))
            self.assertIsNone(call.get("image_position_ids"))
            self.assertIsNone(call.get("per_layer_inputs"))
            self.assertIsNone(call.get("inputs_embeds"))
            for name in ("mm_token_type_ids", "token_type_ids"):
                metadata = call.get(name)
                if metadata is not None:
                    torch.testing.assert_close(
                        metadata, torch.zeros((1, 1), dtype=torch.long)
                    )

    def test_padding_uses_last_valid_logits_and_logical_positions_for_each_item(self):
        ctx = _context(global_steps=2)
        ctx.calibration_inputs = [
            {
                "input_ids": torch.tensor([[2, 3, 0, 0], [0, 4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]]),
                "mm_token_type_ids": torch.zeros((2, 4), dtype=torch.long),
            }
        ]
        ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        prefill, first, second = ctx.model.calls
        torch.testing.assert_close(
            prefill["position_ids"], torch.tensor([[0, 1, 0, 0], [0, 0, 1, 2]])
        )
        torch.testing.assert_close(first["input_ids"], torch.tensor([[4], [7]]))
        torch.testing.assert_close(second["input_ids"], torch.tensor([[5], [8]]))
        torch.testing.assert_close(first["position_ids"], torch.tensor([[2], [3]]))
        torch.testing.assert_close(second["position_ids"], torch.tensor([[3], [4]]))
        torch.testing.assert_close(
            second["attention_mask"],
            torch.tensor([[1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 1]]),
        )
        for call in (first, second):
            torch.testing.assert_close(
                call["mm_token_type_ids"], torch.zeros((2, 1), dtype=torch.long)
            )

    def test_sliding_attention_rejects_right_padding_before_forward(self):
        ctx = _context(global_steps=2)
        ctx.model.config.text_config.layer_types = [
            "sliding_attention",
            "full_attention",
        ]
        ctx.calibration_inputs = [
            {
                "input_ids": torch.tensor([[2, 3, 0, 0], [0, 4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]]),
            }
        ]
        with self.assertRaisesRegex(ValueError, "(right.*padding|sliding)"):
            ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertEqual(ctx.model.calls, [])

    def test_sliding_attention_preserves_left_padding_and_logical_positions(self):
        ctx = _context(global_steps=2)
        ctx.model.config.text_config.layer_types = [
            "sliding_attention",
            "full_attention",
        ]
        ctx.calibration_inputs = [
            {
                "input_ids": torch.tensor([[0, 0, 2, 3], [0, 4, 5, 6]]),
                "attention_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
            }
        ]
        ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertEqual(len(ctx.model.calls), 3)
        torch.testing.assert_close(
            ctx.model.calls[1]["input_ids"], torch.tensor([[4], [7]])
        )
        torch.testing.assert_close(
            ctx.model.calls[1]["position_ids"], torch.tensor([[2], [3]])
        )
        torch.testing.assert_close(
            ctx.model.calls[2]["position_ids"], torch.tensor([[3], [4]])
        )
        torch.testing.assert_close(
            ctx.model.calls[2]["attention_mask"],
            torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]]),
        )

    def test_unsupported_prefill_state_is_rejected_before_forward(self):
        invalid_fields = {
            "past_key_values": _Cache(2),
            "shared_kv_states": {0: object()},
            "cache_position": torch.tensor([0, 1, 2, 3]),
            "inputs_embeds": torch.ones(1, 4, 3),
            "per_layer_inputs": torch.ones(1, 4, 2, 3),
        }
        for name, value in invalid_fields.items():
            with self.subTest(field=name):
                ctx = _context(global_steps=2)
                sample = _batch()
                sample[name] = value
                ctx.calibration_inputs = [sample]
                with self.assertRaisesRegex(ValueError, name):
                    ctx.adapter.calibrate_prepared_model(
                        ctx, ctx.model, {"name": "ptq"}
                    )
                self.assertEqual(ctx.model.calls, [])

    def test_invalid_masks_fail_without_dropping_a_batch_item(self):
        for mask in (
            [[0, 0, 0, 0]],
            [[1, 0, 1, 0]],
            [[1, 1, 2, 0]],
            [[1, 1, 1]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
        ):
            with self.subTest(mask=mask):
                ctx = _context(global_steps=2)
                sample = _batch()
                sample["input_ids"] = sample["input_ids"].expand(len(mask), -1)
                sample["attention_mask"] = torch.tensor(mask)
                ctx.calibration_inputs = [sample]
                with self.assertRaisesRegex(ValueError, "(mask|padding)"):
                    ctx.adapter.calibrate_prepared_model(
                        ctx, ctx.model, {"name": "ptq"}
                    )
                self.assertEqual(ctx.model.calls, [])

    def test_custom_positions_are_rejected_before_forward(self):
        ctx = _context(global_steps=2)
        sample = _batch()
        sample["position_ids"] = torch.tensor([[4, 5, 6, 7]])
        ctx.calibration_inputs = [sample]
        with self.assertRaisesRegex(ValueError, "position_ids"):
            ctx.adapter.calibrate_prepared_model(ctx, ctx.model, {"name": "ptq"})
        self.assertEqual(ctx.model.calls, [])

    def test_invalid_position_shape_fails_with_a_clear_error(self):
        for positions in (torch.tensor([[0, 1, 2]]), torch.zeros(1, 1, 4)):
            with self.subTest(shape=positions.shape):
                ctx = _context(global_steps=2)
                sample = _batch()
                sample["position_ids"] = positions
                ctx.calibration_inputs = [sample]
                with self.assertRaisesRegex(ValueError, "position_ids"):
                    ctx.adapter.calibrate_prepared_model(
                        ctx, ctx.model, {"name": "ptq"}
                    )
                self.assertEqual(ctx.model.calls, [])

    def test_eos_does_not_reduce_the_number_of_decode_forwards(self):
        model = _RecordingModel(eos=True)
        ctx = _context(global_steps=2, model=model)
        ctx.calibration_inputs = [_batch()]
        ctx.adapter.calibrate_prepared_model(ctx, model, {"name": "ptq"})
        self.assertEqual(len(model.calls), 3)
        for call in model.calls[1:]:
            torch.testing.assert_close(call["input_ids"], torch.tensor([[1]]))

    def test_missing_cache_is_an_error_and_never_recomputes_the_prefix(self):
        model = _RecordingModel(return_cache=False)
        ctx = _context(global_steps=2, model=model)
        ctx.calibration_inputs = [_batch()]
        with self.assertRaisesRegex((ValueError, RuntimeError), "cache"):
            ctx.adapter.calibrate_prepared_model(ctx, model, {"name": "ptq"})
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)

    def test_cache_missing_after_decode_is_also_an_error(self):
        model = _RecordingModel(missing_cache_call=2)
        ctx = _context(global_steps=2, model=model)
        ctx.calibration_inputs = [_batch()]
        with self.assertRaisesRegex((ValueError, RuntimeError), "cache"):
            ctx.adapter.calibrate_prepared_model(ctx, model, {"name": "ptq"})
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[1]["input_ids"].shape, (1, 1))
        self.assertIs(model.calls[1]["past_key_values"], model.caches[0])


if __name__ == "__main__":
    unittest.main()

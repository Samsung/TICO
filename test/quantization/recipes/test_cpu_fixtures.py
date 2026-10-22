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

"""Real tiny HF fixture and teacher-forced calibration-budget regressions."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from tico.quantization import prepare
from tico.quantization.recipes.debug.cpu_fixtures import (
    fixture_context,
    qwen_batch,
    qwen_model,
)
from tico.quantization.recipes.debug.cpu_followup import _command


class TestCPUFixtures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_qwen_image_and_realistic_head_dimension_are_bounded(self):
        """A real image reaches a bounded vision model with valid mRoPE."""
        for head_dim in (32, 128):
            with self.subTest(head_dim=head_dim):
                model = qwen_model(head_dim=head_dim)
                self.assertLess(
                    sum(parameter.numel() for parameter in model.parameters()),
                    2_000_000,
                )
                with torch.no_grad():
                    result = model(**qwen_batch(), use_cache=True, return_dict=True)
                self.assertEqual(tuple(result.logits.shape), (1, 8, 128))
                self.assertEqual(result.past_key_values.get_seq_length(), 8)
                self.assertTrue(torch.isfinite(result.logits).all())

    def test_llama_decode_calibration_reserves_teacher_forcing_token_budget(self):
        """Calibration consumes each dataset ID once, including short samples."""
        for length, requested in ((8, 0), (8, 2), (8, 20), (1, 20)):
            with self.subTest(length=length, requested=requested):
                ctx = fixture_context("llama")
                sample = ctx.calibration_inputs[0][:, :length]
                ctx.calibration_inputs = [sample]
                original_config = copy.deepcopy(ctx.cfg)
                stage = dict(ctx.cfg["pipeline"][0], decode_calibration_steps=requested)
                prepared = prepare(ctx.model, ctx.adapter.build_ptq_config(ctx, stage))
                calls = []

                def record(_module, args, kwargs):
                    ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
                    calls.append(ids.detach().clone())

                handle = prepared.register_forward_pre_hook(record, with_kwargs=True)
                try:
                    ctx.adapter.calibrate_prepared_model(ctx, prepared, stage)
                finally:
                    handle.remove()
                self.assertTrue(torch.equal(torch.cat(calls, dim=-1), sample))
                steps = min(requested, length - 1)
                self.assertEqual(
                    [value.shape[-1] for value in calls], [length - steps] + [1] * steps
                )
                self.assertEqual(ctx.cfg, original_config)

    def test_llama_rejects_negative_decode_calibration_before_execution(self):
        ctx = fixture_context("llama")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ctx.adapter.calibrate_prepared_model(
                ctx, ctx.model, {"decode_calibration_steps": -1}
            )

    def test_fixed_cache_capacity_and_padded_slot_masks(self):
        """Only the valid prefix and current-token slot are visible."""
        from tico.quantization.recipes.debug.cpu_static_cache import decode_mask

        for capacity in (16, 32):
            for past_len in (0, 1, capacity // 2, capacity - 1):
                mask = decode_mask(capacity=capacity, past_len=past_len, fill=-120.0)
                self.assertEqual(tuple(mask.shape), (1, 1, 1, capacity))
                self.assertTrue((mask[..., :past_len] == 0).all())
                self.assertTrue((mask[..., past_len:-1] == -120.0).all())
                self.assertEqual(mask[..., -1].item(), 0.0)
            for past_len in (-1, capacity, capacity + 1):
                with self.assertRaisesRegex(ValueError, "no room"):
                    decode_mask(capacity=capacity, past_len=past_len, fill=-120.0)

    def test_runner_rejects_empty_or_skipped_unit_stage(self):
        """UNIT_MOCK: selected unittest suites cannot pass without execution."""
        import json
        from unittest.mock import patch

        from tico.quantization.recipes.debug.cpu_followup import main

        class Skipped(unittest.TestCase):
            @unittest.skip("Deliberate fixture skip")
            def runTest(self):
                pass

        for suite in (unittest.TestSuite(), unittest.TestSuite([Skipped()])):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "stage"
                with patch.object(
                    unittest.TestLoader, "loadTestsFromNames", return_value=suite
                ):
                    code = main(["--stage", "_tests", "--output", str(output)])
                self.assertEqual(code, 1)
                self.assertEqual(
                    json.loads((output / "summary.json").read_text())["status"],
                    "INCOMPLETE",
                )

    def test_runner_preserves_child_failure_exit_code(self):
        """A failing child has a nonzero recorded result and retained stdout."""
        import sys

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = _command(
                root,
                "failure",
                [
                    sys.executable,
                    "-c",
                    "print('failure evidence'); raise SystemExit(7)",
                ],
                cwd=root,
            )
            self.assertEqual(result["returncode"], 7)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("failure evidence", (root / "failure.log").read_text())

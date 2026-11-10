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

"""Exercise actual runtime helpers with nonzero HF sine and an explicit ABI."""

import unittest
from types import SimpleNamespace

import torch


class _RecordingRotary:
    """Return fresh dense/partial tables, not a last-dimension-one dummy."""

    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.snapshots = {}

    def __call__(self, x, position_ids, layer_type):
        width = 8 if layer_type == "full_attention" else 4
        frequencies = torch.linspace(0.1, 0.7, width // 2, device=x.device)
        angles = (position_ids.to(x.device).float().unsqueeze(-1) + 1) * frequencies
        if layer_type == "full_attention":
            angles[..., width // 4 :] = 0.0  # Inactive partial-RoPE channels.
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        self.inputs[layer_type] = position_ids.clone()
        self.outputs[layer_type] = (cos, sin)
        self.snapshots[layer_type] = (cos.clone(), sin.clone())
        return cos, sin


class _RuntimeModel(torch.nn.Module):
    def __init__(self, rotary):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.model = SimpleNamespace(language_model=SimpleNamespace(rotary_emb=rotary))


def _runtime(convention):
    rotary = _RecordingRotary()
    runtime = SimpleNamespace(
        rope_convention=convention,
        model=_RuntimeModel(rotary),
        text_config=SimpleNamespace(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=3,
        ),
        layout=SimpleNamespace(max_seq=8),
        device=torch.device("cpu"),
        past_len=3,
        attention_mask_fill_value=-37.5,
    )
    return runtime, rotary


class TestStaticGemma4RoPEInputContract(unittest.TestCase):
    def _assert_tables(self, tables, rotary, convention):
        for kind, (cos, sin) in tables.items():
            raw_cos, raw_sin = rotary.snapshots[kind]
            expected = raw_sin.clone()
            if convention == "pre_negated_sin":
                expected[..., : expected.size(-1) // 2] *= -1
            torch.testing.assert_close(cos, raw_cos, atol=0, rtol=0)
            torch.testing.assert_close(sin, expected, atol=0, rtol=0)
            self.assertEqual(sin.size(-1) % 2, 0)
            self.assertTrue(torch.count_nonzero(raw_sin) > 0)
            # Preparing the NPU input must not change the reference table.
            torch.testing.assert_close(rotary.outputs[kind][1], raw_sin, atol=0, rtol=0)

    def test_prefill_prepares_sine_without_changing_masks(self):
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        baseline_masks = None
        for convention in ("hf", "pre_negated_sin"):
            with self.subTest(convention=convention):
                runtime, rotary = _runtime(convention)
                masks, tables = StaticGemma4Runtime.build_prefill_masks_and_rope(
                    runtime,
                    torch.ones(1, 8, dtype=torch.long),
                    torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]]),
                )
                self._assert_tables(tables, rotary, convention)
                self.assertEqual(masks["full_attention"][0, 0, 0, 3], -37.5)
                if baseline_masks is not None:
                    for kind in masks:
                        torch.testing.assert_close(
                            masks[kind], baseline_masks[kind], atol=0, rtol=0
                        )
                baseline_masks = masks

    def test_decode_prepares_sine_at_past_len(self):
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        for convention in ("hf", "pre_negated_sin"):
            with self.subTest(convention=convention):
                runtime, rotary = _runtime(convention)
                masks, tables = StaticGemma4Runtime.build_decode_masks_and_rope(
                    runtime, batch_size=1, dtype=torch.float32
                )
                self._assert_tables(tables, rotary, convention)
                for positions in rotary.inputs.values():
                    torch.testing.assert_close(positions, torch.tensor([[3]]))
                self.assertEqual(masks["full_attention"][0, 0, 3], -37.5)
                self.assertEqual(masks["full_attention"][0, 0, 7], 0.0)
                self.assertEqual(masks["sliding_attention"][0, 0, 0], -37.5)

    def test_missing_convention_is_not_silently_interpreted_as_hf(self):
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
        )

        runtime, _ = _runtime("pre_negated_sin")
        del runtime.rope_convention
        with self.assertRaisesRegex(AttributeError, "rope_convention"):
            StaticGemma4Runtime.build_decode_masks_and_rope(
                runtime, batch_size=1, dtype=torch.float32
            )


if __name__ == "__main__":
    unittest.main()

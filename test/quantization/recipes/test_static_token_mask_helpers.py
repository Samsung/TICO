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

import unittest

import tico.quantization.recipes.debug.static_llama_runtime as llama_runtime
import tico.quantization.recipes.debug.static_qwen3_vl_runtime as qwen_runtime

import torch


class TestStaticTokenMaskHelpers(unittest.TestCase):
    """Characterize shared token-mask semantics through both existing paths."""

    RUNTIMES = (
        (
            "llama",
            llama_runtime,
            llama_runtime._build_position_ids_from_valid_token_mask,
        ),
        (
            "qwen3_vl",
            qwen_runtime,
            qwen_runtime._build_text_position_ids_from_valid_token_mask,
        ),
    )

    def test_attention_mask_takes_precedence_and_does_not_mutate_inputs(self):
        input_ids = torch.tensor([[0, 5, 6], [7, 0, 9]])
        attention_mask = torch.tensor([[1, 0, -2], [0, 3, 0]])
        original_ids = input_ids.clone()
        original_mask = attention_mask.clone()
        expected = torch.tensor([[True, False, True], [False, True, False]])
        for family, runtime, _position_ids in self.RUNTIMES:
            with self.subTest(family=family):
                valid = runtime._normalize_valid_token_mask(
                    input_ids,
                    attention_mask,
                    pad_token_id=0,
                    device=torch.device("cpu"),
                )
                self.assertTrue(torch.equal(valid, expected))
                self.assertEqual(valid.dtype, torch.bool)
                self.assertEqual(valid.device, torch.device("cpu"))
                self.assertTrue(torch.equal(input_ids, original_ids))
                self.assertTrue(torch.equal(attention_mask, original_mask))
                valid.logical_not_()
                self.assertTrue(torch.equal(attention_mask, original_mask))

    def test_boolean_attention_mask_preserves_same_device_ownership(self):
        for family, runtime, _position_ids in self.RUNTIMES:
            with self.subTest(family=family):
                input_ids = torch.tensor([[1, 2, 0]])
                attention_mask = torch.tensor([[True, True, False]])
                valid = runtime._normalize_valid_token_mask(
                    input_ids,
                    attention_mask,
                    pad_token_id=0,
                    device=torch.device("cpu"),
                )
                self.assertIs(valid, attention_mask)

    def test_without_attention_mask_uses_only_configured_pad_token(self):
        input_ids = torch.tensor([[0, 4, 0], [8, 9, 0]])
        original = input_ids.clone()
        for family, runtime, _position_ids in self.RUNTIMES:
            for pad_token_id, expected in (
                (0, [[False, True, False], [True, True, False]]),
                (None, [[True, True, True], [True, True, True]]),
            ):
                with self.subTest(family=family, pad_token_id=pad_token_id):
                    valid = runtime._normalize_valid_token_mask(
                        input_ids,
                        None,
                        pad_token_id=pad_token_id,
                        device=torch.device("cpu"),
                    )
                    self.assertEqual(valid.tolist(), expected)
                    self.assertEqual(valid.dtype, torch.bool)
                    valid.logical_not_()
                    self.assertTrue(torch.equal(input_ids, original))

    def test_shape_mismatch_retains_diagnostic(self):
        for family, runtime, _position_ids in self.RUNTIMES:
            with self.subTest(family=family):
                with self.assertRaisesRegex(
                    ValueError,
                    r"attention_mask must have the same shape as input_ids\. "
                    r"Got attention_mask=\(1, 2\), input_ids=\(1, 3\)\.",
                ):
                    runtime._normalize_valid_token_mask(
                        torch.tensor([[1, 2, 3]]),
                        torch.tensor([[1, 0]]),
                        pad_token_id=None,
                        device=torch.device("cpu"),
                    )

    def test_empty_sequence_and_each_all_padding_row_are_rejected(self):
        cases = (
            (torch.empty((2, 0), dtype=torch.long), None, None),
            (torch.tensor([[0, 0], [1, 0]]), None, 0),
            (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 0], [0, 0]]), None),
        )
        for family, runtime, _position_ids in self.RUNTIMES:
            for input_ids, attention_mask, pad_token_id in cases:
                with self.subTest(
                    family=family, shape=tuple(input_ids.shape), mask=attention_mask
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Each batch row must contain at least one real token",
                    ):
                        runtime._normalize_valid_token_mask(
                            input_ids,
                            attention_mask,
                            pad_token_id=pad_token_id,
                            device=torch.device("cpu"),
                        )

    def test_empty_batch_stays_empty(self):
        input_ids = torch.empty((0, 3), dtype=torch.long)
        for family, runtime, position_ids in self.RUNTIMES:
            with self.subTest(family=family):
                valid = runtime._normalize_valid_token_mask(
                    input_ids,
                    None,
                    pad_token_id=None,
                    device=torch.device("cpu"),
                )
                self.assertEqual(tuple(valid.shape), (0, 3))
                self.assertEqual(tuple(position_ids(valid).shape), (0, 3))
                self.assertEqual(
                    tuple(runtime._last_valid_token_indices(valid).shape), (0,)
                )

    def test_contiguous_left_and_right_padding_and_unpadded_rows(self):
        cases = (
            ("right", [[True, True, False], [True, False, False]]),
            ("left", [[False, True, True], [False, False, True]]),
            ("right", [[True, True, True]]),
            ("left", [[True, True, True]]),
            ("right", [[False, False, False]]),
            ("left", [[False, False, False]]),
        )
        for family, runtime, _position_ids in self.RUNTIMES:
            for padding_side, rows in cases:
                with self.subTest(family=family, padding_side=padding_side, rows=rows):
                    valid = torch.tensor(rows)
                    before = valid.clone()
                    runtime._validate_padding_layout(valid, padding_side)
                    self.assertTrue(torch.equal(valid, before))

    def test_mismatched_and_interior_padding_retain_diagnostic(self):
        cases = (
            ("right", [[False, True, True]]),
            ("left", [[True, True, False]]),
            ("right", [[True, False, True]]),
            ("left", [[True, False, True]]),
        )
        for family, runtime, _position_ids in self.RUNTIMES:
            for padding_side, rows in cases:
                with self.subTest(family=family, padding_side=padding_side, rows=rows):
                    valid = torch.tensor(rows)
                    before = valid.clone()
                    with self.assertRaisesRegex(
                        ValueError,
                        "Input padding layout does not match padding_side. "
                        f"Expected contiguous {padding_side} padding for shape "
                        r"\(B=1, S=3\)\.",
                    ):
                        runtime._validate_padding_layout(valid, padding_side)
                    self.assertTrue(torch.equal(valid, before))

    def test_unsupported_padding_side_is_rejected(self):
        for family, runtime, _position_ids in self.RUNTIMES:
            with self.subTest(family=family):
                with self.assertRaisesRegex(
                    ValueError, "padding_side must be 'left' or 'right'. got 'middle'"
                ):
                    runtime._validate_padding_layout(torch.tensor([[True]]), "middle")

    def test_position_ids_and_last_indices_preserve_values_dtype_and_inputs(self):
        valid = torch.tensor(
            [
                [True, True, False, False],
                [False, False, True, True],
                [False, True, False, True],
                [False, False, False, False],
            ]
        )
        before = valid.clone()
        expected_positions = torch.tensor(
            [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
        expected_indices = torch.tensor([1, 3, 3, 0])
        for family, runtime, position_ids in self.RUNTIMES:
            with self.subTest(family=family):
                positions = position_ids(valid)
                indices = runtime._last_valid_token_indices(valid)
                self.assertTrue(torch.equal(positions, expected_positions))
                self.assertTrue(torch.equal(indices, expected_indices))
                self.assertEqual(positions.dtype, torch.long)
                self.assertEqual(indices.dtype, torch.long)
                self.assertEqual(positions.device, valid.device)
                self.assertEqual(indices.device, valid.device)
                positions.zero_()
                indices.zero_()
                self.assertTrue(torch.equal(valid, before))


if __name__ == "__main__":
    unittest.main()

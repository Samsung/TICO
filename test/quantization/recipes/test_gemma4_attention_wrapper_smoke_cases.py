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

"""Tests for mode-specific Gemma4 attention wrapper-smoke cases."""

import unittest

from tico.quantization.recipes.debug.wrapper_smoke import case_names, get_case
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    _has_gemma4,
    Gemma4TextAttentionDecodeCase,
    Gemma4TextAttentionSharedKVDecodeCase,
    Gemma4TextSlidingAttentionDecodeCase,
    Gemma4TextSlidingAttentionPrefillCase,
)


_EXPECTED_CASES = {
    "gemma4_text_attention_prefill",
    "gemma4_text_attention_decode",
    "gemma4_text_attention_sliding_prefill",
    "gemma4_text_attention_sliding_decode",
    "gemma4_text_attention_k_eq_v_prefill",
    "gemma4_text_attention_shared_kv_prefill",
    "gemma4_text_attention_shared_kv_decode",
}

_AMBIGUOUS_LEGACY_CASES = {
    "gemma4_text_attention",
    "gemma4_text_attention_sliding",
    "gemma4_text_attention_k_eq_v",
    "gemma4_text_attention_shared_kv",
}


class TestGemma4AttentionSmokeRegistry(unittest.TestCase):
    """Validate the public wrapper-smoke case names."""

    def test_mode_specific_cases_are_registered(self):
        """Expose every supported Gemma4 attention execution contract."""
        names = set(case_names())
        self.assertTrue(_EXPECTED_CASES.issubset(names))

    def test_ambiguous_attention_case_names_are_removed(self):
        """Reject case names that do not identify prefill or decode mode."""
        names = set(case_names())
        self.assertTrue(_AMBIGUOUS_LEGACY_CASES.isdisjoint(names))

    def test_registry_returns_mode_specific_case(self):
        """Return the exact decode case requested by the CLI."""
        case = get_case("gemma4_text_attention_sliding_decode")
        self.assertEqual(case.name, "gemma4_text_attention_sliding_decode")
        self.assertEqual(case.export_mode, "decode")  # type: ignore[attr-defined]


_GEMMA4_AVAILABILITY = _has_gemma4()


@unittest.skipUnless(
    _GEMMA4_AVAILABILITY.available,
    _GEMMA4_AVAILABILITY.reason or "Gemma4 is unavailable",
)
class TestGemma4AttentionDecodeSmokeInputs(unittest.TestCase):
    """Validate fixed-shape decode inputs before quantization and export."""

    def test_full_decode_uses_past_cache_and_one_query_token(self):
        """Create a single-token query with a fixed-capacity past cache."""
        case = Gemma4TextAttentionDecodeCase()
        module, _ = case.build({})
        sample = case.eval_input(module, {})
        kwargs = dict(sample.kwargs)
        past_key, past_value = kwargs["past_key_value"]

        self.assertEqual(kwargs["hidden_states"].shape[1], 1)
        self.assertEqual(kwargs["attention_mask"].shape, (1, 1, 1, case.max_seq))
        self.assertEqual(past_key.shape[2], case.max_seq - 1)
        self.assertEqual(past_value.shape, past_key.shape)
        self.assertTrue(kwargs["use_cache"])

    def test_shared_decode_uses_full_shared_cache(self):
        """Provide full shared K/V without a layer-owned past-cache input."""
        case = Gemma4TextAttentionSharedKVDecodeCase()
        module, _ = case.build({})
        sample = case.eval_input(module, {})
        kwargs = dict(sample.kwargs)
        shared_key, shared_value = kwargs["shared_key_value"]

        self.assertNotIn("past_key_value", kwargs)
        self.assertEqual(kwargs["hidden_states"].shape[1], 1)
        self.assertEqual(shared_key.shape[2], case.max_seq)
        self.assertEqual(shared_value.shape, shared_key.shape)

    def test_sliding_prefill_masks_tokens_before_the_window(self):
        """Exercise left-window masking in the sliding prefill case."""
        case = Gemma4TextSlidingAttentionPrefillCase()
        module, _ = case.build({})
        mask = case.eval_input(module, {}).kwargs["attention_mask"]

        self.assertEqual(case.sliding_window, 4)
        self.assertTrue((mask[0, 0, -1, :4] == -120.0).all())
        self.assertTrue((mask[0, 0, -1, 4:] == 0.0).all())

    def test_sliding_decode_uses_the_last_sliding_mask_row(self):
        """Use a single decode row with the same fixed sliding-window policy."""
        case = Gemma4TextSlidingAttentionDecodeCase()
        module, _ = case.build({})
        mask = case.eval_input(module, {}).kwargs["attention_mask"]

        self.assertEqual(mask.shape, (1, 1, 1, case.max_seq))
        self.assertTrue((mask[0, 0, 0, :4] == -120.0).all())
        self.assertTrue((mask[0, 0, 0, 4:] == 0.0).all())


if __name__ == "__main__":
    unittest.main()

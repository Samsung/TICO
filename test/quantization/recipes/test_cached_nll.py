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

"""Cached-decode NLL scores exactly the configured targets through a real cache."""

import copy
import math
import unittest
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

import torch
import torch.nn.functional as F
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.debug.cpu_fixtures import gemma_model
from tico.quantization.recipes.evaluation.cached_nll import (
    CachedNLLProtocol,
    chunk_token_ranges,
    evaluate_cached_nll,
    evaluate_text_cached_nll,
    score_cached_chunk,
)
from tico.quantization.wrapq.dtypes import DType


def _independent_target_nll(model, ids: torch.Tensor, first_target: int):
    """Score ``ids[first_target:]`` from one uncached full-sequence forward."""
    with torch.inference_mode():
        logits = (
            model(input_ids=ids.unsqueeze(0), use_cache=False, return_dict=True)
            .logits[0]
            .float()
        )
    positions = logits[first_target - 1 : -1]
    targets = ids[first_target:]
    return F.cross_entropy(positions, targets, reduction="none").double()


class _FakeTokenizer:
    """Deterministic local tokenizer: ``t<id>`` words map to their IDs."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "t5 t6 t7"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [int(token[1:]) for token in str(text).split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class TestCachedNLL(unittest.TestCase):
    fp: ClassVar[torch.nn.Module]
    stream: ClassVar[torch.Tensor]

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5)
        cls.fp = gemma_model(seed=1073)
        cls.stream = torch.randint(
            2, 127, (96,), generator=torch.Generator().manual_seed(9)
        )

    def test_ranges_are_contiguous_non_overlapping_and_bounded(self):
        protocol = CachedNLLProtocol(
            prefix_tokens=6, target_tokens=4, n_chunks=3, start_offset=2
        )
        self.assertEqual(
            chunk_token_ranges(32, protocol), [(2, 12), (12, 22), (22, 32)]
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            chunk_token_ranges(31, protocol)
        invalid: tuple[dict[str, Any], ...] = (
            dict(prefix_tokens=0, target_tokens=4, n_chunks=1),
            dict(prefix_tokens=4, target_tokens=0, n_chunks=1),
            dict(prefix_tokens=4, target_tokens=4, n_chunks=0),
            dict(prefix_tokens=4, target_tokens=4, n_chunks=1, start_offset=-1),
        )
        for bad in invalid:
            with self.assertRaises(ValueError):
                CachedNLLProtocol(**bad)

    def test_cached_targets_match_uncached_alignment_for_fp_model(self):
        protocol = CachedNLLProtocol(
            prefix_tokens=10, target_tokens=6, n_chunks=3, chat_prefix=False
        )
        result = evaluate_cached_nll(
            self.fp, self.stream, protocol, device="cpu", show_progress=False
        )
        self.assertEqual(result["scored_tokens"], 18)
        self.assertEqual(len(result["chunks"]), 3)
        self.assertAlmostEqual(result["ppl"], math.exp(result["mean_nll"]), places=9)
        total = 0.0
        for chunk, (begin, end) in zip(
            result["chunks"], chunk_token_ranges(96, protocol)
        ):
            ids = self.stream[begin:end]
            expected = _independent_target_nll(self.fp, ids, protocol.prefix_tokens)
            actual = torch.tensor(chunk["target_nll"], dtype=torch.float64)
            self.assertEqual(actual.numel(), 6)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            # Prefill-only scores the prefix tokens that have a predecessor.
            prefix_expected = _independent_target_nll(
                self.fp, ids[: protocol.prefix_tokens], 1
            )
            self.assertEqual(chunk["prefill_tokens"], 9)
            self.assertAlmostEqual(
                chunk["prefill_nll_sum"], float(prefix_expected.sum()), places=4
            )
            total += float(expected.sum())
        self.assertAlmostEqual(result["mean_nll"], total / 18, places=5)
        self.assertEqual(result["prefill_only"]["scored_tokens"], 27)

    def test_quantized_wrapper_scores_through_its_own_cache(self):
        qcfg = PTQConfig(
            activation=affine(DType.int(16)),
            weight=affine(DType.int(16)),
            model_args={"profile": "npu_export", "text": {"max_seq": 64}},
        )
        prepared = prepare(copy.deepcopy(self.fp), qcfg, inplace=True)
        prepared.enable_calibration()
        with torch.no_grad():
            ids = self.stream[:12].unsqueeze(0)
            out = prepared(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            next_ids = self.stream[12:13].unsqueeze(0)
            mask = torch.ones(1, 13, dtype=torch.long)
            prepared(
                input_ids=next_ids,
                attention_mask=mask,
                position_ids=torch.tensor([[12]]),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        quantized = convert(prepared, inplace=True).eval()
        protocol = CachedNLLProtocol(
            prefix_tokens=8, target_tokens=5, n_chunks=2, chat_prefix=False
        )
        result = evaluate_cached_nll(
            quantized, self.stream, protocol, device="cpu", show_progress=False
        )
        self.assertEqual(result["scored_tokens"], 10)
        for chunk, (begin, end) in zip(
            result["chunks"], chunk_token_ranges(96, protocol)
        ):
            ids = self.stream[begin:end]
            expected = _independent_target_nll(quantized, ids, protocol.prefix_tokens)
            actual = torch.tensor(chunk["target_nll"], dtype=torch.float64)
            # Cached and uncached fake-quant paths share grids; only the
            # cache storage/compute boundaries differ by design.
            self.assertTrue(torch.isfinite(actual).all())
            self.assertLess(float((actual - expected).abs().max()), 0.5)

    def test_chat_prefix_tokens_are_context_not_targets(self):
        tokenizer = _FakeTokenizer()
        words = " ".join(f"t{int(v)}" for v in self.stream[:40])
        dataset = [{"text": words}]
        protocol = CachedNLLProtocol(
            prefix_tokens=7, target_tokens=3, n_chunks=2, chat_prefix=True
        )
        with mock.patch(
            "tico.quantization.evaluation.vlm_eval_utils.get_dataset",
            return_value=(dataset, None),
        ):
            result = evaluate_text_cached_nll(
                model=self.fp,
                processor_or_tokenizer=SimpleNamespace(
                    tokenizer=tokenizer,
                    apply_chat_template=tokenizer.apply_chat_template,
                ),
                dataset_name="wikitext2",
                split="test",
                protocol=protocol,
                device="cpu",
                show_progress=False,
            )
        self.assertEqual(result["context_tokens"], 3)
        self.assertEqual(result["scored_tokens"], 6)
        context = torch.tensor([5, 6, 7])
        for chunk, (begin, end) in zip(
            result["chunks"], chunk_token_ranges(40, protocol)
        ):
            ids = torch.cat((context, self.stream[begin:end]))
            expected = _independent_target_nll(self.fp, ids, 3 + protocol.prefix_tokens)
            torch.testing.assert_close(
                torch.tensor(chunk["target_nll"], dtype=torch.float64),
                expected,
                rtol=1e-5,
                atol=1e-5,
            )
            # With a chat prefix every corpus prefix token has a predecessor.
            self.assertEqual(chunk["prefill_tokens"], 7)

    def test_missing_cache_is_an_explicit_error(self):
        class NoCache(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, **kwargs):
                kwargs["use_cache"] = False
                return self.inner(**kwargs)

        with self.assertRaisesRegex(RuntimeError, "requires past_key_values"):
            score_cached_chunk(
                NoCache(self.fp),
                context_ids=torch.zeros((0,), dtype=torch.long),
                prefix_ids=self.stream[:4],
                target_ids=self.stream[4:6],
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()

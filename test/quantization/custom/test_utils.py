# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import unittest
from types import SimpleNamespace
from typing import Optional

import torch
from tico.experimental.quantization.custom.utils import perplexity, reduce_except
from torch import nn


class TestReduceExcept(unittest.TestCase):
    def test_keep_dim0(self):
        # (C, N) : keep channel dim = 0
        x = torch.tensor([[1.0, 2.0, -3.0], [4.0, -5.0, 0.5]])

        mins, maxs = reduce_except(x, dim_to_keep=0)

        self.assertTrue(torch.equal(mins, torch.tensor([-3.0, -5.0])))
        self.assertTrue(torch.equal(maxs, torch.tensor([2.0, 4.0])))

    def test_keep_middle_dim(self):
        # Shape (B, C, L) – keep dim 1
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mins, maxs = reduce_except(x, 1)

        # Reference using PyTorch’s built-in reductions
        ref_min = x.amin(dim=(0, 2))
        ref_max = x.amax(dim=(0, 2))

        self.assertTrue(torch.equal(mins, ref_min))
        self.assertTrue(torch.equal(maxs, ref_max))

    def test_keep_negative_index(self):
        # Negative index should work the same as positive one
        x = torch.tensor([[7.0, -2.0, 3.0]])
        mins_pos, maxs_pos = reduce_except(x, 1)  # keep last dim
        mins_neg, maxs_neg = reduce_except(x, -1)  # same dim via -1

        self.assertTrue(
            torch.equal(mins_pos, mins_neg),
            f"mins_pos: {mins_pos}, mins_neg: {mins_neg}",
        )
        self.assertTrue(torch.equal(maxs_pos, maxs_neg))


"""
unittest suite for `perplexity`

This test checks three things:

1. **API sanity** — the function returns a Python float > 0.
2. **Short-sequence equivalence** — if the input length ≤ `max_length`,
   the sliding-window PPL must equal the single-pass PPL.
3. **Window/stride invariance** — for a short sequence (≤ `max_length`)
   changing the stride must **not** change the result.

A lightweight dummy causal-LM is used so the tests run quickly on CPU.
"""
# ────────────────────────────────────────────────────────────
#   Dummy causal language model
# ────────────────────────────────────────────────────────────
class DummyLM(nn.Module):
    """
    Minimal causal LM that supports the Hugging-Face style signature
    `forward(input_ids, labels=None) -> Namespace(loss, logits)`.
    If *labels* are supplied, it performs the internal 1-token shift
    before computing `CrossEntropyLoss` (ignore_index = -100).
    """

    def __init__(self, vocab_size: int, hidden_size: int, n_positions: int):
        super().__init__()
        self.config = SimpleNamespace(n_positions=n_positions, hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    # ---------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        """
        Parameters
        ----------
        input_ids : Tensor[B, T]
        labels    : Tensor[B, T] or None
            If provided, this method emulates HF CausalLM by shifting
            logits/labels internally before computing CE loss.
        """
        emb = self.embed(input_ids)  # (B, T, H)
        logits = self.fc(emb)  # (B, T, V)

        if labels is None:
            return SimpleNamespace(logits=logits)

        # --- internal 1-token shift (HF behaviour) -----------------
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # -----------------------------------------------------------

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return SimpleNamespace(loss=loss, logits=logits)


# ─────────────────────────────────────────────────────────────
# Unit-test class
# ─────────────────────────────────────────────────────────────
class TestPerplexitySlidingWindow(unittest.TestCase):
    """
    All tests run entirely on CPU and complete in <1 s.
    """

    VOCAB: int = 16
    HIDDEN: int = 8
    CONTEXT: int = 32  # max_length for DummyLM

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.device = torch.device("cpu")
        cls.model = (
            DummyLM(
                vocab_size=cls.VOCAB,
                hidden_size=cls.HIDDEN,
                n_positions=cls.CONTEXT,
            )
            .to(cls.device)
            .eval()
        )

    # ─────────────────────────────────────────────────────────
    # 1. API sanity
    # ─────────────────────────────────────────────────────────
    def test_returns_positive_float(self) -> None:
        seq = torch.randint(0, self.VOCAB, (1, 50), device=self.device)
        ppl = perplexity(
            self.model,
            seq,
            self.device,
            max_length=self.CONTEXT,
            stride=24,
            show_progress=False,
        )
        self.assertIsInstance(ppl, float)
        self.assertGreater(ppl, 0.0)

    # ─────────────────────────────────────────────────────────
    # 2. Short-sequence equivalence
    # ─────────────────────────────────────────────────────────
    def test_short_sequence_equivalence(self) -> None:
        seq_len = self.CONTEXT  # exactly fills one window
        seq = torch.randint(0, self.VOCAB, (1, seq_len), device=self.device)

        # ---- reference exact perplexity (manual shift) ----------
        with torch.no_grad():
            logits = self.model(seq).logits  # (1, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = seq[:, 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss()
        ref_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        ref_ppl = torch.exp(ref_loss).item()

        # ---- sliding-window perplexity at arbitrary stride ------
        test_ppl = perplexity(
            self.model,
            seq,
            self.device,
            max_length=self.CONTEXT,
            stride=8,
            show_progress=False,
        )

        self.assertAlmostEqual(
            ref_ppl,
            test_ppl,
            places=6,
            msg=f"ref={ref_ppl:.6f}, test={test_ppl:.6f}",
        )

    # ─────────────────────────────────────────────────────────
    # 3. Stride invariance on short sequences
    # ─────────────────────────────────────────────────────────
    def test_stride_invariance_short(self) -> None:
        seq = torch.randint(0, self.VOCAB, (1, self.CONTEXT // 2), device=self.device)

        ppls: list[float] = []
        for stride in (1, 4, 16):
            ppl = perplexity(
                self.model,
                seq,
                self.device,
                max_length=self.CONTEXT,
                stride=stride,
                show_progress=False,
            )
            ppls.append(float(ppl))

        spread = max(ppls) - min(ppls)
        self.assertLess(
            spread,
            1e-6,
            msg=f"PPLs differ by {spread}: {ppls}",
        )

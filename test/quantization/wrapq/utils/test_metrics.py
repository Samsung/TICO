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

import torch
import torch.nn.functional as F
from tico.quantization.wrapq.utils.metrics import (
    _CHAT_PREFIX_INSTRUCTION,
    perplexity,
    perplexity_chat_prefix,
)
from torch import nn

"""
unittest suite for `perplexity`

This test checks three things:

1. API sanity — the function returns a Python float > 0.
2. Short-sequence equivalence — if the input length ≤ `max_length`,
   the sliding-window PPL must equal the single-pass PPL.
3. Window/stride invariance — for a short sequence (≤ `max_length`)
   changing the stride must NOT change the result.

A lightweight dummy causal-LM is used so the tests run quickly on CPU.
"""


# ────────────────────────────────────────────────────────────
#   Dummy causal language model
# ────────────────────────────────────────────────────────────
class DummyLM(nn.Module):
    """
    Minimal causal LM that supports the Hugging-Face style signature
    `forward(input_ids, labels=None) -> Namespace(loss, logits)`.
    If labels are supplied, it performs the internal 1-token shift
    before computing `CrossEntropyLoss` (ignore_index = -100).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_positions: int,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            n_positions=n_positions, hidden_size=hidden_size, ignore_index=ignore_index
        )
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.logits_to_keep_history: list[int | torch.Tensor] = []

    # ---------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> SimpleNamespace:
        """
        Parameters
        ----------
        input_ids : Tensor[B, T]
        labels    : Tensor[B, T] or None
            If provided, this method emulates HF CausalLM by shifting
            logits/labels internally before computing CE loss.
        attention_mask, use_cache : ignored
            Accepted for compatibility with perplexity evaluation helpers.
        logits_to_keep : int or Tensor
            Select the hidden-state positions projected by the LM head,
            matching Gemma4's integer-suffix and tensor-index semantics.
        """
        hidden_states = self.embed(input_ids)  # (B, T, H)
        self.logits_to_keep_history.append(logits_to_keep)

        # Match Gemma4: slice hidden states before applying the LM head so
        # unnecessary vocabulary projections are not computed.
        if isinstance(logits_to_keep, int) and logits_to_keep:
            slice_indices = slice(-logits_to_keep, None)
        elif isinstance(logits_to_keep, torch.Tensor):
            slice_indices = logits_to_keep
        else:
            slice_indices = slice(None)

        logits = self.fc(hidden_states[:, slice_indices, :])

        if labels is None:
            return SimpleNamespace(logits=logits)

        # --- internal 1-token shift (HF behaviour) -----------------
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # -----------------------------------------------------------

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
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

    device: torch.device
    model: DummyLM

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

    def test_non_default_ignore_index(self):
        model = DummyLM(self.VOCAB, self.HIDDEN, self.CONTEXT, ignore_index=123)
        seq = torch.randint(0, self.VOCAB, (1, self.CONTEXT), device=self.device)
        ppl = perplexity(
            model,
            seq,
            self.device,
            max_length=self.CONTEXT,
            stride=8,
            show_progress=False,
            ignore_index=123,
        )
        self.assertIsInstance(ppl, float)


# ────────────────────────────────────────────────────────────
#   Dummy tokenizer with apply_chat_template
# ────────────────────────────────────────────────────────────
class DummyTokenizer:
    """
    Minimal tokenizer that supports `apply_chat_template` and basic
    `__call__` for perplexity_chat_prefix tests.

    Vocabulary: token id 0 = <pad>, 1..VOCAB-1 = real tokens.
    Each character in the text is mapped to (ord(c) % (VOCAB-1)) + 1.
    """

    def __init__(self, vocab_size: int = 16):
        self.vocab_size = vocab_size
        self.bos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Render a simple chat prefix: '<bos>user: {text}\\nmodel: '."""
        text = messages[0]["content"]
        prefix = f"<bos>user: {text}\nmodel: "
        if tokenize:
            # Return a dict mimicking tokenizer output
            ids = [self._char_to_id(c) for c in prefix]
            return {"input_ids": torch.tensor([ids])}
        return prefix

    def __call__(self, text=None, return_tensors="pt", add_special_tokens=False):
        """Tokenize text into input_ids tensor [1, seq_len]."""
        if text is None:
            raise ValueError("text is required")
        ids = [self._char_to_id(c) for c in text]
        return {"input_ids": torch.tensor([ids])}

    def _char_to_id(self, c: str) -> int:
        return (ord(c) % (self.vocab_size - 1)) + 1


# ─────────────────────────────────────────────────────────────
# Unit-test class for perplexity_chat_prefix
# ─────────────────────────────────────────────────────────────
class TestPerplexityChatPrefix(unittest.TestCase):
    """
    Tests for perplexity_chat_prefix.

    All tests run on CPU with a DummyLM and DummyTokenizer.
    """

    VOCAB: int = 16
    HIDDEN: int = 8
    MAX_SEQ_LEN: int = 256

    device: torch.device
    model: DummyLM
    tokenizer: DummyTokenizer

    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.device = torch.device("cpu")
        cls.model = (
            DummyLM(
                vocab_size=cls.VOCAB,
                hidden_size=cls.HIDDEN,
                n_positions=cls.MAX_SEQ_LEN,
            )
            .to(cls.device)
            .eval()
        )
        cls.tokenizer = DummyTokenizer(vocab_size=cls.VOCAB)

    def _make_dataset(self, text: str) -> list[dict]:
        return [{"text": text}]

    # ─────────────────────────────────────────────────────────
    # 1. API sanity — returns a positive float
    # ─────────────────────────────────────────────────────────
    def test_returns_positive_float(self) -> None:
        dataset = self._make_dataset("Hello world this is a test sentence.")
        ppl = perplexity_chat_prefix(
            self.model,
            self.tokenizer,
            dataset,
            stride=8,
            max_seq_len=self.MAX_SEQ_LEN,
            device=self.device,
            show_progress=False,
        )
        self.assertIsInstance(ppl, float)
        self.assertGreater(ppl, 0.0)

    # ─────────────────────────────────────────────────────────
    # 2. Empty dataset raises ValueError
    # ─────────────────────────────────────────────────────────
    def test_empty_dataset_raises(self) -> None:
        with self.assertRaises(ValueError):
            perplexity_chat_prefix(
                self.model,
                self.tokenizer,
                [],
                stride=8,
                max_seq_len=self.MAX_SEQ_LEN,
                device=self.device,
                show_progress=False,
            )

    # ─────────────────────────────────────────────────────────
    # 3. stride > window_size raises ValueError
    # ─────────────────────────────────────────────────────────
    def test_stride_too_large_raises(self) -> None:
        dataset = self._make_dataset("Some text here.")
        with self.assertRaises(ValueError):
            perplexity_chat_prefix(
                self.model,
                self.tokenizer,
                dataset,
                stride=1000,
                max_seq_len=self.MAX_SEQ_LEN,
                device=self.device,
                show_progress=False,
            )

    # ─────────────────────────────────────────────────────────
    # 4. max_seq_len too small for prefix raises ValueError
    # ─────────────────────────────────────────────────────────
    def test_max_seq_len_too_small_raises(self) -> None:
        dataset = self._make_dataset("Some text here.")
        with self.assertRaises(ValueError):
            perplexity_chat_prefix(
                self.model,
                self.tokenizer,
                dataset,
                stride=1,
                max_seq_len=2,  # too small for the chat prefix
                device=self.device,
                show_progress=False,
            )

    # ─────────────────────────────────────────────────────────
    # 5. Optimized suffix logits match a full-logits masked-CE
    #    reference across overlapping windows.
    # ─────────────────────────────────────────────────────────
    def test_overlapping_windows_match_full_logits_reference(self) -> None:
        """Optimized suffix logits should match full-logits masked CE."""
        # Fourteen character tokens with window_size=8 and stride=3 produce
        # three overlapping windows whose fresh target lengths are 8, 3, 3.
        text = "abcdefghijklmn"
        dataset = self._make_dataset(text)

        prefix_str = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": _CHAT_PREFIX_INSTRUCTION,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_enc = self.tokenizer(
            text=prefix_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prefix_ids = prefix_enc["input_ids"].to(self.device)

        content_enc = self.tokenizer(
            text=text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        content_ids = content_enc["input_ids"].to(self.device)

        prefix_len = prefix_ids.shape[1]
        window_size = 8
        stride = 3
        max_seq_len = prefix_len + window_size

        self.model.logits_to_keep_history.clear()

        optimized_ppl = perplexity_chat_prefix(
            self.model,
            self.tokenizer,
            dataset,
            stride=stride,
            max_seq_len=max_seq_len,
            device=self.device,
            show_progress=False,
        )

        # Each T-token target suffix requests T + 1 logits. The final logit
        # is discarded because it predicts the token following the window.
        self.assertEqual(
            self.model.logits_to_keep_history,
            [9, 4, 4],
        )

        # Build a full-logits reference using the original label-masking
        # formulation. This verifies both causal alignment and the rule that
        # every content token is scored exactly once.
        sequence_length = content_ids.shape[1]
        previous_end = 0
        total_nll = torch.zeros(
            (),
            dtype=torch.float64,
            device=self.device,
        )
        total_target_tokens = 0

        with torch.no_grad():
            for begin in range(0, sequence_length, stride):
                end = min(begin + window_size, sequence_length)
                target_length = end - previous_end
                if target_length <= 0:
                    break

                content_window_ids = content_ids[:, begin:end]
                model_input_ids = torch.cat(
                    [prefix_ids, content_window_ids],
                    dim=1,
                )

                labels = torch.full_like(model_input_ids, -100)
                labels[:, -target_length:] = model_input_ids[:, -target_length:]

                full_logits = self.model(
                    model_input_ids,
                    logits_to_keep=0,
                ).logits
                shift_logits = full_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                window_nll = F.cross_entropy(
                    shift_logits.float().reshape(
                        -1,
                        shift_logits.size(-1),
                    ),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                target_tokens = int((shift_labels != -100).sum().item())
                total_nll += window_nll.double()
                total_target_tokens += target_tokens

                previous_end = end
                if end == sequence_length:
                    break

        reference_ppl = torch.exp(total_nll / total_target_tokens).item()

        # The prefix provides context for the first corpus token but is never
        # itself scored. Therefore all corpus tokens contribute exactly once.
        self.assertEqual(total_target_tokens, sequence_length)

        self.assertAlmostEqual(
            optimized_ppl,
            reference_ppl,
            places=5,
            msg=(f"optimized={optimized_ppl:.6f}, " f"reference={reference_ppl:.6f}"),
        )

    # ─────────────────────────────────────────────────────────
    # 6. Overlapping windows score every content token exactly
    #    once, including the final partial window.
    # ─────────────────────────────────────────────────────────
    def test_overlapping_windows_score_each_token_once(self) -> None:
        """Every corpus token should be scored once across all windows."""
        text = "abcdefghijklmnop"  # 16 content tokens
        dataset = self._make_dataset(text)

        prefix_str = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": _CHAT_PREFIX_INSTRUCTION,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = self.tokenizer(
            text=prefix_str,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)
        content_ids = self.tokenizer(
            text=text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(self.device)

        prefix_len = prefix_ids.shape[1]
        content_len = content_ids.shape[1]

        # For 16 content tokens:
        #
        # window 0: [0, 8)   -> score [0, 8)   : 8 tokens
        # window 1: [3, 11)  -> score [8, 11)  : 3 tokens
        # window 2: [6, 14)  -> score [11, 14) : 3 tokens
        # window 3: [9, 16)  -> score [14, 16) : 2 tokens
        #
        # The final window has only seven content tokens and only two
        # fresh targets, so it also covers the final partial-window case.
        window_size = 8
        stride = 3
        max_seq_len = prefix_len + window_size

        calls: list[tuple[torch.Tensor, torch.Tensor, bool, int]] = []

        def record_call(_module, _args, kwargs):
            calls.append(
                (
                    kwargs["input_ids"].detach().clone(),
                    kwargs["attention_mask"].detach().clone(),
                    bool(kwargs["use_cache"]),
                    int(kwargs["logits_to_keep"]),
                )
            )

        hook = self.model.register_forward_pre_hook(
            record_call,
            with_kwargs=True,
        )
        try:
            actual_ppl = perplexity_chat_prefix(
                self.model,
                self.tokenizer,
                dataset,
                stride=stride,
                max_seq_len=max_seq_len,
                device=self.device,
                show_progress=False,
            )
        finally:
            hook.remove()

        expected_content_lengths = [8, 8, 8, 7]
        expected_target_lengths = [8, 3, 3, 2]
        expected_logits_to_keep = [9, 4, 4, 3]

        self.assertEqual(len(calls), len(expected_content_lengths))
        self.assertEqual(
            [input_ids.shape[1] - prefix_len for input_ids, _, _, _ in calls],
            expected_content_lengths,
        )
        self.assertEqual(
            [logits_to_keep for _, _, _, logits_to_keep in calls],
            expected_logits_to_keep,
        )
        self.assertTrue(all(not use_cache for _, _, use_cache, _ in calls))

        previous_end = 0
        reference_nll = torch.zeros(
            (),
            dtype=torch.float64,
            device=self.device,
        )
        reference_tokens = 0
        scored_positions: list[int] = []

        for call_index, begin in enumerate(range(0, content_len, stride)):
            end = min(begin + window_size, content_len)
            target_length = end - previous_end
            if target_length <= 0:
                break

            expected_window_ids = content_ids[:, begin:end]
            expected_input_ids = torch.cat(
                [prefix_ids, expected_window_ids],
                dim=1,
            )

            actual_input_ids, actual_attention_mask, _, _ = calls[call_index]

            self.assertTrue(
                torch.equal(actual_input_ids, expected_input_ids),
                msg=f"Unexpected input_ids in window {call_index}",
            )
            self.assertTrue(
                torch.equal(
                    actual_attention_mask,
                    torch.ones_like(expected_input_ids, dtype=torch.long),
                ),
                msg=f"Unexpected attention_mask in window {call_index}",
            )

            labels = torch.full_like(expected_input_ids, -100)
            labels[:, -target_length:] = expected_input_ids[:, -target_length:]

            with torch.no_grad():
                full_logits = self.model(
                    expected_input_ids,
                    logits_to_keep=0,
                ).logits

            shift_logits = full_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            reference_nll += F.cross_entropy(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            ).double()

            target_tokens = int((shift_labels != -100).sum().item())
            self.assertEqual(target_tokens, expected_target_lengths[call_index])
            reference_tokens += target_tokens
            scored_positions.extend(range(previous_end, end))

            previous_end = end
            if end == content_len:
                break

        # This checks both properties:
        #   1. no corpus position was skipped;
        #   2. no corpus position was scored more than once.
        self.assertEqual(scored_positions, list(range(content_len)))
        self.assertEqual(reference_tokens, content_len)

        reference_ppl = torch.exp(reference_nll / reference_tokens).item()

        self.assertAlmostEqual(
            actual_ppl,
            reference_ppl,
            places=5,
            msg=(f"actual={actual_ppl:.6f}, " f"reference={reference_ppl:.6f}"),
        )

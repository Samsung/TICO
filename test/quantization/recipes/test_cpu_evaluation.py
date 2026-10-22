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

"""Offline real-model scoring regressions; random-model PPL is not a quality score."""

import contextlib
import copy
import io
import math
import unittest
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import torch
import torch.nn.functional as F
from datasets import Dataset
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine

from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.cpu_fixtures import (
    fixture_context,
    gemma_model,
    llama_model,
)
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.evaluation.llm import evaluate_perplexity
from tico.quantization.recipes.evaluation.vlm import evaluate_vlm_text_ppl
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


def _tokenizer():
    """Build a genuine local tokenizer with exactly known token IDs."""
    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update({f"t{index}": index for index in range(2, 128)})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>"
    )


def _models(family):
    """Use standard PTQ while keeping an independent FP reference before wrapping."""
    context = fixture_context(family)
    reference = copy.deepcopy(context.model)
    stage = context.cfg["pipeline"][0]
    config = context.adapter.build_ptq_config(context, stage)
    no_quant = PTQWrapper(copy.deepcopy(reference), qcfg=config).eval()
    with contextlib.redirect_stdout(io.StringIO()):
        PTQStage().run(context, stage)
    return {"fp": reference, "no_quant": no_quant, "quant_w4a16": context.model}


def _independent_nll(model, encoding, *, window, stride):
    """Score explicit fresh token positions from real logits, without model loss."""
    ids = encoding.input_ids
    mask = encoding.attention_mask
    scored_positions = []
    nll = 0.0
    previous_end = 0
    with torch.no_grad():
        for start in range(0, ids.shape[1], stride):
            end = min(start + window, ids.shape[1])
            logits = model(
                ids[:, start:end],
                attention_mask=mask[:, start:end],
                use_cache=False,
                return_dict=True,
            ).logits
            for position in range(max(start + 1, previous_end), end):
                if mask[0, position] == 0 or mask[0, position - 1] == 0:
                    continue
                nll += float(
                    F.cross_entropy(
                        logits[:, position - start - 1, :].float(),
                        ids[:, position],
                        reduction="sum",
                    )
                )
                scored_positions.append(position)
            previous_end = end
            if end == ids.shape[1]:
                break
    return nll, scored_positions


class TestCPUPerplexity(unittest.TestCase):
    """Exercise actual HF logits, WrapQ, and both recipe PPL helper loops."""

    previous_threads: ClassVar[int]
    models: ClassVar[dict[str, torch.nn.Module]]
    tokenizer: ClassVar[PreTrainedTokenizerFast]
    dataset: ClassVar[Dataset]

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.models = _models("llama")
        cls.tokenizer = _tokenizer()
        cls.dataset = Dataset.from_dict(
            {"text": ["t80 t81 t82 t83 t84 t85 t86", "t87 t88 t89 t90 t91 t92 t93"]}
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_actual_recipe_helpers_match_independent_fresh_token_nll(self):
        encoding = self.tokenizer(
            "\n\n".join(self.dataset["text"]), return_tensors="pt"
        )
        self.assertEqual(encoding.input_ids.tolist(), [list(range(80, 94))])
        frozen_before = _state_contract(self.models["quant_w4a16"])
        results = {}
        for name, model in self.models.items():
            with self.subTest(model=name):
                nll, positions = _independent_nll(model, encoding, window=8, stride=3)
                self.assertEqual(positions, list(range(1, 14)))
                reference = math.exp(nll / len(positions))
                with patch("datasets.load_dataset", return_value=self.dataset):
                    llm = evaluate_perplexity(
                        model=model,
                        tokenizer=self.tokenizer,
                        device="cpu",
                        cache_dir=None,
                        max_seq_len=8,
                        stride=3,
                    )
                with patch(
                    "tico.quantization.recipes.evaluation.vlm.get_dataset",
                    return_value=(self.dataset, None),
                ):
                    vlm = evaluate_vlm_text_ppl(
                        model=model,
                        processor=SimpleNamespace(tokenizer=self.tokenizer),
                        dataset_name="wikitext2",
                        split="test",
                        device="cpu",
                        stride=3,
                        max_seq_len=8,
                        show_progress=False,
                    )
                self.assertTrue(math.isfinite(llm))
                self.assertAlmostEqual(math.log(llm), math.log(reference), places=6)
                self.assertEqual(llm, vlm)
                results[name] = llm
        self.assertAlmostEqual(results["fp"], results["no_quant"], places=4)
        self.assertEqual(frozen_before, _state_contract(self.models["quant_w4a16"]))
        print(f"CPU_PPL real HF/WrapQ W4A16, window=8 stride=3 tokens=13: {results}")

    def test_padding_and_ignored_overlap_labels_match_real_logit_nll(self):
        frozen_before = _state_contract(self.models["quant_w4a16"])
        for side in ("left", "right"):
            self.tokenizer.padding_side = side
            encoding = self.tokenizer(
                "t80 t81 t82 t83 t84 t85 t86 t87 t88 t89 t90 t91 t92 t93",
                return_tensors="pt",
                padding="max_length",
                max_length=18,
            )
            for name, model in self.models.items():
                with self.subTest(side=side, model=name):
                    nll, positions = _independent_nll(
                        model, encoding, window=8, stride=3
                    )
                    expected_positions = (
                        list(range(5, 18)) if side == "left" else list(range(1, 14))
                    )
                    self.assertEqual(positions, expected_positions)
                    result = perplexity(
                        model,
                        encoding,
                        "cpu",
                        max_length=8,
                        stride=3,
                        show_progress=False,
                    )
                    self.assertAlmostEqual(
                        math.log(result), nll / len(positions), places=6
                    )
        self.assertEqual(frozen_before, _state_contract(self.models["quant_w4a16"]))

    def test_no_valid_next_token_is_an_explicit_error(self):
        for text, length in (("", 4), ("t2", 4)):
            encoding = self.tokenizer(
                text, return_tensors="pt", padding="max_length", max_length=length
            )
            with self.subTest(text=text), self.assertRaisesRegex(
                ValueError, "No target tokens"
            ):
                perplexity(
                    self.models["fp"],
                    encoding,
                    "cpu",
                    max_length=8,
                    stride=3,
                    show_progress=False,
                )

    def test_qwen_actual_text_scoring_preserves_frozen_state(self):
        models = _models("qwen3_vl")
        encoding = self.tokenizer(
            "\n\n".join(self.dataset["text"]), return_tensors="pt"
        )
        frozen_before = _state_contract(models["quant_w4a16"])
        values = {}
        for name, model in models.items():
            with self.subTest(model=name):
                nll, positions = _independent_nll(model, encoding, window=8, stride=3)
                self.assertEqual(positions, list(range(1, 14)))
                with patch(
                    "tico.quantization.recipes.evaluation.vlm.get_dataset",
                    return_value=(self.dataset, None),
                ):
                    value = evaluate_vlm_text_ppl(
                        model=model,
                        processor=SimpleNamespace(tokenizer=self.tokenizer),
                        dataset_name="wikitext2",
                        split="test",
                        device="cpu",
                        stride=3,
                        max_seq_len=8,
                        show_progress=False,
                    )
                self.assertAlmostEqual(math.log(value), nll / len(positions), places=6)
                values[name] = value
        self.assertAlmostEqual(values["fp"], values["no_quant"], places=4)
        self.assertEqual(frozen_before, _state_contract(models["quant_w4a16"]))
        print(
            f"CPU_PPL Qwen real HF/WrapQ W4A16, window=8 stride=3 tokens=13: {values}"
        )

    def test_gemma_bounded_text_smoke_crosses_sliding_window(self):
        reference = gemma_model()
        parameter_count = sum(parameter.numel() for parameter in reference.parameters())
        self.assertLess(parameter_count, 200_000)
        config = PTQConfig(
            activation=affine(DType.int(16)), weight=affine(DType.int(4))
        )
        no_quant = PTQWrapper(copy.deepcopy(reference), qcfg=config).eval()
        quant = PTQWrapper(copy.deepcopy(reference), qcfg=config).eval()
        quant.enable_calibration()
        with torch.no_grad():
            quant(torch.arange(5, 13).unsqueeze(0), use_cache=False)
        quant.freeze_qparams()
        frozen_before = _state_contract(quant)
        encoding = self.tokenizer(
            "\n\n".join(self.dataset["text"]), return_tensors="pt"
        )
        values = {}
        for name, model in {
            "fp": reference,
            "no_quant": no_quant,
            "quant_w4a16": quant,
        }.items():
            with self.subTest(model=name):
                nll, positions = _independent_nll(model, encoding, window=12, stride=4)
                self.assertEqual(positions, list(range(1, 14)))
                value = perplexity(
                    model, encoding, "cpu", max_length=12, stride=4, show_progress=False
                )
                self.assertAlmostEqual(math.log(value), nll / len(positions), places=6)
                values[name] = value
        self.assertAlmostEqual(values["fp"], values["no_quant"], places=4)
        self.assertEqual(frozen_before, _state_contract(quant))
        print(
            "CPU_PPL Gemma dense text-only, PLE/shared-KV disabled, "
            f"parameters={parameter_count} float32_bytes={parameter_count * 4}, "
            f"sliding_window=8 window=12 stride=4 tokens=13: {values}"
        )


class TestAdapterPPLPlumbing(unittest.TestCase):
    """UNIT_MOCK: isolate adapter argument routing; real scoring is tested above."""

    def test_all_adapters_keep_context_window_and_cache_independent(self):
        for adapter in (LlamaAdapter(), Qwen3VLAdapter(), Gemma4Adapter()):
            for cache in (False, True):
                with self.subTest(family=adapter.family, use_cache=cache):
                    model = llama_model()
                    tokenizer = _tokenizer()
                    task_key = "perplexity" if adapter.family == "llama" else "ppl"
                    cfg: dict[str, Any] = {
                        "model": {"family": adapter.family},
                        "runtime": {"device": "cpu", "show_progress": False},
                        "calibration": {"seq_len": 4},
                        "export": {"max_seq_len": 16},
                        "evaluation": {
                            "enabled": True,
                            "selected_tasks": ["ppl"],
                            "max_seq_len": 48,
                            "use_cache": cache,
                            task_key: {"max_seq_len": 8, "stride": 3},
                        },
                    }
                    ctx = RecipeContext(
                        cfg=cfg,
                        adapter=adapter,
                        model=model,
                        tokenizer=tokenizer,
                        processor=SimpleNamespace(tokenizer=tokenizer),
                    )
                    target = (
                        "evaluate_perplexity"
                        if adapter.family == "llama"
                        else "evaluate_vlm_text_ppl"
                    )
                    with patch(
                        f"tico.quantization.recipes.adapters.{adapter.family}.{target}",
                        return_value=1.0,
                    ) as helper, contextlib.redirect_stdout(io.StringIO()):
                        adapter.evaluate(ctx)
                    self.assertEqual(helper.call_args.kwargs["max_seq_len"], 8)
                    self.assertEqual(helper.call_args.kwargs["stride"], 3)
                    self.assertEqual(model.config.use_cache, cache)
                    self.assertEqual(model.generation_config.use_cache, cache)
                    self.assertEqual(
                        ctx.artifacts["evaluation_protocol"]["context"]["resolved"], 48
                    )
                    self.assertEqual(cfg["calibration"]["seq_len"], 4)
                    self.assertEqual(cfg["export"]["max_seq_len"], 16)


if __name__ == "__main__":
    unittest.main()

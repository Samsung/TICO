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

"""Actual download-free HF generation through frozen CPU PTQ wrappers."""

import copy
import json
import unittest
from typing import ClassVar

import torch

from tico.quantization.recipes.debug.cpu_fixtures import (
    fixture_context,
    qwen_batch,
    qwen_model,
)
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm import (
    QuantLlamaForCausalLM,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import (
    QuantQwen3VLForConditionalGeneration,
)


def frozen_snapshot(model):
    """Record persistent weights, observer buffers, and wrapper execution policy."""
    tensors = _state_contract(model)
    policy = {
        name: tuple(
            (attr, str(getattr(module, attr)))
            for attr in (
                "_mode",
                "force_export",
                "return_type",
                "cache_output_mode",
                "attn_options",
                "profile",
            )
            if hasattr(module, attr)
        )
        for name, module in model.named_modules()
    }
    return tensors, {
        "modules": policy,
        "config": copy.deepcopy(model.config.to_dict()),
        "generation_config": copy.deepcopy(model.generation_config.to_dict()),
    }


def manual_greedy(model, batch, *, use_cache, steps=4, qwen=False):
    """Run independent forward/argmax steps with HF's last-logit projection policy."""
    batch = copy.deepcopy(batch)
    ids = batch.pop("input_ids")
    mask = batch.pop("attention_mask")
    mm_types = batch.pop("mm_token_type_ids", None)
    source = getattr(model, "module", model)
    if qwen and "pixel_values" in batch:
        positions, _ = source.model.get_rope_index(
            ids,
            attention_mask=mask,
            mm_token_type_ids=mm_types,
            image_grid_thw=batch["image_grid_thw"],
        )
    else:
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
        if qwen:
            positions = positions.unsqueeze(0).expand(3, -1, -1)
    past = None
    scores = []
    for step in range(steps):
        current = ids[:, -1:] if use_cache and step else ids
        current_positions = positions[..., -1:] if use_cache and step else positions
        visual = batch if not (use_cache and step) else {}
        # GenerationMixin uses one projected row; keep the forward contract identical.
        output = model(
            input_ids=current,
            attention_mask=mask,
            position_ids=current_positions,
            past_key_values=past,
            use_cache=use_cache,
            return_dict=True,
            logits_to_keep=1,
            **visual,
        )
        logits = output.logits[:, -1].detach()
        scores.append(logits)
        ids = torch.cat((ids, logits.argmax(-1, keepdim=True)), dim=-1)
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=-1)
        positions = torch.cat((positions, positions[..., -1:] + 1), dim=-1)
        past = output.past_key_values if use_cache else None
    return ids, scores


class TestCPUGeneration(unittest.TestCase):
    previous_threads: ClassVar[int]

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _models(self, family, *, wide_image=False):
        ctx = fixture_context(family)
        if wide_image:
            ctx.model = qwen_model(head_dim=128)
            ctx.cfg["model_args"]["vision"]["grid_thw"] = [1, 2, 4]
            ctx.calibration_inputs = [
                qwen_batch(image=False),
                qwen_batch(grid_thw=(1, 2, 4)),
            ]
        reference = copy.deepcopy(ctx.model).eval()
        wrapper_cls = (
            QuantLlamaForCausalLM
            if family == "llama"
            else QuantQwen3VLForConditionalGeneration
        )
        no_quant = wrapper_cls(
            copy.deepcopy(reference),
            qcfg=ctx.adapter.build_ptq_config(ctx, ctx.cfg["pipeline"][0]),
        ).eval()
        if family == "llama":
            ctx.cfg["pipeline"][0]["decode_calibration_steps"] = 2
        PTQStage().run(ctx, ctx.cfg["pipeline"][0])
        quant = ctx.model.wrapped
        self.assertIs(quant._mode, Mode.QUANT)
        return reference, no_quant, quant

    def _exercise(self, family, *, image=False, wide_image=False):
        reference, no_quant, quant = self._models(family, wide_image=wide_image)
        before = frozen_snapshot(quant)
        qwen = family == "qwen3_vl"
        batch = qwen_batch(image=image, grid_thw=(1, 2, 4) if wide_image else (1, 2, 2))
        if wide_image:
            positions, _ = reference.model.get_rope_index(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image_grid_thw=batch["image_grid_thw"],
                mm_token_type_ids=batch["mm_token_type_ids"],
            )
            self.assertFalse(torch.equal(positions[0], positions[2]))
        if qwen:
            batch["mm_token_type_ids"] = (batch["input_ids"] == 121).long()
        conditions = (
            ((False, True),)
            if wide_image
            else ((False, True), (True, True), (False, False), (True, False))
        )
        for padded, use_cache in conditions:
            with self.subTest(
                family=family, image=image, padded=padded, use_cache=use_cache
            ):
                sample = copy.deepcopy(batch)
                if padded:
                    sample["input_ids"] = torch.cat(
                        (torch.zeros(1, 2, dtype=torch.long), sample["input_ids"]),
                        dim=-1,
                    )
                    sample["attention_mask"] = torch.cat(
                        (torch.zeros(1, 2, dtype=torch.long), sample["attention_mask"]),
                        dim=-1,
                    )
                    if qwen:
                        sample["mm_token_type_ids"] = torch.cat(
                            (
                                torch.zeros(1, 2, dtype=torch.long),
                                sample["mm_token_type_ids"],
                            ),
                            dim=-1,
                        )
                settings = dict(
                    do_sample=False,
                    num_beams=1,
                    min_new_tokens=4,
                    max_new_tokens=4,
                    eos_token_id=None,
                    pad_token_id=0,
                    use_cache=use_cache,
                    return_dict_in_generate=True,
                    output_scores=True,
                    logits_to_keep=1,
                )
                with torch.no_grad():
                    fp_output = reference.generate(**copy.deepcopy(sample), **settings)
                    no_quant_output = no_quant.generate(
                        **copy.deepcopy(sample), **settings
                    )
                    self.assertTrue(
                        torch.equal(fp_output.sequences, no_quant_output.sequences)
                    )
                    self.assertEqual(len(fp_output.scores), 4)
                    self.assertEqual(len(no_quant_output.scores), 4)
                    for expected, actual in zip(
                        fp_output.scores, no_quant_output.scores
                    ):
                        torch.testing.assert_close(
                            actual, expected, atol=1e-5, rtol=1e-4
                        )
                    visits = []
                    vision_visits = []
                    leaf_visits = []
                    leaf_input_lengths = []

                    def record_leaf(module, args):
                        leaf_visits.append(module.wrapped._mode)
                        leaf_input_lengths.append(int(args[0].shape[-2]))

                    leaf_handle = quant.lm_head.register_forward_pre_hook(record_leaf)
                    handle = quant.model.register_forward_pre_hook(
                        lambda _module, _args, kwargs: visits.append(
                            int(kwargs["input_ids"].shape[-1])
                        ),
                        with_kwargs=True,
                    )
                    vision_handle = None
                    if qwen:
                        vision_handle = (
                            quant.model.wrapped.visual.register_forward_pre_hook(
                                lambda *_: vision_visits.append(True)
                            )
                        )
                    try:
                        output = quant.generate(**copy.deepcopy(sample), **settings)
                    finally:
                        handle.remove()
                        leaf_handle.remove()
                        if vision_handle is not None:
                            vision_handle.remove()
                    self.assertEqual(len(output.scores), 4)
                    self.assertEqual(leaf_visits, [Mode.QUANT] * 4)
                    self.assertEqual(leaf_input_lengths, [1] * 4)
                    self.assertEqual(
                        visits,
                        [sample["input_ids"].shape[-1], 1, 1, 1]
                        if use_cache
                        else list(
                            range(
                                sample["input_ids"].shape[-1],
                                sample["input_ids"].shape[-1] + 4,
                            )
                        ),
                    )
                    if image:
                        self.assertEqual(len(vision_visits), 1 if use_cache else 4)
                    manual_head_lengths = []
                    manual_head_handle = quant.lm_head.register_forward_pre_hook(
                        lambda _module, args: manual_head_lengths.append(
                            int(args[0].shape[-2])
                        )
                    )
                    try:
                        manual_ids, manual_scores = manual_greedy(
                            quant, sample, use_cache=use_cache, qwen=qwen
                        )
                    finally:
                        manual_head_handle.remove()
                    self.assertEqual(manual_head_lengths, [1] * 4)
                    self.assertTrue(torch.equal(output.sequences, manual_ids))
                    for generated, manual in zip(output.scores, manual_scores):
                        torch.testing.assert_close(
                            generated, manual, atol=1e-5, rtol=1e-4
                        )
                    self.assertTrue(
                        all(torch.isfinite(score).all() for score in output.scores)
                    )
                    print(
                        json.dumps(
                            {
                                "event": "actual_generation",
                                "family": family,
                                "image": image,
                                "head_dim": reference.config.text_config.head_dim
                                if qwen
                                else reference.config.head_dim,
                                "parameter_count": sum(
                                    parameter.numel()
                                    for parameter in reference.parameters()
                                ),
                                "padded": padded,
                                "use_cache": use_cache,
                                "input_ids": sample["input_ids"].tolist(),
                                "quant_output_ids": output.sequences.tolist(),
                                "prefill_decode_lengths": visits,
                                "quant_leaf_modes": [mode.name for mode in leaf_visits],
                                "lm_head_input_lengths": leaf_input_lengths,
                                "logits_to_keep": 1,
                                "vision_calls": len(vision_visits),
                                "manual_max_abs_logit_error": max(
                                    float((generated - manual).abs().max())
                                    for generated, manual in zip(
                                        output.scores, manual_scores
                                    )
                                ),
                                "fp_no_quant_max_abs_logit_error": max(
                                    float((expected - actual).abs().max())
                                    for expected, actual in zip(
                                        fp_output.scores, no_quant_output.scores
                                    )
                                ),
                            }
                        ),
                        flush=True,
                    )
        if wide_image:
            fresh_text = qwen_batch(image=False)
            fresh_text["mm_token_type_ids"] = torch.zeros_like(fresh_text["input_ids"])
            with torch.no_grad():
                expected_ids, expected_scores = manual_greedy(
                    quant, fresh_text, use_cache=True, qwen=True
                )
                text_output = quant.generate(**fresh_text, **settings)
                repeated_image = quant.generate(**copy.deepcopy(batch), **settings)
            self.assertTrue(torch.equal(text_output.sequences, expected_ids))
            self.assertTrue(torch.equal(repeated_image.sequences, output.sequences))
            for actual, expected in zip(text_output.scores, expected_scores):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
            print(
                json.dumps(
                    {
                        "event": "generation_request_isolation",
                        "family": family,
                        "requests": ["wide_image", "text_only", "wide_image"],
                        "status": "PASS",
                    }
                ),
                flush=True,
            )
        after = frozen_snapshot(quant)
        self.assertEqual(before[1], after[1])
        self.assertEqual(before[0], after[0])
        self.assertTrue(
            all(
                module._mode is Mode.QUANT
                for module in quant.modules()
                if isinstance(module, QuantModuleBase)
            )
        )

    def test_llama_generation(self):
        self._exercise("llama")

    def test_qwen_text_generation(self):
        self._exercise("qwen3_vl")

    def test_qwen_image_generation(self):
        self._exercise("qwen3_vl", image=True)

    def test_qwen_head128_non_square_image_mrope(self):
        self._exercise("qwen3_vl", image=True, wide_image=True)

    def test_experts_setter_matches_dense_hf_contract(self):
        for family in ("llama", "qwen3_vl"):
            reference, no_quant, _ = self._models(family)
            with self.subTest(family=family):
                for requested in (
                    "eager",
                    {"": "eager"},
                    reference.get_experts_implementation(),
                ):
                    self.assertEqual(
                        reference.set_experts_implementation(requested),
                        no_quant.set_experts_implementation(requested),
                    )
                    self.assertEqual(
                        reference.get_experts_implementation(),
                        no_quant.get_experts_implementation(),
                    )
                for requested in ("unsupported-for-test", "grouped_mm"):
                    with self.assertRaises(ValueError):
                        reference.set_experts_implementation(requested)
                    with self.assertRaises(ValueError):
                        no_quant.set_experts_implementation(requested)
                before = len(list(no_quant.parameters()))
                no_quant.set_experts_implementation(
                    no_quant.get_experts_implementation()
                )
                self.assertEqual(before, len(list(no_quant.parameters())))

    def test_llama_legacy_checkpoint_metadata_and_generation_inputs(self):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_utils import PreTrainedModel

        reference, model, _ = self._models("llama")
        self.assertFalse(
            any(isinstance(module, PreTrainedModel) for module in model.modules())
        )
        del model._generation_model_class
        self.assertEqual(
            model.get_experts_implementation(), reference.get_experts_implementation()
        )
        ids = torch.tensor([[5, 6, 7, 8, 9]])
        past = ((torch.zeros(1, 1, 3, 32), torch.zeros(1, 1, 3, 32)),) * 2
        prepared = model.prepare_inputs_for_generation(
            ids,
            2,
            past,
            torch.ones_like(ids),
            position_ids=torch.arange(5).unsqueeze(0),
            cache_position=torch.arange(5),
            logits_to_keep=1,
        )
        self.assertTrue(torch.equal(prepared["input_ids"], ids[:, -2:]))
        self.assertTrue(torch.equal(prepared["position_ids"], torch.tensor([[3, 4]])))
        self.assertTrue(torch.equal(prepared["cache_position"], torch.tensor([3, 4])))
        self.assertEqual(prepared["logits_to_keep"], 1)
        embeddings = reference.get_input_embeddings()(ids)
        prepared = model.prepare_inputs_for_generation(
            torch.empty(1, 0, dtype=torch.long),
            5,
            DynamicCache(config=reference.config),
            torch.ones_like(ids),
            embeddings,
            True,
        )
        self.assertTrue(torch.equal(prepared["inputs_embeds"], embeddings))
        self.assertEqual(prepared["cache_position"].numel(), 5)
        with torch.no_grad():
            settings = dict(
                do_sample=False,
                max_new_tokens=4,
                min_new_tokens=4,
                eos_token_id=None,
                pad_token_id=0,
            )
            expected = reference.generate(
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(ids),
                **settings,
            )
            actual = model.generate(
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(ids),
                **settings,
            )
        self.assertTrue(torch.equal(expected, actual))

    def test_eos_stops_actual_generation(self):
        _, _, model = self._models("llama")
        batch = qwen_batch(image=False)
        with torch.no_grad():
            first = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=1,
                eos_token_id=None,
                pad_token_id=0,
            )
            stop = int(first[0, -1])
            stopped = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=5,
                eos_token_id=stop,
                pad_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
        self.assertEqual(len(stopped.scores), 1)
        self.assertTrue(torch.equal(stopped.sequences, first))


if __name__ == "__main__":
    unittest.main()

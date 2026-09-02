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

"""Slow integration checks against real Gemma4 assistant checkpoints.

These tests never download models. They run only when the corresponding
environment variables point at local checkpoints:

    GEMMA4_ASSISTANT_PATH  — Gemma4 assistant checkpoint (draft model)
    GEMMA4_TARGET_PATH     — Gemma4 target checkpoint (verification model)
"""

import copy
import os
import unittest

import torch


ASSISTANT_PATH = os.environ.get("GEMMA4_ASSISTANT_PATH")
TARGET_PATH = os.environ.get("GEMMA4_TARGET_PATH")

_ASSISTANT_SKIP_MSG = (
    "set GEMMA4_ASSISTANT_PATH to a local Gemma4 assistant checkpoint to run "
    "this integration test"
)
_TARGET_SKIP_MSG = (
    "set GEMMA4_TARGET_PATH and GEMMA4_ASSISTANT_PATH to local Gemma4 "
    "checkpoints to run this integration test"
)


def _load_assistant():
    from transformers import AutoModelForCausalLM

    return (
        AutoModelForCausalLM.from_pretrained(ASSISTANT_PATH, dtype=torch.float32)
        .cpu()
        .eval()
    )


def _make_real_geometry_sample(model, kv_len: int = 32) -> dict:
    from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
        assistant_layer_type_head_dim,
        assistant_shared_kv_num_heads,
        extract_assistant_text_config,
    )

    text_cfg = extract_assistant_text_config(model.config)
    kv_heads = assistant_shared_kv_num_heads(text_cfg)
    return {
        "inputs_embeds": torch.randn(1, 1, 2 * int(model.config.backbone_hidden_size)),
        "position_ids": torch.tensor([[kv_len - 1]]),
        "attention_mask": torch.ones(1, kv_len, dtype=torch.long),
        "shared_kv_states": {
            layer_type: (
                torch.randn(
                    1,
                    kv_heads,
                    kv_len,
                    assistant_layer_type_head_dim(text_cfg, layer_type),
                ),
                torch.randn(
                    1,
                    kv_heads,
                    kv_len,
                    assistant_layer_type_head_dim(text_cfg, layer_type),
                ),
            )
            for layer_type in ("full_attention", "sliding_attention")
        },
        "use_cache": False,
    }


@unittest.skipUnless(ASSISTANT_PATH, _ASSISTANT_SKIP_MSG)
class TestRealGemma4AssistantCheckpoint(unittest.TestCase):
    """FP parity and PTQ flow against the real assistant checkpoint."""

    def test_fp_wrapper_parity_and_ptq_flow(self):
        from tico.quantization import convert, prepare
        from tico.quantization.config.gemma4_assistant_builders import (
            build_gemma4_assistant_ptq_config,
        )
        from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
            validate_gemma4_assistant_architecture,
        )

        torch.manual_seed(42)
        model = _load_assistant()
        validate_gemma4_assistant_architecture(model)
        reference = copy.deepcopy(model)
        sample = _make_real_geometry_sample(model)

        with torch.no_grad():
            ref = reference(**sample)

        text_config = model.config.get_text_config()
        prepared = prepare(
            model,
            build_gemma4_assistant_ptq_config(
                num_hidden_layers=int(text_config.num_hidden_layers),
                model_args={
                    "assistant": {"full_kv_length": 64, "sliding_kv_length": 64}
                },
            ),
        )
        with torch.no_grad():
            out = prepared(**sample)

        torch.testing.assert_close(
            out.last_hidden_state, ref.last_hidden_state, atol=1e-4, rtol=1e-4
        )
        self.assertEqual(
            int(out.logits.argmax(-1).item()), int(ref.logits.argmax(-1).item())
        )

        with torch.no_grad():
            prepared(**_make_real_geometry_sample(reference, kv_len=48))
        converted = convert(prepared)
        with torch.no_grad():
            quant_out = converted(**sample)
        self.assertTrue(torch.isfinite(quant_out.logits).all())

        missing = [
            name
            for name, obs in converted.wrapped.named_observers()
            if hasattr(obs, "has_qparams") and not obs.has_qparams
        ]
        self.assertEqual(missing, [])


@unittest.skipUnless(ASSISTANT_PATH and TARGET_PATH, _TARGET_SKIP_MSG)
class TestRealGemma4AssistedGeneration(unittest.TestCase):
    """Target + assistant greedy assisted decoding on a small prompt set."""

    def test_greedy_assisted_generation_matches_target_only(self):
        from tico.quantization import convert, prepare
        from tico.quantization.config.gemma4_assistant_builders import (
            build_gemma4_assistant_ptq_config,
        )
        from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
            Gemma4AssistantGenerationAdapter,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
        # Greedy assisted decoding reproduces the target-only sequence only
        # when target verification is numerically stable. bfloat16 chunked
        # verification (K+1 tokens per forward) can flip near-tie argmax
        # decisions even with the unmodified HF FP assistant, so the exact
        # sequence-equality contract is checked with a float32 target.
        target = (
            AutoModelForCausalLM.from_pretrained(TARGET_PATH, dtype=torch.float32)
            .to(device)
            .eval()
        )
        assistant = _load_assistant()

        text_config = assistant.config.get_text_config()
        prepared = prepare(
            assistant,
            build_gemma4_assistant_ptq_config(
                num_hidden_layers=int(text_config.num_hidden_layers),
                model_args={
                    "assistant": {
                        "full_kv_length": 512,
                        "sliding_kv_length": 640,
                    }
                },
            ),
        )
        # HF assisted generation expects the assistant on the same device as
        # the target model.
        prepared = prepared.to(device)
        generation_adapter = Gemma4AssistantGenerationAdapter(prepared)
        generation_adapter.generation_config.num_assistant_tokens = 4
        generation_adapter.generation_config.num_assistant_tokens_schedule = "constant"

        prompt = tokenizer("The quick brown fox", return_tensors="pt").to(device)
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        with torch.no_grad():
            baseline = target.generate(
                **prompt,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            # Calibration pass: real assisted generation with the prepared
            # assistant collecting observer statistics.
            target.generate(
                **prompt,
                assistant_model=generation_adapter,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        converted = convert(prepared)
        quant_adapter = Gemma4AssistantGenerationAdapter(converted)
        quant_adapter.generation_config.num_assistant_tokens = 4
        quant_adapter.generation_config.num_assistant_tokens_schedule = "constant"
        with torch.no_grad():
            assisted = target.generate(
                **prompt,
                assistant_model=quant_adapter,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        # Greedy assisted decoding must reproduce the target-only sequence.
        self.assertTrue(torch.equal(baseline, assisted))


if __name__ == "__main__":
    unittest.main()

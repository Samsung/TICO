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

import argparse
from typing import List, Tuple

import torch

from lm_eval.utils import make_table

from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.examples.static_llama_layer_runtime import (
    _build_rope_templates_from_config,
    LayerCache,
)


def pad_input(input, pad_token, max_seq_len, pads_on_left=True):
    """Pad a tensor to a maximum sequence length using the specified pad token."""
    pads = torch.full(
        (input.shape[0], max_seq_len - input.shape[1]),
        fill_value=pad_token,
        device=input.device,
    )

    if pads_on_left is True:
        res = torch.cat((pads, input), dim=1)
    else:
        res = torch.cat((input, pads), dim=1)

    return res


class PrefillDecodeUtils:
    def __init__(self, max_seq_len, config, device):
        self.max_seq_len = max_seq_len
        self.device = device
        self.pos_embeds = _build_rope_templates_from_config(
            config, max_seq=max_seq_len, device=device, dtype=torch.float32
        )

    def build_attention_mask_for_padded_input(
        self, pad_mask: torch.Tensor, prefill: bool, mask_value: float = -120.0
    ):
        dtype = torch.float32

        max_seq_len = self.max_seq_len - 1 if prefill is True else self.max_seq_len
        assert pad_mask is not None

        causal_mask = torch.full(
            (1, max_seq_len, max_seq_len), 1, device=self.device, dtype=dtype
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask == 0  # negating
        res_mask = torch.logical_and(pad_mask, causal_mask)
        mask_res = torch.zeros(
            (1, max_seq_len, max_seq_len), device=self.device, dtype=dtype
        )
        mask_res = mask_res.masked_fill(~res_mask, mask_value)
        return mask_res

    def build_position_embeddings_for_padded_input(
        self, position_embeddings, position_ids
    ):
        sl_cos, sl_sin = position_embeddings
        sl_cos = sl_cos[..., :-1, :]
        sl_sin = sl_sin[..., :-1, :]
        cos = sl_cos[..., position_ids.to(sl_cos.device).squeeze(0), :]
        sin = sl_sin[..., position_ids.to(sl_sin.device).squeeze(0), :]

        return (cos, sin)

    def get_input_for_decode_model(self, input, past_key_values, pad_mask, cur_seq_len):
        next_token = torch.tensor([[input[..., -1]]], device=input.device)

        attention_mask = self.build_attention_mask_for_padded_input(
            pad_mask, prefill=False
        )
        attention_mask = attention_mask[..., -1, :].unsqueeze(0)  # last row
        position_embeddings = self.build_position_embeddings_for_padded_input(
            self.pos_embeds, position_ids=torch.tensor([[cur_seq_len - 1]])
        )

        # fill in input
        inputs = {}
        inputs["input_ids"] = next_token
        inputs["attention_mask"] = attention_mask
        inputs["position_embeddings"] = position_embeddings
        inputs["past_key_values"] = past_key_values
        inputs["use_cache"] = True

        return inputs


DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}


class GreedyDecoder:
    """Greedy decoder for causal language models.

    Args:
        model: The transformer model used for generation.
        tokenizer: Tokenizer to convert text to token IDs.
        device: Torch device on which the model and tensors reside.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt, max_new_tokens):
        """Generate tokens greedily up to max_length.

        Args:
            prompt (str): Input text prompt.
            max_length (int): Maximum length of generated sequence.

        Returns:
            torch.Tensor: Tensor of token IDs including the prompt and generated tokens.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        eos_token_id = self.tokenizer.eos_token_id

        produced_tokens = 0
        with torch.no_grad():
            while produced_tokens < max_new_tokens:
                logits = self.model(inputs).logits
                next_token = torch.tensor(
                    [[torch.argmax(logits[..., -1, :])]], device=inputs.device
                )
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                inputs = torch.cat([inputs, next_token], dim=1)
                produced_tokens += 1

        return inputs


class PrefillDecodeGreedyDecoder:
    """
    Greedy generator to mimick prefill-decode pipeline on the devide.

    This decoder performs an initial *prefill* pass on the padded prompt to obtain
    key/value caches (KV cache) from the model, then iteratively generates new
    tokens using a greedy strategy while updating the cache.

    Args:
        model: The transformer model used for both prefill and decode phases.
        tokenizer: Tokenizer for converting text to token ids and vice‑versa.
        max_seq_len (int): Maximum sequence length the model can handle.
        config: Configuration object used to build RoPE position embeddings.
        device: Torch device (e.g., ``torch.device('cuda')``) on which the model
            runs.
        use_right_padding: pad input sequence to the right or left (left is default)

    The workflow is:
    1. the input prompt to ``max_seq_len - 1`` and build the attention mask.
    2. Run a *prefill* forward pass to obtain initial KV caches.
    3. Iteratively generate tokens:
       * Select the token with the highest logit (greedy).
       * Update the KV cache with the newly generated token.
       * Adjust the attention mask and position embeddings.
       * Stop when ``max_new_tokens`` is reached or an EOS token is produced.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_seq_len,
        config,
        device,
        use_right_padding: bool = False,
    ):
        self.prefill_model = model
        self.decode_model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.use_right_padding = use_right_padding
        self.helper = PrefillDecodeUtils(max_seq_len, config, self.device)
        self.pos_embeds = _build_rope_templates_from_config(
            config, max_seq=max_seq_len, device=device, dtype=torch.float32
        )

    def generate_right_padding(self, prompt, max_new_tokens):
        """
        Generate tokens using right‑padded prefill‑decode pipeline.

        Parameters
        ----------
        prompt : str
            Input text prompt to be tokenised.
        max_new_tokens : int
            Maximum number of new tokens to generate after the prompt.

        Returns
        -------
        torch.Tensor
            Tensor containing the original prompt token IDs followed by the generated token IDs.

        Notes
        -----
        The method pads the input on the right,
        (<PROMPT_TOKEN1>...<PROMPT_TOKENN><PAD><PAD>...<PAD>)
        builds the appropriate attention mask, runs a prefill
        pass to obtain the KV‑cache, and then iteratively decodes
        tokens while updating the cache as if new token was added
        to the original one. It differs from huggingface decoding
        mode which adds new kv-cache tuples strictly to the end.
        Generation stops early if the EOS token is produced.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        assert isinstance(inputs, torch.Tensor)

        eos_token_id = self.tokenizer.eos_token_id

        generated = inputs.clone()
        cur_seq_len = inputs.shape[-1]
        prefill_max_seq_len = self.max_seq_len - 1
        prefill_input = pad_input(
            inputs, self.tokenizer.pad_token_id, prefill_max_seq_len, pads_on_left=False
        )
        pad_mask = (prefill_input != self.tokenizer.pad_token_id).to(inputs.device)
        attn_mask = self.helper.build_attention_mask_for_padded_input(
            pad_mask=pad_mask, prefill=True
        )

        prefill_position_ids = pad_mask.long().cumsum(-1) - 1
        prefill_position_ids.masked_fill_(pad_mask == 0, 0)
        position_embeddings = self.helper.build_position_embeddings_for_padded_input(
            self.pos_embeds, prefill_position_ids
        )
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []

        with torch.no_grad():
            outputs = self.prefill_model(
                prefill_input,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values
        token_index_in_logits = (
            cur_seq_len - 1
        )  # token at -1 will be just some <PAD> or <NEW_LINE>

        self.prefill_model = self.prefill_model.cpu()
        self.decode_model = self.decode_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        produced_tokens = 0
        with torch.no_grad():
            while produced_tokens < max_new_tokens:
                next_token = torch.tensor(
                    [[torch.argmax(logits[..., token_index_in_logits, :])]],
                    device=self.device,
                )
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                generated = torch.cat([generated, next_token], dim=1)
                dec_pad_mask = torch.concatenate(
                    (pad_mask, torch.tensor([[False]], device=pad_mask.device)), dim=-1
                )

                cur_seq_len += 1
                produced_tokens += 1
                dec_inputs = self.helper.get_input_for_decode_model(
                    next_token,
                    past_key_values=past_key_values,
                    pad_mask=dec_pad_mask,
                    cur_seq_len=cur_seq_len,
                )
                outputs = self.decode_model(**dec_inputs)
                logits = outputs.logits
                new_key_values = outputs.past_key_values
                token_index_in_logits = -1

                # insert new key-value at seq_len as if decoded token was present at original prompt
                for idx in range(len(new_key_values)):
                    past_key_values[idx][0][
                        :, :, cur_seq_len - 1 : cur_seq_len, :
                    ] = new_key_values[idx][0][:, :, -1:, :]
                    past_key_values[idx][1][
                        :, :, cur_seq_len - 1 : cur_seq_len, :
                    ] = new_key_values[idx][1][:, :, -1:, :]
                # update mask accordingly as if decoded token was present at original prompt
                pad_mask[..., cur_seq_len - 1] = True

        return generated

    def generate_left_padding(self, prompt, max_new_tokens):
        """
        Generate tokens for the given prompt using a greedy decoding strategy.

        Args:
            prompt (str): The input text prompt to be tokenised and processed.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Tensor containing the original prompt token ids followed
            by the generated token ids.

        The method performs a prefill pass on padded to the left sequence
        (<PAD><PAD>...<PAD><PROMPT_TOKEN1>...<PROMPT_TOKENN>)
        to obtain initial key/value caches, then iteratively generates tokens,
        updating the cache and attention masks after each step.
        Generation stops early if the EOS token is produced.
        This way of generation corresponds to decoding used in huggingface.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        assert isinstance(inputs, torch.Tensor)

        eos_token_id = self.tokenizer.eos_token_id

        generated = inputs.clone()
        cur_seq_len = inputs.shape[-1]
        prefill_max_seq_len = self.max_seq_len - 1
        prefill_input = pad_input(
            inputs, self.tokenizer.pad_token_id, prefill_max_seq_len
        )
        pad_mask = (prefill_input != self.tokenizer.pad_token_id).to(inputs.device)
        attn_mask = self.helper.build_attention_mask_for_padded_input(
            pad_mask=pad_mask, prefill=True
        )

        prefill_position_ids = pad_mask.long().cumsum(-1) - 1
        prefill_position_ids.masked_fill_(pad_mask == 0, 0)
        position_embeddings = self.helper.build_position_embeddings_for_padded_input(
            self.pos_embeds, prefill_position_ids
        )
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = []

        with torch.no_grad():
            outputs = self.prefill_model(
                prefill_input,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                use_cache=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        self.prefill_model = self.prefill_model.cpu()
        self.decode_model = self.decode_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        produced_tokens = 0
        with torch.no_grad():
            while produced_tokens < max_new_tokens:
                next_token = torch.tensor(
                    [[torch.argmax(logits[..., -1, :])]], device=self.device
                )
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                generated = torch.cat([generated, next_token], dim=1)
                pad_mask = torch.concatenate(
                    (pad_mask, torch.tensor([[True]], device=pad_mask.device)), dim=-1
                )
                cur_seq_len += 1
                produced_tokens += 1

                dec_inputs = self.helper.get_input_for_decode_model(
                    next_token,
                    past_key_values=past_key_values,
                    pad_mask=pad_mask,
                    cur_seq_len=cur_seq_len,
                )
                outputs = self.decode_model(**dec_inputs)
                logits = outputs.logits
                new_key_values = outputs.past_key_values

                # shift new_key_values to the left to get updated past_key_values
                for i in range(prefill_max_seq_len):
                    for idx in range(len(new_key_values)):
                        past_key_values[idx][0][:, :, i : i + 1, :] = new_key_values[
                            idx
                        ][0][:, :, i + 1 : i + 2, :]
                        past_key_values[idx][1][:, :, i : i + 1, :] = new_key_values[
                            idx
                        ][1][:, :, i + 1 : i + 2, :]

                pad_mask = pad_mask[..., 1:]  # update pad_mask accordingly

        return generated

    def generate(self, prompt, max_new_tokens):
        if self.use_right_padding:
            return self.generate_right_padding(prompt, max_new_tokens)

        return self.generate_left_padding(prompt, max_new_tokens)


class OriginalModelEvaluator:
    """
    Evaluates the original full‑precision Llama model.

    This evaluator loads the model in FP32 (or the dtype specified by ``args.dtype``),
    runs optional benchmark tasks via ``lm_eval`` and can generate a sample output
    for a provided prompt.

    Args:
        args: Namespace object containing command‑line arguments (model path,
              device, dtype, evaluation tasks, prompt, etc.).
        tokenizer: Tokenizer instance compatible with the model.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device(args.device)
        self.dtype = DTYPE_MAP[args.dtype]

    def evaluate(self):
        """Load the full‑precision model and optionally run evaluation tasks.

        The method:
        1. Loads the model with the specified dtype.
        2. If ``args.eval_tasks`` is provided, runs ``evaluate_llm_on_tasks``.
        3. If a prompt is supplied, generates text using the model.
        4. Cleans up GPU memory after execution.
        """
        # Load FP model
        model = (
            AutoModelForCausalLM.from_pretrained(
                self.args.model,
                dtype=self.dtype,
                trust_remote_code=self.args.trust_remote_code,
                token=self.args.hf_token,
                cache_dir=self.args.cache_dir,
            )
            .cpu()
            .eval()
        )

        if self.args.eval_tasks is not None:
            config = model.config
            max_seq_len = config.max_position_embeddings
            results = evaluate_llm_on_tasks(
                model,
                self.tokenizer,
                self.args.eval_tasks,
                max_length=max_seq_len,
            )
            print("Original RESULTS ARE:")
            print(make_table(results))

        if self.args.prompt is not None:
            max_seq_len = min(2048, model.config.max_position_embeddings)
            inputs = self.tokenizer(
                self.args.prompt,
                return_tensors="pt",
                max_length=max_seq_len - 1,
                padding="max_length",
                padding_side="left",
            ).to(self.device)
            out_ids = model.to(self.args.device).generate(
                **inputs,
                max_length=max_seq_len + self.args.max_new_tokens,
                do_sample=False,
            )
            output = self.tokenizer.decode(out_ids.squeeze(), skip_special_tokens=True)
            print(f"Original model prompt: {output}")

        # Cleanup
        model = model.cpu()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class QuantizedModelEvaluator:
    """
    Evaluates the fake‑quantized Llama model.

    This evaluator loads a model that has been quantized using the fake‑quantization
    pipeline, runs optional benchmark tasks via ``lm_eval`` and can generate a sample
    output for a provided prompt.

    Args:
        args: Namespace object containing command‑line arguments (model path,
              device, quantized model path, evaluation tasks, prompt, etc.).
        tokenizer: Tokenizer instance compatible with the model.
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device(args.device)

    def evaluate(self):
        """Load the fake‑quantized model and optionally run evaluation tasks.

        The method:
        1. Loads the quantized model from ``args.fk_model_path``.
        2. If ``args.eval_tasks`` is provided, runs ``evaluate_llm_on_tasks``.
        3. If a prompt is supplied, generates text using the model.
        4. Cleans up GPU memory after execution.
        """
        # Load fake‑quantized model
        fk_model = (
            torch.load(self.args.fk_model_path, weights_only=False)
            .eval()
            .to(self.args.device)
        )
        config = fk_model.wrapped.config
        max_seq_len = config.max_position_embeddings

        if self.args.eval_tasks is not None:
            results = evaluate_llm_on_tasks(
                fk_model,
                self.tokenizer,
                self.args.eval_tasks,
                max_length=max_seq_len,
            )
            print("Quantized RESULTS ARE:")
            print(make_table(results))

        # Choose decoder
        if not self.args.prefill_decode:
            decoder = GreedyDecoder(fk_model, self.tokenizer, self.args.device)
        else:
            # Use original model for prefill‑decode if available
            decoder = PrefillDecodeGreedyDecoder(
                fk_model,
                self.tokenizer,
                max_seq_len,
                config,
                self.args.device,
                self.args.use_right_padding,
            )  # type: ignore[assignment]

        out_ids = decoder.generate(
            self.args.prompt, max_new_tokens=self.args.max_new_tokens
        )
        output = self.tokenizer.decode(out_ids.squeeze(), skip_special_tokens=True)
        print(f"Fake quantized model prompt: {output}")

        # Cleanup
        fk_model = (
            fk_model.cpu()
            if not isinstance(fk_model, tuple)
            else (fk_model[0].cpu(), fk_model[1].cpu())
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fake-quantized Llama model"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HF repo name or local path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda|cpu|mps).",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="float32",
        help="Model dtype for load.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache_dir for using model/datasets loading",
    )
    parser.add_argument(
        "--fk_model_path", type=str, required=True, help="Path to fake_quantized model"
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help="tasks to be evaluated using lm_eval, e.g. `winogrande,arc_easy,arc_challenge,openbookqa,mmlu_pro,ifeval,bbh`",
    )
    parser.add_argument(
        "--skip_fp_eval",
        action="store_true",
        help="Skip original model evaluation.",
    )
    parser.add_argument(
        "--prefill_decode",
        action="store_true",
        help="Model is calibrated for prefill_decode pipeline.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt to decode",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Maximum new tokens to produce"
    )
    parser.add_argument(
        "--use_right_padding",
        action="store_true",
        help="Use right padding instead of left (default) for prefill-decode pipeline.",
    )
    return parser.parse_args()


def main():

    args = parse_args()
    print(args)

    # -------------------------------------------------------------------------
    # Basic setup
    # -------------------------------------------------------------------------
    device = torch.device(args.device)

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"fk_model_path    : {args.fk_model_path}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    if not args.skip_fp_eval:
        # -------------------------------------------------------------------------
        # FP model evaluation
        # -------------------------------------------------------------------------
        OriginalModelEvaluator(args, tokenizer).evaluate()

    # -------------------------------------------------------------------------
    # Quantized model evaluation
    # -------------------------------------------------------------------------
    QuantizedModelEvaluator(args, tokenizer).evaluate()


if __name__ == "__main__":
    main()

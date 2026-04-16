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
import torch

from lm_eval.utils import make_table

from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.examples.static_llama_layer_runtime import (
    _build_decode_attention_mask,
    _build_rope_templates_from_config,
    _slice_rope,
)

from tico.quantization.wrapq.examples.quantize_full_qmodel_with_gptq import pad_input

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"

@torch.no_grad()
class GreedyDecoder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(self, prompt, max_length):
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
         while inputs.shape[-1] < max_length:
            logits = self.model(inputs).logits
            next_token = torch.tensor([[torch.argmax(logits[..., -1, :])]], device=inputs.device)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            inputs = torch.cat([inputs, next_token], dim=1)
        
        return inputs
    
def pad_input_to_left(input, pad_token, max_seq_len):
    """Pad a tensor to a maximum sequence length using the specified pad token."""
    pads = torch.full(
        (input.shape[0], max_seq_len - input.shape[1]),
        fill_value=pad_token,
        device=input.device,
    )
    return torch.cat((pads, input), dim=1)

class PrefillDecodeGreedyDecoder:
    def __init__(self, model, orig_model, tokenizer, max_seq_len, config, device):
        prefill_model, decode_model = model
        self.prefill_model = prefill_model
        self.decode_model = decode_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        self.orig_model = orig_model
        self.rope_cos, self.rope_sin = _build_rope_templates_from_config(
            config, max_seq=max_seq_len, device=device, dtype=torch.float32
        )
    
    
    def generate_left_padding(self, prompt, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        assert isinstance(inputs, torch.Tensor)
        
        eos_token_id = self.tokenizer.eos_token_id
        
        generated = inputs.clone()
        cur_seq_len = inputs.shape[-1]
        prefill_max_seq_len = self.max_seq_len - 1
        prefill_input = pad_input_to_left(inputs, self.tokenizer.pad_token_id, prefill_max_seq_len)
        attn_mask = self.build_prefill_padded_attention_mask(cur_seq_len, prefill_max_seq_len, self.device, right_padding=False)
        position_embeddings = self.build_prefill_position_embeddings(cur_seq_len, prefill_max_seq_len, self.device, right_padding=False)
        
        with torch.no_grad():
            outputs = self.prefill_model(prefill_input, attention_mask = attn_mask, position_embeddings=position_embeddings, use_cache = True)
            
          #  orig_inputs = self.tokenizer(prompt, return_tensors="pt", max_length=prefill_max_seq_len, padding='max_length', padding_side="left").to(self.device)
          #  orig_attn_mask = orig_inputs["attention_mask"]
          #  orig_position_ids = orig_attn_mask.long().cumsum(-1) - 1
          #  orig_position_ids.masked_fill_(orig_attn_mask == 0, 0)
          #  orig_inputs["position_ids"] = orig_position_ids
          #  #orig_outs = self.orig_model.to(self.device)(**orig_inputs)
        
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        self.prefill_model = self.prefill_model.cpu()
        self.decode_model = self.decode_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        produced_tokens = 0
        with torch.no_grad():
         while produced_tokens < max_new_tokens:
            next_token = torch.tensor([[torch.argmax(logits[..., -1, :])]], device=self.device)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            generated = torch.cat([generated, next_token], dim=1)
            cur_seq_len += 1
            produced_tokens += 1
        
            dec_inputs = self.get_input_for_decode_model(next_token, past_key_values=past_key_values, cur_seq_len = cur_seq_len, right_padding=False)
            outputs = self.decode_model(**dec_inputs)
            logits = outputs.logits
            new_key_values = outputs.past_key_values
            # shift past_key_values
            for i in range(prefill_max_seq_len - 1):
                #cur_seq_idx = prefill_max_seq_len - cur_seq_len + i - 1
                for idx in range(len(new_key_values)):
                    past_key_values[idx][0][:, :, i : i + 1, :] =\
                        past_key_values[idx][0][:, :, i + 1: i + 2, :]
                    past_key_values[idx][1][:, :, i : i + 1, :] =\
                        past_key_values[idx][1][:, :, i + 1 : i + 2, :]
                
            # update past_key_values
            for idx in range(len(new_key_values)):
                past_key_values[idx][0][:, :, -1 :, :] = new_key_values[idx][0]
                past_key_values[idx][1][:, :, -1 :, :] = new_key_values[idx][1]
                
        
        return generated
        
    def generate_right_padding(self, prompt, max_new_tokens):
        
        inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        assert isinstance(inputs, torch.Tensor)
        
        eos_token_id = self.tokenizer.eos_token_id
        
        generated = inputs.clone()
        cur_seq_len = inputs.shape[-1]
        prefill_max_seq_len = self.max_seq_len - 1
        prefill_input = pad_input(inputs, self.tokenizer.pad_token_id, prefill_max_seq_len)
        attn_mask = self.build_prefill_padded_attention_mask(cur_seq_len, prefill_max_seq_len, self.device, right_padding=True)
        position_embeddings = self.build_prefill_position_embeddings(cur_seq_len, prefill_max_seq_len, self.device, right_padding=True)
        
        with torch.no_grad():
            outputs = self.prefill_model(prefill_input, attention_mask = attn_mask, position_embeddings=position_embeddings, use_cache = True)
            
           # orig_inputs = self.tokenizer(prompt, return_tensors="pt", max_length=prefill_max_seq_len, padding='max_length', padding_side="right").to(self.device)
           # orig_attn_mask = orig_inputs["attention_mask"]
           # orig_position_ids = orig_attn_mask.long().cumsum(-1) - 1
           # orig_position_ids.masked_fill_(orig_attn_mask == 0, 0)
           # orig_inputs["position_ids"] = orig_position_ids
           # orig_outs = self.orig_model.to(self.device)(**orig_inputs)
          
       
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        
        self.prefill_model = self.prefill_model.cpu()
        self.decode_model = self.decode_model.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        produced_tokens = 0
        with torch.no_grad():
         while produced_tokens < max_new_tokens:
            next_token = torch.tensor([[torch.argmax(logits[..., -1, :])]], device=self.device)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            generated = torch.cat([generated, next_token], dim=1)
            cur_seq_len += 1
            produced_tokens += 1

            dec_inputs = self.get_input_for_decode_model(next_token, past_key_values=past_key_values, cur_seq_len = cur_seq_len-1, right_padding=True)
            outputs = self.decode_model(**dec_inputs)
            logits = outputs.logits
            new_key_values = outputs.past_key_values
            # shift past_key_values
            for i in range(prefill_max_seq_len - 1):
                #cur_seq_idx = prefill_max_seq_len - cur_seq_len + i - 1
                for idx in range(len(new_key_values)):
                    past_key_values[idx][0][:, :, i : i + 1, :] =\
                        past_key_values[idx][0][:, :, i + 1: i + 2, :]
                    past_key_values[idx][1][:, :, i : i + 1, :] =\
                        past_key_values[idx][1][:, :, i + 1 : i + 2, :]
                
            # update past_key_values to be the last added 
            for idx in range(len(new_key_values)):
                past_key_values[idx][0][:, :, -1 :, :] = new_key_values[idx][0]
                past_key_values[idx][1][:, :, -1 :, :] = new_key_values[idx][1]
                

        
        return generated
    
    def generate(self, prompt, max_new_tokens):
        return self.generate_left_padding(prompt, max_new_tokens) 
        #return self.generate_right_padding(prompt, max_new_tokens)
        
    def build_prefill_padded_attention_mask(self, cur_seq_len, max_seq_len, device, right_padding = False, mask_value:float = -120.0):
        dtype = torch.float32
        mask = torch.full((1, max_seq_len, max_seq_len), mask_value, device=device, dtype=dtype)
        
        if right_padding:
            for i in range(max_seq_len):
                for j in range(max_seq_len):
                    if i >= j and j < cur_seq_len:
                        mask[..., i, j] = 0
        else:
            for i in range(max_seq_len):
                for j in range(max_seq_len):
                    if i >= j and j >= max_seq_len - cur_seq_len:
                        mask[..., i, j] = 0
                        
        return mask
    
    def build_prefill_position_embeddings(self, cur_seq_len, max_seq_len, device, right_padding = False):
        dtype = torch.float32
        position_embeddings = self.prefill_model.wrapped.model.wrapped.get_position_embeddings_for(dtype, device)
        cos = torch.ones_like(position_embeddings[0][:, : max_seq_len, :])
        sin = torch.zeros_like(position_embeddings[1][:, : max_seq_len, :])
        
        sl_cos, sl_sin = position_embeddings    
        sl_cos = sl_cos[:, : cur_seq_len, :]
        sl_sin = sl_sin[:, : cur_seq_len, :]
        if right_padding is True:
            cos[..., :cur_seq_len, :] = sl_cos
            sin[..., :cur_seq_len, :] = sl_sin
        else:
            # left padding
            cos[..., max_seq_len - cur_seq_len : max_seq_len, :] = sl_cos
            sin[..., max_seq_len - cur_seq_len : max_seq_len, :] = sl_sin
            
        return (cos, sin)
    
    def get_input_for_decode_model(self, next_token, past_key_values, cur_seq_len, right_padding=False):
        dtype = torch.float32
        if right_padding:
            attention_mask = _build_decode_attention_mask(
                batch_size=1,
                past_len=cur_seq_len,
                max_seq=self.max_seq_len,
                device=self.device,
                dtype=dtype,
            )

            
            position_embeddings = _slice_rope(
                self.rope_cos,
                self.rope_sin,
                position=cur_seq_len,
                batch_size=1,
                device=self.device,
                dtype=dtype,
            )
        else:
            attention_mask = self.build_prefill_padded_attention_mask(cur_seq_len, self.max_seq_len, self.device)
            attention_mask = attention_mask[..., -1, :].unsqueeze(0)
            position_embeddings = self.build_prefill_position_embeddings(cur_seq_len, self.max_seq_len, self.device)
            (cos, sin) = position_embeddings
            cos = cos[..., -1, :].unsqueeze(0)
            sin = sin[..., -1, :].unsqueeze(0)
            position_embeddings = (cos, sin)

        # fill in input
        inputs = {}
        inputs["input_ids"] = next_token
        inputs["attention_mask"] = attention_mask
        inputs["position_embeddings"] = position_embeddings
        inputs["past_key_values"] = past_key_values
        return inputs
                
def main():
    parser = argparse.ArgumentParser(
        description="Try a fake-quantized models"
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
        "--prompt", type=str, default="The capital of France is", help="Prompt to decode"
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help="tasks to be evaluated using lm_eval, e.g. `winogrande,arc_easy,arc_challenge,openbookqa,mmlu_pro,ifeval,bbh`",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128, help="Maximum new okens to produce"
    )
    
    args = parser.parse_args()
    print(args)

    # Basic setup
    
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"Prompt           : {args.prompt}")
    print()

    # -------------------------------------------------------------------------
    # 2. Load the FP backbone and tokenizer
    # -------------------------------------------------------------------------
    print("Loading FP model …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    ).cpu().eval()
    
    if tokenizer.pad_token is None:
        print(
            "Warning: tokenizer doesn't have pad_token. Prefill-decoding scheme may fail."
        )
        tokenizer.pad_token = tokenizer.eos_token
            
    fk_model = torch.load(args.fk_model_path, weights_only=False)
    
    if isinstance(fk_model, tuple):
        fk_model = (fk_model[0].eval().cpu(), fk_model[1].eval().cpu())
        config = fk_model[0].wrapped.config
    else:
        fk_model.eval()
        fk_model = fk_model.cpu()
        config = fk_model.wrapped.config

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    max_seq_len = config.max_position_embeddings
    inputs = tokenizer(args.prompt, return_tensors="pt", max_length=max_seq_len - 1, padding='max_length', padding_side="left").to(device) #just try with right padding below
    #inputs = tokenizer(args.prompt, return_tensors="pt", max_length=max_seq_len - 1, padding='max_length', padding_side="right", device=args.device).to(device)
        
    model.config.use_cache = True
    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            model, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        print("Original RESULTS ARE:")
        print(make_table(results))
        
    out_ids = model.to(args.device).generate(**inputs, max_length=max_seq_len + args.max_new_tokens, do_sample = False)
    output = tokenizer.decode(out_ids.squeeze(), skip_special_tokens=True)
    print(f"Original model prompt: {output}")
    model = model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    fk_model = fk_model.to(args.device) if not isinstance(fk_model, tuple) else (fk_model[0].to(args.device), fk_model[1].cpu())
    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            fk_model, tokenizer, args.eval_tasks, max_length=max_seq_len
        )
        print("Quantized RESULTS ARE:")
        print(make_table(results))
      
    fk_decoder = GreedyDecoder(fk_model, tokenizer, args.device) if not isinstance(fk_model, tuple) else PrefillDecodeGreedyDecoder(fk_model, model, tokenizer, max_seq_len, config, args.device)
    
    out_ids = fk_decoder.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    output = tokenizer.decode(out_ids.squeeze(), skip_special_tokens=True)
    print(f"Fake quantized model prompt: {output}")
    fk_model = fk_model.cpu() if not isinstance(fk_model, tuple) else (fk_model[0].cpu(), fk_model[1].cpu())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    main()

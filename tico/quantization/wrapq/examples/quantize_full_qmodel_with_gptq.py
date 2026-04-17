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

# =============================================================================
# PTQ + GPTQ HYBRID QUANTIZATION PIPELINE
# -----------------------------------------------------------------------------
# This script shows how to:
#   1. Load a pretrained FP Llama-3 model.
#   2. Run GPTQ to quantize weights only (optional).
#   3. Wrap every Transformer layer with a PTQWrapper to quantize activations.
#   4. Calibrate activations observers in a single pass over a text corpus.
#   5. Inject GPTQ’s per-tensor weight scales / zero-points into the PTQ graph.
#   6. Freeze all Q-params and compute Wikitext-2 perplexity.
#   7. Save model/layers (optional)
# =============================================================================

import argparse
import copy
import pathlib
import random
from typing import Any

import types

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer

import tico
from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.builders import build_llm_ptq_config
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.spinquant import SpinQuantConfig
from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.examples.static_llama_layer_runtime import (
    _build_decode_attention_mask,
    _build_rope_templates_from_config,
    _slice_rope,
)
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from tico.utils.utils import SuppressWarning, move_to_device

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}

# Hardcoded dataset settings
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

left_pad = False#True

# -------------------------------------------------------------------------
# Helper — copy GPTQ (scale, zp) into PTQ observers
# -------------------------------------------------------------------------
def inject_gptq_qparams(
    root: torch.nn.Module,
    gptq_quantizers: dict[str, Any],  # {fp_name: quantizer}
    weight_obs_name: str = "weight",
    *,
    verbose: bool = False,
):
    """
    Inject GPTQ (scale, zero-point) into PTQ observers.

    When verbose=True, prints a summary of matched / missed / unused entries.
    """
    seen = set()
    missed_modules = []

    for m in root.modules():
        if not isinstance(m, QuantModuleBase):
            continue
        if m.fp_name is None:
            continue

        quantizer = gptq_quantizers.get(m.fp_name)
        obs = m.get_observer(weight_obs_name)

        # Only care about modules that should have weight observers
        if obs is None:
            continue

        if quantizer is None:
            missed_modules.append(m.fp_name)
            continue

        assert isinstance(obs, AffineObserverBase)
        obs.load_qparams(quantizer.scale, quantizer.zero, lock=True)

def pad_input(input, pad_token, max_seq_len, right: bool = True):
    """Pad a tensor to a maximum sequence length using the specified pad token."""
    pads = torch.full(
        (input.shape[0], max_seq_len - input.shape[1]),
        fill_value=pad_token,
        device=input.device,
    )
    if right is True:
        res = torch.cat((input, pads), dim=1)
    else:
        res = torch.cat((pads, input), dim=1)

    return res
class PrefillDecodeUtils:
    def __init__(self, max_seq_len, config, device):
        self.max_seq_len = max_seq_len
        self.device = device
        self.pos_embeds = _build_rope_templates_from_config(
            config, max_seq=max_seq_len, device=device, dtype=torch.float32
        )

    def build_attention_mask_for_padded_input(self, cur_seq_len, right_padding = False, mask_value:float = -120.0, prefill: bool = True, pad_mask = None):
        dtype = torch.float32
      #  mask = torch.full((1, max_seq_len, max_seq_len), mask_value, device=device, dtype=dtype)
      #  
      #  #return mask
      #  if right_padding:
      #      for i in range(max_seq_len):
      #          for j in range(max_seq_len):
      #              if i >= j and j < cur_seq_len:
      #                  mask[..., i, j] = 0
      #  else:
      #      for i in range(max_seq_len):
      #          for j in range(max_seq_len):
      #              if i >= j and j >= max_seq_len - cur_seq_len:
      #                  mask[..., i, j] = 0

        max_seq_len = self.max_seq_len -  1 if prefill is True else self.max_seq_len
        if pad_mask is None:
            pad_mask = [i < cur_seq_len for i in range(max_seq_len)] if right_padding else [i >= max_seq_len - cur_seq_len for i in range(max_seq_len)]
            pad_mask = torch.tensor(pad_mask, dtype = torch.bool, device = self.device)
        causal_mask = torch.full((1, max_seq_len, max_seq_len), 1, device=self.device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask == 0 # negating
        res_mask = torch.logical_and(pad_mask, causal_mask)
        mask_res = torch.zeros((1, max_seq_len, max_seq_len), device=self.device, dtype=dtype)
        mask_res = mask_res.masked_fill(~res_mask, mask_value)
        return mask_res
        #assert torch.equal(mask_res, mask)
        
        #return mask
    
    def build_position_embeddings_for_padded_input(self, position_embeddings, cur_seq_len, position_ids, right_padding = False, prefill: bool = True):
        dtype = torch.float32
        #position_embeddings = self.prefill_model.wrapped.model.wrapped.get_position_embeddings_for(dtype, self.device)
        #max_seq_len = self.max_seq_len -  1 if prefill is True else self.max_seq_len
        #cos = torch.ones_like(position_embeddings[0][:, : max_seq_len, :])
        #sin = torch.zeros_like(position_embeddings[1][:, : max_seq_len, :])
        #
        #sl_cos, sl_sin = position_embeddings    
        #sl_cos = sl_cos[:, : cur_seq_len, :]
        #sl_sin = sl_sin[:, : cur_seq_len, :]
        #if right_padding is True:
        #    cos[..., :cur_seq_len, :] = sl_cos
        #    sin[..., :cur_seq_len, :] = sl_sin
        #else:
        #    # left padding
        #    cos[..., max_seq_len - cur_seq_len : max_seq_len, :] = sl_cos
        #    sin[..., max_seq_len - cur_seq_len : max_seq_len, :] = sl_sin
        #    
        #return (cos, sin)
        
        sl_cos, sl_sin = position_embeddings
        sl_cos = sl_cos[..., :-1, :]
        sl_sin = sl_sin[..., :-1, :]
        cos = sl_cos[..., position_ids.to(sl_cos.device).squeeze(0), :]
        sin = sl_sin[..., position_ids.to(sl_sin.device).squeeze(0), :]
        
        #cos = torch.gather(sl_cos, index = position_ids.to(sl_cos.device).squeeze(0).unsqueeze(1))
        #sin = torch.gather(sl_sin, index = position_ids.to(sl_sin.device).squeeze(0).unsqueeze(1))
        return (cos, sin)
    
    def get_input_for_decode_model(self, input, past_key_values, cur_seq_len, right_padding=False, pad_mask = None):
        dtype = torch.float32
        next_token = torch.tensor([[input[..., -1]]], device = input.device)
        if right_padding:
          #  attention_mask = _build_decode_attention_mask(
          #      batch_size=1,
          #      past_len=cur_seq_len,
          #      max_seq=self.max_seq_len,
          #      device=self.device,
          #      dtype=dtype,
          #  )
##
          #  position_embeddings = _slice_rope(
          #      self.pos_embeds[0],  # cos
          #      self.pos_embeds[1],  # sin
          #      position=cur_seq_len,
          #      batch_size=1,
          #      device=self.device,
          #      dtype=dtype,
          #  )
            attention_mask = self.build_attention_mask_for_padded_input(cur_seq_len, right_padding=True, prefill=False, pad_mask = pad_mask)
            attention_mask = attention_mask[..., -1, :].unsqueeze(0) # last row
            #position_embeddings = self.build_position_embeddings_for_padded_input(self.pos_embeds, cur_seq_len, right_padding=True, prefill=False)
            #(cos, sin) = position_embeddings
            #cos = cos[..., -1, :].unsqueeze(0)
            #sin = sin[..., -1, :].unsqueeze(0)
            #position_embeddings = (cos, sin)
            position_embeddings = self.build_position_embeddings_for_padded_input(self.pos_embeds, cur_seq_len, position_ids=torch.tensor([[cur_seq_len - 1]]), right_padding=True, prefill=False)
        else:
            attention_mask = self.build_attention_mask_for_padded_input(cur_seq_len, right_padding=False, prefill=False, pad_mask = pad_mask)
            attention_mask = attention_mask[..., -1, :].unsqueeze(0) # last row
            #position_embeddings = self.build_position_embeddings_for_padded_input(self.pos_embeds, cur_seq_len, position_ids=torch.tensor([[cur_seq_len]]), right_padding=False, prefill=False )
            #(cos, sin) = position_embeddings
            #cos = cos[..., -1, :].unsqueeze(0)
            #sin = sin[..., -1, :].unsqueeze(0)
            #position_embeddings = (cos, sin)
            position_embeddings = self.build_position_embeddings_for_padded_input(self.pos_embeds, cur_seq_len, position_ids=torch.tensor([[cur_seq_len - 1]]), right_padding=False, prefill=False)

        # fill in input
        inputs = {}
        inputs["input_ids"] = next_token
        inputs["attention_mask"] = attention_mask
        inputs["position_embeddings"] = position_embeddings
        inputs["past_key_values"] = past_key_values
        inputs["use_cache"] = True

        return inputs
        
def get_decode_input(
    prefill_model,
    calib_input,
    pad_token_id,
    ropes,
    max_seq_len,
    device,
    helper,
):
    """Prepare inputs for the decode model using prefill KV‑cache and rotary embeddings."""
    prefill_input = calib_input[..., :-1]
    prefill_seq_len = prefill_input.shape[-1]

    prefill_max_seq_len = max_seq_len - 1
    prefill_input = pad_input(prefill_input, pad_token_id, prefill_max_seq_len, right = not left_pad)
    prefill_attn_mask = (prefill_input != pad_token_id).to(device)
    attn_mask = helper.build_attention_mask_for_padded_input(prefill_seq_len, right_padding=not left_pad, pad_mask = prefill_attn_mask)
    
    prefill_position_ids = prefill_attn_mask.long().cumsum(-1) - 1
    prefill_position_ids.masked_fill_(prefill_attn_mask == 0, 1)
    position_embeddings = helper.build_position_embeddings_for_padded_input(ropes, prefill_seq_len, prefill_position_ids, right_padding=not left_pad)
    
    with torch.no_grad():
        # run prefill model to get kv-cache
        outputs = prefill_model(prefill_input.to(device), attention_mask = attn_mask, position_embeddings=position_embeddings, use_cache = True)

    # fill inputs for decode model
    next_token = calib_input[..., -1:]
    prefill_input = torch.concat([prefill_input, next_token], dim=1)
    attn_mask = (prefill_input != pad_token_id).to(device)
    
    dec_inputs = helper.get_input_for_decode_model(prefill_input, past_key_values=outputs.past_key_values, cur_seq_len = prefill_seq_len + 1, right_padding=not left_pad, pad_mask = attn_mask)
    return dec_inputs
    #dtype=torch.float32
    #attention_mask = _build_decode_attention_mask(
    #    batch_size=1,
    #    past_len=prefill_seq_len,
    #    max_seq=max_seq_len,
    #    device=device,
    #    dtype=dtype,
    #)
#
    #rope_cos, rope_sin = ropes
    #position_embeddings = _slice_rope(
    #    rope_cos,
    #    rope_sin,
    #    position=prefill_seq_len - 1,
    #    batch_size=1,
    #    device=device,
    #    dtype=dtype,
    #)

    # fill in input
    inputs = {}
    inputs["input_ids"] = torch.tensor([[next_token]])
    inputs["attention_mask"] = attention_mask
    inputs["position_embeddings"] = position_embeddings
    inputs["past_key_values"] = outputs.past_key_values
    return inputs

def evaluate_ppl_of_prefill_decode_model_on_dataset(
    model,
    dataset,
    pad_token_id,
    max_seq_len,
    seed=0,
    device: str = "cuda",
):
    """Compute perplexity for the prefill-decode logic."""

    config = (
        model.config
        if hasattr(model, "config")
        else model.wrapped.config
    )
    rope_cos, rope_sin = _build_rope_templates_from_config(
        config, max_seq=max_seq_len, device=device, dtype=torch.float32
    )

    
    if hasattr(model, "device") and model.device.type != device.type:
        if hasattr(model, "to"):
            model.to(device)

    torch.manual_seed(seed)
    nlls = []
    helper = PrefillDecodeUtils(max_seq_len, config, device)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            if isinstance(batch, torch.Tensor):

                prefill_seq_len = (
                    torch.randint(3, max_seq_len - 1, (1,)).cpu().item()
                ) # max_seq_len - 1# cropped input length
                prefill_input = batch[..., :prefill_seq_len]  # cropped input

                last_token = batch[..., prefill_seq_len].cpu().unsqueeze(0)
                #torch.tensor(
                #    [[torch.argmax(ref_output.logits[:, -1, :], dim=-1).cpu()]]
                #)
                if hasattr(model, "wrapped"):
                    inputs = get_decode_input(
                        model,
                        prefill_input,
                        pad_token_id,
                        (rope_cos, rope_sin),
                        max_seq_len,
                        device,
                        helper
                    )
                    inputs = move_to_device(inputs, device)
                    output = model(**inputs)
                else:
                    input = pad_input(
                        prefill_input[..., :-1], pad_token_id, max_seq_len - 1, right = not left_pad
                    )
                    prefill_attn_mask = input != pad_token_id
                    prefill_position_ids = prefill_attn_mask.long().cumsum(-1) - 1
                    prefill_position_ids.masked_fill_(prefill_attn_mask == 0, 1)
                    prefill_ouput = model(
                        input.to(model.device), 
                        attention_mask = prefill_attn_mask.to(model.device), 
                        position_ids = prefill_position_ids.to(model.device), 
                        use_cache=True
                    )
                    next_token = prefill_input[..., -1:]
                    decode_attention_mask = torch.ones((1, max_seq_len))
                    decode_attention_mask[..., :input.shape[-1]] = input != pad_token_id
                    decode_position_ids = torch.tensor([[prefill_seq_len-1]])
                    
                    output = model(
                        next_token.to(model.device),
                        past_key_values=prefill_ouput.past_key_values,
                        attention_mask = decode_attention_mask.to(model.device),
                        position_ids = decode_position_ids.to(model.device),
                        use_cache=True,
                    )

            lm_logits = output.logits

            if torch.isfinite(lm_logits).all():
                labels = last_token[0].to(device)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    lm_logits.reshape(-1, lm_logits.size(-1)),
                    labels.view(-1),
                )
                nlls.append(loss)

            torch.cuda.empty_cache()
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl


def save_model_to(q_m, calib_inputs, args, prefill = True, kwargs=None):
    """ Save the whole model in circle format  """
    q_m.eval()
    q_m.cpu()
    save_circle_to_folder = args.output_dir
    suffix = "" if args.prefill_decode is False else "_prefill" if prefill is True else "_decode"

    save_path = pathlib.Path(save_circle_to_folder, f"model{suffix}.q.circle")
    print(f"saving the whole {'decode-' if prefill is False else 'prefill-' if args.prefill_decode else ''}model to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(q_m, (calib_inputs[0],), kwargs=kwargs, strict=False)

            cm.save(save_path)

# -----------------------------------------------------------------------------
# copied from tico/quantization/wrapq/examples/llama/quantize_decoder_layer_decode.py
# TODO reduce code duplication
# -----------------------------------------------------------------------------
def make_random_decode_batch(config, device, max_seq):
    D = config.hidden_size
    B = 1
    head_dim = getattr(config, "head_dim", D // config.num_attention_heads)
    n_kv = config.num_key_value_heads

    # Single-token hidden state.
    x = torch.randn(B, 1, D, device=device)

    # RoPE tables for the *current token* only.
    cos = torch.randn(B, 1, head_dim, device=device)
    sin = torch.randn(B, 1, head_dim, device=device)
    pos = (cos, sin)

    # Additive mask of final static width: (B, 1, MAX_SEQ)
    # Simulate that only the first L_eff positions are valid and the rest are padding.
    L_eff = torch.randint(low=1, high=max_seq + 1, size=(1,)).item()
    mask = torch.zeros(B, 1, max_seq, device=device, dtype=torch.float32)
    if L_eff < max_seq:
        mask[:, :, L_eff:] = float("-120")

    # Static-sized past KV (already RoPE-applied for past tokens).
    past_k = torch.randn(B, n_kv, max_seq - 1, head_dim, device=device)
    past_v = torch.randn(B, n_kv, max_seq - 1, head_dim, device=device)
    past = (past_k, past_v)

    return x, pos, mask, past

def save_layers_to(q_m, args, prefill = True):
    """ Save all layers of the model in circle format  """
    max_seq_len = args.max_seq_len
    save_layers_to_folder = args.output_dir
    q_m.eval()
    q_m.cpu()

    if not hasattr(q_m, "wrapped"):
        print("Saving layers currently is supported only for PTQ quantized model")
        return

    layers = q_m.wrapped.model.wrapped.layers
    config = q_m.wrapped.config
    suffix = "" if args.prefill_decode is False else "prefill_" if prefill is True else "decode_"
    for i, qlayer in enumerate(layers):
        save_path = pathlib.Path(save_layers_to_folder, f"decoder_layer_{suffix}{i}.q.circle")
        B, D = 1, config.hidden_size
        if args.prefill_decode is False:
            S = max_seq_len
            variant = "prefill"
        elif prefill is True:
            S = max_seq_len - 1
            variant = "prefill"
        else:
            # decode
            S = 1
            variant = "decode"
            
        if prefill:
            example_hidden = torch.randn(B, S, D)
            attention_mask = (
                qlayer.wrapped.causal_mask_template[..., :S, :S].squeeze(0).to("cpu")
            )
            dtype = example_hidden.dtype
            pos_embeds = qlayer.wrapped._slice_rope(
                start=0, seq_len=S, device="cpu", dtype=dtype
            )
            kwargs={"attention_mask": attention_mask, "position_embeddings": pos_embeds }
        else:
            example_hidden, pos_embeds, attention_mask, past_kv = make_random_decode_batch(config, "cpu", args.max_seq_len)
            kwargs={"attention_mask": attention_mask, "position_embeddings": pos_embeds, "past_key_value": past_kv}

        print(f"Saving {suffix}model layer_{i} to {save_path.resolve()}")
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                # Pass attention_mask and position_embeddings as inputs to avoid
                # storing them per layer and increasing model size.
                cm = tico.convert(
                    qlayer.wrapped.as_export_module(variant, return_kv=args.prefill_decode).eval(),
                    (example_hidden,),
                    kwargs=kwargs
                )
                
        cm.save(save_path)


def calibrate_ptq_observers(
    q_m: torch.nn.Module,
    calib_inputs: list[torch.Tensor],
    *,
    device: torch.device,
    decode_calibration_steps: int = 0,
    no_tqdm: bool = False,
):
    """
    Calibrate PTQ observers on prefill and optional decode paths.

    The prefill phase uses full-sequence inputs. The optional decode
    phase runs a short manual autoregressive loop with `use_cache=True`
    so cache-related observers can see realistic decode-time values as well.

    Args:
        q_m: PTQ-prepared model.
        calib_inputs: List of token tensors with shape [1, seq_len].
        device: Device used for calibration.
        decode_calibration_steps: Number of decode steps to run after each
            prefill pass. Set to 0 to disable decode calibration.
        no_tqdm: If True, disable progress bars.
    """
    q_m.eval()

    iterator = calib_inputs
    if not no_tqdm:
        iterator = tqdm.tqdm(calib_inputs, desc="PTQ calibration")

    with torch.no_grad():
        for inp in iterator:
            inp = inp.to(device)

            # Prefill calibration
            if decode_calibration_steps <= 0:
                q_m(inp)
                continue

            # Prefill with cache enabled so decode can continue from it.
            outputs = q_m(
                input_ids=inp,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Short decode calibration for cache-related observers.
            for _ in range(decode_calibration_steps):
                outputs = q_m(
                    input_ids=next_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

                # Greedy next token is enough for calibration purposes.
                next_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)


def quantize_using_PTQ(q_m, calib_inputs, args):
    """
    Wrap the model with PTQ wrappers, calibrate observers, and convert it.
    """
    print("Wrapping layers with PTQWrapper …")

    qcfg = build_llm_ptq_config(
        model_type="llama",
        num_hidden_layers=len(q_m.model.layers),
        activation_dtype=DType.int(16),
        default_qscheme=QScheme.PER_TENSOR_SYMM,
        linear_weight_bits=args.linear_weight_bits,
        embedding_weight_bits=args.embedding_weight_bits,
        lm_head_weight_bits=args.lm_head_weight_bits,
        norm_weight_dtype=DType.int(16),
        strict_wrap=True,
    )
    q_m = prepare(q_m, qcfg)

    print("Calibrating PTQ observers…")

    if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
        inject_gptq_qparams(q_m, q_m.quantizers, verbose=args.verbose)
    elif (
        hasattr(q_m, "wrapped")
        and hasattr(q_m.wrapped, "quantizers")
        and isinstance(q_m.wrapped.quantizers, dict)
    ):
        inject_gptq_qparams(q_m.wrapped, q_m.wrapped.quantizers, verbose=args.verbose)
    else:
        print(
            "[Warn] q_m.quantizers not found or not a dict; skipping GPTQ qparam injection."
        )

    device = torch.device(args.device)
    calibrate_ptq_observers(
        q_m,
        calib_inputs,
        device=device,
        decode_calibration_steps=args.decode_calibration_steps,
        no_tqdm=args.no_tqdm,
    )

    q_m = convert(q_m)
    return q_m


def evaluate(q_m, tokenizer, dataset_test, args, quantized: bool):
    # -------------------------------------------------------------------------
    # Evaluate perplexity on Wikitext-2
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl = perplexity(
        q_m, enc, args.device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    help_str = "int16" if quantized is True else "FP32"
    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ {help_str} : {ppl:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            q_m, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        print("Quantized RESULTS ARE:")
        print(make_table(results))


def get_sensitivities_info_name(model, dataset, seed, n_samples):
    """
    Build a filename for stored sensitivity calibration results.
    """
    model_name = model.config.name_or_path.replace("/", "_")

    name = (
        "."
        + "/sensitivities_for_"
        + model_name
        + "_"
        + dataset
        + "_"
        + str(n_samples)
        + "_"
        + str(seed)
        + ".pt"
    )
    return name


def get_ptq_model_name(model, args):
    """
    Build a filename for a saved PTQ checkpoint.
    """
    model_name = model.config.name_or_path.replace("/", "_")

    name = (
        f"PTQ_{model_name}_"
        + ("SpinQuant_" if args.no_spinquant is False else "")
        + ("GPTQ_" if args.no_GPTQ is False else "")
        + (f"{args.gptq_mse}_" if args.no_GPTQ is False else "")
        + str(args.nsamples_for_qcalibration)
        + "_"
        + str(args.seed)
        + ".pt"
    )
    return name

class QModelProcessor:
    """Base processor handling tokenization, GPTQ, and evaluation logic."""

    def __init__(self, model, tokenizer, args):
        """Initialize the processor with model, tokenizer, and arguments."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(args.device)
        self.args = args

    def get_tokenized_inputs(self, dataset, shuffle=True):
        """Tokenize the dataset into fixed‑length chunks for calibration."""
        text = " ".join(dataset["text"])
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokenized_inputs = []
        nsamples = self.args.nsamples_for_qcalibration
        seqlen = self.model.config.max_position_embeddings
        if shuffle is True:
            random.seed(self.args.seed)
        else:
            stride = min((ids.shape[1] - seqlen - 1) // nsamples, seqlen)
        for index in range(nsamples):
            if shuffle is True:
                i = random.randint(0, ids.shape[1] - seqlen - 1)
            else:
                i = index * stride
            j = i + seqlen
            inp = ids[:, i:j]
            tokenized_inputs.append(inp.cpu())
        return tokenized_inputs

    def run_gptq(self, calib_inputs):
        """Run GPTQ weight‑only quantization on the model using calibration inputs."""
        print("Applying GPTQ …")

        sens = None
        if self.args.gptq_mse is not None and self.args.gptq_mse == "smse":
            if self.args.sensitivity_path is not None:
                sens = torch.load(self.args.sensitivity_path)
            else:
                calibrator = SensitivityCalibrator(self.model, calib_inputs)
                sens = calibrator.compute_sensitivity_info()
                if self.args.output_dir is not None and "sensitivity" in self.args.save:
                    save_name = get_sensitivities_info_name(
                        self.model, "wikitext", self.args.seed, len(calib_inputs)
                    )
                    save_path = pathlib.Path(self.args.output_dir, save_name)
                    print(f"Saving calibrated_sensitivities to {save_path}")
                    torch.save(sens, save_path)
                    
        gptq_config = GPTQConfig(
            weight_bits=self.args.linear_weight_bits,
            perchannel=True,
            mse=self.args.gptq_mse,
            sensitivity=sens,
        )
        q_m = prepare(self.model, gptq_config, inplace=True)
        with torch.no_grad():
            for inp in calib_inputs:
                q_m(inp.to(self.device))

        q_m = convert(q_m, inplace=True)  # materialize INT-weight tensors
        return q_m

    def run_ptq(self, q_m, calib_inputs):
        assert(False)
    
    def _run_ptq(self, q_m, calib_inputs):
        q_m = quantize_using_PTQ(q_m, calib_inputs, self.args)
        if self.args.output_dir is not None and "ptq_checkpoint" in self.args.save:
            save_name = get_ptq_model_name(self.model, self.args)
            save_path = pathlib.Path(self.args.output_dir, save_name)
            print(f"Saving PTQ model to {save_path}")
            torch.save(q_m, save_path)
        return q_m
 
    def evaluate_original(self, dataset_test):
        """Evaluate the original ( model on the test dataset."""
        return evaluate(
            self.model, self.tokenizer, dataset_test, self.args, quantized=False
        )

    def evaluate_quantized(self, dataset_test):
        """Placeholder for evaluating the quantized model (implementation elsewhere)."""
        assert False

    def save_quantized(self, model, calib_inputs):
        """Placeholder for saving quantgization artifacts (implementation elsewhere)."""
        assert False


class PrefillQModelProcessor(QModelProcessor):
    """
    Processor for simple model (just-prefill-model) which doesn't use kv cache.
    """

    def __init__(self, model, tokenizer, args):
        """Initialize the prefill‑decode processor, setting up rope embeddings and handling tokenizer pad token."""
        super().__init__(model, tokenizer, args)

    def run_ptq(self, q_m, calib_inputs):
        return super()._run_ptq(q_m, calib_inputs)

    def evaluate_quantized(self, model, dataset_test):
        evaluate(model, self.tokenizer, dataset_test, self.args, quantized=True)

    def save_quantized(self, model, calib_inputs):
        if self.args.output_dir is not None and "circle_per_layer" in self.args.save:
            save_layers_to(model, self.args)

        if self.args.output_dir is not None and "circle_full" in self.args.save:
            calib_inputs = list(
                torch.stack(calib_inputs).reshape(-1, 1, self.args.max_seq_len)
            )
            save_model_to(model, calib_inputs, self.args)


class PrefillDecodeQModelProcessor(QModelProcessor):
    """
    Processor for Prefill-Decode models.
    Prefill-model computes kv-cache for the user input then each new token is produced by decode-model wit upadted kv-cache.
    """

    def __init__(self, model, tokenizer, args):
        """Initialize the prefill‑decode processor, handling tokenizer pad token and preparing rotary embeddings."""
        super().__init__(model, tokenizer, args)
        if tokenizer.pad_token is None:
            print(
                "Warning: tokenizer doesn't have pad_token. Prefill-decoding scheme may fail."
            )
            tokenizer.pad_token = tokenizer.eos_token

        rope_cos, rope_sin = _build_rope_templates_from_config(
            self.model.config,
            max_seq=self.args.max_seq_len,
            device=self.device,
            dtype=torch.float32,
        )
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin

        # debug padding

        #    inputs = tokenizer("Hello!", return_tensors="pt", max_length=args.max_seq_len - 1, padding='max_length', padding_side="right").input_ids.to(device)
        #    #inputs = tokenizer("Hello! How are you?", return_tensors="pt").input_ids.to(device)
        #    model.config.use_cache = True
        #    model.config._attn_implementation = "eager"
        #    out_ids = model.generate(inputs)
        #    output = tokenizer.decode(out_ids.squeeze(), skip_special_tokens=True)

    #        prefill_decode_ppl = evaluate_ppl_of_prefill_decode_model_on_dataset(q_m, q_m, calib_inputs, pad_token_id=tokenizer.pad_token_id, max_seq_len = args.max_seq_len, seed = args.seed, device = device)
    #
    #        print("\n┌── Wikitext-2 prefill_decode initial calibration perplexity──")
    #        print(f"│ FP32 : {prefill_decode_ppl:8.2f}")
    #        print("└───────────────────────────────────────────")

    def run_ptq(self, q_m, calib_inputs):
        """Run PTQ for the prefill‑decode pipeline, calibrating for padding and kv-cache."""
        
        # ?? pre_ptq_model = copy.deepcopy(q_m).to("cpu")  # to be used in decode quntizing

        # get prefill_model
        q_m = super()._run_ptq(q_m, calib_inputs)

        prefill_decode_ppl = evaluate_ppl_of_prefill_decode_model_on_dataset(
            q_m,
            calib_inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            max_seq_len=self.args.max_seq_len,
            seed=self.args.seed,
            device=self.device,
        )

        print(
            "\n┌── Wikitext-2 prefill_prefill train calibration perplexity ─────────────"
        )
        print(f"│ int16 : {prefill_decode_ppl:8.2f}")
        print("└───────────────────────────────────────────")

        torch.manual_seed(self.args.seed)

        return q_m

    def evaluate_original(self, dataset_test):
        """Evaluate the original (FP) model using the prefill‑decode pipeline."""
        super().evaluate_original(dataset_test)

        test_inputs = self.get_tokenized_inputs(dataset_test, shuffle=False)
        prefill_decode_ppl = evaluate_ppl_of_prefill_decode_model_on_dataset(
            self.model,
            test_inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            max_seq_len=self.args.max_seq_len,
            seed=self.args.seed,
            device=self.device,
        )

        print("\n┌── Wikitext-2 prefill_prefill original test perplexity ─────────────")
        print(f"│ FP32 : {prefill_decode_ppl:8.2f}")
        print("└───────────────────────────────────────────")

    def evaluate_quantized(self, model, dataset_test):
        """Evaluate the quantized prefill‑decode model on the test dataset."""
        
        evaluate(model, self.tokenizer, dataset_test, self.args, quantized=True)

        test_inputs = self.get_tokenized_inputs(dataset_test, shuffle=False)
        prefill_decode_ppl = evaluate_ppl_of_prefill_decode_model_on_dataset(
            model,
            test_inputs,
            pad_token_id=self.tokenizer.pad_token_id,
            max_seq_len=self.args.max_seq_len,
            seed=self.args.seed,
            device=self.device,
        )
        print("\n┌── Wikitext-2 prefill_decode quantized test perplexity ─────────────")
        print(f"│ int16 : {prefill_decode_ppl:8.2f}")
        print("└───────────────────────────────────────────")

    def save_quantized(self, model, calib_inputs):
        """Save the quantized prefill and decode models (and optionally their layers) to disk."""
        
        model.wrapped.config.use_cache = True
        if self.args.output_dir is not None and "circle_per_layer" in self.args.save:
            save_layers_to(model, self.args, prefill=True)
            save_layers_to(model, self.args, prefill=False)

        if self.args.output_dir is not None and "circle_full" in self.args.save:
            model = model.to("cpu")
            calib_inputs = list(
                torch.stack(calib_inputs).reshape(-1, 1, self.args.max_seq_len)
            )
            # save prefill model
            save_model_to(model, [calib_inputs[0][..., :-1]], self.args, prefill=True)  # seq_len = max_seq_len - 1

            # compute example input
            prefill_seq_len = (
                torch.randint(3, self.args.max_seq_len - 1, (1,)).cpu().item()
            )  # cropped input length
            prefill_input = calib_inputs[0][..., :prefill_seq_len].to(
                "cpu"
            )  # cropped input

            inputs = get_decode_input(
                model,
                prefill_input,
                self.tokenizer.pad_token_id,
                (self.rope_cos.cpu(), self.rope_sin.cpu()),
                self.args.max_seq_len,
                "cpu",
                helper=PrefillDecodeUtils(self.args.max_seq_len, model.wrapped.config, "cpu")
            )
            
            assert "input_ids" in inputs
            input = inputs.pop("input_ids")

            # save decode model
            save_model_to(model, [input], self.args, prefill=False, kwargs = inputs) # seq_len = 1


def get_qmodel_processor(model, tokenizer, args):
    if args.prefill_decode:
        return PrefillDecodeQModelProcessor(model, tokenizer, args)

    return PrefillQModelProcessor(model, tokenizer, args)


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ+PTQ pipeline (weight-only + activation)",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )
    parser.add_argument(
        "--no-tqdm", action="store_true", help="Disable tqdm progress bars."
    )
    parser.add_argument(
        "--no_GPTQ",
        action="store_true",
        default=False,
        help="Don't use GPTQ",
    )
    parser.add_argument(
        "--no_spinquant",
        action="store_true",
        default=False,
        help="Disable SpinQuant preprocessing.",
    )
    parser.add_argument(
        "--no_PTQ",
        action="store_true",
        default=False,
        help="Leave model float",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Save specified artifacts to output_dir",
    )
    parser.add_argument(
        "--save",
        nargs="*",
        type=str,
        choices=["circle_full", "circle_per_layer", "ptq_checkpoint", "sensitivity"],
        help="which artifacts should be saved to output_dir",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache_dir for using model/datasets loading",
    )
    parser.add_argument(
        "--nsamples_for_qcalibration",
        type=int,
        default="128",  # almost standard
        help="number of samples to be used in GPTQ/PTQ calibration",
    )
    parser.add_argument(
        "--linear_weight_bits",
        type=int,
        default=4,
        help="Number of bits to be used in quantizer for matmul weight quantization",
    )
    parser.add_argument(
        "--gptq_mse",
        type=str,
        default=None,
        choices=["mse", "smse"],
        help="Whether and how to use mse in gptq (none/mse/smse/)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="seq_len to use in model evaluation and conversion to circle",
    )
    parser.add_argument(
        "--calibrate_seq_len",
        type=int,
        default=2048,
        help="seq_len to use in quantized model calibration. More the better",
    )
    parser.add_argument(
        "--decode_calibration_steps",
        type=int,
        default=0,
        help=(
            "Number of short decode steps to run after each prefill calibration pass. "
            "Set to 0 to disable decode-path calibration."
        ),
    )
    parser.add_argument(
        "--embedding_weight_bits",
        type=int,
        default=8,
        help="Number of bits to be used to quantize input Embedding",
    )
    parser.add_argument(
        "--lm_head_weight_bits",
        type=int,
        default=4,
        help="Number of bits to be used to quantize lm_head",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help="tasks to be evaluated using lm_eval, e.g. `winogrande,arc_easy,arc_challenge,openbookqa,mmlu_pro,ifeval,bbh`",
    )
    parser.add_argument(
        "--sensitivity_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prefill_decode",
        action="store_true",
        default=False,
        help="Wether to use cache",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging for debugging (e.g., GPTQ injection coverage)",
    )
    args = parser.parse_args()
    print(args)

    # Basic setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print()

    # -------------------------------------------------------------------------
    # 2. Load the FP backbone and tokenizer
    # -------------------------------------------------------------------------
    print("Loading FP model …")
    dev_map = "balanced" if args.device != "cpu" else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        legacy=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        device_map=dev_map,
    ).eval()

    if not args.no_spinquant:
        print("Applying SpinQuant preprocessing …")
        model = prepare(model, SpinQuantConfig())
        model = convert(model)
    else:
        print("Skipping SpinQuant preprocessing …")

    if args.calibrate_seq_len is not None:
        model.config.max_position_embeddings = min(
            model.config.max_position_embeddings, args.calibrate_seq_len
        )

    dataset_test = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT, cache_dir=args.cache_dir
    )

    # -------------------------------------------------------------------------
    # Create a processor for the model
    # -------------------------------------------------------------------------
    qmodel_processor = get_qmodel_processor(model, tokenizer, args)

    # -------------------------------------------------------------------------
    # Compute original metrics to estimate metrics degradation
    # -------------------------------------------------------------------------
    qmodel_processor.evaluate_original(dataset_test)

    # -------------------------------------------------------------------------
    # Prepare calibration dataset
    # -------------------------------------------------------------------------
    dataset_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
    calib_txt = " ".join(dataset_train["text"])
    train_ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)
    calib_inputs = []
    nsamples = args.nsamples_for_qcalibration
    seqlen = model.config.max_position_embeddings - args.decode_calibration_steps
    if seqlen <= 0:
        raise ValueError(
            "decode_calibration_steps must be smaller than max_position_embeddings"
        )

    random.seed(args.seed)
    for _ in range(nsamples):
        i = random.randint(0, train_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = train_ids[:, i:j]
        calib_inputs.append(inp.cpu())

    # -------------------------------------------------------------------------
    # Run GPTQ (weight-only) pass
    # -------------------------------------------------------------------------
    if not args.no_GPTQ:
        q_m = qmodel_processor.run_gptq(calib_inputs)
    else:
        q_m = model

    # -------------------------------------------------------------------------
    # Wrap every layer with PTQWrapper
    # -------------------------------------------------------------------------
    if not args.no_PTQ:
        q_m = qmodel_processor.run_ptq(q_m, calib_inputs)

    # -------------------------------------------------------------------------
    # Compute quantized model metrics to estimate metrics degradation
    # -------------------------------------------------------------------------
    qmodel_processor.evaluate_quantized(q_m, dataset_test)

    # -------------------------------------------------------------------------
    # Save layers and model
    # -------------------------------------------------------------------------
    qmodel_processor.save_quantized(q_m, calib_inputs)


if __name__ == "__main__":
    main()

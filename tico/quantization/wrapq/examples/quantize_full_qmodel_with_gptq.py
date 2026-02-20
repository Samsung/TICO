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
import pathlib
import random

import types

from typing import Any, List, Optional, Tuple, Union

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import KwargsForCausalLM, LlamaForCausalLM
from transformers.processing_utils import Unpack

import tico

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from tico.utils.utils import SuppressWarning

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Hardcoded dataset settings
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


# -------------------------------------------------------------------------
# Helper — copy GPTQ (scale, zp) into PTQ observers
# -------------------------------------------------------------------------
def inject_gptq_qparams(
    root: torch.nn.Module,
    gptq_quantizers: dict[str, Any],  # {fp_name: quantizer}
    weight_obs_name: str = "weight",
):
    """
    For every `QuantModuleBase` whose `fp_name` matches a GPTQ key,
    locate the observer called `weight_obs_name` and overwrite its
    (scale, zero-point), then lock them against further updates.
    """
    for m in root.modules():
        if not isinstance(m, QuantModuleBase):
            continue
        if m.fp_name is None:
            continue
        quantizer = gptq_quantizers.get(m.fp_name)
        if quantizer is None:
            continue
        obs = m.get_observer(weight_obs_name)
        if obs is None:
            continue
        assert isinstance(obs, AffineObserverBase)
        # GPTQ quantizer attributes
        obs.load_qparams(quantizer.scale, quantizer.zero, lock=True)


# -------------------------------------------------------------------------
# Save model/layers in circle format
# -------------------------------------------------------------------------
def save_circles_to(q_m, calib_inputs, save_circle_to_folder):
    q_m.eval()
    q_m.cpu()
    save_path = pathlib.Path(save_circle_to_folder, "embedding.q.circle")
    pathlib.Path()
    print(f"saving input embedding to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(
                q_m.model.embed_tokens,
                (calib_inputs[0],),
                strict=False,
            )
            cm.save(save_path)

    save_path = pathlib.Path(save_circle_to_folder, "lm_head.q.circle")
    print(f"saving lm_head to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            B, S, D = 1, q_m.config.max_position_embeddings, q_m.config.hidden_size
            example_hidden = torch.randn(B, S, D)
            cm = tico.convert(
                q_m.lm_head,
                (example_hidden,),
                strict=False,
            )
            cm.save(save_path)

    print("saving layers")
    for i in range(len(q_m.model.layers)):
        save_path = pathlib.Path(save_circle_to_folder, f"decoder_layer_{i}.q.circle")
        print(f"saving model layer_{i} to {save_path.resolve()}")
        B, S, D = 1, q_m.config.max_position_embeddings, q_m.config.hidden_size
        example_hidden = torch.randn(B, S, D)

        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                cm = tico.convert(
                    q_m.model.layers[i],
                    (example_hidden,),
                    strict=False,
                )
        cm.save(save_path)

    save_path = pathlib.Path(save_circle_to_folder, "model.model.q.circle")
    print(f"saving model.model to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(q_m.model, (calib_inputs[0],), strict=False)

            cm.save(save_path)

    save_path = pathlib.Path(save_circle_to_folder, "model.q.circle")
    print(f"saving the whole model to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(q_m, (calib_inputs[0],), strict=False)

            cm.save(save_path)


def quantize_using_PTQ(q_m, calib_inputs, args):
    print("Wrapping layers with PTQWrapper …")

    w_cfg = {
        "mlp": {
            "gate_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "up_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "down_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
        },
        "self_attn": {
            "q_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "k_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "v_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "o_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
        },
        "input_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16)},
        },
        "post_attention_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16)},
        },
    }

    cfg = PTQConfig(
        default_dtype=DType.int(16),
        default_qscheme=QScheme.PER_TENSOR_SYMM,
        overrides={
            "model.embeddings": {
                "weight": {
                    "dtype": (
                        DType.uint(args.embedding_weight_bits)
                        if args.embedding_weight_bits < 16
                        else DType.int(args.embedding_weight_bits)
                    ),
                },
            },
            "lm_head": {
                "weight": {
                    "dtype": (
                        DType.uint(args.lm_head_weight_bits)
                        if args.lm_head_weight_bits < 16
                        else DType.int(args.lm_head_weight_bits)
                    ),
                },
            },
            "model.norm": {
                "weight": {"dtype": DType.int(16)},
            },
        },
    )
    for i in range(len(q_m.model.layers)):
        child_scope = f"layer{i}"
        cfg.overrides[child_scope] = w_cfg  # type: ignore[index]

    qcfg = cfg
    prepare(q_m, qcfg)

    # -------------------------------------------------------------------------
    # Single-pass activation calibration
    # -------------------------------------------------------------------------
    print("Calibrating PTQ obeservers…")

    # Overwrite weight observers with GPTQ statistics
    if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
        inject_gptq_qparams(q_m, q_m.quantizers)
    else:
        print(
            "[Warn] q_m.quantizers not found or not a dict; skipping GPTQ qparam injection."
        )

    device = torch.device(args.device)
    with torch.no_grad():
        for inp in tqdm.tqdm(calib_inputs):
            q_m(inp.to(device))

    # Freeze all Q-params (scale, zero-point)
    q_m = convert(q_m)

    return q_m


def fix_inputs(model, tokenizer, input_ids):
    if tokenizer.pad_token_id is not None:
        pads = torch.full(
            (
                input_ids.shape[0],
                model.config.max_position_embeddings - input_ids.shape[1],
            ),
            fill_value=tokenizer.pad_token_id,
            device=input_ids.device,
        )
    elif tokenizer.eos_token_id is not None:
        pads = torch.full(
            (
                input_ids.shape[0],
                model.config.max_position_embeddings - input_ids.shape[1],
            ),
            fill_value=tokenizer.eos_token_id,
            device=input_ids.device,
        )
    else:
        raise RuntimeError(
            "failed to pad sequence - tokenizer doesn't have pad_token_id/eos_token_id"
        )

    return torch.cat((input_ids, pads), dim=1)


class LLamaWithFixedInput(LlamaForCausalLM):
    def __init__(self, parent: LlamaForCausalLM, tokenizer):
        assert parent.config is not None, "config is a must have"
        super().__init__(parent.config)
        self.__dict__.update(parent.__dict__)

        def forward(
            self,
            input_ids: torch.LongTensor = None,  # type: ignore[assignment]
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            **kwargs: Unpack[KwargsForCausalLM],
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            # fixed input size, due to position_ids fixed
            orig_len = input_ids.shape[-1]
            input_ids = fix_inputs(self, self.tokenizer, input_ids)
            if labels is not None:
                labels = fix_inputs(self, self.tokenizer, labels)
            res = super().forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
                cache_position,
                logits_to_keep,
                **kwargs,
            )
            # we need to trim to the original size
            res.logits = res.logits[..., :orig_len, :]
            return res

        self.forward = types.MethodType(forward, self)
        self.tokenizer = tokenizer


def evaluate(q_m, tokenizer, dataset_test, args):
    # -------------------------------------------------------------------------
    # Evaluate perplexity on Wikitext-2
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_uint8 = perplexity(
        q_m, enc, args.device, stride=q_m.config.max_position_embeddings
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ int16 : {ppl_uint8:8.2f}")
    print("└───────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ+PTQ pipeline (weight-only + activation)"
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
        "--no_PTQ",
        action="store_true",
        default=False,
        help="Leave model float",
    )
    parser.add_argument(
        "--save_circle_to_folder",
        type=str,
        default=None,
        help="Save embedding/lm_head/all_layers/model.model/the_whole_model to the folder specified",
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
        action="store_true",
        default=False,
        help="Whether to use mse in gptq",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="constraint for max_position_embeddings",
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
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
        )
        .to(device)
        .eval()
    )

    model.config.use_cache = False  # TODO use args for it
    if args.max_seq_len is not None:
        model.config.max_position_embeddings = min(
            model.config.max_position_embeddings, args.max_seq_len
        )

    dataset_test = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT, cache_dir=args.cache_dir
    )

    print("\nCalculating original perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_fp32 = perplexity(
        model, enc, device, stride=model.config.max_position_embeddings
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ FP32 : {ppl_fp32:8.2f}")
    print("└───────────────────────────────────────────")

    # -------------------------------------------------------------------------
    # Prepare calibration dataset
    # -------------------------------------------------------------------------
    dataset_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
    calib_txt = " ".join(dataset_train["text"])
    train_ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)
    calib_inputs = []
    nsamples = args.nsamples_for_qcalibration
    seqlen = model.config.max_position_embeddings
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
        if not args.no_GPTQ:
            print("Applying GPTQ …")

        gptq_config = GPTQConfig(
            weight_bits=args.linear_weight_bits, perchannel=True, mse=args.gptq_mse
        )
        q_m = prepare(model, gptq_config, inplace=True)
        with torch.no_grad():
            for inp in calib_inputs:
                q_m(inp.to(args.device))

        q_m = convert(q_m, inplace=True)  # materialize INT-weight tensors
    else:
        q_m = model

    # -------------------------------------------------------------------------
    # Wrap every layer with PTQWrapper
    # -------------------------------------------------------------------------
    if not args.no_PTQ:
        q_m = quantize_using_PTQ(q_m, calib_inputs, args)

    # after PTQ quantizer only fixed-length input sequences are valid
    evaluate(LLamaWithFixedInput(q_m, tokenizer), tokenizer, dataset_test, args)

    if args.save_circle_to_folder is not None:
        save_circles_to(q_m, calib_inputs, args.save_circle_to_folder)


if __name__ == "__main__":
    main()

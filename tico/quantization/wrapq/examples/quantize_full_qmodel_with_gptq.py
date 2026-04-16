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
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.smoothquant import SmoothQuantConfig
from tico.quantization.config.spinquant import SpinQuantConfig
from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from tico.utils.utils import SuppressWarning

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

def evaluate_ppl_of_model_on_dataset(model, dataset, device: str = "cuda"):
    if hasattr(model, "device") and model.device.type != device.type:
        if hasattr(model, "to"):
            model.to(device)
    nlls = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataset):
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
                output = model(
                    batch.to(device),
                )
            else:
                raise RuntimeError("Unknown input in ppl_eval_on_dataset")

            if hasattr(output, "logits"):
                lm_logits = output.logits
            elif len(output) > 1:
                lm_logits = torch.tensor(output[0])
            else:
                lm_logits = torch.tensor(output)

            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                if isinstance(batch, torch.Tensor):
                    shift_labels = batch[:, 1:].contiguous()
                else:
                    assert isinstance(batch, tuple)
                    shift_labels = batch[0][:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nlls.append(loss)
                del shift_logits, shift_labels
                shift_logits = shift_labels = None  # type: ignore[assignment]

            del batch, lm_logits, output
            lm_logits = output = batch = None  # noqa: F841
            torch.cuda.empty_cache()

    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl

# -------------------------------------------------------------------------
# Save model/layers in circle format
# -------------------------------------------------------------------------
def save_model_to(q_m, calib_inputs, save_circle_to_folder):
    q_m.eval()
    q_m.cpu()
    #  save_path = pathlib.Path(save_circle_to_folder, "embedding.q.circle")
    #  pathlib.Path()
    #  print(f"saving input embedding to {save_path.resolve()}")
    #  with torch.no_grad():
    #      with SuppressWarning(UserWarning, ".*"):
    #          cm = tico.convert(
    #              q_m.model.embed_tokens,
    #              (calib_inputs[0],),
    #              strict=False,
    #          )
    #          cm.save(save_path)
    #
    #  save_path = pathlib.Path(save_circle_to_folder, "lm_head.q.circle")
    #  print(f"saving lm_head to {save_path.resolve()}")
    #  with torch.no_grad():
    #      with SuppressWarning(UserWarning, ".*"):
    #          B, S, D = 1, q_m.config.max_position_embeddings, q_m.config.hidden_size
    #          example_hidden = torch.randn(B, S, D)
    #          cm = tico.convert(
    #              q_m.lm_head,
    #              (example_hidden,),
    #              strict=False,
    #          )
    #          cm.save(save_path)
    #
    
    save_path = pathlib.Path(save_circle_to_folder, "model.q.circle")
    print(f"saving the whole model to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(q_m, (calib_inputs[0],), strict=False)

            cm.save(save_path)


def save_layers_to(q_m, max_seq_len, save_layers_to_folder):
    q_m.eval()
    q_m.cpu()

    if not hasattr(q_m, "wrapped"):
        print("Saving layers currently is supported only for PTQ quantized model")
        return

    layers = q_m.wrapped.model.wrapped.layers
    config = q_m.wrapped.config
    for i, qlayer in enumerate(layers):
        save_path = pathlib.Path(save_layers_to_folder, f"decoder_layer_{i}.q.circle")
        B, S, D = 1, max_seq_len, config.hidden_size
        example_hidden = torch.randn(B, S, D)

        attention_mask = (
            qlayer.wrapped.causal_mask_template[..., :S, :S].squeeze(0).to("cpu")
        )
        dtype = example_hidden.dtype
        pos_embeds = qlayer.wrapped._slice_rope(
            start=0, seq_len=S, device="cpu", dtype=dtype
        )

        print(f"Saving model layer_{i} to {save_path.resolve()}")
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                # Pass attention_mask and position_embeddings as inputs to avoid
                # storing them per layer and increasing model size.
                cm = tico.convert(
                    qlayer.wrapped.as_export_module("prefill").eval(),
                    (example_hidden,),
                    kwargs={
                        "attention_mask": attention_mask,
                        "position_embeddings": pos_embeds,
                    },
                )
        cm.save(save_path)


def quantize_using_PTQ(q_m, calib_inputs, args):
    print("Wrapping layers with PTQWrapper …")

    linear_observer = (
        MinMaxObserver
        if args.linear_io_qdtype == "int16"
        else MXObserver if args.linear_io_qdtype == "mxint8" else None
    )
    w_cfg = {
        "mlp": {
            "gate_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
            "up_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
            "down_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
        },
        "self_attn": {
            "q_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
            "k_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
            "v_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
            "o_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": linear_observer},
                "act_out": {"observer": linear_observer},
            },
            "scale": {"observer": MinMaxObserver},
            "mask_add": {"observer": MinMaxObserver},
            "softmax": {"observer": MinMaxObserver},
            "logits_raw": {"observer": linear_observer},
        },
        "self_attn_residual_act_out": {"observer": MinMaxObserver},
        # "act_last_residual_out" : {"observer":MinMaxObserver},
        "input_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16), "observer": MinMaxObserver},
            "act_in": {"observer": MinMaxObserver},
            "act_out": {"observer": MinMaxObserver},
        },
        "post_attention_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16), "observer": MinMaxObserver},
            "act_in": {"observer": MinMaxObserver},
            "act_out": {"observer": MinMaxObserver},
        },
    }

    default_observer = (
        MinMaxObserver
        if args.default_io_qdtype == "int16"
        else MXObserver if args.linear_io_qdtype == "mxint8" else None
    )
    cfg = PTQConfig(
        default_dtype=DType.int(16),
        default_qscheme=QScheme.PER_TENSOR_SYMM,
        default_observer=default_observer,  # type: ignore[arg-type]
        overrides={
            "model": {
                "embed_tokens": {
                    "weight": {
                        "dtype": (
                            DType.uint(args.embedding_weight_bits)
                            if args.embedding_weight_bits < 16
                            else DType.int(args.embedding_weight_bits)
                        ),
                        "observer": MinMaxObserver,
                    },
                },
                "layers": {},
                "norm": {
                    "weight": {"dtype": DType.int(16)},
                },
                "act_out": {"observer": MinMaxObserver},
            },  # embeddings to 8-bits
            "lm_head": {
                "weight": {
                    "dtype": (
                        DType.uint(args.lm_head_weight_bits)
                        if args.lm_head_weight_bits < 16
                        else DType.int(args.lm_head_weight_bits)
                    ),
                    "observer": MinMaxObserver,
                },
                "act_in": {"observer": MinMaxObserver},
                "act_out": {"observer": MinMaxObserver},
            },
            "model.norm": {
                "weight": {"dtype": DType.int(16), "observer": MinMaxObserver},
                "act_in": {"observer": MinMaxObserver},
                "act_out": {"observer": MinMaxObserver},
            },
        },
    )
    for i in range(len(q_m.model.layers)):
        child_scope = f"{i}"
        cfg.overrides["model"]["layers"][child_scope] = w_cfg  # type: ignore[index]

    if args.default_io_qdtype != "float32":
        # hack to keep model.norm in `int16`
        cfg.overrides["model"]["layers"][f"{len(q_m.model.layers) - 1}"]["act_mlp_residual_out"] = {  # type: ignore[index]
            "observer": default_observer
        }
    qcfg = cfg
    q_m = prepare(q_m, qcfg)
    torch.cuda.empty_cache()
    # -------------------------------------------------------------------------
    # Single-pass activation calibration
    # -------------------------------------------------------------------------
    print("Calibrating PTQ obeservers…")

    # Overwrite weight observers with GPTQ statistics
    if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
        inject_gptq_qparams(q_m, q_m.quantizers)
    elif (
        hasattr(q_m, "wrapped")
        and hasattr(q_m.wrapped, "quantizers")
        and isinstance(q_m.wrapped.quantizers, dict)
    ):
        inject_gptq_qparams(q_m.wrapped, q_m.wrapped.quantizers)
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
    torch.cuda.empty_cache()
    
    return q_m


def evaluate(q_m, tokenizer, dataset_test, args):
    # -------------------------------------------------------------------------
    # Evaluate perplexity on Wikitext-2
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_uint8 = perplexity(
        q_m, enc, args.device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ {args.default_io_qdtype} : {ppl_uint8:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            q_m, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        print("Quantized RESULTS ARE:")
        print(make_table(results))

    # to prevent export errors let's evaluate ppl on exported fake_quantized model
    with torch.no_grad():
        q_m.eval()
        q_m.cpu()
        test_ids = enc.input_ids[0]
        test_ids_batch = []
        if hasattr(q_m, "config"):
            assert hasattr(q_m, "config")
            model_config = q_m.config
        else:
            assert hasattr(q_m.wrapped, "config")
            model_config = q_m.wrapped.config
        if hasattr(model_config, "text_config"):
            model_config = model_config.text_config
        assert hasattr(model_config, "max_position_embeddings")
        assert isinstance(model_config.max_position_embeddings, int)
        max_length = model_config.max_position_embeddings
        nsamples = test_ids.numel() // max_length

        for i in range(nsamples):
            batch = test_ids[(i * max_length) : ((i + 1) * max_length)]  # noqa E203
            test_ids_batch.append(batch.unsqueeze(0))

        rnd_input = torch.randint_like(
            test_ids_batch[0], 0, tokenizer.vocab_size - 1
        )  # just random ids
        device = "cuda"
        exported_program = torch.export.export(
            q_m.to(device),
            (rnd_input.to(device),),
            kwargs=None,
            dynamic_shapes=None,
            strict=False,
        )
        ppl = evaluate_ppl_of_model_on_dataset(
            exported_program.module(), test_ids_batch, device=device
        )
        print("\n┌── Wikitext-2 test perplexity ─────────────")
        print(f"│ exported_{args.default_io_qdtype} : {ppl:8.2f}")
        print("└───────────────────────────────────────────")


def get_sensitivities_info_name(model, dataset, seed, n_samples):
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
        "--use-cache",
        dest="use_cache",
        action="store_true",
        default=False,
        help="Use model KV cache if enabled (off by default).",
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
        "--no_SMOOTHQUANT",
        action="store_true",
        default=False,
        help="Don't use smoothquant",
    )
    parser.add_argument(
        "--smoothquant_alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for smoothquant%",
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
        "--default_io_qdtype",
        type=str,
        default="int16",
        help="which activation types are supposed as default for PTQ (`int16`/`mxint8` are supported for now)",
    )
    parser.add_argument(
        "--linear_io_qdtype",
        type=str,
        default="int16",
        help="which activation types are supposed for matmuls for PTQ (`int16`/`mxint8` are supported for now)",
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
        choices=["mse", "smse", "smse_for_gptq", "mse_for_gptq"],
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
    print(f"Use HF cache?    : {args.use_cache}")
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

    model.config.use_cache = False  # TODO use args for it
    if args.calibrate_seq_len is not None:
        model.config.max_position_embeddings = min(
            model.config.max_position_embeddings, args.calibrate_seq_len
        )

    dataset_test = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT)

    print("\nCalculating original perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_fp32 = perplexity(
        model, enc, device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ FP32 : {ppl_fp32:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            model, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        print("Original RESULTS ARE:")
        print(make_table(results))

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
        
    train_ppl_fp32 = evaluate_ppl_of_model_on_dataset(
            model, calib_inputs, device=device
        )
    print("\n┌── Wikitext-2 train perplexity ─────────────")
    print(f"│ FP32 : {train_ppl_fp32:8.2f}")
    print("└───────────────────────────────────────────")
    
    if False:#not args.no_SMOOTHQUANT:
        print("Applying SmoothQuant …")
        # attach observers
        model = prepare(model, SmoothQuantConfig(alpha=args.smoothquant_alpha))

        # run calibration
        for inp in calib_inputs:
            model(inp.to(args.device))

        # apply smoothing
        q_m = convert(model)
    else:
        q_m = model

    # -------------------------------------------------------------------------
    # Run GPTQ (weight-only) pass
    # -------------------------------------------------------------------------
    if not args.no_GPTQ:
        print("Applying GPTQ …")

        sens = None
        if args.gptq_mse is not None and (
            args.gptq_mse == "smse" or args.gptq_mse == "smse_for_gptq"
        ):
            if args.sensitivity_path is not None:
                sens = torch.load(args.sensitivity_path)
            else:
                calibrator = SensitivityCalibrator(model, calib_inputs)
                sens = calibrator.compute_sensitivity_info()
                if args.output_dir is not None and args.save is not None and "sensitivity" in args.save:
                    save_name = get_sensitivities_info_name(
                        model, "wikitext", args.seed, len(calib_inputs)
                    )
                    save_path = pathlib.Path(args.output_dir, save_name)
                    print(f"Saving calibrated_sensitivities to {save_path}")
                    torch.save(sens, save_path)

        gptq_config = GPTQConfig(
            weight_bits=args.linear_weight_bits,
            perchannel=True,
            mse=args.gptq_mse,
            sensitivity=sens,
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

        if args.output_dir is not None and  args.save is not None and "ptq_checkpoint" in args.save:
            save_name = get_ptq_model_name(model, args)
            save_path = pathlib.Path(args.output_dir, save_name)
            print(f"Saving PTQ model to {save_path}")
            torch.save(q_m, save_path)

    train_ppl_ioqdtype = evaluate_ppl_of_model_on_dataset(
            q_m, calib_inputs, device=device
        )
    print("\n┌── Wikitext-2 train perplexity ─────────────")
    print(f"│ {args.default_io_qdtype} : {train_ppl_ioqdtype:8.2f}")
    print("└───────────────────────────────────────────")

    

    # after PTQ quantizer only fixed-length input sequences are valid
    evaluate(q_m, tokenizer, dataset_test, args)

    if args.output_dir is not None and args.save is not None and "circle_per_layer" in args.save:
        save_layers_to(q_m, args.max_seq_len, args.output_dir)

    if args.output_dir is not None and args.save is not None and "circle_full" in args.save:
        calib_inputs = list(torch.stack(calib_inputs).reshape(-1, 1, args.max_seq_len))
        save_model_to(q_m, calib_inputs, args.output_dir)


if __name__ == "__main__":
    main()

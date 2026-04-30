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
from typing import Any

import torch
import tqdm
from datasets import load_dataset
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer

import tico
from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.builders import build_llm_ptq_config
from tico.quantization.config.cle import CLEConfig
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.spinquant import SpinQuantConfig
from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
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


def parse_args():
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
        "--enable_CLE",
        action="store_true",
        help="Enable Cross-Layer Equalization preprocessing.",
    )
    parser.add_argument(
        "--cle_pairs",
        nargs="+",
        default=[
            "model.layers.*.mlp.up_proj:model.layers.*.mlp.down_proj",
        ],
        help=(
            "Manual CLE layer pairs. Each pair must be formatted as "
            "`first_layer:second_layer`. Exact names and wildcard patterns are supported. "
            "Example: `model.layers.*.mlp.up_proj:model.layers.*.mlp.down_proj`."
        ),
    )
    parser.add_argument(
        "--cle_method",
        choices=["absmax", "range"],
        default="absmax",
        help="Range method used for Cross-Layer Equalization.",
    )
    parser.add_argument(
        "--cle_max_iter",
        type=int,
        default=1,
        help="Number of CLE iterations.",
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
        "--verbose",
        action="store_true",
        help="Verbose logging for debugging (e.g., GPTQ injection coverage)",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Pad input tensor to a maximum sequence length using the specified pad token.
# -------------------------------------------------------------------------
def pad_input(input, pad_token, max_seq_len):
    """Pad a tensor to a maximum sequence length using the specified pad token."""
    pads = torch.full(
        (input.shape[0], max_seq_len - input.shape[1]),
        fill_value=pad_token,
        device=input.device,
    )

    res = torch.cat((input, pads), dim=1)

    return res


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
        seen.add(m.fp_name)

    unused = set(gptq_quantizers.keys()) - seen

    if verbose:
        print("\n[GPTQ → PTQ injection summary]")
        print(f"  matched : {len(seen)}")
        print(f"  missed  : {len(missed_modules)}")
        print(f"  unused  : {len(unused)}")

        # Print samples (not all, to avoid spam)
        def _print_sample(title, items):
            items = list(items)
            if not items:
                return
            print(f"\n  {title}:")
            for name in items[:10]:
                print(f"    - {name}")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")

        _print_sample("missed modules", missed_modules)
        _print_sample("unused GPTQ entries", unused)


def parse_cle_pairs(raw_pairs: list[str] | None) -> list[tuple[str, str]]:
    """
    Parse command-line CLE pairs.

    Each pair must be formatted as `first_layer:second_layer`.
    Both exact module names and wildcard patterns are supported.

    Examples:
        model.layers.*.mlp.up_proj:model.layers.*.mlp.down_proj
        model.layers.0.mlp.up_proj:model.layers.0.mlp.down_proj
    """
    if raw_pairs is None:
        return []

    pairs = []
    for raw_pair in raw_pairs:
        if ":" not in raw_pair:
            raise ValueError(
                "Each CLE pair must be formatted as `first_layer:second_layer`. "
                f"Got: {raw_pair}"
            )

        first_name, second_name = raw_pair.split(":", maxsplit=1)
        first_name = first_name.strip()
        second_name = second_name.strip()

        if not first_name or not second_name:
            raise ValueError(f"Invalid CLE pair: {raw_pair}")

        pairs.append((first_name, second_name))

    return pairs


def save_model_to(
    q_m, calib_input, save_circle_to_folder, prefill_decode: bool = False
):
    """
    Export and save the whole quantized model in circle format.
    """
    q_m.eval()
    q_m.cpu()
    suffix = "prefill" if prefill_decode else ""
    model_name = "model_prefill" if prefill_decode else "model"
    save_path = pathlib.Path(save_circle_to_folder, f"{model_name}.q.circle")
    print(f"saving the whole {model_name} to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(
                q_m.wrapped.as_export_module(
                    "prefill", return_kv=prefill_decode
                ).eval(),
                (calib_input,),
                strict=False,
            )
            cm.save(save_path)

    if prefill_decode is True:
        model_name = f"model_decode"
        save_path = pathlib.Path(save_circle_to_folder, f"{model_name}.q.circle")
        print(f"saving the whole {model_name} to {save_path.resolve()}")
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                token = torch.Tensor([[calib_input[..., 0]]], device="cpu").to(
                    dtype=calib_input.dtype
                )  # no matter which token

                config = q_m.wrapped.config
                D = config.hidden_size
                head_dim = getattr(config, "head_dim", D // config.num_attention_heads)
                n_kv = config.num_key_value_heads
                max_seq_len = calib_input.shape[-1]
                past_kv = [
                    (
                        torch.randn(1, n_kv, max_seq_len - 1, head_dim, device="cpu"),
                        torch.randn(1, n_kv, max_seq_len - 1, head_dim, device="cpu"),
                    )
                    for _ in range(config.num_hidden_layers)
                ]
                cm = tico.convert(
                    q_m.wrapped.as_export_module("decode").eval(),
                    (token, past_kv),
                    strict=False,
                )
                cm.save(save_path)


# -----------------------------------------------------------------------------
# copied from quantize_decoder_layer_decode.py
# -----------------------------------------------------------------------------
def make_random_decode_batch(model, B, DEVICE, MAX_SEQ):
    # TODO reduce code duplication
    D = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", D // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads

    # Single-token hidden state.
    x = torch.randn(B, 1, D, device=DEVICE)

    # RoPE tables for the *current token* only.
    cos = torch.randn(B, 1, head_dim, device=DEVICE)
    sin = torch.randn(B, 1, head_dim, device=DEVICE)
    pos = (cos, sin)

    # Additive mask of final static width: (B, 1, MAX_SEQ)
    # Simulate that only the first L_eff positions are valid and the rest are padding.
    L_eff = torch.randint(low=1, high=MAX_SEQ + 1, size=(1,)).item()
    mask = torch.zeros(B, 1, MAX_SEQ, device=DEVICE, dtype=torch.float32)
    if L_eff < MAX_SEQ:
        mask[:, :, L_eff:] = float("-120")

    # Static-sized past KV (already RoPE-applied for past tokens).
    past_k = torch.randn(B, n_kv, MAX_SEQ - 1, head_dim, device=DEVICE)
    past_v = torch.randn(B, n_kv, MAX_SEQ - 1, head_dim, device=DEVICE)
    past = (past_k, past_v)

    return x, pos, mask, past


def save_layers_to(
    q_m, max_seq_len, save_layers_to_folder, prefill_decode: bool = False
):
    """
    Export and save quantized decoder layers one by one in circle format.
    """
    q_m.eval()
    q_m.cpu()

    if not hasattr(q_m, "wrapped"):
        print("Saving layers currently is supported only for PTQ quantized model")
        return

    layers = q_m.wrapped.model.wrapped.layers
    config = q_m.wrapped.config
    for i, qlayer in enumerate(layers):
        suffix = "prefill_" if prefill_decode else ""
        layer_name = f"decoder_layer_{suffix}{i}"
        save_path = pathlib.Path(save_layers_to_folder, f"{layer_name}.q.circle")
        B, S, D = 1, max_seq_len, config.hidden_size
        example_hidden = torch.randn(B, S, D)

        attention_mask = (
            qlayer.wrapped.causal_mask_template[..., :S, :S].squeeze(0).to("cpu")
        )
        dtype = example_hidden.dtype
        pos_embeds = qlayer.wrapped._slice_rope(
            start=0, seq_len=S, device="cpu", dtype=dtype
        )

        print(f"Saving {layer_name} to {save_path.resolve()}")
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                # Pass attention_mask and position_embeddings as inputs to avoid
                # storing them per layer and increasing model size.
                cm = tico.convert(
                    qlayer.wrapped.as_export_module(
                        "prefill", return_kv=prefill_decode
                    ).eval(),
                    (example_hidden,),
                    kwargs={
                        "attention_mask": attention_mask,
                        "position_embeddings": pos_embeds,
                    },
                )
        cm.save(save_path)

        if prefill_decode is True:
            layer_name = f"decoder_layer_decode_{i}"
            save_path = pathlib.Path(save_layers_to_folder, f"{layer_name}.q.circle")
            print(f"Saving {layer_name} to {save_path.resolve()}")
            with torch.no_grad():
                with SuppressWarning(UserWarning, ".*"):
                    ex_hid, pos_embeds, attn_mask, past = make_random_decode_batch(
                        q_m.wrapped, B=1, DEVICE="cpu", MAX_SEQ=max_seq_len
                    )
                    cm = tico.convert(
                        qlayer.wrapped.as_export_module("decode").eval(),
                        (ex_hid,),  # hidden_states
                        {
                            "attention_mask": attn_mask,
                            "past_key_value": past,
                            "position_embeddings": pos_embeds,
                        },
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
    if args.no_PTQ:
        return q_m

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


def evaluate(q_m, tokenizer, dataset_test, args):
    """
    Evaluate the quantized model with perplexity and optional lm-eval tasks.
    """
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_uint8 = perplexity(
        q_m, enc, args.device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ int16 : {ppl_uint8:8.2f}")
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
        + ("CLE_" if args.enable_CLE else "")
        + ("GPTQ_" if args.no_GPTQ is False else "")
        + (f"{args.gptq_mse}_" if args.no_GPTQ is False else "")
        + str(args.nsamples_for_qcalibration)
        + "_"
        + str(args.seed)
        + ".pt"
    )
    return name


def should_save(args, artifact: str) -> bool:
    """
    Return True when a specific artifact should be saved.
    """
    return (
        args.output_dir is not None and args.save is not None and artifact in args.save
    )


def setup_runtime(args) -> tuple[torch.device, torch.dtype]:
    """
    Initialize deterministic settings and resolve runtime device / dtype.
    """
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    return device, dtype


def print_config(args, device: torch.device) -> None:
    """
    Print the effective high-level runtime configuration.
    """
    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print()


def load_model_and_tokenizer(args, dtype: torch.dtype):
    """
    Load the floating-point model backbone and tokenizer.
    """
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

    return model, tokenizer


def apply_spinquant(model, args):
    """
    Optionally apply SpinQuant preprocessing.
    """
    if args.no_spinquant:
        print("Skipping SpinQuant preprocessing …")
        return model

    print("Applying SpinQuant preprocessing …")
    model = prepare(model, SpinQuantConfig())
    return convert(model)


def apply_cle(model, args):
    """
    Optionally apply Cross-Layer Equalization preprocessing.
    """
    if not args.enable_CLE:
        print("Skipping Cross-Layer Equalization preprocessing …")
        return model

    cle_pairs = parse_cle_pairs(args.cle_pairs)
    if not cle_pairs:
        raise ValueError(
            "CLE is enabled, but no CLE pairs were provided. "
            "Pass pairs with `--cle_pairs first_layer:second_layer ...`."
        )

    print("Applying Cross-Layer Equalization preprocessing …")
    cle_config = CLEConfig(
        pairs=cle_pairs,
        method=args.cle_method,
        max_iter=args.cle_max_iter,
        show_progress=not args.no_tqdm,
    )
    model = prepare(model, cle_config)
    return convert(model)


def configure_max_position_embeddings(model, args) -> None:
    """
    Clamp model max_position_embeddings when a calibration sequence length is set.
    """
    if args.calibrate_seq_len is None:
        return

    model.config.max_position_embeddings = min(
        model.config.max_position_embeddings,
        args.calibrate_seq_len,
    )


def load_eval_dataset(args):
    """
    Load the fixed Wikitext evaluation split.
    """
    return load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=TEST_SPLIT,
        cache_dir=args.cache_dir,
    )


def evaluate_original_model(
    model, tokenizer, dataset_test, args, device: torch.device
) -> None:
    """
    Evaluate the original floating-point model before quantization.
    """
    print("\nCalculating original perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_fp32 = perplexity(
        model,
        enc,
        device,
        max_length=args.max_seq_len,
        stride=args.max_seq_len,
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ FP32 : {ppl_fp32:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            model,
            tokenizer,
            args.eval_tasks,
            max_length=args.max_seq_len,
        )
        print("Original RESULTS ARE:")
        print(make_table(results))


def build_calibration_inputs(
    model, tokenizer, args, device: torch.device
) -> list[torch.Tensor]:
    """
    Build random fixed-length calibration samples from the Wikitext train split.
    """
    dataset_train = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=TRAIN_SPLIT,
        cache_dir=args.cache_dir,
    )
    calib_txt = " ".join(dataset_train["text"])
    train_ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)

    nsamples = args.nsamples_for_qcalibration
    seqlen = model.config.max_position_embeddings - args.decode_calibration_steps
    if seqlen <= 0:
        raise ValueError(
            "decode_calibration_steps must be smaller than max_position_embeddings"
        )

    random.seed(args.seed)
    calib_inputs = []
    for _ in range(nsamples):
        i = random.randint(0, train_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        calib_inputs.append(train_ids[:, i:j].cpu())

    return calib_inputs


def compute_or_load_sensitivity(model, calib_inputs, args):
    """
    Load or compute sensitivity information for SMSE GPTQ.
    """
    if args.gptq_mse != "smse":
        return None

    if args.sensitivity_path is not None:
        return torch.load(args.sensitivity_path)

    calibrator = SensitivityCalibrator(model, calib_inputs)
    sens = calibrator.compute_sensitivity_info()

    if should_save(args, "sensitivity"):
        save_name = get_sensitivities_info_name(
            model,
            "wikitext",
            args.seed,
            len(calib_inputs),
        )
        save_path = pathlib.Path(args.output_dir, save_name)
        print(f"Saving calibrated_sensitivities to {save_path}")
        torch.save(sens, save_path)

    return sens


def quantize_using_GPTQ(model, calib_inputs, args):
    """
    Run the optional GPTQ weight-only quantization pass.
    """
    if args.no_GPTQ:
        return model

    print("Applying GPTQ …")

    # use_cache increases VRAM usage significantly, but GPTQ does not use decoding.
    prev_use_cache = model.config.use_cache
    model.config.use_cache = False

    sens = compute_or_load_sensitivity(model, calib_inputs, args)
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

    q_m = convert(q_m, inplace=True)
    model.config.use_cache = prev_use_cache
    return q_m


def save_ptq_checkpoint(q_m, model, args) -> None:
    """
    Save the PTQ checkpoint when requested by CLI flags.
    """
    if not should_save(args, "ptq_checkpoint"):
        return

    save_name = get_ptq_model_name(model, args)
    save_path = pathlib.Path(args.output_dir, save_name)
    print(f"Saving PTQ model to {save_path}")
    torch.save(q_m, save_path)


def save_requested_artifacts(q_m, tokenizer, calib_inputs, args) -> None:
    """
    Save requested Circle artifacts after final evaluation.
    """
    if should_save(args, "circle_per_layer"):
        save_layers_to(
            q_m,
            args.max_seq_len,
            args.output_dir,
            prefill_decode=args.decode_calibration_steps != 0,
        )

    if should_save(args, "circle_full"):
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        calib_input = pad_input(
            calib_inputs[0], pad_token_id, max_seq_len=args.max_seq_len
        )
        save_model_to(
            q_m,
            calib_input,
            args.output_dir,
            prefill_decode=args.decode_calibration_steps != 0,
        )


def run_pipeline(args) -> None:
    """
    Run the full Llama GPTQ + PTQ quantization pipeline.
    """
    print(args)

    device, dtype = setup_runtime(args)
    print_config(args, device)

    model, tokenizer = load_model_and_tokenizer(args, dtype)
    model = apply_spinquant(model, args)
    model = apply_cle(model, args)
    configure_max_position_embeddings(model, args)

    dataset_test = load_eval_dataset(args)
    evaluate_original_model(model, tokenizer, dataset_test, args, device)

    calib_inputs = build_calibration_inputs(model, tokenizer, args, device)
    q_m = quantize_using_GPTQ(model, calib_inputs, args)
    q_m = quantize_using_PTQ(q_m, calib_inputs, args)
    save_ptq_checkpoint(q_m, model, args)

    evaluate(q_m, tokenizer, dataset_test, args)
    save_requested_artifacts(q_m, tokenizer, calib_inputs, args)


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()

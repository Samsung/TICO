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
#   7. Save model/layers (optional).
#
# Llama attention execution profiles
# -----------------------------------------------------------------------------
#   --profile npu_export
#       Preserves the current NPU-export-oriented attention graph.
#
#   --profile reference_eval
#       Uses a Hugging Face-like attention path that is better suited for quick
#       GPU evaluation and regression checks. Circle export is intentionally
#       restricted to npu_export in this example.
# =============================================================================

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pathlib
import random
from typing import Any, Optional

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
from tico.quantization.config.cle import CLEConfig
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.llama_attention import (
    DEFAULT_EXECUTION_PROFILE,
    SUPPORTED_EXECUTION_PROFILES,
)
from tico.quantization.config.specs import affine
from tico.quantization.config.spinquant import SpinQuantConfig
from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaLMHeadExportAdapter,
    LlamaTokenEmbeddingExportAdapter,
    make_token_embedding_dynamic_shapes,
    make_token_embedding_example_input,
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from tico.utils.utils import SuppressWarning

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}

_SUPPORTED_WEIGHT_BITS = (4, 8, 16)


def _weight_dtype_from_bits(bits: int) -> DType:
    """Return the PTQ weight dtype for a supported bit-width."""
    if bits in (4, 8):
        return DType.uint(bits)
    if bits == 16:
        return DType.int(bits)

    raise ValueError(
        f"Unsupported weight bit-width: {bits}. "
        f"Expected one of {_SUPPORTED_WEIGHT_BITS}."
    )


# Hardcoded dataset settings
DATASET_NAME = "Salesforce/wikitext"
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
        "--gptq_lm_head",
        action="store_true",
        default=False,
        help=(
            "Apply GPTQ to lm_head. Disabled by default because "
            "lm_head.weight can be tied with the input embedding table."
        ),
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
        default=128,  # almost standard
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
        help="Whether and how to use mse in gptq (none/mse/smse/smse_for_gptq/mse_for_gptq)",
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
        default=8,
        help=(
            "Number of bits to be used to quantize lm_head."
            "For tied embedding/lm_head it must be the same as embedding_weight_bits."
        ),
    )
    parser.add_argument(
        "--spin_rotation_weight_bits",
        type=int,
        default=16,
        help=(
            "Number of bits to be used to quantize SpinLlama rotation weights "
            "created by SpinQuant, namely model.rotate_embedding.weight and "
            "rotate_lm_head.weight. This option is used only when SpinQuant is enabled."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=list(SUPPORTED_EXECUTION_PROFILES),
        default=DEFAULT_EXECUTION_PROFILE,
        help=(
            "Use 'reference_eval' for a GPU-friendly, HF-like attention path. "
            "Use 'npu_export' for the NPU-export-oriented attention graph."
        ),
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
    parser.add_argument(
        "--gptq_use_orig_model_inference",
        action="store_true",
        default=False,
        help="Run inputs for the next layer on original model to stabilize GPTQ",
    )
    parser.add_argument(
        "--gptq_percdamp",
        type=float,
        default=0.01,
        help="Dampening parameter to be used in GPTQ. It helps to avoid degenerate,"
        "ill-conditioned matrices and serve as a tradeoff between GPTQ and ordinary min-max quantizer.",
    )
    return parser.parse_args()


# -------------------------------------------------------------------------
# Pad input tensor to a maximum sequence length using the specified pad token.
# -------------------------------------------------------------------------
def pad_input(input, pad_token, max_seq_len):
    """Pad a tensor to a maximum sequence length using the specified pad token."""

    if input.shape[1] > max_seq_len:
        input = input[:, :max_seq_len]

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


# -------------------------------------------------------------------------
# Helper — clear gptq quantizers after injection
# -------------------------------------------------------------------------
def clear_gptq_quantizers(model: torch.nn.Module) -> None:
    """Remove GPTQ quantizer attributes from the model to free memory.

    This helper clears the ``quantizers`` attribute from both the top-level model
    and, if present, from the wrapped sub‑model. It is typically called after
    GPTQ quantizers injection is complete and the quantizers are no longer needed.
    """
    if hasattr(model, "quantizers"):
        delattr(model, "quantizers")
    if hasattr(model, "wrapped") and hasattr(model.wrapped, "quantizers"):
        delattr(model.wrapped, "quantizers")


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


def _weights_share_storage(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    """Return True if two weight tensors share the exact same storage slice."""
    if left is right:
        return True

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False

    if left.device != right.device:
        return False

    if left.device.type == "meta" or right.device.type == "meta":
        return False

    if left.numel() == 0 or right.numel() == 0:
        return False

    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
    )


def has_tied_input_output_embeddings(model: torch.nn.Module) -> bool:
    """Return True if the input embedding and LM head weights are tied."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)

    if not callable(get_input_embeddings) or not callable(get_output_embeddings):
        return False

    input_embeddings = get_input_embeddings()
    output_embeddings = get_output_embeddings()

    if input_embeddings is None or output_embeddings is None:
        return False

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)

    if input_weight is None or output_weight is None:
        return False

    return _weights_share_storage(input_weight, output_weight)


def validate_tied_embedding_weight_bits(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> None:
    """
    Reject different embedding and LM head bit-widths for tied weights.

    Args:
        model: Model whose input embedding and output projection are inspected.
        args: Parsed command-line arguments.

    Raises:
        ValueError: If the model ties input embedding and LM head weights while
            `--embedding_weight_bits` and `--lm_head_weight_bits` differ.
    """
    if args.embedding_weight_bits == args.lm_head_weight_bits:
        return

    if not has_tied_input_output_embeddings(model):
        return

    raise ValueError(
        "Cannot use different bit-widths for tied input embedding and lm_head "
        "weights: "
        f"--embedding_weight_bits={args.embedding_weight_bits}, "
        f"--lm_head_weight_bits={args.lm_head_weight_bits}. "
        "Set both options to the same value or use a model with untied "
        "input/output embeddings."
    )


def build_gptq_config(
    args,
    sensitivity: dict[str, torch.Tensor] | None = None,
) -> GPTQConfig:
    """
    Build a GPTQ configuration from command-line arguments.

    GPTQ for lm_head is disabled by default because many causal language models
    tie `lm_head.weight` with the input embedding table. Users can enable it
    explicitly with `--gptq_lm_head`.
    """
    weight_bits_overrides: dict[str, int] = {}

    if args.gptq_lm_head:
        weight_bits_overrides["lm_head"] = args.lm_head_weight_bits

    return GPTQConfig(
        show_progress=not args.no_tqdm,
        weight_bits=args.linear_weight_bits,
        weight_bits_overrides=weight_bits_overrides,
        mse=args.gptq_mse,
        sensitivity=sensitivity,
        quantize_lm_head=args.gptq_lm_head,
        use_orig_model_inference=args.gptq_use_orig_model_inference,
        percdamp=args.gptq_percdamp,
    )


def save_model_to(
    q_m, calib_input, save_circle_to_folder, prefill_decode: bool = False
):
    """
    Export and save the whole quantized model in circle format.
    """
    q_m.eval()
    q_m.cpu()
    model_name = "model_prefill" if prefill_decode else "model"
    save_path = pathlib.Path(save_circle_to_folder, f"{model_name}.q.circle")
    print(f"saving the whole {model_name} to {save_path.resolve()}")
    config = q_m.wrapped.config
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            qmodel = q_m.wrapped.model.wrapped
            if prefill_decode is True:
                # kwargs for padding
                S = calib_input.shape[-1]
                attention_mask = (
                    qmodel.causal_mask_template[..., :S, :S].squeeze(0).to("cpu")
                )
                pos_embeds = (
                    qmodel.rope_cos_template[:, :S, :].to("cpu"),
                    qmodel.rope_sin_template[::S, :].to("cpu"),
                )
                kwargs = {
                    "attention_mask": attention_mask,
                    "position_embeddings": pos_embeds,
                }
            else:
                kwargs = {}

            cm = tico.convert(
                q_m.wrapped.as_export_module(
                    "prefill", return_kv=prefill_decode
                ).eval(),
                (calib_input,),
                kwargs=kwargs,
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
                # kwargs for padding
                attention_mask = make_random_decode_attn_mask(1, max_seq_len, "cpu")
                pos_embeds = make_random_position_embeddings(1, head_dim, "cpu")

                cm = tico.convert(
                    q_m.wrapped.as_export_module("decode").eval(),
                    (token, past_kv),
                    kwargs={
                        "attention_mask": attention_mask,
                        "position_embeddings": pos_embeds,
                    },
                    strict=False,
                )
                cm.save(save_path)


def make_random_position_embeddings(B, head_dim, DEVICE):
    """Create random RoPE tables for one decode step."""
    cos = torch.randn(B, 1, head_dim, device=DEVICE)
    sin = torch.randn(B, 1, head_dim, device=DEVICE)
    return (cos, sin)


def make_random_decode_attn_mask(B, MAX_SEQ, DEVICE):
    # Additive mask of final static width: (B, 1, MAX_SEQ)
    # Simulate that only the first L_eff positions are valid and the rest are padding.
    L_eff = torch.randint(low=1, high=MAX_SEQ + 1, size=(1,)).item()
    mask = torch.zeros(B, 1, MAX_SEQ, device=DEVICE, dtype=torch.float32)
    if L_eff < MAX_SEQ:
        mask[:, :, L_eff:] = float("-120")
    return mask


# -----------------------------------------------------------------------------
# copied from quantize_decoder_layer_decode.py
# -----------------------------------------------------------------------------
def make_random_decode_batch(model, B, DEVICE, MAX_SEQ):
    """Create a synthetic decode batch for per-layer export."""
    # TODO reduce code duplication
    D = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", D // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads

    # Single-token hidden state.
    x = torch.randn(B, 1, D, device=DEVICE)
    pos = make_random_position_embeddings(B, head_dim, DEVICE)
    mask = make_random_decode_attn_mask(B, MAX_SEQ, DEVICE)

    # Static-sized past KV (already RoPE-applied for past tokens).
    past_k = torch.randn(B, n_kv, MAX_SEQ - 1, head_dim, device=DEVICE)
    past_v = torch.randn(B, n_kv, MAX_SEQ - 1, head_dim, device=DEVICE)
    past = (past_k, past_v)

    return x, pos, mask, past


def save_export_module_to(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    save_path: pathlib.Path,
    artifact_name: str,
    *,
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Optional[Any] = None,
    strict: bool = False,
) -> None:
    """Convert an export module to Circle and save it."""
    print(f"Saving {artifact_name} to {save_path.resolve()}")

    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(
                module.eval(),
                example_inputs,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )

    cm.save(save_path)


def save_token_embedding_to(
    qmodel: torch.nn.Module,
    max_seq_len: int,
    save_layers_to_folder: str | pathlib.Path,
) -> None:
    """
    Export and save the token embedding stage with a dynamic sequence dimension.

    The generated Circle model is shared by prefill and decode runtime paths.

    Circle contract:
        input_ids:     `(1, S)`
        hidden_states: `(1, S, hidden_size)`

    The sequence dimension `S` is dynamic and bounded by
    `1 <= S <= max_seq_len`.
    """
    register_fake_quant_meta_kernels_for_dynamic_export()

    artifact_name = "token_embedding"
    save_path = pathlib.Path(save_layers_to_folder, f"{artifact_name}.q.circle")

    example_input_ids = make_token_embedding_example_input(
        qmodel=qmodel,
        max_seq_len=max_seq_len,
    )
    dynamic_shapes = make_token_embedding_dynamic_shapes(max_seq_len)

    save_export_module_to(
        LlamaTokenEmbeddingExportAdapter(qmodel),
        (example_input_ids,),
        save_path,
        artifact_name,
        dynamic_shapes=dynamic_shapes,
    )


def save_lm_head_to(
    qmodel: torch.nn.Module,
    save_layers_to_folder: str | pathlib.Path,
) -> None:
    """
    Export and save the shared single-token LM head stage.

    This artifact is used for both:
        - the last real token after prefill
        - every decode token

    Circle contract:
        hidden_states: `(1, 1, hidden_size)`
        logits:        `(1, 1, vocab_size)`

    The runtime should slice or gather the last real prefill hidden state before
    calling this artifact.
    """
    artifact_name = "lm_head"
    save_path = pathlib.Path(save_layers_to_folder, f"{artifact_name}.q.circle")
    example_hidden = torch.randn(
        1,
        1,
        int(qmodel.config.hidden_size),
        device="cpu",
    )

    save_export_module_to(
        LlamaLMHeadExportAdapter(qmodel),
        (example_hidden,),
        save_path,
        artifact_name,
    )


def save_layers_to(
    q_m, max_seq_len, save_layers_to_folder, prefill_decode: bool = False
):
    """
    Export and save quantized token embedding, decoder layers, and LM head.

    Artifacts:
        - `token_embedding.q.circle`
            Shared by prefill and decode. Its sequence dimension is dynamic.

        - `decoder_layer_prefill_{i}.q.circle` and
          `decoder_layer_decode_{i}.q.circle` when `prefill_decode=True`.

        - `decoder_layer_{i}.q.circle` when `prefill_decode=False`.

        - `lm_head.q.circle`
            Shared single-token final norm and LM head stage.
    """
    q_m.eval()
    q_m.cpu()

    if not hasattr(q_m, "wrapped"):
        print("Saving layers currently is supported only for PTQ quantized model")
        return

    if max_seq_len is None:
        raise ValueError("max_seq_len must be set for per-layer Circle export.")

    max_seq_len = int(max_seq_len)
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    qmodel = q_m.wrapped
    layers = qmodel.model.wrapped.layers
    config = qmodel.config

    # Token embedding runs on CPU in the target runtime, so export it once with
    # dynamic sequence length. This one artifact covers both prefill and decode.
    save_token_embedding_to(
        qmodel=qmodel,
        max_seq_len=max_seq_len,
        save_layers_to_folder=save_layers_to_folder,
    )

    for i, qlayer in enumerate(layers):
        suffix = "prefill_" if prefill_decode else ""
        layer_name = f"decoder_layer_{suffix}{i}"
        save_path = pathlib.Path(save_layers_to_folder, f"{layer_name}.q.circle")
        B, S, D = 1, max_seq_len, config.hidden_size
        example_hidden = torch.randn(B, S, D, device="cpu")

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

    # The runtime only needs logits for one token:
    #   - the last real token after prefill
    #   - the current token during decode
    # Therefore one shared single-token LM head artifact is enough.
    save_lm_head_to(
        qmodel=qmodel,
        save_layers_to_folder=save_layers_to_folder,
    )


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
    print(f"Using PTQ execution profile: {args.profile}")

    qcfg = build_llm_ptq_config(
        model_type="llama",
        num_hidden_layers=len(q_m.model.layers),
        activation=affine(DType.int(16)),
        weight=affine(_weight_dtype_from_bits(16)),
        linear_weight=affine(_weight_dtype_from_bits(args.linear_weight_bits)),
        embedding_weight=affine(_weight_dtype_from_bits(args.embedding_weight_bits)),
        lm_head_weight=affine(_weight_dtype_from_bits(args.lm_head_weight_bits)),
        spin_rotation_weight=(
            None
            if args.no_spinquant
            else affine(_weight_dtype_from_bits(args.spin_rotation_weight_bits))
        ),
        norm_weight=affine(_weight_dtype_from_bits(16)),
        strict_wrap=True,
        profile=args.profile,
    )
    q_m = prepare(q_m, qcfg)

    print("Calibrating PTQ observers…")

    if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
        inject_gptq_qparams(q_m, q_m.quantizers, verbose=args.verbose)
        clear_gptq_quantizers(q_m)
    elif (
        hasattr(q_m, "wrapped")
        and hasattr(q_m.wrapped, "quantizers")
        and isinstance(q_m.wrapped.quantizers, dict)
    ):
        inject_gptq_qparams(q_m.wrapped, q_m.wrapped.quantizers, verbose=args.verbose)
        clear_gptq_quantizers(q_m)
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
    dataset_name = dataset.replace("/", "_")

    name = (
        "."
        + "/sensitivities_for_"
        + model_name
        + "_"
        + dataset_name
        + "_"
        + str(n_samples)
        + "_"
        + str(seed)
        + ".pt"
    )
    return name


def get_ptq_model_name(model, dataset, args):
    """
    Build a filename for a saved PTQ checkpoint.
    """
    model_name = model.config.name_or_path.replace("/", "_")
    dataset_name = dataset.replace("/", "_")

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


def circle_export_requested(args) -> bool:
    """Return True if any Circle export artifact is requested."""
    return should_save(args, "circle_full") or should_save(args, "circle_per_layer")


def validate_export_profile(args) -> None:
    """
    Reject Circle export when the model was prepared with a non-export profile.
    """
    if not circle_export_requested(args):
        return

    if args.profile != "npu_export":
        raise ValueError(
            "Circle export in this example requires --profile npu_export. "
            "Use --profile reference_eval for fast GPU evaluation without "
            "circle_full/circle_per_layer saving, or rerun calibration with "
            "--profile npu_export before exporting."
        )


def setup_runtime(args) -> tuple[torch.device, torch.dtype]:
    """
    Initialize deterministic settings and resolve runtime device / dtype.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]
    return device, dtype


def print_config(args, device: torch.device) -> None:
    """
    Print the effective high-level runtime configuration.
    """
    print("=== Config ===")
    print(f"Model                  : {args.model}")
    print(f"Device                 : {args.device}")
    print(f"DType                  : {args.dtype}")
    print(f"Seed                   : {args.seed}")
    print(f"GPTQ enabled           : {not args.no_GPTQ}")
    print(f"GPTQ lm_head enabled   : {args.gptq_lm_head}")
    print(f"PTQ enabled            : {not args.no_PTQ}")
    print(f"SpinQuant enabled      : {not args.no_spinquant}")
    print(f"CLE enabled            : {args.enable_CLE}")
    print(f"Linear weight bits     : {args.linear_weight_bits}")
    print(f"Embedding weight bits  : {args.embedding_weight_bits}")
    print(f"LM head weight bits    : {args.lm_head_weight_bits}")
    print(
        "Spin rotation bits     : "
        f"{args.spin_rotation_weight_bits if not args.no_spinquant else 'disabled'}"
    )
    print(f"Calibration samples    : {args.nsamples_for_qcalibration}")
    print(f"Calibration seq length : {args.calibrate_seq_len}")
    print(f"Max seq length         : {args.max_seq_len}")
    print(f"Profile                : {args.profile}")
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
    Load or compute sensitivity information for sensitivity-based GPTQ.
    """
    if args.gptq_mse not in ("smse", "smse_for_gptq"):
        return None

    if args.sensitivity_path is not None:
        path = pathlib.Path(args.sensitivity_path)
        if path.exists():
            print(f"Loading sensitivity information from {path.resolve()}")
            return torch.load(path)

    print("Computing sensitivity information for GPTQ SMSE ...")
    calibrator = SensitivityCalibrator(model, calib_inputs)
    sens = calibrator.compute_sensitivity_info()

    if should_save(args, "sensitivity"):
        default_path = pathlib.Path(
            get_sensitivities_info_name(
                model,
                DATASET_NAME,
                args.seed,
                args.nsamples_for_qcalibration,
            )
        )
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / default_path.name
        print(f"Saving sensitivity information to {save_path.resolve()}")
        torch.save(sens, save_path)

    return sens


def apply_gptq(model, calib_inputs, args):
    """
    Optionally run GPTQ weight-only quantization.
    """
    if args.no_GPTQ:
        print("Skipping GPTQ ...")
        return model

    print("Applying GPTQ ...")
    sens = compute_or_load_sensitivity(model, calib_inputs, args)
    gptq_config = build_gptq_config(args, sensitivity=sens)

    q_m = prepare(model, gptq_config, inplace=True)

    iterator = calib_inputs
    if not args.no_tqdm:
        iterator = tqdm.tqdm(calib_inputs, desc="GPTQ calibration")

    with torch.no_grad():
        for inp in iterator:
            q_m(inp.to(args.device))

    return convert(q_m, inplace=True)


def get_pad_token_id(tokenizer) -> int:
    """
    Return a usable pad token id for export example inputs.
    """
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def get_export_input(calib_inputs, tokenizer, args) -> torch.Tensor:
    """
    Build the token tensor used for full-model export.
    """
    example = calib_inputs[0].cpu()
    if args.max_seq_len is None:
        return example
    return pad_input(example, get_pad_token_id(tokenizer), args.max_seq_len).cpu()


def save_requested_artifacts(q_m, tokenizer, calib_inputs, args) -> None:
    """
    Save requested artifacts after PTQ conversion.
    """
    if args.output_dir is None or args.save is None:
        return

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if should_save(args, "ptq_checkpoint"):
        save_path = output_dir / get_ptq_model_name(q_m.wrapped, DATASET_NAME, args)
        print(f"Saving PTQ checkpoint to {save_path.resolve()}")
        torch.save(q_m, save_path)

    if should_save(args, "circle_full"):
        export_input = get_export_input(calib_inputs, tokenizer, args)
        save_model_to(
            q_m,
            export_input,
            output_dir,
            prefill_decode=args.decode_calibration_steps > 0,
        )

    if should_save(args, "circle_per_layer"):
        max_seq_len = args.max_seq_len or q_m.wrapped.config.max_position_embeddings
        save_layers_to(
            q_m,
            max_seq_len,
            output_dir,
            prefill_decode=args.decode_calibration_steps > 0,
        )


def main():
    args = parse_args()
    print(args)
    validate_export_profile(args)

    device, dtype = setup_runtime(args)
    print_config(args, device)

    model, tokenizer = load_model_and_tokenizer(args, dtype)
    validate_tied_embedding_weight_bits(model, args)
    configure_max_position_embeddings(model, args)

    dataset_test = load_eval_dataset(args)
    evaluate_original_model(model, tokenizer, dataset_test, args, device)

    calib_inputs = build_calibration_inputs(model, tokenizer, args, device)

    model = apply_spinquant(model, args)
    model = apply_cle(model, args)
    model = apply_gptq(model, calib_inputs, args)

    q_m = quantize_using_PTQ(model, calib_inputs, args)

    evaluate(q_m, tokenizer, dataset_test, args)
    save_requested_artifacts(q_m, tokenizer, calib_inputs, args)


if __name__ == "__main__":
    main()

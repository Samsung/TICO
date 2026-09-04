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
import math
import os
import sys
import types

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pathlib
import random
from typing import Any, Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
from torch import Tensor

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
from tico.quantization.config.llama_gptq import LlamaGPTQConfig
from tico.quantization.config.llama_attention import (
    DEFAULT_EXECUTION_PROFILE,
    SUPPORTED_EXECUTION_PROFILES,
)
from tico.quantization.config.specs import affine, mx
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

# Hardcoded dataset settings
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# MMLU calibration dataset settings
MMLU_DATASET_NAME = "cais/mmlu"
MMLU_DATASET_CONFIG = "all"
MMLU_CALIB_SPLIT = "test"

# TruthfulQA calibration dataset settings
TRUTHFULQA_DATASET_NAME = "truthful_qa"
TRUTHFULQA_DATASET_CONFIG = "multiple_choice"
TRUTHFULQA_CALIB_SPLIT = "validation"

# HellaSwag calibration dataset settings
HELLASWAG_DATASET_NAME = "Rowan/hellaswag"
HELLASWAG_CALIB_SPLIT = "train"

# PIQA calibration dataset settings
PIQA_DATASET_NAME = "chargoddard/piqa-train-10k"
PIQA_CALIB_SPLIT = "train"


def build_parser() -> argparse.ArgumentParser:
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
        choices=["circle_full", "circle_per_layer", "ptq_checkpoint", "sensitivity", "calibration_dataset"],
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
        "--calibration_samples_to_use",
        type=int,
        default=None,
        help="number of samples to actually use from the calibration dataset for quantization (allows compressing the dataset used). When not specified, uses all nsamples_for_qcalibration samples.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for calibration set preparation and processing",
    )
    parser.add_argument(
        "--linear_weight_bits",
        type=int,
        default=4,
        help="Number of bits to be used in quantizer for matmul weight quantization",
    )
    parser.add_argument(
        "--linear_io_qdtype",
        type=str,
        default="int16",
        help="which activation types are supposed for matmuls for PTQ (`int16`/`mxint8` are supported for now)",
    )
    parser.add_argument(
        "--softmax_io_qdtype",
        type=str,
        default="int16",
        help="which activation types are supposed for softmax for PTQ (`int16`/`mxint8` are supported for now)",
    )
    parser.add_argument(
        "--norm_io_qdtype",
        type=str,
        default="int16",
        help="which activation types are supposed for rmsnorm for PTQ (`int16`/`mxint8` are supported for now)",
    )
    parser.add_argument(
        "--spinquant_io_qdtype",
        type=str,
        default=None,
        help="which activation types are supposed for SpinQuant rotation I/O (input/output of rotate_embedding and rotate_lm_head). Defaults to linear_io_qdtype if not specified.",
    )
    parser.add_argument(
        "--lm_head_io_qdtype",
        type=str,
        default=None,
        help="which activation types are supposed for output norm + lm_head I/O (input/output of final norm and lm_head). Defaults to linear_io_qdtype if not specified.",
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
        "--calibration_dataset_path",
        type=str,
        default=None,
        help="Path to a pre-saved calibration dataset (.pt file). When provided, skip dataset loading and load calibration inputs directly.",
    )
    parser.add_argument(
        "--calibration_dataset_mix",
        nargs="+",
        type=str,
        default=["wikitext"],
        help=(
            "Calibration dataset(s) to use for calibration data generation. "
            "Accepts one or more `name:proportion` pairs, e.g. "
            "`--calibration_dataset_mix wikitext:0.7 mmlu:0.3`. "
            "Proportions are optional and normalized to sum to 1.0; "
            "a single name without a proportion uses the full dataset, e.g. "
            "`--calibration_dataset_mix wikitext` or `--calibration_dataset_mix mmlu`. "
            "Supported datasets: wikitext, mmlu, truthfulqa, hellaswag, piqa. "
            "Ignored when --calibration_dataset_path is set."
        ),
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
    parser.add_argument(
        "--gptq_v2",
        action="store_true",
        default=False,
        help="Enable GPTQv2 (uses FP inference for collecting inputs during quantization).",
    )
    parser.add_argument(
        "--llama_gptq",
        action="store_true",
        default=False,
        help="Use LlamaGPTQConfig instead of GPTQConfig for Llama-specific GPTQ quantization.",
    )
    parser.add_argument(
        "--gptq_adaptive_percdamp",
        action="store_true",
        default=False,
        help="Enable adaptive percdamp based on Hessian condition number.",
    )
    parser.add_argument(
        "--gptq_cond_threshold_good",
        type=float,
        default=100000.0,
        help="Condition number threshold for good matrices to be used in adaptive percdamp (default: 100000.0). Matrices with condition number below this threshold use minimal damping.",
    )
    parser.add_argument(
        "--llama_gptq_sequential",
        action="store_true",
        default=False,
        help="Enable sequential processing of layer groups in LlamaGPTQ (default: True). Very slow but more accurate.",
    )
    parser.add_argument(
        "--llama_gptq_no_ptq",
        action="store_true",
        default=False,
        help="Run LlamaGPTQ without PTQ wrapping (LlamaGPTQ-only path, skips activation quantization).",
    )
    parser.add_argument(
        "--gptq_use_iterate",
        action="store_true",
        default=False,
        help="Use iterate_GPTQ instead of the main block-based loop (same approach as fpi_gptq.py).",
    )
    parser.add_argument(
        "--llama_gptq_use_subgroup_runner",
        action="store_true",
        default=False,
        help="Use SubgroupRunner for efficient subgroup-level inference during LlamaGPTQ quantization (default: False). When enabled, runs only the necessary submodules for each subgroup instead of the full layer, significantly reducing redundant computation.",
    )
    parser.add_argument(
        "--visualize_calibration_compression",
        action="store_true",
        default=False,
        help="Save 2D visualization PNGs for calibration dataset compression (per-layer and final compression clustering).",
    )
    parser.add_argument(
        "--assess_cluster_consistency",
        action="store_true",
        default=False,
        help=(
            "Assess cross-layer cluster consistency of CalibrationSetCompressor. "
            "Collects K-Means labels for every decoder layer and computes "
            "Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) "
            "between all layer pairs, plus a sample co-occurrence matrix. "
            "Saves heatmap visualizations and prints a summary."
        ),
    )

    parser.add_argument(
        "--calibration_compression_n_components",
        type=int,
        default=2,
        help=(
            "Number of projection components for calibration dataset compression. "
            "Higher values preserve more activation structure. "
            "Clamped to <= --calibration_samples_to_use (and >= 2)."
        ),
    )

    parser.add_argument(
        "--gptq_double_precision",
        action="store_true",
        default=False,
        help=(
            "Use float64 (double) for Hessian accumulation to make GPTQ results "
            "stable. Float32 accumulation causes different rounding "
            "depending on how samples are grouped into batches. Default: False "
            "(float32, backward compatible). Enable for exact/reproducible results."
        ),
    )
    parser.add_argument(
        "--gptq_saturation_threshold",
        type=float,
        default=None,
        help=(
            "Early stopping threshold for Hessian saturation. "
            "When the relative change in effective rank (participation ratio) "
            "r_eff = trace(H)²/||H||_F² drops below this value, GPTQ stops "
            "collecting batches. The metric is dimension-agnostic, so the same "
            "threshold works across model sizes. None = disabled. Typical value: 1e-3."
        ),
    )
    parser.add_argument(
        "--gptq_parallel_workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes for GPTQ layer quantization. "
            "0 = sequential (default). Requires --gptq_use_orig_model_inference. "
            "Each worker process runs on GPU using multiprocessing spawn."
        ),
    )

    parser.add_argument(
        "--compression_feature_mode",
        type=str,
        default="mean_pool",
        choices=["flatten", "mean_pool", "mean_pool_std"],
        help=(
            "How to extract per-sample features from layer activations. "
            "'flatten' = raw flatten (original, unstable). "
            "'mean_pool' = mean over seq_len (stable, default). "
            "'mean_pool_std' = concat(mean, std)."
        ),
    )
    parser.add_argument(
        "--compression_projection_mode",
        type=str,
        default="pca",
        choices=["qr_random", "pca", "none", "sparse_random"],
        help=(
            "How to project features before K-Means. "
            "'qr_random' = random QR projection (original, different per layer). "
            "'pca' = PCA fit on current layer's features (shared, stable, default). "
            "'none' = no projection, use raw features. "
            "'sparse_random' = sparse random projection (original flatten behavior)."
        ),
    )
    parser.add_argument(
        "--compression_normalize_features",
        type=str,
        default="true",
        choices=["true", "false"],
        help=(
            "Whether to L2-normalize features before K-Means (spherical K-Means). "
            "Default: true. Use 'false' to disable."
        ),
    )
    parser.add_argument(
        "--compression_kmeans_n_init",
        type=int,
        default=50,
        help=(
            "Number of K-Means restarts. More restarts = more stable clustering. "
            "Default: 50."
        ),
    )
    parser.add_argument(
        "--compression_concat_layers",
        action="store_true",
        default=False,
        help=(
            "Concatenate per-sample features from ALL layers and cluster on the "
            "combined representation, instead of picking the single best layer "
            "by silhouette score. Uses information from every layer "
            "simultaneously for more robust clustering."
        ),
    )
    parser.add_argument(
        "--ppl_filter_percentile",
        type=float,
        default=None,
        help=(
            "Filter calibration samples by model perplexity. "
            "When set (e.g. 80.0), keep only samples with PPL below this "
            "percentile, removing the most uncertain samples. "
            "None = no filtering (default)."
        ),
    )
    return parser


def parse_args():
    parser = build_parser()
    return parser.parse_args()


def print_cmd(args) -> None:
    """
    Print the command-line invocation that reproduces the current run.

    Only non-default arguments are shown so the output stays concise and
    focused on what the user actually changed.  ``store_true`` flags are
    included only when enabled.  List-valued arguments (e.g. ``--save``,
    ``--calibration_dataset_mix``) are space-joined.
    """
    parser = build_parser()
    script = sys.argv[0]
    parts = [f"python {script}"]

    for action in parser._actions:
        if action.dest == "help":
            continue

        value = getattr(args, action.dest, None)
        default = action.default

        # store_true flags: include only when True (i.e. explicitly passed)
        if isinstance(action, argparse._StoreTrueAction):
            if value is True:
                parts.append(action.option_strings[0])
            continue

        # Skip arguments still at their default
        if value == default:
            continue

        # Skip None values that default to None
        if value is None:
            continue

        flag = action.option_strings[0]
        if isinstance(value, list):
            parts.append(f"{flag} {' '.join(str(v) for v in value)}")
        else:
            parts.append(f"{flag} {value}")

    cmd = " \\\n  ".join(parts)
    print("\n=== Reproduce with: ===")
    print(cmd)
    print("=" * 40)


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


def print_minmax_values(model: torch.nn.Module) -> None:
    """
    Print min/max values from all PTQ observers in the quantized model.

    This function traverses the model hierarchy and prints the min/max statistics
    collected by each AffineObserverBase instance. Useful for debugging and
    inspecting quantization ranges after calibration.

    For per-tensor observers, prints scalar min/max values.
    For per-channel observers, prints the global min/max range and channel shape.

    Args:
        model: A PTQ-quantized model with observers containing min/max statistics.

    Example usage:
        # After calibration and before/after conversion:
        print_minmax_values(q_m)
    """
    from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
    from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

    print("\n" + "=" * 80)
    print("PTQ Model Min/Max Values")
    print("=" * 80)
    print(f"{'Module Name':<50} | {'Observer':<25} | Min/Max Values")
    print("-" * 80)

    count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, QuantModuleBase):
            continue

        for obs_name, obs in module.named_observers(recurse=True):
            if not isinstance(obs, AffineObserverBase):
                continue

            if not hasattr(obs, "min_val") or not hasattr(obs, "max_val"):
                continue

            min_val = obs.min_val
            max_val = obs.max_val

            # Format output based on per-tensor vs per-channel
            if min_val.numel() == 1:
                # Per-tensor: scalar values
                values_str = f"min={min_val.item():.6f}, max={max_val.item():.6f}"
            else:
                # Per-channel: show shape and range
                values_str = (
                    f"min={min_val.min().item():.6f}..{max_val.max().item():.6f} "
                    f"(shape={tuple(min_val.shape)})"
                )

            print(f"{module_name:<50} | {obs_name:<25} | {values_str}")
            count += 1

    print("-" * 80)
    print(f"Total observers: {count}")
    print("=" * 80 + "\n")


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
    sample_weights: list[float] | None = None,
):
    """
    Build a GPTQ configuration from command-line arguments.

    GPTQ for lm_head is disabled by default because many causal language models
    tie `lm_head.weight` with the input embedding table. Users can enable it
    explicitly with `--gptq_lm_head`.

    If `--llama_gptq` or `--llama_gptq_sequential` is specified, returns a LlamaGPTQConfig instead of
    GPTQConfig for Llama-specific GPTQ quantization.
    """
    weight_bits_overrides: dict[str, int] = {}

    if args.gptq_lm_head:
        weight_bits_overrides["lm_head"] = args.lm_head_weight_bits

    if args.llama_gptq or args.llama_gptq_sequential:
        config = LlamaGPTQConfig(
            show_progress=not args.no_tqdm,
            weight_bits=args.linear_weight_bits,
            weight_bits_overrides=weight_bits_overrides,
            mse=args.gptq_mse,
            sensitivity=sensitivity,
            quantize_lm_head=args.gptq_lm_head,
            quantize_rotate_lm_head=not args.no_spinquant,
            use_orig_model_inference=args.gptq_use_orig_model_inference,
            percdamp=args.gptq_percdamp,
            verbose=args.verbose,
            gptq_v2=args.gptq_v2,
            adaptive_percdamp=args.gptq_adaptive_percdamp,
            cond_threshold_good=args.gptq_cond_threshold_good,
            sequential=args.llama_gptq_sequential,
            use_iterate=args.gptq_use_iterate,
            use_subgroup_runner=args.llama_gptq_use_subgroup_runner,
            sample_weights=sample_weights,
            saturation_threshold=args.gptq_saturation_threshold,
            double_precision=args.gptq_double_precision,
            parallel_workers=args.gptq_parallel_workers,
        )
        return config
    else:
        config = GPTQConfig(
            show_progress=not args.no_tqdm,
            weight_bits=args.linear_weight_bits,
            weight_bits_overrides=weight_bits_overrides,
            mse=args.gptq_mse,
            sensitivity=sensitivity,
            quantize_lm_head=args.gptq_lm_head,
            use_orig_model_inference=args.gptq_use_orig_model_inference,
            percdamp=args.gptq_percdamp,
            verbose=args.verbose,
            gptq_v2=args.gptq_v2,
            adaptive_percdamp=args.gptq_adaptive_percdamp,
            cond_threshold_good=args.gptq_cond_threshold_good,
            use_iterate=args.gptq_use_iterate,
            sample_weights=sample_weights,
            saturation_threshold=args.gptq_saturation_threshold,
            double_precision=args.gptq_double_precision,
            parallel_workers=args.gptq_parallel_workers,
        )
        return config


def save_model_to(
    q_m, calib_input, save_circle_to_folder, prefill_decode: bool = False
):
    """
    Export and save the whole quantized model in circle format.
    """
    if not hasattr(q_m, "wrapped"):
        print("Saving whole model circle is supported only for PTQ quantized model")
        return

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


class StopForward(Exception):
    """Custom exception used to stop the forward pass after capturing embedding outputs."""
    pass


class CalibrationSetCompressor:
    """
    Compress calibration dataset using K-Means clustering.
    
    Uses layerwise activations from decoder layers as features for clustering,
    providing semantically meaningful similarity metrics compared to raw token IDs.
    Inspired by FPInputsCache in llama_quantizer.py.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        compress_to_samples: int,
        seed: int = 42,
        device: Optional[torch.device] = None,
        n_layers_to_use: int = 1,  # Number of decoder layers to extract activations from
        n_components: int = 64,
        visualize: bool = False,
        output_dir: Optional[str | pathlib.Path] = None,
        feature_mode: str = "mean_pool",
        projection_mode: str = "pca",
        normalize_features: bool = True,
        kmeans_n_init: int = 50,
        concat_layers: bool = False,
    ):

        self.model = model
        self.compress_to_samples = compress_to_samples
        self.seed = seed
        self.device = device or next(model.parameters()).device
        self.n_layers_to_use = n_layers_to_use
        # Clamp n_components to be at least 2 and at most compress_to_samples
        self.n_components = max(2, min(n_components, compress_to_samples // 2))
        self.visualize = visualize
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        # --- New stability parameters ---
        # How to extract a per-sample feature vector from [seq_len, hidden_dim] activations
        #   "flatten"     — raw flatten (original, unstable)
        #   "mean_pool"   — mean over seq_len → [hidden_dim] (stable, default)
        #   "mean_pool_std" — concat(mean, std) → [2*hidden_dim]
        self.feature_mode = feature_mode
        # How to project features before K-Means
        #   "qr_random" — random QR projection (original, different per layer)
        #   "pca"       — PCA fit on all layers' features (shared, stable)
        #   "none"      — no projection, use raw pooled features
        self.projection_mode = projection_mode
        # L2-normalize features before K-Means (spherical K-Means)
        self.normalize_features = normalize_features
        # Number of K-Means restarts (more = more stable)
        self.kmeans_n_init = kmeans_n_init
        # When True, concatenate features from all layers and cluster on the
        # combined representation instead of picking the single best layer.
        self.concat_layers = concat_layers
        # Cached shared projection matrix (for qr_random with shared_projection)
        self._shared_proj_matrix: Optional[Tensor] = None
        # Cached PCA model (for pca mode)
        self._shared_pca: Optional[Any] = None

        # When using "flatten" feature mode, restore the original behavior
        # (sparse random projection, no normalization, n_init=10) to
        # reproduce results from before the configurable compression parameters.
        if self.feature_mode == "flatten":
            self.projection_mode = "sparse_random"
            self.normalize_features = False
            self.kmeans_n_init = 10

    def _get_decoder_layers(self) -> list[torch.nn.Module]:
        """
        Get decoder layers from the model, handling PTQ wrappers.
        
        Returns list of decoder layer modules.
        """
        # Handle PTQ-wrapped models
        if hasattr(self.model, 'wrapped'):
            if hasattr(self.model.wrapped, 'model'):
                # QuantLlamaForCausalLM -> PTQWrapper -> QuantLlamaModel
                llama_model = self.model.wrapped.model.wrapped
                return list(llama_model.layers)
        elif hasattr(self.model, 'model'):
            # Standard LlamaForCausalLM
            llama_model = self.model.model
            return list(llama_model.layers)
        return []

    def _sparse_random_project(self, features: list[Tensor]) -> np.ndarray:
        """
        Project high-dimensional activation data to ``self.n_components`` dimensions on GPU.

        This replaces sklearn's ``SparseRandomProjection`` (which runs on CPU)
        with an equivalent GPU-accelerated computation.  The random projection
        matrix follows the same distribution as sklearn:

          - ``+sqrt(s) / sqrt(n_components)``  with probability 1 / 2s
          -  ``0``                             with probability 1 - 1 / s
          - ``-sqrt(s) / sqrt(n_components)``  with probability 1 / 2s

        where ``s = 1 / density`` and ``density = 1 / sqrt(n_features)`` (the
        sklearn default for ``density='auto'``).

        Projects features one at a time to keep GPU memory usage low (only one
        feature + the projection matrix are on GPU at any time).

        Args:
            features: List of 1-D CPU tensors, each of shape ``[n_features]``.

        Returns:
            A numpy array of shape ``[n_samples, n_components]`` (on CPU)
            suitable for downstream KMeans / silhouette_score.
        """
        n_features = features[0].numel()
        n_components = self.n_components

        # density = 1 / sqrt(n_features), same as sklearn's 'auto'
        density = 1.0 / math.sqrt(n_features)
        s = 1.0 / density  # = sqrt(n_features)

        # Scale factor: sqrt(1 / density) / sqrt(n_components)
        scale = math.sqrt(1.0 / density) / math.sqrt(n_components)

        # Generate the sparse random projection matrix on GPU.
        # Each entry is +1, -1, or 0 with probabilities matching sklearn.
        # P(nonzero) = density, and among nonzeros P(+1) = P(-1) = 0.5.
        # Uses the global torch RNG (seed set in compress()).
        rand = torch.rand(n_components, n_features, device=self.device)

        # +1 where rand < density/2, -1 where rand in [density/2, density), 0 otherwise
        components = torch.zeros(
            n_components, n_features, device=self.device, dtype=features[0].dtype
        )
        components[rand < density / 2.0] = 1.0
        components[(rand >= density / 2.0) & (rand < density)] = -1.0
        components *= scale

        # Project each feature one at a time to keep GPU memory low.
        projected = []
        for feat in features:
            # [n_features] @ [n_features, n_components] -> [n_components]
            proj = feat.to(self.device) @ components.T
            projected.append(proj.cpu())
        data_proj = torch.stack(projected, dim=0).numpy()  # [n_samples, n_components]

        return data_proj


    def _qr_random_project(self, features: list[Tensor]) -> np.ndarray:
        """
        Project high-dimensional activation data to ``self.n_components`` dimensions on GPU via QR factorization.

        Generates a random Gaussian matrix of shape ``(n_features, n_components)``
        and computes its QR decomposition to obtain ``n_components`` orthonormal
        projection directions.  This preserves pairwise distances better than a
        sparse random projection and is simpler to compute.

        Projects features one at a time to keep GPU memory usage low (only one
        feature + the projection matrix are on GPU at any time).

        Uses the global torch RNG, so ``torch.manual_seed`` should be set
        before calling this method for reproducibility.

        Args:
            features: List of 1-D CPU tensors, each of shape ``[n_features]``.

        Returns:
            A numpy array of shape ``[n_samples, n_components]`` (on CPU)
            suitable for downstream KMeans / silhouette_score.
        """
        n_features = features[0].numel()
        n_components = self.n_components

        # Generate a random Gaussian matrix on GPU (uses global RNG).
        G = torch.randn(
            n_features, n_components, device=self.device, dtype=features[0].dtype
        )

        # QR decomposition: Q has orthonormal columns of shape (n_features, n_components).
        Q, _ = torch.linalg.qr(G)

        # Project each feature one at a time to keep GPU memory low.
        projected = []
        for feat in features:
            # [n_features] @ [n_features, n_components] -> [n_components]
            proj = feat.to(self.device) @ Q
            projected.append(proj.cpu())
        data_proj = torch.stack(projected, dim=0).numpy()  # [n_samples, n_components]

        return data_proj

    def _extract_features(
        self,
        layer_outputs: list[Tensor],
    ) -> list[Tensor]:
        """
        Extract a per-sample feature vector from layer activation outputs.

        Supports multiple feature extraction modes controlled by
        ``self.feature_mode``:

          - ``"flatten"``: raw flatten of [seq_len, hidden_dim] → [seq_len*hidden_dim]
            (original, unstable — high-dimensional, mixes position and feature)

          - ``"mean_pool"``: mean over the sequence dimension → [hidden_dim]
            (stable — captures average semantic content, removes positional noise)

          - ``"mean_pool_std"``: concat(mean, std) → [2*hidden_dim]
            (captures both central tendency and spread)

        Args:
            layer_outputs: List of activation tensors, each [batch, seq_len, hidden_dim]
                or [seq_len, hidden_dim].

        Returns:
            List of 1-D feature tensors (one per individual sample).
        """
        features: list[Tensor] = []

        for act in layer_outputs:
            # Unbatch: split batch dimension into individual samples
            if act.dim() == 3:
                samples = [act[i] for i in range(act.shape[0])]
            else:
                samples = [act]

            for sample in samples:
                # sample shape: [seq_len, hidden_dim]
                if self.feature_mode == "flatten":
                    features.append(sample.flatten())
                elif self.feature_mode == "mean_pool":
                    # Mean over sequence dimension → [hidden_dim]
                    features.append(sample.mean(dim=0))
                elif self.feature_mode == "mean_pool_std":
                    # Concat mean and std → [2 * hidden_dim]
                    mean = sample.mean(dim=0)
                    std = sample.std(dim=0)
                    features.append(torch.cat([mean, std], dim=0))
                else:
                    raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

        return features

    def _project_and_normalize(
        self,
        features: list[Tensor],
        all_layer_features: Optional[list[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Project features to ``self.n_components`` dimensions and optionally L2-normalize.

        This method centralizes the projection logic so that the same approach
        is used in both ``_find_best_layer_layerwise`` and
        ``_collect_all_layer_labels``.

        Projection modes (``self.projection_mode``):

          - ``"qr_random"``: random QR projection (original, different per call)
          - ``"pca"``: PCA fit on concatenated features from all layers
            (shared, stable).  When ``all_layer_features`` is provided, the PCA
            is fit on that data; otherwise it is fit on the current layer's
            features.
          - ``"none"``: no projection, use raw features (only normalization)

        After projection, if ``self.normalize_features`` is True, each row is
        L2-normalized (spherical K-Means).

        Args:
            features: List of 1-D CPU tensors for the current layer.
            all_layer_features: Optional concatenated features from all layers
                (for PCA fitting).  Shape [n_samples_total, n_features].

        Returns:
            A numpy array of shape [n_samples, n_components] (on CPU).
        """
        # Stack features into a single matrix
        feat_matrix = torch.stack(features, dim=0).numpy().astype(np.float64)

        if self.projection_mode == "qr_random":
            # Use the original QR random projection (per-layer, not shared)
            data_proj = self._qr_random_project(features)
        elif self.projection_mode == "sparse_random":
            # Sparse random projection (original flatten behavior)
            data_proj = self._sparse_random_project(features)
        elif self.projection_mode == "pca":
            # GPU-accelerated PCA via torch.pca_lowrank (replaces sklearn PCA)
            if all_layer_features is not None:
                fit_data = all_layer_features
            else:
                fit_data = feat_matrix

            n_comp = min(self.n_components, fit_data.shape[0], fit_data.shape[1])

            # Move to GPU, center, and compute PCA via SVD
            fit_gpu = torch.from_numpy(fit_data).to(self.device)
            fit_gpu = fit_gpu - fit_gpu.mean(dim=0, keepdim=True)
            _, _, V = torch.pca_lowrank(fit_gpu, q=n_comp)

            # Project the current layer's features using the same components
            feat_gpu = torch.from_numpy(feat_matrix).to(self.device)
            feat_gpu = feat_gpu - feat_gpu.mean(dim=0, keepdim=True)
            data_proj = (feat_gpu @ V).cpu().numpy()
        elif self.projection_mode == "none":

            # No projection — use raw features
            data_proj = feat_matrix
        else:
            raise ValueError(f"Unknown projection_mode: {self.projection_mode}")

        # L2-normalize each row (spherical K-Means)
        if self.normalize_features:
            norms = np.linalg.norm(data_proj, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            data_proj = data_proj / norms

        return data_proj

    def _fit_shared_pca(
        self,
        all_layer_features: list[list[Tensor]],
    ) -> None:
        """
        Fit a shared PCA model on concatenated features from all layers.

        Uses ``torch.pca_lowrank`` on GPU instead of sklearn PCA.
        The principal components (V) are stored in ``self._shared_pca`` as a
        GPU tensor for later projection.

        Args:
            all_layer_features: List (one per layer) of lists of feature tensors.
        """
        # Concatenate all layers' features into one GPU tensor
        all_feats = []
        for layer_feats in all_layer_features:
            for feat in layer_feats:
                all_feats.append(feat)
        concat_feats = torch.stack(all_feats, dim=0).to(self.device).to(torch.float64)

        n_comp = min(self.n_components, concat_feats.shape[0], concat_feats.shape[1])

        # Center and compute PCA via SVD on GPU
        concat_centered = concat_feats - concat_feats.mean(dim=0, keepdim=True)
        _, _, V = torch.pca_lowrank(concat_centered, q=n_comp)

        # Store the principal components for later projection
        self._shared_pca = V  # [n_features, n_comp]

        print(
            f"  Fitted shared PCA on {concat_feats.shape[0]} samples, "
            f"{concat_feats.shape[1]} features → {n_comp} components"
        )

    def _project_with_shared_pca(
        self,
        features: list[Tensor],
    ) -> np.ndarray:
        """
        Project features using the pre-fitted shared PCA components.

        Args:
            features: List of 1-D CPU tensors.

        Returns:
            Numpy array [n_samples, n_components].
        """
        feat_gpu = torch.stack(features, dim=0).to(self.device).to(torch.float64)
        feat_gpu = feat_gpu - feat_gpu.mean(dim=0, keepdim=True)
        data_proj = (feat_gpu @ self._shared_pca).cpu().numpy()

        if self.normalize_features:
            norms = np.linalg.norm(data_proj, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            data_proj = data_proj / norms

        return data_proj


    def _find_best_layer_layerwise(
        self,
        calib_inputs: list[Tensor],
        visualize: bool = False,
    ) -> tuple[int, np.ndarray, Any, Any]:
        """
        Find best layer for clustering using sequential layer-by-layer inference.

        
        Memory-efficient approach following LlamaGPTQQuantizer pattern:

        1. Override first decoder layer's forward to capture cache_args/cache_kwargs
        2. For each layer sequentially:
           a. Run layer with cached hidden states from cache_args
           b. Capture output activations
           c. Project to 2D and compute silhouette score
           d. If best: keep 2D data + labels
           e. Update cache_args[0] with layer output for next layer
           f. Discard old activations
        
        Args:
            calib_inputs: List of calibration tensors [N, seq_len]
            visualize: If True, save per-layer 2D clustering visualizations
            
        Returns:
            Tuple of:
            - best_layer_idx: Index of best layer (-1 if no layers found)
            - best_data_2d: 2D projected data for best layer [n_samples, 2]
            - best_labels: Cluster labels for best layer
            - best_kmeans: Fitted KMeans model for best layer
        """
        layers = self._get_decoder_layers()
        
        if not layers:
            print("  Warning: No decoder layers found")
            return -1, None, None, None
        
        best_score = -1.0
        best_layer_idx = -1
        best_data_2d = None
        best_labels = None
        best_kmeans = None
        
        self.model.eval()
        
        # Get first decoder layer (same pattern as LlamaGPTQQuantizer.prepare)
        first_layer = layers[0]
        
        # Cache args and kwargs from first layer (same pattern as LlamaGPTQQuantizer.prepare)
        cache_args: List[List[Any]] = []
        cache_kwargs: Dict[str, List[Any]] = {}
        
        # Store original forwards
        orig_layer_forward = first_layer.forward
        orig_model_forward = self.model.forward
        
        # Define catcher that stores args and kwargs, then raises StopForward
        # (same pattern as LlamaGPTQQuantizer.prepare)
        def layer_catcher(layer, *args, **kwargs):
            # Store positional args (hidden_states is first arg)
            for idx, item in enumerate(args):
                if (idx + 1) > len(cache_args):
                    cache_args.append([])
                cache_args[idx].append(item.detach().cpu())
            # Store keyword args
            for k, v in kwargs.items():
                if k not in cache_kwargs:
                    cache_kwargs[k] = []
                cache_kwargs[k].append(v.detach().cpu() if hasattr(v, 'detach') else v)
            # Raise exception to stop further execution after capturing
            raise StopForward
        
        # Replace first layer forward temporarily (same pattern as LlamaGPTQQuantizer.prepare)
        first_layer.forward = types.MethodType(layer_catcher, first_layer)
        
        # Wrap model.forward to catch StopForward (same pattern as LlamaGPTQQuantizer.prepare)
        def model_forward_wrapper(_model, *m_args, **m_kwargs):
            try:
                return orig_model_forward(*m_args, **m_kwargs)
            except StopForward:
                # Stop after first layer capture; return None
                return None
        
        self.model.forward = types.MethodType(model_forward_wrapper, self.model)
        
        # Run model to populate cache_args and cache_kwargs
        # Execution stops after first layer due to StopForward
        with torch.no_grad():
            for inp in calib_inputs:
                inp_device = inp.to(self.device)
                self.model(inp_device, use_cache=False)
        
        # Restore original forwards
        first_layer.forward = orig_layer_forward
        self.model.forward = orig_model_forward
        
        # Now process each layer sequentially using cached args
        for layer_idx, layer in enumerate(tqdm.tqdm(layers, desc="Finding best samples")):
            # Run THIS layer on cached hidden states
            layer_outputs: list[Tensor] = []
            
            with torch.no_grad():
                for batch_idx in range(len(cache_args[0])):
                    hs = cache_args[0][batch_idx].to(self.device)
                    # Run layer - LlamaDecoderLayer takes hidden_states as first arg + kwargs
                    # Use captured kwargs (attention_mask, position_embeddings, etc.)
                    layer_kwargs = {k: v[batch_idx] for k, v in cache_kwargs.items()}
                    out = layer(hs, **layer_kwargs)
                    if isinstance(out, tuple):
                        layer_outputs.append(out[0].cpu())
                    else:
                        layer_outputs.append(out.cpu())
            
            # Extract per-sample features using the configured feature_mode
            # (mean_pool by default for stability, or flatten for original behavior)
            features = self._extract_features(layer_outputs)

            # Build mapping from individual sample index back to (batch_idx, sub_idx)
            sample_to_batch_map = []
            for batch_idx, act in enumerate(layer_outputs):
                n_sub = act.shape[0] if act.dim() == 3 else 1
                for sub_idx in range(n_sub):
                    sample_to_batch_map.append((batch_idx, sub_idx))

            n_individual_samples = len(features)
            
            # Project and normalize features using the configured projection_mode
            data_proj = self._project_and_normalize(features)

            # K-Means
            kmeans = KMeans(
                n_clusters=self.compress_to_samples,
                random_state=self.seed,
                n_init=self.kmeans_n_init,
                max_iter=300,
            )


            labels = kmeans.fit_predict(data_proj)
            
            # Silhouette score - use actual individual sample count
            if self.compress_to_samples > 1 and self.compress_to_samples < n_individual_samples:
                score = silhouette_score(data_proj, labels)
                print(f"  Layer {layer_idx}: silhouette_score={score:.4f} (from {n_individual_samples} samples)")
                
                # Visualize if requested

                if visualize:
                    save_path = f"calibration_layer_{layer_idx}_activations_2d.png"
                    self._visualize_compression_2d(data_proj, kmeans, save_path)
                
                if score > best_score:
                    best_score = score
                    best_layer_idx = layer_idx
                    best_data_2d = data_proj.copy()
                    best_labels = labels.copy()
                    best_kmeans = kmeans
                    best_sample_map = sample_to_batch_map
            
            # Update cache_args[0] for next layer (current layer's output)
            cache_args[0] = layer_outputs
            
            del data_proj, labels, kmeans, features, layer_outputs, sample_to_batch_map



        
        if best_layer_idx >= 0:
            print(f"Best layer: {best_layer_idx} with silhouette_score={best_score:.4f}")
        
        return best_layer_idx, best_data_2d, best_labels, best_kmeans, best_sample_map

    def _cluster_concatenated_layers(
        self,
        calib_inputs: list[Tensor],
        visualize: bool = False,
    ) -> tuple[int, np.ndarray, Any, Any, list[tuple[int, int]]]:
        """
        Cluster using concatenated features from ALL layers.

        Instead of picking the single best layer by silhouette score, this
        method collects per-sample features from every decoder layer and
        concatenates them into a single feature vector per sample.  The
        concatenated features are then projected (PCA), normalized, and
        clustered with K-Means.

        This uses information from all layers simultaneously, which should
        produce more robust and representative clustering than any single
        layer.

        Memory: with mean_pool features, each sample is [hidden_dim] per
        layer.  Storing all layers: n_samples × n_layers × hidden_dim × 8
        bytes — trivially fits in GPU memory for typical model sizes.

        Args:
            calib_inputs: List of calibration tensors [N, seq_len]
            visualize: If True, save a 2D visualization of the final clustering

        Returns:
            Tuple of:
            - best_layer_idx: Always -1 (not applicable for concatenated mode)
            - best_data_2d: Projected concatenated data [n_samples, n_components]
            - best_labels: Cluster labels
            - best_kmeans: Fitted KMeans model
            - best_sample_map: Mapping from sample index to (batch_idx, sub_idx)
        """
        layers = self._get_decoder_layers()

        if not layers:
            print("  Warning: No decoder layers found")
            return -1, None, None, None, None

        self.model.eval()

        # Get first decoder layer (same pattern as _find_best_layer_layerwise)
        first_layer = layers[0]

        cache_args: List[List[Any]] = []
        cache_kwargs: Dict[str, List[Any]] = {}

        orig_layer_forward = first_layer.forward
        orig_model_forward = self.model.forward

        def layer_catcher(layer, *args, **kwargs):
            for idx, item in enumerate(args):
                if (idx + 1) > len(cache_args):
                    cache_args.append([])
                cache_args[idx].append(item.detach().cpu())
            for k, v in kwargs.items():
                if k not in cache_kwargs:
                    cache_kwargs[k] = []
                cache_kwargs[k].append(v.detach().cpu() if hasattr(v, 'detach') else v)
            raise StopForward

        first_layer.forward = types.MethodType(layer_catcher, first_layer)

        def model_forward_wrapper(_model, *m_args, **m_kwargs):
            try:
                return orig_model_forward(*m_args, **m_kwargs)
            except StopForward:
                return None

        self.model.forward = types.MethodType(model_forward_wrapper, self.model)

        with torch.no_grad():
            for inp in calib_inputs:
                inp_device = inp.to(self.device)
                self.model(inp_device, use_cache=False)

        first_layer.forward = orig_layer_forward
        self.model.forward = orig_model_forward

        # Collect features from all layers
        all_layer_features: list[list[Tensor]] = []
        sample_to_batch_map: list[tuple[int, int]] = []

        for layer_idx, layer in enumerate(
            tqdm.tqdm(layers, desc="Collecting all-layer features")
        ):
            layer_outputs: list[Tensor] = []

            with torch.no_grad():
                for batch_idx in range(len(cache_args[0])):
                    hs = cache_args[0][batch_idx].to(self.device)
                    layer_kwargs = {k: v[batch_idx] for k, v in cache_kwargs.items()}
                    out = layer(hs, **layer_kwargs)
                    if isinstance(out, tuple):
                        layer_outputs.append(out[0].cpu())
                    else:
                        layer_outputs.append(out.cpu())

            # Extract per-sample features for this layer
            features = self._extract_features(layer_outputs)
            all_layer_features.append(features)

            # Build sample_to_batch_map (only needed once, same for all layers)
            if layer_idx == 0:
                for batch_idx, act in enumerate(layer_outputs):
                    n_sub = act.shape[0] if act.dim() == 3 else 1
                    for sub_idx in range(n_sub):
                        sample_to_batch_map.append((batch_idx, sub_idx))

            # Update cache_args[0] for next layer
            cache_args[0] = layer_outputs
            del layer_outputs, features

        n_individual_samples = len(all_layer_features[0])
        n_layers = len(all_layer_features)

        # Concatenate features from all layers per sample
        # Each sample's feature: [n_layers * feature_dim]
        print(
            f"  Concatenating features from {n_layers} layers "
            f"({n_individual_samples} samples)..."
        )
        concat_features: list[Tensor] = []
        for sample_idx in range(n_individual_samples):
            sample_feats = [
                all_layer_features[layer_idx][sample_idx]
                for layer_idx in range(n_layers)
            ]
            concat_features.append(torch.cat(sample_feats, dim=0))

        # Free per-layer features (no longer needed)
        del all_layer_features

        # Project and normalize the concatenated features
        data_proj = self._project_and_normalize(concat_features)

        # K-Means
        kmeans = KMeans(
            n_clusters=self.compress_to_samples,
            random_state=self.seed,
            n_init=self.kmeans_n_init,
            max_iter=300,
        )
        labels = kmeans.fit_predict(data_proj)

        # Silhouette score
        if self.compress_to_samples > 1 and self.compress_to_samples < n_individual_samples:
            score = silhouette_score(data_proj, labels)
            print(
                f"  Concatenated clustering: silhouette_score={score:.4f} "
                f"(from {n_individual_samples} samples, {n_layers} layers)"
            )

        if visualize:
            self._visualize_compression_2d(data_proj, kmeans, "calibration_compression_2d.png")

        del concat_features

        # Return with best_layer_idx = -1 (not applicable)
        return -1, data_proj, labels, kmeans, sample_to_batch_map

    def _visualize_compression_2d(

        self,
        data_proj: np.ndarray,
        kmeans: KMeans,
        save_path: str,
    ) -> None:
        """
        Visualize clustering with gray points and red centroid markers.

        When the projected data has more than 2 dimensions, only the first two
        components are used for the scatter plot (with a note in the title).

        Args:
            data_proj: Projected data [n_samples, n_components]
            kmeans: Fitted KMeans model
            save_path: Path to save the PNG visualization
        """
        # Resolve save_path against output_dir when set
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = pathlib.Path(self.output_dir, save_path)

        _, ax = plt.subplots(figsize=(10, 8))

        # Plot all points in light gray (simple background) using first 2 dims
        ax.scatter(
            data_proj[:, 0],
            data_proj[:, 1],
            c='lightgray',
            alpha=0.5,
            s=50,
            edgecolors='w',
            linewidth=0.5,
        )

        # Mark cluster centroids with red X (first 2 dims)
        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            c='red',
            marker='X',
            s=200,
            label='Centroids',
            edgecolors='black',
            linewidths=2,
        )

        ax.set_xlabel('Projected Dimension 1', fontsize=12)
        ax.set_ylabel('Projected Dimension 2', fontsize=12)
        n_dims = data_proj.shape[1]
        title = f'Compression - {self.compress_to_samples} Clusters'
        if n_dims > 2:
            title += f' (showing first 2 of {n_dims} dims)'
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved compression visualization to {save_path}")

    
    def _select_representatives(
        self,
        data_2d: np.ndarray,
        labels: np.ndarray,
        kmeans: KMeans,
        calib_inputs: list[Tensor],
        sample_to_batch_map: Optional[list[tuple[int, int]]] = None,
    ) -> tuple[list[Tensor], list[float]]:
        """
        Select representative samples from 2D clustering results.
        
        For each cluster:
        1. Find all samples in the cluster
        2. Compute weight = cluster_size / n_samples
        3. Select sample closest to centroid as representative
        
        Args:
            data_2d: 2D projected data [n_samples, 2]
            labels: Cluster labels for each sample
            kmeans: Fitted KMeans model
            calib_inputs: Original calibration inputs (may be batched)
            sample_to_batch_map: Optional mapping from individual sample index 
                to (batch_idx, sub_idx) for extracting samples from batched inputs.
                If None, assumes calib_inputs contains individual samples.
            
        Returns:
            Tuple of:
            - Compressed list of compress_to_samples representative samples
            - List of weights (one per sample, proportional to cluster size)
        """
        n_samples = len(data_2d)  # Use actual number of samples in clustering
        representatives = []
        weights = []
        
        for k in range(self.compress_to_samples):
            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = cluster_mask.sum()
            
            # Weight = proportion of samples in this cluster
            weight = cluster_size / n_samples
            assert(len(cluster_indices) > 0)
            
            
            # Find sample closest to centroid within cluster
            cluster_data_2d = data_2d[cluster_mask]
            centroid = kmeans.cluster_centers_[k]
            distances = np.linalg.norm(cluster_data_2d - centroid, axis=1)
            best_local_idx = np.argmin(distances)
            best_idx = cluster_indices[best_local_idx]
            
            # Extract sample using the map if provided (for batched inputs)
            if sample_to_batch_map is not None:
                batch_idx, sub_idx = sample_to_batch_map[best_idx]
                representatives.append(calib_inputs[batch_idx][sub_idx:sub_idx+1, ...])
            else:
                representatives.append(calib_inputs[best_idx])
            weights.append(weight)
        
        print(f"  Sample weights: min={min(weights):.4f}, max={max(weights):.4f}")
        
        return representatives, weights
    
    def compress(
        self,
        calib_inputs: list[Tensor],
    ) -> tuple[list[Tensor], list[float]]:
        """
        Compress calibration dataset using K-Means on 2D-projected activations.
        
        Uses memory-efficient layerwise approach: processes one layer at a time
        instead of storing all layer activations simultaneously.
        
        Args:
            calib_inputs: List of calibration tensors [N, seq_len]
            
        Returns:
            Tuple of:
            - Compressed list of compress_to_samples representative samples
            - List of weights (one per sample, proportional to cluster size)
        """
        original_batch_size = calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1
        n_individual_samples = len(calib_inputs) * original_batch_size

        if n_individual_samples <= self.compress_to_samples:
            # Return uniform weights for uncompressed case
            weights = [1.0 / n_individual_samples] * n_individual_samples
            return calib_inputs, weights

        
        # Set global torch seed for reproducible random projections
        torch.manual_seed(self.seed)

        if self.concat_layers:
            # Concatenate features from ALL layers and cluster on the combined representation
            print(f"  Concatenating all-layer features from {len(calib_inputs) * original_batch_size} samples...")
            best_layer_idx, best_data_2d, best_labels, best_kmeans, best_sample_map = self._cluster_concatenated_layers(
                calib_inputs, visualize=self.visualize
            )
        else:
            # Find best layer using memory-efficient layerwise approach
            print(f"  Finding best compressed dataset from {len(calib_inputs) * original_batch_size} samples...")
            best_layer_idx, best_data_2d, best_labels, best_kmeans, best_sample_map = self._find_best_layer_layerwise(
                calib_inputs, visualize=self.visualize
            )

        
        # Visualize final compression result if requested
        if self.visualize and best_data_2d is not None:
            self._visualize_compression_2d(best_data_2d, best_kmeans, "calibration_compression_2d.png")
        
        if best_data_2d is None:
            # Fallback to simple subsampling
            print("  Warning: Clustering failed, using simple subsampling")

            # Unbatch into individual samples for subsampling
            individual_inputs = []
            for batched_input in calib_inputs:
                for i in range(batched_input.shape[0]):
                    individual_inputs.append(batched_input[i:i+1, ...])
            step = len(individual_inputs) // self.compress_to_samples
            uniform_weight = 1.0 / self.compress_to_samples
            representatives = []
            weights = []
            for i in range(self.compress_to_samples):
                idx = min(i * step, len(individual_inputs) - 1)
                representatives.append(individual_inputs[idx])
                weights.append(uniform_weight)
            return representatives, weights

        
        # Select representatives from best layer's clustering
        print(f"  Selecting {self.compress_to_samples} representatives from best layer...")
        representatives, weights = self._select_representatives(
            best_data_2d, best_labels, best_kmeans, calib_inputs, best_sample_map
        )
        
        return representatives, weights

    def _collect_all_layer_labels(
        self,
        calib_inputs: list[Tensor],
    ) -> tuple[list[np.ndarray], list[float], int]:
        """
        Collect K-Means cluster labels for *every* decoder layer.

        This reuses the same memory-efficient layerwise inference loop as
        ``_find_best_layer_layerwise`` but stores the labels (and silhouette
        scores) for all layers instead of only the best one.

        Args:
            calib_inputs: List of calibration tensors [N, seq_len]

        Returns:
            Tuple of:
            - all_labels: List of label arrays, one per layer
            - all_silhouettes: List of silhouette scores, one per layer
            - n_individual_samples: Number of individual samples that were clustered
        """
        layers = self._get_decoder_layers()

        if not layers:
            print("  Warning: No decoder layers found")
            return [], [], 0

        all_labels: list[np.ndarray] = []
        all_silhouettes: list[float] = []

        self.model.eval()

        # Get first decoder layer (same pattern as LlamaGPTQQuantizer.prepare)
        first_layer = layers[0]

        # Cache args and kwargs from first layer
        cache_args: List[List[Any]] = []
        cache_kwargs: Dict[str, List[Any]] = {}

        # Store original forwards
        orig_layer_forward = first_layer.forward
        orig_model_forward = self.model.forward

        # Define catcher that stores args and kwargs, then raises StopForward
        def layer_catcher(layer, *args, **kwargs):
            for idx, item in enumerate(args):
                if (idx + 1) > len(cache_args):
                    cache_args.append([])
                cache_args[idx].append(item.detach().cpu())
            for k, v in kwargs.items():
                if k not in cache_kwargs:
                    cache_kwargs[k] = []
                cache_kwargs[k].append(v.detach().cpu() if hasattr(v, 'detach') else v)
            raise StopForward

        first_layer.forward = types.MethodType(layer_catcher, first_layer)

        def model_forward_wrapper(_model, *m_args, **m_kwargs):
            try:
                return orig_model_forward(*m_args, **m_kwargs)
            except StopForward:
                return None

        self.model.forward = types.MethodType(model_forward_wrapper, self.model)

        # Run model to populate cache_args and cache_kwargs
        with torch.no_grad():
            for inp in calib_inputs:
                inp_device = inp.to(self.device)
                self.model(inp_device, use_cache=False)

        # Restore original forwards
        first_layer.forward = orig_layer_forward
        self.model.forward = orig_model_forward

        n_individual_samples = 0

        for layer_idx, layer in enumerate(
            tqdm.tqdm(layers, desc="Collecting per-layer labels")
        ):
            layer_outputs: list[Tensor] = []

            with torch.no_grad():
                for batch_idx in range(len(cache_args[0])):
                    hs = cache_args[0][batch_idx].to(self.device)
                    layer_kwargs = {k: v[batch_idx] for k, v in cache_kwargs.items()}
                    out = layer(hs, **layer_kwargs)
                    if isinstance(out, tuple):
                        layer_outputs.append(out[0].cpu())
                    else:
                        layer_outputs.append(out.cpu())

            # Extract per-sample features using the configured feature_mode
            features = self._extract_features(layer_outputs)

            n_individual_samples = len(features)

            # Project and normalize features using the configured projection_mode
            data_proj = self._project_and_normalize(features)

            kmeans = KMeans(
                n_clusters=self.compress_to_samples,
                random_state=self.seed,
                n_init=self.kmeans_n_init,
                max_iter=300,
            )
            labels = kmeans.fit_predict(data_proj)


            all_labels.append(labels.copy())

            if (
                self.compress_to_samples > 1
                and self.compress_to_samples < n_individual_samples
            ):
                score = silhouette_score(data_proj, labels)
                all_silhouettes.append(score)
                print(
                    f"  Layer {layer_idx}: silhouette_score={score:.4f} "
                    f"(from {n_individual_samples} samples)"
                )
            else:
                all_silhouettes.append(float("nan"))

            # Update cache_args[0] for next layer
            cache_args[0] = layer_outputs

            del data_proj, labels, kmeans, features, layer_outputs

        return all_labels, all_silhouettes, n_individual_samples

    def _compute_pairwise_agreement(
        self,
        all_labels: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise ARI and NMI matrices between all layers.

        Args:
            all_labels: List of label arrays, one per layer.

        Returns:
            Tuple of (ari_matrix, nmi_matrix), each of shape [n_layers, n_layers].
        """
        n_layers = len(all_labels)
        ari_matrix = np.zeros((n_layers, n_layers))
        nmi_matrix = np.zeros((n_layers, n_layers))

        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    ari_matrix[i, j] = 1.0
                    nmi_matrix[i, j] = 1.0
                elif j > i:
                    ari = adjusted_rand_score(all_labels[i], all_labels[j])
                    nmi = normalized_mutual_info_score(all_labels[i], all_labels[j])
                    ari_matrix[i, j] = ari
                    ari_matrix[j, i] = ari
                    nmi_matrix[i, j] = nmi
                    nmi_matrix[j, i] = nmi

        return ari_matrix, nmi_matrix

    def _compute_co_occurrence_matrix(
        self,
        all_labels: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute the sample co-occurrence matrix.

        For each pair of samples (i, j), the entry is the fraction of layers
        in which i and j are assigned to the same cluster.

        Args:
            all_labels: List of label arrays, one per layer.

        Returns:
            Co-occurrence matrix of shape [n_samples, n_samples].
        """
        n_layers = len(all_labels)
        n_samples = len(all_labels[0])

        # Build binary co-cluster indicator per layer, then average
        co_occurrence = np.zeros((n_samples, n_samples), dtype=np.float64)

        for labels in all_labels:
            # co_cluster[i, j] = 1 if labels[i] == labels[j]
            same = (labels[:, None] == labels[None, :]).astype(np.float64)
            co_occurrence += same

        co_occurrence /= n_layers
        return co_occurrence

    def _visualize_heatmap(
        self,
        matrix: np.ndarray,
        title: str,
        save_path: str,
        *,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        """
        Save a heatmap visualization of a 2D matrix.

        Args:
            matrix: 2D array to visualize.
            title: Plot title.
            save_path: Filename for the PNG.
            cmap: Matplotlib colormap name.
            vmin: Minimum value for color scale.
            vmax: Maximum value for color scale.
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = pathlib.Path(self.output_dir, save_path)

        n = matrix.shape[0]
        fig_size = max(8, min(n * 0.4, 20))

        _, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(
            matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax
        )
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add tick labels for layer-pair heatmaps
        if n <= 40:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(range(n), fontsize=7)
            ax.set_yticklabels(range(n), fontsize=7)
            ax.set_xlabel("Layer", fontsize=12)
            ax.set_ylabel("Layer", fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved heatmap to {save_path}")

    def _visualize_silhouette_line(
        self,
        all_silhouettes: list[float],
        save_path: str,
    ) -> None:
        """
        Save a line plot of per-layer silhouette scores.

        Args:
            all_silhouettes: List of silhouette scores, one per layer.
            save_path: Filename for the PNG.
        """
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = pathlib.Path(self.output_dir, save_path)

        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            range(len(all_silhouettes)),
            all_silhouettes,
            marker="o",
            linewidth=2,
            markersize=6,
        )
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Silhouette Score", fontsize=12)
        ax.set_title("Per-Layer Silhouette Score", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(all_silhouettes)))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved silhouette line plot to {save_path}")

    def assess_cluster_consistency(
        self,
        calib_inputs: list[Tensor],
    ) -> dict:
        """
        Assess cross-layer cluster consistency of the calibration set compression.

        Collects K-Means labels for every decoder layer, then computes:

        1. **Pairwise ARI (Adjusted Rand Index)** between all layer pairs —
           measures how often two layers agree on which sample-pairs are
           co-clustered, corrected for chance.  Range: [-1, 1], 1 = perfect.

        2. **Pairwise NMI (Normalized Mutual Information)** between all layer
           pairs — information-theoretic measure of shared clustering
           structure.  Range: [0, 1], 1 = identical.

        3. **Sample co-occurrence matrix** — for each sample pair (i, j), the
           fraction of layers in which they land in the same cluster.

        4. **Per-layer silhouette scores** — already computed during label
           collection, visualized together here.

        Saves heatmap visualizations (when ``self.visualize`` is True or
        ``self.output_dir`` is set) and prints a summary.

        Args:
            calib_inputs: List of calibration tensors [N, seq_len].

        Returns:
            A dictionary with keys:
              - ``all_labels``: list of per-layer label arrays
              - ``all_silhouettes``: list of per-layer silhouette scores
              - ``ari_matrix``: [n_layers, n_layers] ARI matrix
              - ``nmi_matrix``: [n_layers, n_layers] NMI matrix
              - ``co_occurrence``: [n_samples, n_samples] co-occurrence matrix
              - ``mean_ari``: mean ARI across all layer pairs
              - ``mean_nmi``: mean NMI across all layer pairs
              - ``mean_consecutive_ari``: mean ARI between consecutive layers
              - ``mean_consecutive_nmi``: mean NMI between consecutive layers
        """
        original_batch_size = (
            calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1
        )
        n_individual_samples = len(calib_inputs) * original_batch_size

        if n_individual_samples <= self.compress_to_samples:
            print(
                "  Warning: Not enough samples for clustering "
                f"({n_individual_samples} <= {self.compress_to_samples}). "
                "Skipping consistency assessment."
            )
            return {}

        # Set global torch seed for reproducible random projections
        torch.manual_seed(self.seed)

        print(
            f"  Collecting per-layer cluster labels from "
            f"{n_individual_samples} samples..."
        )
        all_labels, all_silhouettes, n_clustered = self._collect_all_layer_labels(
            calib_inputs
        )

        if not all_labels:
            print("  Warning: No labels collected. Skipping consistency assessment.")
            return {}

        n_layers = len(all_labels)

        # Compute pairwise agreement matrices
        print("  Computing pairwise ARI and NMI matrices...")
        ari_matrix, nmi_matrix = self._compute_pairwise_agreement(all_labels)

        # Compute co-occurrence matrix
        print("  Computing sample co-occurrence matrix...")
        co_occurrence = self._compute_co_occurrence_matrix(all_labels)

        # Summary statistics
        # Exclude diagonal (self-agreement = 1.0) from means
        mask = ~np.eye(n_layers, dtype=bool)
        mean_ari = ari_matrix[mask].mean()
        mean_nmi = nmi_matrix[mask].mean()

        # Consecutive-layer agreement
        if n_layers > 1:
            consecutive_aris = [
                ari_matrix[i, i + 1] for i in range(n_layers - 1)
            ]
            consecutive_nmis = [
                nmi_matrix[i, i + 1] for i in range(n_layers - 1)
            ]
            mean_consecutive_ari = float(np.mean(consecutive_aris))
            mean_consecutive_nmi = float(np.mean(consecutive_nmis))
        else:
            mean_consecutive_ari = float("nan")
            mean_consecutive_nmi = float("nan")

        # Find best and worst layer pairs
        ari_no_diag = ari_matrix.copy()
        np.fill_diagonal(ari_no_diag, -2.0)
        best_pair_flat = np.argmax(ari_no_diag)
        worst_pair_flat = np.argmin(ari_no_diag)
        best_i, best_j = np.unravel_index(best_pair_flat, ari_no_diag.shape)
        worst_i, worst_j = np.unravel_index(worst_pair_flat, ari_no_diag.shape)

        # Print summary
        print("\n" + "=" * 70)
        print("Cluster Consistency Assessment Summary")
        print("=" * 70)
        print(f"  Layers analyzed          : {n_layers}")
        print(f"  Samples clustered        : {n_clustered}")
        print(f"  Clusters per layer       : {self.compress_to_samples}")
        print()
        print(f"  Mean ARI (all pairs)      : {mean_ari:.4f}")
        print(f"  Mean NMI (all pairs)      : {mean_nmi:.4f}")
        print(f"  Mean ARI (consecutive)    : {mean_consecutive_ari:.4f}")
        print(f"  Mean NMI (consecutive)    : {mean_consecutive_nmi:.4f}")
        print()
        print(
            f"  Best  layer pair (ARI)    : layers {best_i},{best_j} "
            f"(ARI={ari_matrix[best_i, best_j]:.4f})"
        )
        print(
            f"  Worst layer pair (ARI)    : layers {worst_i},{worst_j} "
            f"(ARI={ari_matrix[worst_i, worst_j]:.4f})"
        )
        print()
        print("  Per-layer silhouette scores:")
        for i, s in enumerate(all_silhouettes):
            print(f"    Layer {i:3d}: {s:.4f}")
        print()

        # Interpretation hints
        if mean_ari > 0.7:
            print("  >> Clusters are STABLE across layers (mean ARI > 0.7).")
            print("     The choice of which layer to cluster on has limited impact.")
        elif mean_ari > 0.4:
            print("  >> Clusters are MODERATELY stable across layers (0.4 < mean ARI <= 0.7).")
            print("     Some drift exists; the 'best layer' heuristic is reasonable.")
        else:
            print("  >> Clusters are UNSTABLE across layers (mean ARI <= 0.4).")
            print("     The choice of which layer to cluster on matters significantly.")
        print("=" * 70 + "\n")

        # Visualizations
        if self.visualize or self.output_dir is not None:
            self._visualize_heatmap(
                ari_matrix,
                "Adjusted Rand Index (ARI) Between Layers",
                "cluster_consistency_ari_heatmap.png",
                cmap="RdYlGn",
                vmin=-0.2,
                vmax=1.0,
            )
            self._visualize_heatmap(
                nmi_matrix,
                "Normalized Mutual Information (NMI) Between Layers",
                "cluster_consistency_nmi_heatmap.png",
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
            )
            self._visualize_heatmap(
                co_occurrence,
                "Sample Co-occurrence Matrix (fraction of layers co-clustered)",
                "cluster_consistency_co_occurrence.png",
                cmap="RdYlBu",
                vmin=0.0,
                vmax=1.0,
            )
            self._visualize_silhouette_line(
                all_silhouettes,
                "cluster_consistency_silhouette.png",
            )

        return {
            "all_labels": all_labels,
            "all_silhouettes": all_silhouettes,
            "ari_matrix": ari_matrix,
            "nmi_matrix": nmi_matrix,
            "co_occurrence": co_occurrence,
            "mean_ari": float(mean_ari),
            "mean_nmi": float(mean_nmi),
            "mean_consecutive_ari": mean_consecutive_ari,
            "mean_consecutive_nmi": mean_consecutive_nmi,
        }


# Explicit mapping from MX dtype strings to element formats
MX_DTYPE_TO_ELEM_FORMAT = {
    "mxint8": "int8",
    "mxfp4": "fp4",
    "mxfp6": "fp6",
    "mxfp8_e4m3": "fp8_e4m3",
    "mxfp8_e5m2": "fp8_e5m2",
}

# Explicit mapping from affine dtype strings to (bits, signed) tuples
AFFINE_DTYPE_TO_CONFIG = {
    "int4": (4, True),
    "int8": (8, True),
    "int16": (16, True),
    "int32": (32, True),
    "uint4": (4, False),
    "uint8": (8, False),
    "uint16": (16, False),
    "uint32": (32, False),
}


def quant_spec_from_dtype_string(dtype_str: str):
    """
    Convert a dtype string to a QuantSpec (either affine or mx).

    For simple data types like "int16", "int8", "uint8", returns affine(...).
    For MX types like "mxint8", "mxfp4", returns mx(...) QuantSpec.

    Args:
        dtype_str: A dtype string such as "int16", "uint8", "mxint8", "mxfp4".

    Returns:
        A QuantSpec instance:
          - affine(DType(...)) for simple integer types
          - mx(...) for microscaling types

    Raises:
        ValueError: For unrecognized dtype strings.
    """
    if dtype_str in MX_DTYPE_TO_ELEM_FORMAT:
        elem_format = MX_DTYPE_TO_ELEM_FORMAT[dtype_str]
        return mx(elem_format=elem_format)

    if dtype_str in AFFINE_DTYPE_TO_CONFIG:
        bits, signed = AFFINE_DTYPE_TO_CONFIG[dtype_str]
        return affine(DType(bits=bits, signed=signed))

    raise ValueError(
        f"Unknown dtype string {dtype_str!r}. "
        f"Expected one of affine: {list(AFFINE_DTYPE_TO_CONFIG.keys())} "
        f"or MX: {list(MX_DTYPE_TO_ELEM_FORMAT.keys())}."
    )


def quantize_using_PTQ(q_m, calib_inputs, args):
    """
    Wrap the model with PTQ wrappers, calibrate observers, and convert it.
    """
    if args.no_PTQ:
        return q_m

    print("Wrapping layers with PTQWrapper …")
    print(f"Using PTQ execution profile: {args.profile}")

    
    linear_spec = quant_spec_from_dtype_string(args.linear_io_qdtype)
    norm_spec = quant_spec_from_dtype_string(args.norm_io_qdtype)
    softmax_spec = quant_spec_from_dtype_string(args.softmax_io_qdtype)
    spinquant_io_spec = (
        quant_spec_from_dtype_string(args.spinquant_io_qdtype)
        if args.spinquant_io_qdtype is not None
        else linear_spec
    )
    lm_head_io_spec = (
        quant_spec_from_dtype_string(args.lm_head_io_qdtype)
        if args.lm_head_io_qdtype is not None
        else linear_spec
    )

    qcfg = build_llm_ptq_config(
        model_type="llama",
        num_hidden_layers=len(q_m.model.layers),
        activation=affine(DType.int(16)),
        linear=linear_spec,
        linear_weight=affine(DType.uint(args.linear_weight_bits)),
        embedding_weight=affine(DType.uint(args.embedding_weight_bits)),
        lm_head_weight=affine(DType.uint(args.lm_head_weight_bits)),
        spin_rotation_weight=(
            None
            if args.no_spinquant
            else affine(DType.int(args.spin_rotation_weight_bits))
        ),
        spinquant_io=spinquant_io_spec,
        lm_head_io=lm_head_io_spec,
        norm=norm_spec,
        norm_weight=affine(DType.int(16)),
        softmax=softmax_spec,
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


def quantize_using_PTQ_and_LlamaGPTQ(model, calib_inputs, args, sample_weights=None):
    """
    Combined PTQ + LlamaGPTQ pipeline.

    When ``--use_llama_gptq`` and PTQ are both enabled the execution order
    changes so that LlamaGPTQ can operate on the PTQ-wrapped model:

      1. PTQ ``prepare()``  — wraps every layer with PTQWrapper / observers.
      2. LlamaGPTQ ``prepare()`` + calibration forward passes — collects
         first-layer inputs for GPTQ while also populating activation
         observers in CALIB mode.
      3. LlamaGPTQ ``convert()`` — runs GPTQ weight quantization layer by
         layer.  After each layer it injects the resulting weight qparams
         into the PTQ weight observers and calls ``freeze_qparams()`` so
         subsequent forward passes use fake-quantized (QUANT mode) outputs.
      4. PTQ ``convert()`` — finalises the PTQ graph (all observers already
         frozen).

    Because LlamaGPTQ's forward passes during collection and re-forward
    already drive the PTQ activation observers, **no separate activation
    calibration pass is needed** for this path.
    
    Args:
        sample_weights: Optional list of weights for weighted Hessian accumulation
    """
    # Step 1: PTQ prepare
    print("Wrapping layers with PTQWrapper …")
    print(f"Using PTQ execution profile: {args.profile}")
    assert args.norm_io_qdtype != "int16" #otherwise it is incorrect on layers joint

    linear_spec = quant_spec_from_dtype_string(args.linear_io_qdtype)
    norm_spec = quant_spec_from_dtype_string(args.norm_io_qdtype)
    softmax_spec = quant_spec_from_dtype_string(args.softmax_io_qdtype)
    spinquant_io_spec = (
        quant_spec_from_dtype_string(args.spinquant_io_qdtype)
        if args.spinquant_io_qdtype is not None
        else linear_spec
    )
    lm_head_io_spec = (
        quant_spec_from_dtype_string(args.lm_head_io_qdtype)
        if args.lm_head_io_qdtype is not None
        else linear_spec
    )

    qcfg = build_llm_ptq_config(
        model_type="llama",
        num_hidden_layers=len(model.model.layers),
        activation=affine(DType.int(16)),
        linear=linear_spec,
        linear_weight=affine(DType.uint(args.linear_weight_bits)),
        embedding_weight=affine(DType.uint(args.embedding_weight_bits)),
        lm_head_weight=affine(DType.uint(args.lm_head_weight_bits)),
        spin_rotation_weight=(
            None
            if args.no_spinquant
            else affine(DType.int(args.spin_rotation_weight_bits))
        ),
        spinquant_io=spinquant_io_spec,
        lm_head_io=lm_head_io_spec,
        norm=norm_spec,
        norm_weight=affine(DType.int(16)),
        softmax=softmax_spec,
        strict_wrap=True,
        profile=args.profile,
    )
    q_m = prepare(model, qcfg)

    # Step 2: LlamaGPTQ prepare + calibration
    # Temporarily remove the PTQ quantizer attribute so that the second
    # ``prepare()`` call does not raise "prepare() already has been called."
    # We will restore it after LlamaGPTQ convert().
    ptq_quantizer = getattr(q_m, "tico_quantizer", None)
    if ptq_quantizer is not None:
        delattr(q_m, "tico_quantizer")

    print("Applying LlamaGPTQ on PTQ-wrapped model …")
    sens = compute_or_load_sensitivity(
        model, calib_inputs, args, sample_weights=sample_weights
    )
    gptq_config = build_gptq_config(args, sensitivity=sens, sample_weights=sample_weights)
    q_m = prepare(q_m, gptq_config, inplace=True)

    iterator = calib_inputs
    if not args.no_tqdm:
        iterator = tqdm.tqdm(calib_inputs, desc="LlamaGPTQ calibration")

    with torch.no_grad():
        for inp in iterator:
            q_m(inp.to(args.device))

    # Step 3: LlamaGPTQ convert (includes freeze_qparams per layer)
    q_m = convert(q_m, inplace=True)
    #print_minmax_values(q_m)
    
    # Clean up GPTQ quantizers that are no longer needed
    clear_gptq_quantizers(q_m)

    # Step 4: PTQ convert (all observers are already frozen)
    # Restore the PTQ quantizer that was saved before LlamaGPTQ prepare()
    # so that PTQ convert() can find it.
    #if ptq_quantizer is not None:
    #    setattr(q_m, "tico_quantizer", ptq_quantizer)
    #q_m = convert(q_m)

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

    # to prevent export errors let's evaluate ppl on exported fake_quantized model
   # prev_use_cache = q_m.wrapped.config.use_cache
   # q_m.wrapped.config.use_cache = False
   # eval_exported = False
   # if eval_exported:
   #     with torch.no_grad():
   #         q_m.eval()
   #         q_m.cpu()
   #         test_ids = enc.input_ids[0]
   #         test_ids_batch = []
   #         if hasattr(q_m, "config"):
   #             assert hasattr(q_m, "config")
   #             model_config = q_m.config
   #         else:
   #             assert hasattr(q_m.wrapped, "config")
   #             model_config = q_m.wrapped.config
   #         if hasattr(model_config, "text_config"):
   #             model_config = model_config.text_config
   #         assert hasattr(model_config, "max_position_embeddings")
   #         assert isinstance(model_config.max_position_embeddings, int)
   #         max_length = model_config.max_position_embeddings
   #         nsamples = test_ids.numel() // max_length
#
   #         for i in range(nsamples):
   #             batch = test_ids[(i * max_length) : ((i + 1) * max_length)]  # noqa E203
   #             test_ids_batch.append(batch.unsqueeze(0))
#
   #         rnd_input = torch.randint_like(
   #             test_ids_batch[0], 0, tokenizer.vocab_size - 1
   #         )  # just random ids
   #         device = "cuda"
   #         exported_program = torch.export.export(
   #             q_m.to(device),
   #             (rnd_input.to(device),),
   #             kwargs=None,
   #             dynamic_shapes=None,
   #             strict=False,
   #         )
   #         ppl = evaluate_ppl_of_model_on_dataset(
   #             exported_program.module(), test_ids_batch, device=device
   #         )
   #         print("\n┌── Wikitext-2 test perplexity ─────────────")
   #         print(f"│ exported_int16 : {ppl:8.2f}")
   #         print("└───────────────────────────────────────────")
   # q_m.wrapped.config.use_cache = prev_use_cache


def get_sensitivities_info_name(
    model,
    dataset,
    seed,
    n_samples,
    *,
    seq_len: int | None = None,
    spinquant: bool = False,
    compressed_n_samples: int | None = None,
) -> str:
    """
    Build a filename for stored sensitivity calibration results.

    Args:
        model: Model whose sensitivity was computed.
        dataset: Short name of the calibration dataset.
        seed: Random seed used for dataset generation.
        n_samples: Original number of samples for calibration.
        seq_len: Calibration sequence length used when generating samples.
        spinquant: Whether SpinQuant preprocessing was applied before
            sensitivity computation.  SpinQuant changes model weights, which
            changes activations and thus sensitivity values.
        compressed_n_samples: Number of samples actually used after compression
            (only included when compression was applied).

    Returns:
        Filename string for the sensitivity results.
    """
    model_name = model.config.name_or_path.replace("/", "_")

    parts = ["sensitivities_for", model_name, dataset, str(n_samples)]

    if compressed_n_samples is not None and compressed_n_samples < n_samples:
        parts.append(f"compressed_{compressed_n_samples}")

    if seq_len is not None:
        parts.append(f"sl{seq_len}")

    if spinquant:
        parts.append("SQ")

    parts.append(str(seed))

    name = "_".join(parts) + ".pt"
    return name


def _io_qdtype_short(dtype_str: str | None) -> str:
    """
    Return a short name for an I/O quantization dtype.

    MX-family dtypes (``mxint8``, ``mxfp4``, …) are collapsed to ``MX``.
    All other (affine integer) dtypes are represented as ``int16``.
    """
    if dtype_str is None:
        return "int16"
    if dtype_str.startswith("mx"):
        return "MX"
    return "int16"


def get_ptq_model_name(model, args):
    """
    Build a filename for a saved PTQ checkpoint.

    The name encodes every option that can change the quantized model output
    so that two runs with different settings never collide on the same file.
    """
    model_name = model.config.name_or_path.replace("/", "_")

    parts: list[str] = [f"PTQ_{model_name}"]

    # --- Preprocessing / algorithm flags -----------------------------------
    if not args.no_spinquant:
        parts.append("SQ")
    if args.enable_CLE:
        parts.append("CLE")
    if not args.no_GPTQ:
        parts.append("GPTQ")
        if args.gptq_mse:
            parts.append(args.gptq_mse)
        # GPTQ numeric / variant options
        parts.append(f"pd{args.gptq_percdamp}")
        if args.gptq_adaptive_percdamp:
            parts.append("ad")
            parts.append(f"ctg{args.gptq_cond_threshold_good}")
        if args.gptq_v2:
            parts.append("v2")
        if args.gptq_lm_head:
            parts.append("lmh")
        if args.gptq_use_orig_model_inference:
            parts.append("orig")
        if args.gptq_use_iterate:
            parts.append("iter")
        if args.llama_gptq:
            parts.append("lgptq")
        if args.llama_gptq_sequential:
            parts.append("seq")
        if args.llama_gptq_no_ptq:
            parts.append("noptq")
        if args.llama_gptq_use_subgroup_runner:
            parts.append("sgr")

    # --- Weight / activation bit-widths -------------------------------------
    parts.append(f"wb{args.linear_weight_bits}")
    parts.append(f"eb{args.embedding_weight_bits}")
    parts.append(f"lhb{args.lm_head_weight_bits}")
    if not args.no_spinquant:
        parts.append(f"spqb{args.spin_rotation_weight_bits}")

    # --- I/O quantization dtype (single representative) --------------------
    parts.append(_io_qdtype_short(args.linear_io_qdtype))


    # --- Calibration options ------------------------------------------------
    parts.append(str(args.nsamples_for_qcalibration))
    parts.append(f"bs{args.batch}")
    if (
        args.calibration_samples_to_use is not None
        and args.calibration_samples_to_use != args.nsamples_for_qcalibration
    ):
        parts.append(f"used{args.calibration_samples_to_use}")
    parts.append(f"sl{args.calibrate_seq_len}")
    parts.append(_get_calib_dataset_short_name(args))
    if args.decode_calibration_steps > 0:
        parts.append(f"dc{args.decode_calibration_steps}")

    # --- Execution profile --------------------------------------------------
    parts.append(args.profile)

    # --- Seed ---------------------------------------------------------------
    parts.append(str(args.seed))

    name = "_".join(parts) + ".pt"
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
    print(f"Calibration dataset    : {_get_calib_dataset_display_name(args)}")
    print(f"Max seq length         : {args.max_seq_len}")
    print(f"Profile                : {args.profile}")
    print()
    print("--- GPTQ ---")
    print(f"GPTQ MSE               : {args.gptq_mse}")
    print(f"GPTQ percdamp          : {args.gptq_percdamp}")
    print(f"GPTQv2                 : {args.gptq_v2}")
    print(f"GPTQ orig model infer  : {args.gptq_use_orig_model_inference}")
    print(f"GPTQ adaptive percdamp : {args.gptq_adaptive_percdamp}")
    print(f"GPTQ cond threshold    : {args.gptq_cond_threshold_good}")
    print(f"GPTQ use iterate       : {args.gptq_use_iterate}")
    print(f"LlamaGPTQ              : {args.llama_gptq}")
    print(f"LlamaGPTQ sequential   : {args.llama_gptq_sequential}")
    print(f"LlamaGPTQ no PTQ       : {args.llama_gptq_no_ptq}")
    print()
    print("--- Activation quantization ---")
    print(f"Linear IO qdtype       : {args.linear_io_qdtype}")
    print()
    print("--- Calibration ---")
    print(f"Batch size             : {args.batch}")
    print(f"Calibration samples    : {args.calibration_samples_to_use}")
    print(f"Compression n_components: {args.calibration_compression_n_components}")
    print(f"Decode calibration     : {args.decode_calibration_steps}")
    print(f"Calibration dataset    : {args.calibration_dataset_path}")
    print(f"Calibration dataset mix: {args.calibration_dataset_mix}")
    print(f"PPL filter percentile  : {args.ppl_filter_percentile}")
    print()
    print("--- Output ---")
    print(f"Output dir             : {args.output_dir}")
    print(f"Save artifacts         : {args.save}")
    print()


def print_pytorch_environment() -> None:
    """
    Print PyTorch-related environment information for reproducibility.

    Includes PyTorch and NumPy versions, as well as CUDA/cuDNN
    availability and device details (when applicable).
    """

    print("=== PyTorch Environment ===")
    print(f"PyTorch version         : {torch.__version__}")
    print(f"NumPy version           : {np.__version__}")
    print(f"CUDA compiled version   : {torch.version.cuda}")
    print(f"HIP/ROCm version        : {torch.version.hip}")

    cudnn_enabled = torch.backends.cudnn.enabled
    cudnn_version = (
        torch.backends.cudnn.version() if cudnn_enabled else "N/A (cuDNN disabled)"
    )
    print(f"cuDNN version           : {cudnn_version}")
    print(f"cuDNN enabled           : {cudnn_enabled}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available          : {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count       : {device_count}")
        if device_count > 0:
            current_idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current_idx)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"CUDA device name        : {props.name}")
            print(f"CUDA total memory (GB)  : {total_mem_gb:.2f}")

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
        #device_map=dev_map,
    ).eval().to(args.device)

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


def get_calibration_dataset_name(
    seed,
    n_samples,
    dataset_name: str = "wiki",
    compressed_n_samples=None,
    *,
    seq_len: int | None = None,
    spinquant: bool = False,
    concat_layers: bool = False,
) -> str:
    """
    Build a filename for stored calibration dataset.

    Args:
        seed: Random seed used for dataset generation
        n_samples: Original number of samples for calibration
        dataset_name: Short name of the calibration dataset (e.g. ``"wiki"``,
            ``"mmlu"``).
        compressed_n_samples: Number of samples after compression (if compression was applied)
        seq_len: Calibration sequence length used when generating samples.
        spinquant: Whether SpinQuant preprocessing was applied before
            compression.  SpinQuant changes model weights, which affects the
            K-Means clustering used for compression, so the selected
            representative samples differ.
        concat_layers: Whether calibration set compression used the
            concatenated-all-layers approach (``True``) or the best-single-layer
            approach (``False``).  Different clustering methods produce
            different representative samples, so this is encoded in the filename.

    Returns:
        Filename string for the calibration dataset
    """
    name_parts = ["calibration_dataset", dataset_name, str(n_samples)]

    # Add compression info if compression was applied
    if compressed_n_samples is not None and compressed_n_samples < n_samples:
        # Indicate which clustering mode was used:
        #   "concat" = concatenated all-layers features
        #   "bestlayer" = best single layer by silhouette score
        cluster_mode = "concat" if concat_layers else "bestlayer"
        name_parts.append(f"compressed_{compressed_n_samples}_{cluster_mode}")

    # Sequence length affects which token windows are sampled
    if seq_len is not None:
        name_parts.append(f"sl{seq_len}")

    # SpinQuant changes model weights, affecting K-Means compression results
    if spinquant:
        name_parts.append("SQ")

    name_parts.append(str(seed))
    name = "_".join(name_parts) + ".pt"

    return name





def _apply_calibration_compression(
    calib_inputs: list[torch.Tensor],
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[torch.Tensor], Optional[list[float]]]:
    """
    Compress calibration dataset using K-Means clustering if --calibration_samples_to_use is specified.
    
    Args:
        calib_inputs: List of calibration tensors (may be batched)
        model: Model used for extracting layer activations
        args: Parsed command-line arguments
        device: Device to run compression on
        
    Returns:
        Tuple of:
        - Compressed list of calibration samples
        - Optional list of sample weights (if compression was applied)
    """
    sample_weights = None
    
    if args.calibration_samples_to_use is None:
        return calib_inputs, sample_weights
    
    if args.calibration_samples_to_use < 1:
        raise ValueError(
            f"--calibration_samples_to_use must be positive, "
            f"got {args.calibration_samples_to_use}"
        )
    
    # Track original batch size for rebatching after compression
    original_batch_size = calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1
    n_individual_samples = len(calib_inputs) * original_batch_size
    
    if args.calibration_samples_to_use >= n_individual_samples:
        print(
            f"[Info] --calibration_samples_to_use ({args.calibration_samples_to_use}) "
            f"is >= available samples ({n_individual_samples}). "
            f"Using all {n_individual_samples} samples (no compression needed)."
        )
        sample_weights = [1.0 / n_individual_samples] * n_individual_samples
        return calib_inputs, sample_weights

    
    # Unbatch: flatten batched inputs into individual samples (batch_size=1)
  #  if original_batch_size > 1:
  #      individual_inputs = []
  #      for batched_input in calib_inputs:
  #          for i in range(batched_input.shape[0]):
  #              individual_inputs.append(batched_input[i:i+1, ...])
  #      calib_inputs_for_compression = individual_inputs
  #      print(f"  Unbatched {len(calib_inputs)} batches into {len(individual_inputs)} individual samples")
  #  else:
  #      calib_inputs_for_compression = calib_inputs
  #  
  #  print(
  #      f"Compressing calibration dataset from {len(calib_inputs_for_compression)} to "
  #      f"{args.calibration_samples_to_use} samples using K-Means clustering..."
  #  )
    
    # Run compression on individual samples
    compressor = CalibrationSetCompressor(
        model=model,
        compress_to_samples=args.calibration_samples_to_use,
        seed=args.seed,
        device=device,
        n_layers_to_use=1,
        n_components=args.calibration_compression_n_components,
        visualize=args.visualize_calibration_compression,
        output_dir=args.output_dir,
        feature_mode=args.compression_feature_mode,
        projection_mode=args.compression_projection_mode,
        normalize_features=(args.compression_normalize_features == "true"),
        kmeans_n_init=args.compression_kmeans_n_init,
        concat_layers=args.compression_concat_layers,
    )

    compressed_inputs, compressed_weights = compressor.compress(calib_inputs)
    
    # Rebatch: regroup compressed individual samples back to original batch_size
    if original_batch_size > 1:
        rebatched_inputs = []
        rebatched_weights = []
        for i in range(0, len(compressed_inputs), original_batch_size):
            batch = compressed_inputs[i:i+original_batch_size]
            batch_w = compressed_weights[i:i+original_batch_size]
            rebatched_inputs.append(torch.stack(batch, dim=0).squeeze())
            rebatched_weights.append(torch.tensor(batch_w))
        calib_inputs = rebatched_inputs
        sample_weights = rebatched_weights
        print(f"  Rebatched {len(compressed_inputs)} individual samples into {len(rebatched_inputs)} batches")
    else:
        calib_inputs = compressed_inputs
        sample_weights = compressed_weights
    
    print(f"Calibration dataset compressed to {len(calib_inputs) * original_batch_size} samples.")
    if sample_weights:
        if isinstance(sample_weights[0], torch.Tensor):
            all_weights = torch.cat(sample_weights).tolist()
            print(f"Sample weights (sum={sum(all_weights):.4f}): min={min(all_weights):.4f}, max={max(all_weights):.4f}")
        else:
            print(f"Sample weights (sum={sum(sample_weights):.4f}): min={min(sample_weights):.4f}, max={max(sample_weights):.4f}")
    
    if should_save(args, "calibration_dataset"):
        # Determine compressed sample count if compression was applied
        # Compression is indicated when the number of samples differs from the original
        # When batch > 1, calib_inputs are batched, so count individual samples
        batch_size = calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1
        n_individual_samples = len(calib_inputs) * batch_size
        compressed_n_samples = (
            n_individual_samples
            if sample_weights is not None and n_individual_samples != args.nsamples_for_qcalibration
            else None
        )

        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / get_calibration_dataset_name(
            args.seed,
            args.nsamples_for_qcalibration,
            dataset_name=_get_calib_dataset_short_name(args),
            compressed_n_samples=compressed_n_samples,
            seq_len=args.calibrate_seq_len,
            spinquant=not args.no_spinquant,
            concat_layers=args.compression_concat_layers,
        )

        print(f"Saving calibration dataset to {save_path.resolve()}")
        torch.save({"calib_inputs": calib_inputs, "sample_weights": sample_weights}, save_path)
        
    return calib_inputs, sample_weights


def format_mmlu_example(
    question: str,
    choices: list[str],
    answer_idx: int,
    subject: str,
) -> str:
    """
    Format a single MMLU example into a text prompt for calibration.

    The output includes the subject, the question, the lettered choices, and
    the correct answer — so the model sees realistic instruction-like text
    during calibration.

    Args:
        question: The question text.
        choices: List of answer choices.
        answer_idx: Index (0-based) of the correct answer in ``choices``.
        subject: The MMLU subject name (e.g. ``"abstract_algebra"``).

    Returns:
        A formatted prompt string.
    """
    subject_display = subject.replace("_", " ").title()
    lines = [f"The following are multiple choice questions (with answers) about {subject_display}."]
    lines.append("")
    lines.append(question)
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice}")
    answer_letter = chr(ord("A") + answer_idx)
    lines.append(f"Answer: {answer_letter}")
    return "\n".join(lines)


def format_piqa_example(
    goal: str,
    sol1: str,
    sol2: str,
    answer_idx: int,
) -> str:
    """
    Format a single PIQA example into a text prompt for calibration.

    The output includes the goal, the two lettered solutions, and the correct
    answer — so the model sees realistic instruction-like text during
    calibration.

    Args:
        goal: The goal/question text.
        sol1: The first solution.
        sol2: The second solution.
        answer_idx: Index (0-based) of the correct solution (0 or 1).

    Returns:
        A formatted prompt string.
    """
    lines = ["The following are multiple choice questions (with answers)."]
    lines.append("")
    lines.append(goal)
    lines.append(f"A. {sol1}")
    lines.append(f"B. {sol2}")
    answer_letter = chr(ord("A") + answer_idx)
    lines.append(f"Answer: {answer_letter}")
    return "\n".join(lines)


def format_hellaswag_example(
    context: str,
    endings: list[str],
    answer_idx: int,
) -> str:
    """
    Format a single HellaSwag example into a text prompt for calibration.

    The output includes the context, the lettered sentence endings, and the
    correct answer — so the model sees realistic instruction-like text during
    calibration.

    Args:
        context: The context sentence (``ctx`` field).
        endings: List of possible sentence endings.
        answer_idx: Index (0-based) of the correct ending in ``endings``.

    Returns:
        A formatted prompt string.
    """
    lines = ["The following are multiple choice questions (with answers)."]
    lines.append("")
    lines.append(context)
    for i, ending in enumerate(endings):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {ending}")
    answer_letter = chr(ord("A") + answer_idx)
    lines.append(f"Answer: {answer_letter}")
    return "\n".join(lines)


def format_truthfulqa_example(
    question: str,
    choices: list[str],
    answer_idx: int,
) -> str:
    """
    Format a single TruthfulQA example into a text prompt for calibration.

    The output includes the question, the lettered choices, and the correct
    answer — so the model sees realistic instruction-like text during
    calibration.

    Args:
        question: The question text.
        choices: List of answer choices.
        answer_idx: Index (0-based) of the correct answer in ``choices``.

    Returns:
        A formatted prompt string.
    """
    lines = ["The following are multiple choice questions (with answers)."]
    lines.append("")
    lines.append(question)
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice}")
    answer_letter = chr(ord("A") + answer_idx)
    lines.append(f"Answer: {answer_letter}")
    return "\n".join(lines)


SUPPORTED_CALIB_DATASETS = ["wikitext", "mmlu", "truthfulqa", "hellaswag", "piqa"]

# Short names used in filenames
_DATASET_SHORT_NAMES = {
    "wikitext": "wiki",
    "mmlu": "mmlu",
    "truthfulqa": "truthfulqa",
    "hellaswag": "hellaswag",
    "piqa": "piqa",
}


def _get_calib_dataset_short_name(args) -> str:
    """
    Return a short filesystem-safe name for the selected calibration dataset(s).

    When ``--calibration_dataset_path`` is set, the name is derived from the
    file stem so that different loaded datasets produce different filenames.

    For a single dataset: ``"wiki"`` or ``"mmlu"``.
    For a mix: ``"wiki0.7_mmlu0.3"`` (proportions appended).
    """
    if args.calibration_dataset_path is not None:
        return pathlib.Path(args.calibration_dataset_path).stem

    mix = _parse_dataset_mix(args)
    if len(mix) == 1:
        return _DATASET_SHORT_NAMES.get(mix[0][0], mix[0][0])
    parts = []
    for name, prop in mix:
        short = _DATASET_SHORT_NAMES.get(name, name)
        parts.append(f"{short}{prop:.2f}")
    return "_".join(parts)


def _get_calib_dataset_display_name(args) -> str:
    """
    Return a human-readable description of the selected calibration dataset(s).

    When ``--calibration_dataset_path`` is set, returns the file path.
    """
    if args.calibration_dataset_path is not None:
        return args.calibration_dataset_path

    mix = _parse_dataset_mix(args)
    if len(mix) == 1:
        return mix[0][0]
    return ", ".join(f"{name}:{prop:.2f}" for name, prop in mix)



def _parse_dataset_mix(args) -> list[tuple[str, float]]:
    """
    Parse the calibration dataset mix from ``--calibration_dataset_mix``.

    Each entry can be either:
      - ``"name"`` — uses the full dataset (proportion 1.0)
      - ``"name:proportion"`` — uses the dataset with the given proportion

    When proportions are provided, they are normalized to sum to 1.0.
    When no proportions are provided (all entries are bare names), each
    dataset gets an equal share.

    Returns:
        A list of ``(dataset_name, proportion)`` tuples with proportions
        normalized to sum to 1.0.
    """
    mix: list[tuple[str, float]] = []

    # Allow both space-separated and comma-separated entries, e.g.:
    #   --calibration_dataset_mix wikitext:0.7 mmlu:0.3
    #   --calibration_dataset_mix "wikitext:0.7, mmlu:0.3"
    raw_entries: list[str] = []
    for entry in args.calibration_dataset_mix:
        raw_entries.extend(part.strip() for part in entry.split(",") if part.strip())

    for entry in raw_entries:
        if ":" in entry:
            name, prop_str = entry.rsplit(":", maxsplit=1)
            name = name.strip()
            prop = float(prop_str.strip())
        else:
            name = entry.strip()
            prop = 1.0

        if name not in SUPPORTED_CALIB_DATASETS:
            raise ValueError(
                f"Unknown calibration dataset {name!r}. "
                f"Supported: {SUPPORTED_CALIB_DATASETS}"
            )
        if prop <= 0:
            raise ValueError(
                f"Proportion for {name!r} must be positive, got {prop}"
            )
        mix.append((name, prop))

    if not mix:
        raise ValueError("--calibration_dataset_mix must not be empty")

    # Normalize proportions to sum to 1.0
    total = sum(p for _, p in mix)
    mix = [(name, p / total) for name, p in mix]
    return mix


def _load_single_dataset_text(dataset_name: str, args) -> str:
    """
    Load the full text for a single calibration dataset.

    Args:
        dataset_name: One of :data:`SUPPORTED_CALIB_DATASETS`.
        args: Parsed command-line arguments (used for cache_dir).

    Returns:
        The full text string for the dataset.
    """
    if dataset_name == "mmlu":
        print("Loading MMLU for calibration …")
        dataset = load_dataset(
            MMLU_DATASET_NAME,
            MMLU_DATASET_CONFIG,
            split=MMLU_CALIB_SPLIT,
            cache_dir=args.cache_dir,
        )
        prompts = []
        for ex in dataset:
            question = ex["question"]
            choices = ex["choices"]
            answer_idx = ex["answer"]
            subject = ex["subject"]
            prompts.append(
                format_mmlu_example(question, choices, answer_idx, subject)
            )
        return "\n\n".join(prompts)

    if dataset_name == "truthfulqa":
        print("Loading TruthfulQA for calibration …")
        dataset = load_dataset(
            TRUTHFULQA_DATASET_NAME,
            TRUTHFULQA_DATASET_CONFIG,
            split=TRUTHFULQA_CALIB_SPLIT,
            cache_dir=args.cache_dir,
        )
        prompts = []
        for ex in dataset:
            question = ex["question"]
            mc1 = ex["mc1_targets"]
            choices = mc1["choices"]
            labels = mc1["labels"]
            # The correct answer is the one with label 1
            answer_idx = labels.index(1) if 1 in labels else 0
            prompts.append(
                format_truthfulqa_example(question, choices, answer_idx)
            )
        return "\n\n".join(prompts)

    if dataset_name == "piqa":
        print("Loading PIQA for calibration …")
        dataset = load_dataset(
            PIQA_DATASET_NAME,
            split=PIQA_CALIB_SPLIT,
            cache_dir=args.cache_dir,
        )
        prompts = []
        for ex in dataset:
            goal = ex["goal"]
            sol1 = ex["sol1"]
            sol2 = ex["sol2"]
            label = ex["label"]
            answer_idx = int(label) if label is not None else 0
            prompts.append(
                format_piqa_example(goal, sol1, sol2, answer_idx)
            )
        return "\n\n".join(prompts)

    if dataset_name == "hellaswag":
        print("Loading HellaSwag for calibration …")
        dataset = load_dataset(
            HELLASWAG_DATASET_NAME,
            split=HELLASWAG_CALIB_SPLIT,
            cache_dir=args.cache_dir,
        )
        prompts = []
        for ex in dataset:
            ctx = ex["ctx"]
            endings = ex["endings"]
            label = ex["label"]
            answer_idx = int(label) if label else 0
            prompts.append(
                format_hellaswag_example(ctx, endings, answer_idx)
            )
        return "\n\n".join(prompts)

    # Default: wikitext
    print("Loading Wikitext for calibration …")
    dataset_train = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=TRAIN_SPLIT,
        cache_dir=args.cache_dir,
    )
    return " ".join(dataset_train["text"])


def _load_calibration_text(args, tokenizer) -> str:
    """
    Load and concatenate calibration text from the selected dataset(s).

    When ``--calibration_dataset_mix`` is provided, each dataset's text is
    tokenized and truncated so that its token count is proportional to the
    requested mix ratio.  The resulting texts are then concatenated.

    When a single dataset is selected, the full text is returned without truncation.

    Args:
        args: Parsed command-line arguments.
        tokenizer: Tokenizer used to measure token counts for proportional
            truncation (only needed when mixing).

    Returns:
        A single string containing the concatenated calibration text.
    """
    mix = _parse_dataset_mix(args)

    if len(mix) == 1:
        # Single dataset — no proportional truncation needed
        return _load_single_dataset_text(mix[0][0], args)

    # Multiple datasets — truncate each to its proportional token count
    print(f"Mixing calibration datasets: {mix}")

    # Load and tokenize each dataset to get token counts
    dataset_texts: list[str] = []
    dataset_token_counts: list[int] = []
    for name, _ in mix:
        text = _load_single_dataset_text(name, args)
        dataset_texts.append(text)
        ids = tokenizer(text, return_tensors="pt").input_ids
        dataset_token_counts.append(ids.shape[1])

    # Target: use the minimum available tokens as the base, then allocate
    # proportionally. This avoids one tiny dataset being over-represented.
    total_available = sum(dataset_token_counts)
    # Use all available tokens, distributing proportionally
    # Each dataset gets: proportion * total_available tokens (capped by availability)
    result_parts: list[str] = []
    for (name, proportion), text, n_tokens in zip(mix, dataset_texts, dataset_token_counts):
        target_tokens = int(proportion * total_available)
        actual_tokens = min(target_tokens, n_tokens)
        if actual_tokens < n_tokens:
            # Truncate: tokenize, take first actual_tokens, decode back to text
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
            truncated_text = tokenizer.decode(ids[:actual_tokens], skip_special_tokens=True)
            result_parts.append(truncated_text)
            print(f"  {name}: {actual_tokens}/{n_tokens} tokens ({proportion:.1%})")
        else:
            result_parts.append(text)
            print(f"  {name}: {n_tokens}/{n_tokens} tokens ({proportion:.1%})")

    return "\n\n".join(result_parts)


def rebatch_calibration_inputs(
    calib_inputs: list[torch.Tensor],
    sample_weights: Optional[list[float]],
    desired_batch_size: int,
) -> tuple[list[torch.Tensor], Optional[list[float]]]:
    """
    Rebatch calibration inputs (and optional sample weights) to a new batch size.

    The saved dataset may have been created with a different ``--batch`` value
    than the current run.  This helper flattens all tensors into individual
    ``[1, seq_len]`` samples and regroups them into batches of
    ``desired_batch_size``.  The last batch may be smaller if the total sample
    count is not divisible by ``desired_batch_size``.

    When ``desired_batch_size`` already matches the saved batch size, the
    inputs and weights are returned unchanged.

    Args:
        calib_inputs: List of calibration tensors (may be batched).
        sample_weights: Optional list of per-sample weights (may be ``None``).
        desired_batch_size: Target batch size to regroup into.

    Returns:
        Tuple of (rebatched_inputs, rebatched_weights).
    """
    saved_batch_size = calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1

    if desired_batch_size == saved_batch_size:
        return calib_inputs, sample_weights

    print(
        f"Rebatching loaded calibration dataset from batch_size={saved_batch_size} "
        f"to batch_size={desired_batch_size}"
    )

    # Flatten all inputs into individual samples [1, seq_len]
    individual_inputs = []
    for batched_input in calib_inputs:
        for i in range(batched_input.shape[0]):
            individual_inputs.append(batched_input[i:i+1, ...])

    # Flatten sample_weights to individual-sample level (if present)
    flat_weights = None
    if sample_weights is not None:
        flat_weights = []
        for w in sample_weights:
            if isinstance(w, torch.Tensor):
                flat_weights.extend(w.reshape(-1).tolist())
            elif isinstance(w, (list, tuple)):
                flat_weights.extend(w)
            else:
                flat_weights.append(w)

    # Regroup into batches of desired_batch_size (same pattern as compression)
    rebatched_inputs = []
    rebatched_weights = []
    for i in range(0, len(individual_inputs), desired_batch_size):
        batch = individual_inputs[i:i+desired_batch_size]
        if desired_batch_size == 1:
            # Keep [1, seq_len] shape (individual inputs already have batch dim)
            rebatched_inputs.append(batch[0])
        else:
            rebatched_inputs.append(torch.stack(batch, dim=0).squeeze())
        if flat_weights is not None:
            batch_w = flat_weights[i:i+desired_batch_size]
            rebatched_weights.append(torch.tensor(batch_w))

    if flat_weights is None:
        rebatched_weights = None

    return rebatched_inputs, rebatched_weights


def filter_calibration_samples_by_ppl(
    model: torch.nn.Module,
    calib_inputs: list[torch.Tensor],
    device: torch.device,
    ppl_filter_percentile: float,
    no_tqdm: bool = False,
) -> list[torch.Tensor]:
    """
    Filter calibration samples by model perplexity.

    Computes per-sample PPL on the FP model and removes samples above the
    given percentile (i.e. removes the most uncertain samples where the
    model is bad at predicting the next token).

    Args:
        model: FP model (before quantization).
        calib_inputs: List of calibration tensors, each [batch, seq_len].
        device: Device to run model on.
        ppl_filter_percentile: Keep samples with PPL below this percentile
            (e.g. 80.0 = remove top 20% highest-PPL samples).
        no_tqdm: Disable progress bar.

    Returns:
        Filtered list of calibration tensors (same batch size as input).
    """
    if not calib_inputs:
        return calib_inputs

    original_batch_size = calib_inputs[0].shape[0]

    # Flatten batched inputs into individual [1, seq_len] samples
    individual_samples: list[torch.Tensor] = []
    for batched_input in calib_inputs:
        for i in range(batched_input.shape[0]):
            individual_samples.append(batched_input[i : i + 1, ...])

    n_total = len(individual_samples)
    print(f"\n[PPL filter] Computing PPL for {n_total} samples ...")

    model.eval()
    ppls: list[float] = []

    iterator = individual_samples
    if not no_tqdm:
        iterator = tqdm.tqdm(individual_samples, desc="PPL filtering")

    with torch.no_grad():
        for sample in iterator:
            sample_dev = sample.to(device)
            outputs = model(sample_dev)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sample_dev[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            ppl = math.exp(loss.item())
            ppls.append(ppl)

            del sample_dev, outputs, logits

    # Print PPL statistics
    ppls_arr = np.array(ppls)
    print(f"[PPL filter] Statistics:")
    print(f"  min    : {ppls_arr.min():.2f}")
    print(f"  max    : {ppls_arr.max():.2f}")
    print(f"  median : {np.median(ppls_arr):.2f}")
    print(f"  mean   : {ppls_arr.mean():.2f}")
    for p in [25, 50, 75, 80, 90, 95]:
        print(f"  p{p:<4} : {np.percentile(ppls_arr, p):.2f}")

    # Robust outlier detection (median + 3*MAD)
    median_ppl = float(np.median(ppls_arr))
    mad = float(np.median(np.abs(ppls_arr - median_ppl)))
    mad_threshold = median_ppl + 3.0 * 1.4826 * mad
    n_mad_outliers = int((ppls_arr > mad_threshold).sum())
    print(f"[PPL filter] Robust outlier detection (median + 3*MAD):")
    print(f"  median : {median_ppl:.2f}")
    print(f"  MAD    : {mad:.2f}")
    print(f"  threshold (median + 3*1.4826*MAD): {mad_threshold:.2f}")
    print(f"  outliers: {n_mad_outliers}/{n_total} ({100.0 * n_mad_outliers / n_total:.1f}%)")

    # Compute PPL threshold:
    # - When ppl_filter_percentile is in [0, 100], use percentile-based filtering
    # - When ppl_filter_percentile is < 0 or > 100, use median + 3*MAD threshold
    if 0.0 <= ppl_filter_percentile <= 100.0:
        ppl_threshold = float(np.percentile(ppls_arr, ppl_filter_percentile))
        print(f"  threshold (p{ppl_filter_percentile}): {ppl_threshold:.2f}")
    else:
        ppl_threshold = mad_threshold
        print(f"  threshold (median + 3*MAD): {ppl_threshold:.2f}")

    # Filter: keep samples with PPL below threshold
    kept_indices = [
        i
        for i in range(n_total)
        if math.isfinite(ppls[i]) and ppls[i] <= ppl_threshold
    ]
    n_kept = len(kept_indices)

    # Trim to a multiple of batch_size so all output batches are full
    n_kept = (n_kept // original_batch_size) * original_batch_size
    kept_indices = kept_indices[:n_kept]

    n_removed = n_total - n_kept
    pct_removed = 100.0 * n_removed / n_total if n_total > 0 else 0.0

    print(f"[PPL filter] Kept {n_kept}/{n_total} samples (removed {n_removed}, {pct_removed:.1f}%)")

    if pct_removed > 50.0:
        print(
            f"[PPL filter] WARNING: More than 50% of samples were removed. "
            f"Consider using a higher percentile (e.g. 90.0 instead of {ppl_filter_percentile})."
        )

    if n_kept == 0:
        print("[PPL filter] WARNING: All samples were filtered out. Keeping original set.")
        return calib_inputs

    # Re-batch filtered samples back to original batch size
    filtered_samples = [individual_samples[i] for i in kept_indices]
    if original_batch_size == 1:
        return filtered_samples

    rebatched: list[torch.Tensor] = []
    for i in range(0, len(filtered_samples), original_batch_size):
        batch = filtered_samples[i : i + original_batch_size]
        assert len(batch) == original_batch_size
        rebatched.append(torch.stack(batch, dim=0).squeeze())

    return rebatched


def build_calibration_inputs(
    model,
    tokenizer,
    args,
    device: torch.device,
) -> tuple[list[torch.Tensor], Optional[list[float]]]:
    """
    Build random fixed-length calibration samples from a text corpus.

    The calibration dataset is selected by ``--calibration_dataset_mix`` (wikitext
    and/or mmlu).  The text is tokenized into one long token stream, then ``nsamples``
    random fixed-length windows of size ``seqlen`` are sampled.

    When batch > 1, samples are grouped into batches of shape [batch_size, seq_len].
    The last batch may be smaller if nsamples is not divisible by batch_size.

    If --calibration_dataset_path is provided, load the calibration inputs directly
    from the specified .pt file instead of generating it.

    Returns:
        - Tuple of (calib_inputs, sample_weights)
          - calib_inputs: List of calibration tensors
          - sample_weights: List of weights (only when loading from file, None when generating)
    """
    if args.calibration_dataset_path is not None:
        calib_path = pathlib.Path(args.calibration_dataset_path)
        if calib_path.exists():
            print(f"Loading calibration dataset from {calib_path.resolve()}")
            loaded_data = torch.load(calib_path, weights_only=False)
            
            # Handle both old format (just calib_inputs) and new format (dict with calib_inputs and sample_weights)
            if isinstance(loaded_data, dict):
                calib_inputs = loaded_data.get("calib_inputs")
                sample_weights = loaded_data.get("sample_weights")
            elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                calib_inputs, sample_weights = loaded_data
            else:
                # Old format: just calib_inputs list
                calib_inputs = loaded_data
                sample_weights = None
            
            if calib_inputs is None:
                raise ValueError(
                    f"Calibration dataset file does not contain 'calib_inputs': {calib_path.resolve()}"
                )

            # Rebatch loaded inputs to match the current --batch setting.
            # The saved dataset may have been created with a different batch size.
            calib_inputs, sample_weights = rebatch_calibration_inputs(
                calib_inputs, sample_weights, args.batch
            )

            return calib_inputs, sample_weights
        else:
            raise FileNotFoundError(
                f"Calibration dataset file not found: {calib_path.resolve()}"
            )

    calib_txt = _load_calibration_text(args, tokenizer)
    train_ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)

    nsamples = args.nsamples_for_qcalibration
    batch_size = args.batch
    seqlen_for_decode = 0 if ((args.llama_gptq or args.llama_gptq_sequential) and not args.llama_gptq_no_ptq) else args.decode_calibration_steps
    seqlen = model.config.max_position_embeddings - seqlen_for_decode
    if seqlen <= 0:
        raise ValueError(
            "decode_calibration_steps must be smaller than max_position_embeddings"
        )

    random.seed(args.seed)
    calib_inputs = []
    for _ in range(0, nsamples, batch_size):
        batch_samples = []
        for _ in range(batch_size):
            if len(calib_inputs) * batch_size + len(batch_samples) >= nsamples:
                break
            i = random.randint(0, train_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            sample = train_ids[:, i:j].cpu()
            if batch_size == 1:
                # Keep original behavior for batch_size == 1: [1, seq_len] tensor
                calib_inputs.append(sample)
            else:
                # Squeeze to remove batch dim before stacking
                batch_samples.append(sample.squeeze(0))
        if batch_samples and batch_size > 1:
            # Stack samples into a batch tensor of shape [batch_size, seq_len]
            batched = torch.stack(batch_samples, dim=0)
            calib_inputs.append(batched)

    # Filter calibration samples by model perplexity if requested.
    # This removes samples where the model is highly uncertain (high PPL),
    # keeping only samples below the given percentile.
    if args.ppl_filter_percentile is not None:
        calib_inputs = filter_calibration_samples_by_ppl(
            model,
            calib_inputs,
            device,
            ppl_filter_percentile=args.ppl_filter_percentile,
            no_tqdm=args.no_tqdm,
        )

    # When generating (not loading), return None for sample_weights
    # The caller should use _apply_calibration_compression if compression is needed
    return calib_inputs, None


def compute_or_load_sensitivity(model, calib_inputs, args, sample_weights=None):
    """
    Load or compute sensitivity information for sensitivity-based GPTQ.

    Args:
        sample_weights: Optional list of per-sample weights for weighted Fisher
            accumulation.  When the calibration dataset has been compressed,
            each representative sample carries a weight proportional to the
            number of original samples it stands for.  Passing the weights
            here ensures the sensitivity (empirical Fisher) is consistent with
            the weighted Hessian used by GPTQ.
    """
    if args.gptq_mse not in ("smse", "smse_for_gptq"):
        return None

    if args.sensitivity_path is not None:
        path = pathlib.Path(args.sensitivity_path)
        if path.exists():
            print(f"Loading sensitivity information from {path.resolve()}")
            return torch.load(path)

    print("Computing sensitivity information for GPTQ SMSE ...")
    calibrator = SensitivityCalibrator(
        model, calib_inputs, sample_weights=sample_weights
    )
    sens = calibrator.compute_sensitivity_info()

    if should_save(args, "sensitivity"):
        # Determine compressed sample count if compression was applied
        batch_size = calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1
        n_individual_samples = len(calib_inputs) * batch_size
        compressed_n_samples = (
            n_individual_samples
            if n_individual_samples != args.nsamples_for_qcalibration
            else None
        )

        default_path = pathlib.Path(
            get_sensitivities_info_name(
                model,
                _get_calib_dataset_short_name(args),
                args.seed,
                args.nsamples_for_qcalibration,
                seq_len=args.calibrate_seq_len,
                spinquant=not args.no_spinquant,
                compressed_n_samples=compressed_n_samples,
            )
        )

        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / default_path.name
        print(f"Saving sensitivity information to {save_path.resolve()}")
        torch.save(sens, save_path)

    return sens


def apply_gptq(model, calib_inputs, args, sample_weights=None):
    """
    Optionally run GPTQ weight-only quantization.
    
    Args:
        sample_weights: Optional list of weights for weighted Hessian accumulation
    """
    if args.no_GPTQ:
        print("Skipping GPTQ ...")
        return model

    print("Applying GPTQ ...")
    sens = compute_or_load_sensitivity(
        model, calib_inputs, args, sample_weights=sample_weights
    )
    gptq_config = build_gptq_config(args, sensitivity=sens, sample_weights=sample_weights)

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
    example = calib_inputs[0][0:1, ...].cpu()
    if args.max_seq_len is None:
        return example
    return pad_input(example, get_pad_token_id(tokenizer), args.max_seq_len).cpu()


def save_requested_artifacts(q_m, tokenizer, calib_inputs, args, sample_weights=None) -> None:
    """
    Save requested artifacts after PTQ conversion.
    """
    if args.output_dir is None or args.save is None:
        return

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if should_save(args, "calibration_dataset"):
        # Determine compressed sample count if compression was applied
        # Compression is indicated when the number of samples differs from the original
        # When batch > 1, calib_inputs are batched, so count individual samples
        batch_size = calib_inputs[0].shape[0] if len(calib_inputs) > 0 else 1
        n_individual_samples = len(calib_inputs) * batch_size
        compressed_n_samples = (
            n_individual_samples
            if sample_weights is not None and n_individual_samples != args.nsamples_for_qcalibration
            else None
        )

        save_path = output_dir / get_calibration_dataset_name(
            args.seed,
            args.nsamples_for_qcalibration,
            dataset_name=_get_calib_dataset_short_name(args),
            compressed_n_samples=compressed_n_samples,
            seq_len=args.calibrate_seq_len,
            spinquant=not args.no_spinquant,
            concat_layers=args.compression_concat_layers,
        )

        print(f"Saving calibration dataset to {save_path.resolve()}")
        torch.save({"calib_inputs": calib_inputs, "sample_weights": sample_weights}, save_path)

    # When --no_PTQ is used, q_m has no PTQ wrapper so there is no .wrapped
    # attribute.  Use q_m directly in that case.
    inner = q_m if args.no_PTQ else q_m.wrapped

    if should_save(args, "ptq_checkpoint"):
        save_path = output_dir / get_ptq_model_name(inner, args)
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
        max_seq_len = args.max_seq_len or inner.config.max_position_embeddings
        save_layers_to(
            q_m,
            max_seq_len,
            output_dir,
            prefill_decode=args.decode_calibration_steps > 0,
        )


def main():
    args = parse_args()
    print(args)
    print_cmd(args)
    validate_export_profile(args)

    device, dtype = setup_runtime(args)
    print_config(args, device)
    print_pytorch_environment()


    model, tokenizer = load_model_and_tokenizer(args, dtype)
    validate_tied_embedding_weight_bits(model, args)
    configure_max_position_embeddings(model, args)

    dataset_test = load_eval_dataset(args)
    evaluate_original_model(model, tokenizer, dataset_test, args, device)

    # Build calibration inputs (includes compression if --calibration_samples_to_use is specified)
    # Returns tuple of (calib_inputs, sample_weights) when loading from file, or (calib_inputs, None) when generating
    calib_inputs, sample_weights = build_calibration_inputs(model, tokenizer, args, device)

    model = apply_spinquant(model, args)
    model = apply_cle(model, args)

    # Assess cross-layer cluster consistency if requested.
    # This must run BEFORE compression so the assessment sees the full set of
    # original samples (e.g. 128) and clusters them into the target number of
    # clusters (e.g. 64).  Running it after compression would leave only the
    # compressed samples, making the assessment meaningless.
    if args.assess_cluster_consistency:
        print("\n=== Assessing Cluster Consistency ===")
        n_clusters_for_assessment = (
            args.calibration_samples_to_use
            if args.calibration_samples_to_use is not None
            else args.nsamples_for_qcalibration // 4
        )
        if n_clusters_for_assessment < 2:
            n_clusters_for_assessment = 2

        consistency_compressor = CalibrationSetCompressor(
            model=model,
            compress_to_samples=n_clusters_for_assessment,
            seed=args.seed,
            device=device,
            n_layers_to_use=1,
            n_components=args.calibration_compression_n_components,
            visualize=args.visualize_calibration_compression,
            output_dir=args.output_dir,
            feature_mode=args.compression_feature_mode,
            projection_mode=args.compression_projection_mode,
            normalize_features=(args.compression_normalize_features == "true"),
            kmeans_n_init=args.compression_kmeans_n_init,
        )
        consistency_compressor.assess_cluster_consistency(calib_inputs)
        print("=== Cluster Consistency Assessment Complete ===\n")

    if args.calibration_dataset_path is None:
        # Compress calibration inputs using K-Means clustering if --calibration_samples_to_use is specified
        calib_inputs, sample_weights = _apply_calibration_compression(
            calib_inputs, model, args, device
        )

    # When --llama_gptq_no_ptq is specified, run LlamaGPTQ without PTQ wrapping.

    # This allows weight-only quantization using LlamaGPTQ improvements.
    if args.llama_gptq_no_ptq and not args.no_GPTQ:
        print("Running LlamaGPTQ without PTQ (weight-only quantization) ...")
        model = apply_gptq(model, calib_inputs, args, sample_weights)
        q_m = quantize_using_PTQ(model, calib_inputs, args)
    # When both LlamaGPTQ and PTQ are enabled, run PTQ prepare first so that
    # LlamaGPTQ operates on the PTQ-wrapped model.  LlamaGPTQ will inject its
    # weight qparams into PTQ observers and freeze them layer-by-layer, so no
    # separate activation calibration pass is needed.
    elif (args.llama_gptq or args.llama_gptq_sequential) and not args.no_PTQ and not args.no_GPTQ:
        q_m = quantize_using_PTQ_and_LlamaGPTQ(model, calib_inputs, args, sample_weights)
    else:
        model = apply_gptq(model, calib_inputs, args, sample_weights)
        q_m = quantize_using_PTQ(model, calib_inputs, args)

    evaluate(q_m, tokenizer, dataset_test, args)
    save_requested_artifacts(q_m, tokenizer, calib_inputs, args, sample_weights)


if __name__ == "__main__":
    main()

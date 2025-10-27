# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

# ============================================================================
# LAYER-WISE DIFF DEBUGGING PIPELINE
# ----------------------------------------------------------------------------
# A quantization debugging pipeline that identifies accuracy regressions
# by comparing UINT vs FP outputs at each layer.
#
#   1. Load a full-precision (FP) LLaMA-3-1B model.
#   2. Wrap each Transformer block with PTQWrapper (activations → fake-quant).
#   3. Capture reference FP layer outputs before quantization.
#   4. Calibrate UINT-8 activation observers in a single pass.
#   5. Freeze quantization parameters (scale, zero-point).
#   6. Re-run inference and compare UINT-8 vs FP outputs per layer.
#   7. Report where quantization hurts the most.
#
# Use this pipeline to trace precision loss layer by layer, and pinpoint
# problematic modules during post-training quantization.
# ============================================================================

import argparse
import sys

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.introspection import (
    build_fqn_map,
    compare_layer_outputs,
    save_fp_outputs,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

# Token-budget presets for activation calibration
TOKENS: dict[str, int] = {
    # Smoke test (<1 min turnaround on CPU/GPU)
    "debug": 2_000,  # ≈16 × 128-seq batches
    # Good default for 1-7B models (≲3 % ppl delta)
    "baseline": 50_000,
    # Production / 4-bit observer smoothing
    "production": 200_000,
}

DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Hardcoded dataset settings
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TRAIN_SPLIT = "train"


def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise diff debugging pipeline for PTQ"
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
        help=f"Model dtype for load.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride used during calibration.",
    )
    parser.add_argument(
        "--calib-preset",
        choices=list(TOKENS.keys()),
        default="debug",
        help="Calibration token budget preset.",
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

    args = parser.parse_args()

    # Basic setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]  # noqa: E999 (kept readable)

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"Stride           : {args.stride}")
    print(
        f"Calib preset     : {args.calib_preset} ({TOKENS[args.calib_preset]:,} tokens)"
    )
    print(f"Use HF cache?    : {args.use_cache}")
    print()

    # -------------------------------------------------------------------------
    # 1. Load the FP backbone and tokenizer
    # -------------------------------------------------------------------------
    print("Loading FP model …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
        )
        .to(device)
        .eval()
    )

    # Disable KV cache to force full forward passes for introspection
    model.config.use_cache = args.use_cache

    # Build module -> FQN map before wrapping
    m_to_fqn = build_fqn_map(model)

    # Prepare calibration inputs (HF Wikitext-2 train split)
    CALIB_TOKENS = TOKENS[args.calib_preset]
    print(f"Calibrating with {CALIB_TOKENS:,} tokens.\n")
    # Use Wikitext-2 train split for calibration.
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)

    # -------------------------------------------------------------------------
    # 2. Wrap every layer with PTQWrapper (UINT-8 activations)
    # -------------------------------------------------------------------------
    print("Wrapping layers with PTQWrapper …")
    qcfg = PTQConfig()  # default: per-tensor UINT8
    prepare(model, qcfg)

    # -------------------------------------------------------------------------
    # 3. Activation calibration plus FP-vs-UINT8 diffing
    # -------------------------------------------------------------------------
    print("Calibrating UINT-8 observers …")
    calib_txt = " ".join(dataset["text"])[:CALIB_TOKENS]
    ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)

    # Save reference FP activations before observers clamp/quantize
    save_handles, act_cache = save_fp_outputs(model)

    iterator = range(0, ids.size(1) - 1, args.stride)
    if not args.no_tqdm:
        iterator = tqdm.tqdm(iterator, desc="Act-Calibration")
    with torch.no_grad():
        for i in iterator:
            inputs = ids[:, i : i + args.stride]
            model(inputs)  # observers collect act. ranges

    # Remove save hooks now that FP activations are cached
    for h in save_handles:
        h.remove()

    # Freeze (scale, zero-point) after calibration
    convert(model)

    # Register diff hooks and measure per-layer deltas
    cmp_handles = compare_layer_outputs(model, act_cache, metrics=["diff", "peir"])
    # Use same inputs for comparison.
    with torch.no_grad():
        model(inputs)

    assert isinstance(cmp_handles, list)
    for h in cmp_handles:
        h.remove()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)

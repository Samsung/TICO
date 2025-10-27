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

# =============================================================================
# QUICK PTQ WORKFLOW (OPTIONAL FP32 BASELINE)
# -----------------------------------------------------------------------------
# Toggle RUN_FP to choose between:
#   • FP32 perplexity measurement only, OR
#   • Full post-training UINT-8 flow (wrap → calibrate → eval).
# =============================================================================

import argparse
import sys

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.metrics import perplexity

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
TEST_SPLIT = "test"


def main():
    parser = argparse.ArgumentParser(description="Quick PTQ example (FP or UINT8)")
    parser.add_argument(
        "--mode",
        choices=["fp", "uint8"],
        default="fp",
        help="Choose FP baseline only or full UINT8 PTQ path.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HF repo name or local path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda|cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="float32",
        help=f"Model dtype for load.",
    )
    parser.add_argument(
        "--stride", type=int, default=512, help="Sliding-window stride for perplexity."
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
        help="Optional HF token for gated/private models.",
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
    # 2) calib-preset default = debug
    parser.add_argument(
        "--calib-preset",
        choices=list(TOKENS.keys()),
        default="debug",
        help="Calibration token budget preset.",
    )

    args = parser.parse_args()

    # Basic setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print("=== Config ===")
    print(f"Mode             : {args.mode}")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"Stride           : {args.stride}")
    print(f"Use HF cache?    : {args.use_cache}")
    print(
        f"Calib preset     : {args.calib_preset} ({TOKENS[args.calib_preset]:,} tokens)"
    )
    print()

    # -------------------------------------------------------------------------
    # 1. Load model and tokenizer
    # -------------------------------------------------------------------------
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

    model.config.use_cache = args.use_cache

    if args.mode == "fp":
        fp_model = model
    else:
        # INT8 PTQ path
        uint8_model = model

        CALIB_TOKENS = TOKENS[args.calib_preset]
        print(f"Calibrating with {CALIB_TOKENS:,} tokens.\n")

        # ---------------------------------------------------------------------
        # 2. Wrap every Transformer layer with PTQWrapper
        # ---------------------------------------------------------------------
        qcfg = PTQConfig()  # all-uint8 defaults
        prepare(uint8_model, qcfg)

        # ---------------------------------------------------------------------
        # 3. Single-pass activation calibration
        # ---------------------------------------------------------------------
        print("Calibrating UINT-8 observers …")
        calib_txt = " ".join(
            load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)["text"]
        )[:CALIB_TOKENS]
        ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)

        # Run inference to collect ranges
        iterator = range(0, ids.size(1) - 1, args.stride)
        if not args.no_tqdm:
            iterator = tqdm.tqdm(iterator, desc="Calibration")
        with torch.no_grad():
            for i in iterator:
                uint8_model(ids[:, i : i + args.stride])

        # Freeze (scale, zero-point)
        convert(uint8_model)

    # -------------------------------------------------------------------------
    # 4. Evaluate perplexity
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    test_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT)
    enc = tokenizer("\n\n".join(test_ds["text"]), return_tensors="pt")

    if args.mode == "fp":
        ppl_fp = perplexity(
            fp_model,
            enc,
            args.device,
            stride=args.stride,
            show_progress=not args.no_tqdm,
        )
    else:
        ppl_int8 = perplexity(
            uint8_model,
            enc,
            args.device,
            stride=args.stride,
            show_progress=not args.no_tqdm,
        )

    # -------------------------------------------------------------------------
    # 5. Report
    # -------------------------------------------------------------------------
    print("\n┌── Wikitext-2 test perplexity ─────────────")
    if args.mode == "fp":
        print(f"│ FP     : {ppl_fp:8.2f}")
    else:
        print(f"│ UINT-8 : {ppl_int8:8.2f}")
    print("└───────────────────────────────────────────")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)

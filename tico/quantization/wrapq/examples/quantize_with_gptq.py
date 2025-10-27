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
# PTQ + GPTQ HYBRID QUANTIZATION PIPELINE
# -----------------------------------------------------------------------------
# This script shows how to:
#   1. Load a pretrained FP Llama-3 model.
#   2. Run GPTQ to quantize weights only.
#   3. Wrap every Transformer layer with a PTQWrapper to quantize activations.
#   4. Calibrate UINT-8 observers in a single pass over a text corpus.
#   5. Inject GPTQ’s per-tensor weight scales / zero-points into the PTQ graph.
#   6. Freeze all Q-params and compute Wikitext-2 perplexity.
# =============================================================================

import argparse
import sys
from typing import Any

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.introspection import build_fqn_map
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


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

# -------------------------------------------------------------------------
# 1. Helper — copy GPTQ (scale, zp) into PTQ observers
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


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ+PTQ pipeline (weight-only + activation UINT8)"
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
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride used for calibration and eval.",
    )
    parser.add_argument(
        "--calib-preset",
        choices=list(TOKENS.keys()),
        default="debug",
        help="Activation calibration token budget preset.",
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
    dtype = DTYPE_MAP[args.dtype]

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
    # 2. Load the FP backbone and tokenizer
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

    model.config.use_cache = args.use_cache

    # Build module -> FQN map BEFORE wrapping
    m_to_fqn = build_fqn_map(model)

    # -------------------------------------------------------------------------
    # 3. Run GPTQ (weight-only) pass
    # -------------------------------------------------------------------------
    print("Applying GPTQ …")
    dataset_test = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT)
    q_m = prepare(model, GPTQConfig(), inplace=True)

    it = (
        dataset_test
        if args.no_tqdm
        else tqdm.tqdm(dataset_test, desc="GPTQ calibration")
    )
    for d in it:
        ids = tokenizer(d["text"], return_tensors="pt").input_ids.to(device)
        q_m(ids)  # observers gather weight stats

    q_m = convert(q_m, inplace=True)  # materialize INT-weight tensors

    # -------------------------------------------------------------------------
    # 4. Wrap every layer with PTQWrapper (activation UINT-8)
    # -------------------------------------------------------------------------
    print("Wrapping layers with PTQWrapper …")
    qcfg = PTQConfig()  # default: per-tensor UINT8
    prepare(q_m, qcfg)

    # -------------------------------------------------------------------------
    # 5. Single-pass activation calibration
    # -------------------------------------------------------------------------
    print("Calibrating UINT-8 observers …")
    CALIB_TOKENS = TOKENS[args.calib_preset]
    print(f"Calibrating with {CALIB_TOKENS:,} tokens.\n")
    dataset_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
    calib_txt = " ".join(dataset_train["text"])[:CALIB_TOKENS]
    train_ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)

    # Overwrite weight observers with GPTQ statistics
    if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
        inject_gptq_qparams(q_m, q_m.quantizers)
    else:
        print(
            "[Warn] q_m.quantizers not found or not a dict; skipping GPTQ qparam injection."
        )

    # Forward passes to collect activation ranges
    iterator = range(0, train_ids.size(1) - 1, args.stride)
    if not args.no_tqdm:
        iterator = tqdm.tqdm(iterator, desc="Act-calibration")
    with torch.no_grad():
        for i in iterator:
            q_m(train_ids[:, i : i + args.stride])

    # Freeze all Q-params (scale, zero-point)
    convert(q_m)

    # -------------------------------------------------------------------------
    # 6. Evaluate perplexity on Wikitext-2
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_uint8 = perplexity(q_m, enc, device, stride=args.stride)

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ UINT-8 : {ppl_uint8:8.2f}")
    print("└───────────────────────────────────────────")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[Error] {e}", file=sys.stderr)
        sys.exit(1)

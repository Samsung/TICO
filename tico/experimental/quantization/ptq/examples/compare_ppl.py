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

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.utils.metrics import perplexity
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper

# -------------------------------------------------------------------------
# 0. Global configuration
# -------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE = 512  # sliding-window stride for perplexity
RUN_FP = True  # set False → run UINT-8 path

# Token-budget presets for activation calibration
TOKENS: dict[str, int] = {
    # Smoke test (<1 min turnaround on CPU/GPU)
    "debug": 2_000,  # ≈16 × 128-seq batches
    # Good default for 1-7B models (≲3 % ppl delta)
    "baseline": 50_000,
    # Production / 4-bit observer smoothing
    "production": 200_000,
}
CALIB_TOKENS = TOKENS["baseline"]
print(f"Calibrating with {CALIB_TOKENS:,} tokens.\n")

# -------------------------------------------------------------------------
# 1. Load model
# -------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if RUN_FP:
    # -- FP32 baseline ------------------------------------------------------
    print("Loading FP32 model …")
    fp_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    fp_model.config.use_cache = False
else:
    # -- UINT-8 pipeline -----------------------------------------------------
    print("Creating UINT-8 clone …")
    uint8_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    uint8_model.config.use_cache = False

    # ---------------------------------------------------------------------
    # 2. Wrap every Transformer layer with PTQWrapper
    # ---------------------------------------------------------------------
    qcfg = QuantConfig()  # all-uint8 defaults

    wrapped_layers = torch.nn.ModuleList()
    for idx, layer in enumerate(uint8_model.model.layers):
        layer_cfg = qcfg.child(f"layer{idx}")
        wrapped_layers.append(PTQWrapper(layer, qcfg=layer_cfg))
    uint8_model.model.layers = wrapped_layers

    # ---------------------------------------------------------------------
    # 3. Single-pass activation calibration
    # ---------------------------------------------------------------------
    print("Calibrating UINT-8 observers …")
    calib_txt = " ".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
    )[:CALIB_TOKENS]
    ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(DEVICE)

    # (a) switch every QuantModuleBase to CALIB mode
    for l in uint8_model.model.layers:
        l.enable_calibration()

    # (b) run inference to collect ranges
    with torch.no_grad():
        for i in tqdm.trange(0, ids.size(1) - 1, STRIDE, desc="Calibration"):
            uint8_model(ids[:, i : i + STRIDE])

    # (c) freeze (scale, zero-point)
    for l in uint8_model.model.layers:
        l.freeze_qparams()

# -------------------------------------------------------------------------
# 4. Evaluate perplexity on Wikitext-2
# -------------------------------------------------------------------------
print("\nCalculating perplexities …")
test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tokenizer("\n\n".join(test_ds["text"]), return_tensors="pt")

if RUN_FP:
    ppl_fp = perplexity(fp_model, enc, DEVICE, stride=STRIDE)
else:
    ppl_int8 = perplexity(uint8_model, enc, DEVICE, stride=STRIDE)

# -------------------------------------------------------------------------
# 5. Report
# -------------------------------------------------------------------------
print("\n┌── Wikitext-2 test perplexity ─────────────")
if RUN_FP:
    print(f"│ FP32  : {ppl_fp:8.2f}")
else:
    print(f"│ UINT-8 : {ppl_int8:8.2f}")
print("└───────────────────────────────────────────")

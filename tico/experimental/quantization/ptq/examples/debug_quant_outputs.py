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

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.utils.introspection import (
    build_fqn_map,
    compare_layer_outputs,
    save_fp_outputs,
)
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper

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

# -------------------------------------------------------------------------
# 0. Global configuration
# -------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE = 512

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
# 1. Load the FP backbone
# -------------------------------------------------------------------------
print("Loading FP model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
model.config.use_cache = False  # disable KV-cache → full forward
m_to_fqn = build_fqn_map(model)  # map modules → fully-qualified names

# Use Wikitext-2 train split for calibration.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# -------------------------------------------------------------------------
# 2. Wrap every layer with PTQWrapper (UINT-8 activations)
# -------------------------------------------------------------------------
print("Wrapping layers with PTQWrapper …")
qcfg = QuantConfig()  # default: per-tensor UINT8

new_layers = torch.nn.ModuleList()
for idx, fp_layer in enumerate(model.model.layers):
    layer_cfg = qcfg.child(f"layer{idx}")
    q_layer = PTQWrapper(
        fp_layer,
        qcfg=layer_cfg,
        fp_name=m_to_fqn.get(fp_layer),
    )
    new_layers.append(q_layer)

model.model.layers = new_layers  # swap in quant wrappers

# -------------------------------------------------------------------------
# 3. Activation calibration plus FP-vs-UINT8 diffing
# -------------------------------------------------------------------------
print("Calibrating UINT-8 observers …")
calib_txt = " ".join(dataset["text"])[:CALIB_TOKENS]
ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(DEVICE)

# (a) Enable CALIB mode on every QuantModuleBase
for l in model.model.layers:
    l.enable_calibration()

# Save reference FP activations before observers clamp/quantize
save_handles, act_cache = save_fp_outputs(model)

with torch.no_grad():
    for i in tqdm.trange(0, ids.size(1) - 1, STRIDE, desc="Act-calibration"):
        inputs = ids[:, i : i + STRIDE]
        model(inputs)  # observers collect act. ranges

# Remove save hooks now that FP activations are cached
for h in save_handles:
    h.remove()

# (b) Freeze (scale, zero-point) after calibration
for l in model.model.layers:
    l.freeze_qparams()

# (c) Register diff hooks and measure per-layer deltas
cmp_handles = compare_layer_outputs(model, act_cache, metrics=["diff", "peir"])
# Use same inputs for comparison.
model(inputs)

assert isinstance(cmp_handles, list)
for h in cmp_handles:
    h.remove()

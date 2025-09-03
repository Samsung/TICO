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

from typing import Any

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import GPTQConfig
from tico.experimental.quantization.ptq.observers.affine_base import AffineObserverBase
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.utils.introspection import build_fqn_map
from tico.experimental.quantization.ptq.utils.metrics import perplexity
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)

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


# -------------------------------------------------------------------------
# 2. Load the FP backbone
# -------------------------------------------------------------------------
print("Loading FP model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
model.config.use_cache = False  # disable KV-cache → full forward
m_to_fqn = build_fqn_map(model)  # map modules → fully-qualified names

# -------------------------------------------------------------------------
# 3. Run GPTQ (weight-only) pass
# -------------------------------------------------------------------------
print("Applying GPTQ …")
dataset = load_dataset("wikiText", "wikitext-2-raw-v1", split="test")
q_m = prepare(model, GPTQConfig(), inplace=True)

for d in tqdm.tqdm(dataset, desc="GPTQ calibration"):
    ids = tokenizer(d["text"], return_tensors="pt").input_ids.to(DEVICE)
    q_m(ids)  # observers gather weight stats

q_m = convert(q_m, inplace=True)  # materialize INT-weight tensors

# -------------------------------------------------------------------------
# 4. Wrap every layer with PTQWrapper (activation UINT-8)
# -------------------------------------------------------------------------
qcfg = QuantConfig()  # default: per-tensor UINT8
new_layers = torch.nn.ModuleList()

for idx, fp_layer in enumerate(q_m.model.layers):
    layer_cfg = qcfg.child(f"layer{idx}")
    q_layer = PTQWrapper(
        fp_layer,
        qcfg=layer_cfg,
        fp_name=m_to_fqn.get(fp_layer),
    )
    new_layers.append(q_layer)

q_m.model.layers = new_layers

# -------------------------------------------------------------------------
# 5. Single-pass activation calibration
# -------------------------------------------------------------------------
print("Calibrating UINT-8 observers …")
calib_txt = " ".join(
    load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
)[:CALIB_TOKENS]
ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(DEVICE)

# (a) Enable CALIB mode on every QuantModuleBase
for l in q_m.model.layers:
    l.enable_calibration()

# (b) Overwrite weight observers with GPTQ statistics
inject_gptq_qparams(q_m, q_m.quantizers)

with torch.no_grad():
    for i in tqdm.trange(0, ids.size(1) - 1, STRIDE, desc="Act-calibration"):
        q_m(ids[:, i : i + STRIDE])  # observers collect act. ranges

# (c) Freeze all Q-params (scale, zp)
for l in q_m.model.layers:
    l.freeze_qparams()

# -------------------------------------------------------------------------
# 6. Evaluate perplexity on Wikitext-2
# -------------------------------------------------------------------------
print("\nCalculating perplexities …")
test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tokenizer("\n\n".join(test_ds["text"]), return_tensors="pt")
ppl_uint8 = perplexity(q_m, enc, DEVICE, stride=STRIDE)

print("\n┌── Wikitext-2 test perplexity ─────────────")
print(f"│ UINT-8 : {ppl_uint8:8.2f}")
print("└───────────────────────────────────────────")

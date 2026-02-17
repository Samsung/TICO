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

import pathlib

import torch
from transformers import AutoModelForVision2Seq, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_mlp import QuantQwen3VLTextMLP
from tico.utils.utils import SuppressWarning

# -------------------------------------------------------------------------
# 0. Load a Qwen3-VL model (text tower) + tokenizer
# -------------------------------------------------------------------------
name = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    name,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

# Make sure pad token exists (Llama often uses eos as pad)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

"""
As max_seq increases, the proportion of padded tokens grows, 
 which directly affects calibration statistics and tends to increase PEIR.

This is not a modeling issue but a calibration artifact: 
 observers see a distribution dominated by padding rather than real tokens.

For a more representative and realistic accuracy evaluation, 
the calibration dataset should be adjusted (e.g., longer or more diverse 
 sequence lengths, or padding-aware calibration) so that activation 
statistics better reflect actual inference workloads.
"""
MAX_SEQ = 128
text_cfg = model.config.text_config
text_cfg.max_position_embeddings = MAX_SEQ

# -------------------------------------------------------------------------
# 1. Replace layer-0's MLP with QuantQwen3VLTextMLP
# -------------------------------------------------------------------------
orig_mlp = model.model.language_model.layers[0].mlp
mlp_q = prepare(orig_mlp, PTQConfig())
mlp_q.eval()

assert isinstance(mlp_q.wrapped, QuantQwen3VLTextMLP)

# -------------------------------------------------------------------------
# 2. Single-pass calibration
# -------------------------------------------------------------------------
CALIB_TENSORS = []
for _ in range(10):
    ct = torch.randn(5, MAX_SEQ, text_cfg.hidden_size)
    CALIB_TENSORS.append(ct)

with torch.no_grad():
    for ct in CALIB_TENSORS:
        _ = mlp_q(ct)

convert(mlp_q)
assert mlp_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
hidden = CALIB_TENSORS[3]

with torch.no_grad():
    quant_out = mlp_q(hidden)
    fp_out = orig_mlp(hidden)

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(quant_out - fp_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp_out, quant_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp_out, quant_out))

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("qwen3vl_text_mlp.q.circle")
B, S, D = 1, MAX_SEQ, text_cfg.hidden_size
example = torch.randn(B, S, D)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(mlp_q, (example,))
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")

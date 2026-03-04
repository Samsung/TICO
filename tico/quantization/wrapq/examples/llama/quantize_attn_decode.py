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
from transformers import AutoModelForCausalLM

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.llama.quant_attn_decode import (
    QuantLlamaAttentionDecode,
)
from tico.utils.utils import SuppressWarning

MODEL_NAME = "Maykeye/TinyLLama-v0"
MAX_S = 256
B = 1
N_CALIB = 16  # number of random calibration batches
DEVICE = "cpu"

torch.set_grad_enabled(False)

# -----------------------------------------------------------------------------
# Build model + replace attention with quant wrapper (decode variant)
# -----------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE)
model.eval()
# make config consistent with static decode length if wrapper reads it
model.config.max_position_embeddings = MAX_S

layer0 = model.model.layers[0]
orig_attn = layer0.self_attn

layer0.self_attn = prepare(orig_attn, PTQConfig(wrapper_variant="decode"))
attn_q = layer0.self_attn
assert isinstance(attn_q.wrapped, QuantLlamaAttentionDecode), type(attn_q.wrapped)

# -----------------------------------------------------------------------------
# Random input generator
# -----------------------------------------------------------------------------
def make_random_decode_batch():
    D = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", D // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads

    x = torch.randn(B, 1, D, device=DEVICE)

    cos = torch.randn(B, 1, head_dim, device=DEVICE)
    sin = torch.randn(B, 1, head_dim, device=DEVICE)
    pos = (cos, sin)

    # additive mask: (B,1,MAX_S)
    # Here we randomly mask some tail region to simulate padding.
    # Keep it simple: choose a random effective length L_eff and mask [L_eff:].
    L_eff = torch.randint(low=1, high=MAX_S + 1, size=(1,)).item()
    mask = torch.zeros(B, 1, MAX_S, device=DEVICE, dtype=torch.float32)
    if L_eff < MAX_S:
        mask[:, :, L_eff:] = float("-120")

    past_k = torch.randn(B, n_kv, MAX_S - 1, head_dim, device=DEVICE)
    past_v = torch.randn(B, n_kv, MAX_S - 1, head_dim, device=DEVICE)
    past = (past_k, past_v)

    return x, pos, mask, past


# -----------------------------------------------------------------------------
# 1) Calibration pass
# -----------------------------------------------------------------------------
with torch.no_grad():
    for _ in range(N_CALIB):
        x, pos, mask, past = make_random_decode_batch()
        fp_out = attn_q(
            x,
            pos,
            attention_mask=mask,
            past_key_value=past,
            use_cache=True,
        )

# -----------------------------------------------------------------------------
# 2) Convert to QUANT (freeze qparams)
# -----------------------------------------------------------------------------
convert(attn_q)
assert attn_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -----------------------------------------------------------------------------
# 3) Diff check: wrapper FP vs wrapper QUANT on the same random batch
# -----------------------------------------------------------------------------
with torch.no_grad():
    fp = fp_out[0]

    int8_out = attn_q(
        x,
        pos,
        attention_mask=mask,
        past_key_value=past,
        use_cache=True,
    )
    int8 = int8_out[0]

print("┌───────────── Quantization Error Summary (Decode Attn / Random) ─────────────")
print(f"│ Mean |diff|: {(int8 - fp).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp, int8) * 100:.6f} %")
print("└────────────────────────────────────────────────────────────────────────────")
print(plot_two_outputs(fp, int8))

# -----------------------------------------------------------------------------
# 4) Export to Circle (static inputs)
# -----------------------------------------------------------------------------
import tico

save_path = pathlib.Path("attn_decode.q.circle")

# example inputs must match forward signature
x_ex, pos_ex, mask_ex, past_ex = make_random_decode_batch()
cos_ex, sin_ex = pos_ex
past_k_ex, past_v_ex = past_ex

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(
        attn_q.eval(),
        (x_ex, (cos_ex, sin_ex), mask_ex, (past_k_ex, past_v_ex)),
    )
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")

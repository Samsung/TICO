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

# Example script to calibrate + convert + export a *decode-only* LlamaDecoderLayer wrapper
# with fully static shapes and external KV-cache delta updates.
#
# Expected wrapper stack:
#   QuantLlamaDecoderLayerDecode (variant="decode")
#     └─ self_attn -> QuantLlamaAttentionDecode (variant="decode")
#
# Notes
# -----
# - The decode wrapper expects:
#     hidden_states:        (B, 1, D)
#     position_embeddings:  (cos, sin) each (B, 1, head_dim)
#     attention_mask:       (B, 1, MAX_S) additive (0 or -120)
#     past_key_value:       (past_k, past_v) each (B, n_kv, MAX_S-1, head_dim)
# - It returns:
#     out:                 (B, 1, D)
#     new_kv_delta:        (new_k, new_v) each (B, n_kv, 1, head_dim)   when use_cache=True
#
# - Host runtime is responsible for writing new_k/new_v into an external KV cache buffer.
#
import pathlib

import torch
from transformers import AutoModelForCausalLM

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer_decode import (
    QuantLlamaDecoderLayerDecode,
)
from tico.utils.utils import SuppressWarning

MODEL_NAME = "Maykeye/TinyLLama-v0"
MAX_S = 256
B = 1
N_CALIB = 16  # number of random calibration batches
DEVICE = "cpu"

torch.set_grad_enabled(False)


# -----------------------------------------------------------------------------
# Build model + replace one decoder layer with quant wrapper (decode variant)
# -----------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE)
model.eval()

# Make config consistent with static decode length if wrappers read it.
model.config.max_position_embeddings = MAX_S

layer0 = model.model.layers[0]
orig_layer = layer0

# Replace the entire decoder layer (decode variant).
layer0_q = prepare(orig_layer, PTQConfig(wrapper_variant="decode"))
model.model.layers[0] = layer0_q

assert hasattr(layer0_q, "wrapped"), "prepare() should return a PTQWrapper"
assert isinstance(layer0_q.wrapped, QuantLlamaDecoderLayerDecode), type(
    layer0_q.wrapped
)


# -----------------------------------------------------------------------------
# Random input generator for the decode decoder-layer
# -----------------------------------------------------------------------------
def make_random_decode_batch():
    D = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", D // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads

    # Single-token hidden state.
    x = torch.randn(B, 1, D, device=DEVICE)

    # RoPE tables for the *current token* only.
    cos = torch.randn(B, 1, head_dim, device=DEVICE)
    sin = torch.randn(B, 1, head_dim, device=DEVICE)
    pos = (cos, sin)

    # Additive mask of final static width: (B, 1, MAX_S)
    # Simulate that only the first L_eff positions are valid and the rest are padding.
    L_eff = torch.randint(low=1, high=MAX_S + 1, size=(1,)).item()
    mask = torch.zeros(B, 1, MAX_S, device=DEVICE, dtype=torch.float32)
    if L_eff < MAX_S:
        mask[:, :, L_eff:] = float("-120")

    # Static-sized past KV (already RoPE-applied for past tokens).
    past_k = torch.randn(B, n_kv, MAX_S - 1, head_dim, device=DEVICE)
    past_v = torch.randn(B, n_kv, MAX_S - 1, head_dim, device=DEVICE)
    past = (past_k, past_v)

    return x, pos, mask, past


# -----------------------------------------------------------------------------
# 1) Calibration pass (collect observer stats)
# -----------------------------------------------------------------------------
with torch.no_grad():
    for _ in range(N_CALIB):
        x, pos, mask, past = make_random_decode_batch()
        fp_out = layer0_q(
            hidden_states=x,
            attention_mask=mask,
            past_key_value=past,
            position_embeddings=pos,
            use_cache=True,
        )


# -----------------------------------------------------------------------------
# 2) Convert to QUANT (freeze qparams)
# -----------------------------------------------------------------------------
convert(layer0_q)
assert layer0_q._mode is Mode.QUANT, "Quantization mode should be active now."


# -----------------------------------------------------------------------------
# 3) Diff check: wrapper FP vs wrapper QUANT on the same random batch
# -----------------------------------------------------------------------------
with torch.no_grad():
    fp_hidden, (fp_new_k, fp_new_v) = fp_out

    # Run once in QUANT mode (post-convert).
    q_out = layer0_q(
        hidden_states=x,
        attention_mask=mask,
        past_key_value=past,
        position_embeddings=pos,
        use_cache=True,
    )

    # The QUANT output is what we have.
    q_hidden, (q_new_k, q_new_v) = q_out

print(
    "┌──────────── Quantization Error Summary (Decode DecoderLayer / Random) ────────────"
)
print(f"│ Mean |diff| (hidden): {(q_hidden - fp_hidden).abs().mean().item():.6f}")
print(f"│ PEIR        (hidden): {compute_peir(fp_hidden, q_hidden) * 100:.6f} %")
print(f"│ Mean |diff| (new_k) : {(q_new_k - fp_new_k).abs().mean().item():.6f}")
print(f"│ PEIR        (new_k) : {compute_peir(fp_new_k, q_new_k) * 100:.6f} %")
print(f"│ Mean |diff| (new_v) : {(q_new_v - fp_new_v).abs().mean().item():.6f}")
print(f"│ PEIR        (new_v) : {compute_peir(fp_new_v, q_new_v) * 100:.6f} %")
print(
    "└──────────────────────────────────────────────────────────────────────────────────"
)
print(plot_two_outputs(fp_hidden, q_hidden))


# -----------------------------------------------------------------------------
# 4) Export to Circle (static inputs)
# -----------------------------------------------------------------------------
import tico

save_path = pathlib.Path("decoder_layer_decode.q.circle")

# Example inputs must match the wrapper's forward signature.
x_ex, pos_ex, mask_ex, past_ex = make_random_decode_batch()
cos_ex, sin_ex = pos_ex
past_k_ex, past_v_ex = past_ex

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(
        layer0_q.eval(),
        (
            x_ex,  # hidden_states
            mask_ex,  # attention_mask
            None,  # position_ids (unused)
            (past_k_ex, past_v_ex),  # past_key_value
            False,  # output_attentions
            True,  # use_cache
            None,  # cache_position (unused)
            (cos_ex, sin_ex),  # position_embeddings
        ),
    )
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")

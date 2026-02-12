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
from transformers import AutoModelForImageTextToText  # since 4.5

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_mlp import (
    QuantQwen3VLVisionMLP,
)
from tico.utils.utils import SuppressWarning

# -------------------------------------------------------------------------
# 0. Load a Qwen3-VL model (text tower) + tokenizer
# -------------------------------------------------------------------------
name = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    name,
    device_map="cpu",
    trust_remote_code=True,
    dtype=torch.float32,
)
model.eval()

# -------------------------------------------------------------------------
# 1. Replace layer-0’s mlp with QuantQwen3VLVisionMLP
# -------------------------------------------------------------------------
orig_mlp = model.model.visual.blocks[0].mlp
mlp_q = prepare(orig_mlp, PTQConfig())
mlp_q.eval()
assert isinstance(mlp_q.wrapped, QuantQwen3VLVisionMLP)

inp_shape = (orig_mlp.intermediate_size, orig_mlp.hidden_size)
# -------------------------------------------------------------------------
# 2. calibration
# -------------------------------------------------------------------------
examples = [
    torch.randn(inp_shape),
    torch.randn(inp_shape),
    torch.randn(inp_shape),
]

with torch.no_grad():
    for example in examples:
        _ = mlp_q(example)

convert(mlp_q)
assert mlp_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
hidden = examples[0]

with torch.no_grad():
    int8_out = mlp_q(hidden)
    fp_out = orig_mlp(hidden)

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(int8_out - fp_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp_out, int8_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp_out, int8_out))

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("qwen3vl_vision_mlp.q.circle")

example = torch.randn(inp_shape)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(mlp_q, (example,))
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")

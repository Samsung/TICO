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
# POST-TRAINING QUANTIZATION EXAMPLE — Simple Linear Model
# -----------------------------------------------------------------------------
# This demo shows a minimal PTQ flow for a toy model:
#   1. Define a simple model with a single Linear layer.
#   2. Replace the FP32 Linear with a QuantLinear wrapper.
#   3. Run a short calibration pass to collect activation statistics.
#   4. Freeze scales / zero-points and switch to INT-simulation mode.
#   5. Compare INT vs FP32 outputs with a mean-absolute-diff check.
#   6. Export the quantized model to a Circle format.
# =============================================================================

import pathlib

import torch
import torch.nn as nn

from tico.experimental.quantization.evaluation.metric import compute_peir
from tico.experimental.quantization.evaluation.utils import plot_two_outputs

from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.wrappers.nn.quant_linear import QuantLinear
from tico.utils.utils import SuppressWarning

# -------------------------------------------------------------------------
# 0. Define a toy model (1 Linear layer only)
# -------------------------------------------------------------------------
class TinyLinearModel(nn.Module):
    """A minimal model: single Linear layer."""

    def __init__(self, in_features=16, out_features=8):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.fc(x)


# Instantiate FP32 model
model = TinyLinearModel()
model.eval()

# Keep FP32 reference for diff check
fp32_layer = model.fc

# -------------------------------------------------------------------------
# 1. Replace the Linear with QuantLinear wrapper
# -------------------------------------------------------------------------
model.fc = QuantLinear(fp32_layer)  # type: ignore[assignment]
# model.fc = PTQWrapper(fp32_layer) (Wrapping helper class)
qlayer = model.fc  # alias for brevity

# -------------------------------------------------------------------------
# 2. Single-pass calibration (collect activation ranges)
# -------------------------------------------------------------------------
assert isinstance(qlayer, QuantLinear)
with torch.no_grad():
    qlayer.enable_calibration()
    for _ in range(16):  # small toy batch
        x = torch.randn(4, 16)  # (batch=4, features=16)
        _ = model(x)
    qlayer.freeze_qparams()  # lock scales & zero-points

assert qlayer._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick INT-sim vs FP32 sanity check
# -------------------------------------------------------------------------
x = torch.randn(2, 16)
with torch.no_grad():
    int8_out = model(x)
    fp32_out = fp32_layer(x)

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(int8_out - fp32_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp32_out, int8_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp32_out, int8_out))

# -------------------------------------------------------------------------
# 4. Export the calibrated model to Circle
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("tiny_linear.q.circle")
example_input = torch.randn(1, 16)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(model, (example_input,))  # forward(x) only
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")

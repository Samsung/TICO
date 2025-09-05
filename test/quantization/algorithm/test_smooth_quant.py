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

import os
import sys
import types
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import SmoothQuantConfig

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class SmoothQuantTest(unittest.TestCase):
    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
    )
    @torch.inference_mode()
    def test_value(self):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
        model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

        # Load data
        dataset = load_dataset("wikiText", "wikitext-2-raw-v1", split="train")
        sample_input = tokenizer(dataset[0]["text"], return_tensors="pt").input_ids

        # base
        base_output = model(sample_input).logits
        base_ln_weight = model.model.layers[0].input_layernorm.weight.clone()

        device = next(model.parameters()).device
        num_samples = 10

        # attach observers
        model = prepare(model, SmoothQuantConfig())

        # run calibration
        for i in range(num_samples):
            input_ids = tokenizer(dataset[i]["text"], return_tensors="pt").input_ids.to(
                device
            )
            model(input_ids)

        # apply smoothing
        q_m = convert(model)

        # target
        target_output = q_m(sample_input).logits
        target_ln_weight = q_m.model.layers[0].input_layernorm.weight

        # Check if weights are updated.
        self.assertFalse(torch.allclose(base_ln_weight, target_ln_weight))

        # Check if output values are same.
        np.testing.assert_allclose(
            actual=base_output,
            desired=target_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Value mismatches.\nbefore result: {base_output}\nafter result: {target_output}",
        )


# ────────────────────────────────────────────────────────────
# Faux fairseq injection (so the handler path works w/o fairseq)
# ────────────────────────────────────────────────────────────
def _install_faux_fairseq():
    """
    Install a minimal fake 'fairseq.modules.transformer_layer' into sys.modules
    with TransformerEncoderLayerBase / TransformerDecoderLayerBase.
    """
    if "fairseq.modules.transformer_layer" in sys.modules:
        return  # already installed by other tests

    fairseq_mod = types.ModuleType("fairseq")
    fairseq_modules_mod = types.ModuleType("fairseq.modules")
    tl_mod = types.ModuleType("fairseq.modules.transformer_layer")

    class TransformerEncoderLayerBase(nn.Module):
        """
        Minimal fairseq-like encoder layer:
          forward: x -> fc1 -> ReLU -> fc2
        """

        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
            self.fc2 = nn.Linear(hidden_dim, in_dim, bias=True)
            # fairseq uses function pointer for activation
            self.activation_fn = F.relu

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.activation_fn(self.fc1(x)))

    class TransformerDecoderLayerBase(TransformerEncoderLayerBase):
        pass

    tl_mod.TransformerEncoderLayerBase = TransformerEncoderLayerBase  # type: ignore[attr-defined]
    tl_mod.TransformerDecoderLayerBase = TransformerDecoderLayerBase  # type: ignore[attr-defined]

    sys.modules["fairseq"] = fairseq_mod
    sys.modules["fairseq.modules"] = fairseq_modules_mod
    sys.modules["fairseq.modules.transformer_layer"] = tl_mod


def _uninstall_faux_fairseq():
    for k in [
        "fairseq.modules.transformer_layer",
        "fairseq.modules",
        "fairseq",
    ]:
        if k in sys.modules:
            del sys.modules[k]


# ────────────────────────────────────────────────────────────
# Tiny model that contains a faux fairseq encoder layer
# ────────────────────────────────────────────────────────────
class TinyFSeqLikeModel(nn.Module):
    """
    Wrap a single faux TransformerEncoderLayerBase under a stable name
    so that observer keys look like "encoder.layer.fc1".
    """

    def __init__(self, in_dim=8, hidden_dim=16):
        super().__init__()
        # Import from the injected faux fairseq
        from fairseq.modules.transformer_layer import (  # type: ignore
            TransformerEncoderLayerBase,
        )

        self.encoder = nn.Module()
        self.encoder.layer = TransformerEncoderLayerBase(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] or [N, D]; observer flattens last dim
        return self.encoder.layer(x)  # type: ignore[operator]


class SmoothQuantOutputHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        _install_faux_fairseq()

    @classmethod
    def tearDownClass(cls):
        # Optional: clean up; comment out if other tests need faux fairseq
        _uninstall_faux_fairseq()

    @torch.inference_mode()
    def test_output_hook_on_fairseq_like_relu_bridge(self):
        """
        Verifies that:
          1) With acts_from='output', the observer collects fc1 outputs,
             and smoothing fuses scaling into (fc1 rows, fc2 cols).
          2) The forward outputs before and after convert() are (near-)identical.
          3) Weights actually changed (non-trivial transformation applied).
        """
        device = "cpu"
        D, H = 8, 16
        B, T = 4, 5

        # Build model
        model = TinyFSeqLikeModel(in_dim=D, hidden_dim=H).to(device)
        model.eval()

        # Sample input
        x = torch.randn(B, T, D, device=device)

        # Baseline forward & weight snapshots
        base_out = model(x)
        assert hasattr(model.encoder.layer, "fc1")
        assert hasattr(model.encoder.layer, "fc2")
        assert isinstance(model.encoder.layer.fc1, nn.Linear)
        assert isinstance(model.encoder.layer.fc2, nn.Linear)
        base_fc1_w = model.encoder.layer.fc1.weight.detach().clone()
        base_fc2_w = model.encoder.layer.fc2.weight.detach().clone()
        base_fc1_b = model.encoder.layer.fc1.bias.detach().clone()
        base_fc2_b = model.encoder.layer.fc2.bias.detach().clone()

        # Prepare with OUTPUT hooks
        cfg = SmoothQuantConfig(alpha=0.6, acts_from="output")
        model_prep = prepare(model, cfg)

        # Run a few calibration passes (output hooks record activations)
        for _ in range(6):
            x_cal = torch.randn(B, T, D, device=device) * (1.0 + 0.5 * torch.randn(1))
            model_prep(x_cal)

        # Convert (applies smoothing)
        model_sq = convert(model_prep)
        model_sq.eval()

        # Post-convert outputs
        tgt_out = model_sq(x)

        # 1) Functional equivalence (allow tiny numerical drift)
        np.testing.assert_allclose(
            actual=tgt_out.cpu().numpy(),
            desired=base_out.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Output mismatch after SmoothQuant (output-hook, fairseq-like).",
        )

        # 2) Weights must have changed (row/col scaling fused)
        assert hasattr(model_sq.encoder.layer, "fc1")
        assert hasattr(model_sq.encoder.layer, "fc2")
        assert isinstance(model_sq.encoder.layer.fc1, nn.Linear)
        assert isinstance(model_sq.encoder.layer.fc2, nn.Linear)
        new_fc1_w = model_sq.encoder.layer.fc1.weight.detach()
        new_fc2_w = model_sq.encoder.layer.fc2.weight.detach()
        new_fc1_b = model_sq.encoder.layer.fc1.bias.detach()
        new_fc2_b = model_sq.encoder.layer.fc2.bias.detach()

        self.assertFalse(
            torch.allclose(base_fc1_w, new_fc1_w),
            msg="fc1.weight did not change after smoothing.",
        )
        self.assertFalse(
            torch.allclose(base_fc2_w, new_fc2_w),
            msg="fc2.weight did not change after smoothing.",
        )
        # Biases also scaled across the bridge on fc1 side
        self.assertFalse(
            torch.allclose(base_fc1_b, new_fc1_b),
            msg="fc1.bias did not change after smoothing.",
        )

        # 3) Optional sanity: ReLU-positive-homogeneity implies non-negative scaling
        #    so weights shouldn't blow up or vanish.
        self.assertTrue(
            torch.isfinite(new_fc1_w).all() and torch.isfinite(new_fc2_w).all()
        )

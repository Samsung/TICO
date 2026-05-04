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

from tico.quantization import convert, prepare
from tico.quantization.config.smoothquant import SmoothQuantConfig

# Initialize the registry with real transformer classes before any faux classes are installed.
# This prevents the registry from registering wrappers for faux/mock classes.
from tico.quantization.wrapq.wrappers.registry import _lazy_init

_lazy_init()

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class SmoothQuantTest(unittest.TestCase):
    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
    )
    @torch.inference_mode()
    def test_value(self):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0", legacy=False)
        model = AutoModelForCausalLM.from_pretrained(
            "Maykeye/TinyLLama-v0", dtype=torch.float32
        )

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


# ────────────────────────────────────────────────────────────
# Faux Qwen3-VL vision components injection
# ────────────────────────────────────────────────────────────

# Store original classes for restoration
_ORIGINAL_QWEN3VL_CLASSES = {}


class FauxQwen3VLVisionAttention(nn.Module):
    """Minimal faux Qwen3VLVisionAttention with combined qkv projection."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)


class FauxQwen3VLVisionMLP(nn.Module):
    """Minimal faux Qwen3VLVisionMLP."""

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)


class FauxQwen3VLVisionBlock(nn.Module):
    """
    Minimal faux Qwen3VLVisionBlock:
      - norm1 (LayerNorm) -> attn (attention with qkv)
      - norm2 (LayerNorm) -> mlp
    """

    def __init__(self, hidden_size: int = 64, intermediate_size: int = 128):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = FauxQwen3VLVisionAttention(hidden_size)
        self.mlp = FauxQwen3VLVisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward (not used in tests)
        return x + self.attn(self.norm1(x)) + self.mlp(self.norm2(x))


class FauxQwen3VLVisionPatchMerger(nn.Module):
    """
    Minimal faux Qwen3VLVisionPatchMerger:
      - norm (LayerNorm) -> linear_fc1 -> act_fn -> linear_fc2
    """

    def __init__(
        self,
        hidden_size: int = 64,
        out_hidden_size: int = 128,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        # norm.weight shape matches hidden_size for standard case
        norm_size = hidden_size if not use_postshuffle_norm else hidden_size * 4
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(hidden_size, out_hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified forward (not used in tests)
        return self.linear_fc2(self.linear_fc1(self.norm(x)))


def _install_faux_qwen3vl():
    """
    Install minimal fake 'transformers.models.qwen3_vl.modeling_qwen3_vl' into sys.modules
    with Qwen3VLVisionBlock and Qwen3VLVisionPatchMerger.
    """
    global _ORIGINAL_QWEN3VL_CLASSES

    # Build the module hierarchy
    transformers_mod = sys.modules.get("transformers", types.ModuleType("transformers"))
    models_mod = sys.modules.get(
        "transformers.models", types.ModuleType("transformers.models")
    )
    qwen3_vl_mod = sys.modules.get(
        "transformers.models.qwen3_vl", types.ModuleType("transformers.models.qwen3_vl")
    )
    modeling_mod = sys.modules.get(
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        types.ModuleType("transformers.models.qwen3_vl.modeling_qwen3_vl"),
    )

    # Store original classes if they exist (for restoration)
    if hasattr(modeling_mod, "Qwen3VLVisionBlock"):
        _ORIGINAL_QWEN3VL_CLASSES[
            "Qwen3VLVisionBlock"
        ] = modeling_mod.Qwen3VLVisionBlock
    if hasattr(modeling_mod, "Qwen3VLVisionPatchMerger"):
        _ORIGINAL_QWEN3VL_CLASSES[
            "Qwen3VLVisionPatchMerger"
        ] = modeling_mod.Qwen3VLVisionPatchMerger

    # Override with faux classes
    modeling_mod.Qwen3VLVisionBlock = FauxQwen3VLVisionBlock  # type: ignore[attr-defined]
    modeling_mod.Qwen3VLVisionPatchMerger = FauxQwen3VLVisionPatchMerger  # type: ignore[attr-defined]
    modeling_mod.FauxQwen3VLVisionAttention = FauxQwen3VLVisionAttention  # type: ignore[attr-defined]
    modeling_mod.FauxQwen3VLVisionMLP = FauxQwen3VLVisionMLP  # type: ignore[attr-defined]

    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.models"] = models_mod
    sys.modules["transformers.models.qwen3_vl"] = qwen3_vl_mod
    sys.modules["transformers.models.qwen3_vl.modeling_qwen3_vl"] = modeling_mod


def _uninstall_faux_qwen3vl():
    """
    Restore original classes in the modeling module instead of deleting modules.

    Deleting modules from sys.modules causes subsequent imports to create NEW class
    objects that don't match the ones registered in the wrapper registry, leading to
    'No quant wrapper for X' errors in subsequent tests.
    """
    global _ORIGINAL_QWEN3VL_CLASSES

    modeling_mod = sys.modules.get("transformers.models.qwen3_vl.modeling_qwen3_vl")
    if modeling_mod is not None:
        # Restore original classes if they were stored
        if "Qwen3VLVisionBlock" in _ORIGINAL_QWEN3VL_CLASSES:
            modeling_mod.Qwen3VLVisionBlock = _ORIGINAL_QWEN3VL_CLASSES["Qwen3VLVisionBlock"]  # type: ignore[attr-defined]
        if "Qwen3VLVisionPatchMerger" in _ORIGINAL_QWEN3VL_CLASSES:
            modeling_mod.Qwen3VLVisionPatchMerger = _ORIGINAL_QWEN3VL_CLASSES["Qwen3VLVisionPatchMerger"]  # type: ignore[attr-defined]

        # Remove faux classes that we added
        if hasattr(modeling_mod, "FauxQwen3VLVisionAttention"):
            delattr(modeling_mod, "FauxQwen3VLVisionAttention")
        if hasattr(modeling_mod, "FauxQwen3VLVisionMLP"):
            delattr(modeling_mod, "FauxQwen3VLVisionMLP")

    # Clear the stored original classes
    _ORIGINAL_QWEN3VL_CLASSES = {}


# ────────────────────────────────────────────────────────────
# Tests for _apply_if_qwen3vl_vision_block
# ────────────────────────────────────────────────────────────
class ApplyIfQwen3VLVisionBlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        _install_faux_qwen3vl()
        # Get the faux classes from sys.modules
        cls.Qwen3VLVisionBlock = sys.modules[  # type: ignore[attr-defined]
            "transformers.models.qwen3_vl.modeling_qwen3_vl"
        ].Qwen3VLVisionBlock

    @classmethod
    def tearDownClass(cls):
        _uninstall_faux_qwen3vl()

    def setUp(self):
        # Import the function after faux module is installed
        from tico.quantization.algorithm.smoothquant.smooth_quant import (
            _apply_if_qwen3vl_vision_block,
        )

        self._apply_if_qwen3vl_vision_block = _apply_if_qwen3vl_vision_block

    @torch.inference_mode()
    def test_returns_true_for_valid_block_with_qkv(self):
        """Test that the function returns True for a valid Qwen3VLVisionBlock with qkv."""
        block = self.Qwen3VLVisionBlock(hidden_size=64, intermediate_size=128)  # type: ignore[attr-defined]

        # Store original weights
        orig_norm1_weight = block.norm1.weight.clone()
        orig_norm2_weight = block.norm2.weight.clone()
        orig_qkv_weight = block.attn.qkv.weight.clone()
        orig_mlp_weight = block.mlp.linear_fc1.weight.clone()

        # Create activation_max with correct keys
        activation_max = {
            "test.attn.qkv": torch.abs(torch.randn(64)) + 0.1,
            "test.mlp.linear_fc1": torch.abs(torch.randn(64)) + 0.1,
        }

        result = self._apply_if_qwen3vl_vision_block("test", block, activation_max, 0.5)

        self.assertTrue(result)
        # Verify weights were modified
        self.assertFalse(torch.allclose(orig_norm1_weight, block.norm1.weight))
        self.assertFalse(torch.allclose(orig_norm2_weight, block.norm2.weight))
        self.assertFalse(torch.allclose(orig_qkv_weight, block.attn.qkv.weight))
        self.assertFalse(torch.allclose(orig_mlp_weight, block.mlp.linear_fc1.weight))


# ────────────────────────────────────────────────────────────
# Tests for _apply_if_qwen3vl_vision_patch_merger
# ────────────────────────────────────────────────────────────
class ApplyIfQwen3VLVisionPatchMergerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        np.random.seed(0)
        _install_faux_qwen3vl()
        # Get the faux classes from sys.modules
        cls.Qwen3VLVisionPatchMerger = sys.modules[  # type: ignore[attr-defined]
            "transformers.models.qwen3_vl.modeling_qwen3_vl"
        ].Qwen3VLVisionPatchMerger

    @classmethod
    def tearDownClass(cls):
        _uninstall_faux_qwen3vl()

    def setUp(self):
        from tico.quantization.algorithm.smoothquant.smooth_quant import (
            _apply_if_qwen3vl_vision_patch_merger,
        )

        self._apply_if_qwen3vl_vision_patch_merger = (
            _apply_if_qwen3vl_vision_patch_merger
        )

    @torch.inference_mode()
    def test_returns_true_and_applies_smoothing_for_valid_merger(self):
        """Test that function returns True and applies smoothing for valid merger."""
        # Create merger with matching dimensions (default case)
        merger = self.Qwen3VLVisionPatchMerger(hidden_size=64, out_hidden_size=128)  # type: ignore[attr-defined]

        # Verify dimensions match
        norm_numel = merger.norm.weight.numel()  # 64
        linear_in = merger.linear_fc1.in_features  # 64
        self.assertEqual(norm_numel, linear_in)

        # Store original weights
        orig_norm_weight = merger.norm.weight.clone()
        orig_fc1_weight = merger.linear_fc1.weight.clone()

        # Create activation_max with correct key
        activation_max = {
            "test.linear_fc1": torch.abs(torch.randn(64)) + 0.1,
        }

        result = self._apply_if_qwen3vl_vision_patch_merger(
            "test", merger, activation_max, 0.5
        )

        self.assertTrue(result)
        # Verify weights were modified
        self.assertFalse(torch.allclose(orig_norm_weight, merger.norm.weight))
        self.assertFalse(torch.allclose(orig_fc1_weight, merger.linear_fc1.weight))

    @torch.inference_mode()
    def test_smoothing_preserves_numerical_equivalence(self):
        """Test that smoothing preserves numerical equivalence of forward pass."""
        merger = self.Qwen3VLVisionPatchMerger(hidden_size=64, out_hidden_size=128)  # type: ignore[attr-defined]
        merger.eval()

        # Create sample input
        x = torch.randn(2, 10, 64)

        # Get baseline output
        with torch.no_grad():
            base_output = merger(x)

        # Store original weights
        orig_norm_weight = merger.norm.weight.clone()
        orig_fc1_weight = merger.linear_fc1.weight.clone()

        # Apply smoothing
        activation_max = {
            "test.linear_fc1": torch.abs(torch.randn(64)) + 0.1,
        }
        result = self._apply_if_qwen3vl_vision_patch_merger(
            "test", merger, activation_max, 0.5
        )

        self.assertTrue(result)

        # Verify weights changed
        self.assertFalse(torch.allclose(orig_norm_weight, merger.norm.weight))
        self.assertFalse(torch.allclose(orig_fc1_weight, merger.linear_fc1.weight))

        # Get output after smoothing
        with torch.no_grad():
            new_output = merger(x)

        # Outputs should be close (smoothquant preserves equivalence)
        np.testing.assert_allclose(
            actual=new_output.numpy(),
            desired=base_output.numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg="Output mismatch after SmoothQuant on PatchMerger.",
        )

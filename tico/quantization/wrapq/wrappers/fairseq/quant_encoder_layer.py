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
#
# -----------------------------------------------------------------------------
# This file includes modifications based on fairseq
#  (https://github.com/facebookresearch/fairseq), originally licensed under
# the MIT License. See the LICENSE file in the fairseq repository for details.
# -----------------------------------------------------------------------------

from typing import Optional

import torch.nn as nn
from torch import Tensor

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.fairseq.quant_mha import (
    QuantFairseqMultiheadAttention,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("fairseq.modules.transformer_layer.TransformerEncoderLayerBase")
class QuantFairseqEncoderLayer(QuantModuleBase):
    """
    Quant-aware drop-in replacement for Fairseq TransformerEncoderLayerBase.

    Design notes (inference-friendly):
    - All training-time logic (dropout, activation-dropout) is removed.
    - I/O shape follows Fairseq convention: [T, B, C].
    - `return_fc` behavior is preserved (returns (x, fc_result) if enabled).
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # --- copy meta / config flags from FP layer (read-only) -------------
        assert hasattr(fp_layer, "embed_dim")
        assert hasattr(fp_layer, "normalize_before")
        self.embed_dim: int = int(fp_layer.embed_dim)  # type: ignore[arg-type]
        self.normalize_before: bool = bool(fp_layer.normalize_before)
        self.return_fc: bool = bool(getattr(fp_layer, "return_fc", False))

        # --- PTQ-wrapped submodules ----------------------------------------
        attn_cfg = qcfg.child("self_attn") if qcfg else None
        fc1_cfg = qcfg.child("fc1") if qcfg else None
        fc2_cfg = qcfg.child("fc2") if qcfg else None
        attn_ln_cfg = qcfg.child("self_attn_layer_norm") if qcfg else None
        final_ln_cfg = qcfg.child("final_layer_norm") if qcfg else None

        assert hasattr(fp_layer, "self_attn") and isinstance(
            fp_layer.self_attn, nn.Module
        )
        assert hasattr(fp_layer, "fc1") and isinstance(fp_layer.fc1, nn.Module)
        assert hasattr(fp_layer, "fc2") and isinstance(fp_layer.fc2, nn.Module)

        self.self_attn = QuantFairseqMultiheadAttention(
            fp_layer.self_attn, qcfg=attn_cfg, fp_name=f"{fp_name}.self_attn"
        )
        self.fc1 = PTQWrapper(fp_layer.fc1, qcfg=fc1_cfg, fp_name=f"{fp_name}.fc1")
        self.fc2 = PTQWrapper(fp_layer.fc2, qcfg=fc2_cfg, fp_name=f"{fp_name}.fc2")

        # LayerNorms
        assert hasattr(fp_layer, "self_attn_layer_norm") and isinstance(
            fp_layer.self_attn_layer_norm, nn.Module
        )
        assert hasattr(fp_layer, "final_layer_norm") and isinstance(
            fp_layer.final_layer_norm, nn.Module
        )
        self.self_attn_layer_norm = PTQWrapper(
            fp_layer.self_attn_layer_norm,
            qcfg=attn_ln_cfg,
            fp_name=f"{fp_name}.self_attn_layer_norm",
        )
        self.final_layer_norm = PTQWrapper(
            fp_layer.final_layer_norm,
            qcfg=final_ln_cfg,
            fp_name=f"{fp_name}.final_layer_norm",
        )

        # Activation function
        self.activation_fn = fp_layer.activation_fn  # type: ignore[operator]  # e.g., GELU/ReLU
        self.obs_activation_fn = self._make_obs("activation_fn")

    # ----------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,  # [T,B,C]
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,  # [T,S] boolean/byte or additive float
    ):
        """
        Returns:
            x' of shape [T, B, C] (or (x', fc_result) when return_fc=True)
        """
        # ---- Self-Attention block (pre-/post-norm kept as in FP layer) ----
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Fairseq MHA expects [T,B,C]; our wrapped module keeps the same API
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,  # additive float [B,S] or None
            need_weights=False,
            attn_mask=attn_mask,  # additive float [T,S] or None
        )
        x = residual + attn_out

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # ---- FFN block (no dropout/activation-dropout) --------------------
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.fc1(x)  # Linear
        x = self.activation_fn(x)  # type: ignore[operator]
        x = self._fq(x, self.obs_activation_fn)
        x = self.fc2(x)  # Linear

        fc_result = x  # keep before residual for optional return

        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.return_fc:
            return x, fc_result
        return x

    def _all_observers(self):
        yield from (self.obs_activation_fn,)
        for m in (
            self.self_attn,
            self.fc1,
            self.fc2,
            self.self_attn_layer_norm,
            self.final_layer_norm,
        ):
            if isinstance(m, QuantModuleBase):
                yield from m._all_observers()

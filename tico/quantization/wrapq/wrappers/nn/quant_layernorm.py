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

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(nn.LayerNorm)
class QuantLayerNorm(QuantModuleBase):
    """
    QuantLayerNorm — drop-in replacement for nn.LayerNorm that quantizes
    the elementary steps:
        1) μ = mean(x, dims)                (mean)
        2) c = x - μ                        (sub)
        3) s = c * c                        (square)
        4) v = mean(s, dims)                (variance)
        5) e = v + eps                      (add-eps)
        6) r = rsqrt(e)                     (rsqrt)
        7) n = c * r                        (normalize)
        8) y = (n * γ) + β                  (affine), with:
             • affine_mul : n * γ
             • affine_add : (n * γ) + β
    """

    def __init__(
        self,
        fp: nn.LayerNorm,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.eps = torch.tensor(self.module.eps)
        # Number of trailing dims participating in normalization
        # (PyTorch stores normalized_shape as a tuple even if an int was passed)
        self._norm_ndim: int = len(fp.normalized_shape)  # safe for int→tuple

        # Activation / intermediate observers
        self.obs_act_in = self._make_obs("act_in")
        self.obs_mean = self._make_obs("mean")
        self.obs_centered = self._make_obs("centered")
        self.obs_square = self._make_obs("square")
        self.obs_var = self._make_obs("var")
        self.obs_eps = self._make_obs("eps")
        self.obs_add_eps = self._make_obs("add_eps")
        self.obs_inv_std = self._make_obs("inv_std")
        self.obs_norm = self._make_obs("norm")
        self.obs_act_outs = self._make_obs("act_out")

        # Optional affine parameter observers (γ, β)
        self.obs_weight = None
        self.obs_bias = None
        self.obs_affine_mul = None
        self.obs_affine_add = None
        if self.module.elementwise_affine:
            if self.module.weight is not None:
                self.obs_weight = self._make_obs("weight")
            if self.module.bias is not None:
                self.obs_bias = self._make_obs("bias")
            # Per-op observers for (n * w) and (+ b)
            self.obs_affine_mul = self._make_obs("affine_mul")
            self.obs_affine_add = self._make_obs("affine_add")

    def enable_calibration(self) -> None:
        """
        Switch to CALIB mode and collect *fixed* ranges for affine params
        immediately, since they do not change across inputs.
        """
        super().enable_calibration()
        if self.module.elementwise_affine:
            if self.obs_weight is not None and self.module.weight is not None:
                self.obs_weight.collect(self.module.weight)
            if self.obs_bias is not None and self.module.bias is not None:
                self.obs_bias.collect(self.module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Determine reduction dims (last self._norm_ndim axes)
        # Example: if x.ndim=4 and norm_ndim=2 → dims=(2,3)
        dims = tuple(range(x.dim() - self._norm_ndim, x.dim()))

        # 0) input
        x_q = self._fq(x, self.obs_act_in)

        # 1) mean
        mu = x_q.mean(dim=dims, keepdim=True)
        mu_q = self._fq(mu, self.obs_mean)

        # 2) center
        c = x_q - mu_q
        c_q = self._fq(c, self.obs_centered)

        # 3) square (elementwise mul)
        s = c_q * c_q
        s_q = self._fq(s, self.obs_square)

        # 4) variance (via squared mean)
        v = s_q.mean(dim=dims, keepdim=True)
        v_q = self._fq(v, self.obs_var)

        # 5) add eps
        eps_q = self._fq(self.eps, self.obs_eps)
        e = v_q + eps_q
        e_q = self._fq(e, self.obs_add_eps)

        # 6) inverse std
        r = torch.rsqrt(e_q)
        r_q = self._fq(r, self.obs_inv_std)

        # 7) normalize
        n = c_q * r_q
        n_q = self._fq(n, self.obs_norm)

        # 8) optional affine
        if self.module.elementwise_affine:
            w = self.module.weight
            b = self.module.bias
            if self._mode is Mode.QUANT:
                if self.obs_weight is not None and w is not None:
                    w = self.obs_weight.fake_quant(w)  # type: ignore[assignment]
                if self.obs_bias is not None and b is not None:
                    b = self.obs_bias.fake_quant(b)  # type: ignore[assignment]
            y = n_q
            # 8a) n * w  (fake-quant the result of the mul)
            if w is not None:
                y = y * w
                if self.obs_affine_mul is not None:
                    y = self._fq(y, self.obs_affine_mul)

            # 8b) (+ b)  (fake-quant the result of the add)
            if b is not None:
                y = y + b
                if self.obs_affine_add is not None:
                    y = self._fq(y, self.obs_affine_add)
        else:
            y = n_q

        # 9) output activation
        return self._fq(y, self.obs_act_outs)

    def _all_observers(self) -> Iterable:
        obs: Tuple = (
            self.obs_act_in,
            self.obs_mean,
            self.obs_centered,
            self.obs_square,
            self.obs_var,
            self.obs_eps,
            self.obs_add_eps,
            self.obs_inv_std,
            self.obs_norm,
            self.obs_act_outs,
        )
        # Insert affine param observers if present
        if self.module.elementwise_affine:
            if self.obs_weight is not None:
                obs = (self.obs_weight,) + obs
            if self.obs_bias is not None:
                obs = obs + (self.obs_bias,)
            if self.obs_affine_mul is not None:
                obs = obs + (self.obs_affine_mul,)
            if self.obs_affine_add is not None:
                obs = obs + (self.obs_affine_add,)
        return obs

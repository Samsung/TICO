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
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaRMSNorm",
)
class QuantLlamaRMSNorm(QuantModuleBase):
    """
    Quant wrapper for LlamaRMSNorm (T5LayerNorm-style RMSNorm).

    Reference forward:
        input_dtype = x.dtype
        x = x.float()
        v = mean(x^2, dim=-1, keepdim=True)
        y = x * rsqrt(v + eps)
        out = weight * y.to(input_dtype)

    We quantize the elementary steps:
        0) x_q = fq(x)                          (act_in)
        1) x_fp32 = x_q.float()
        2) s = x_fp32 * x_fp32                 (square)
        3) v = mean(s, dim=-1)                 (var)
        4) e = v + eps                         (add_eps)
        5) r = rsqrt(e)                        (inv_std)
        6) n = x_fp32 * r                      (norm)
        7) y = (n.to(dtype) * w)               (affine_mul)
        8) out_q = fq(y)                       (act_out)
    """

    def __init__(
        self,
        fp: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp
        self.eps = torch.tensor(self.module.variance_epsilon)

        # Observers
        self.obs_act_in = self._make_obs("act_in")
        self.obs_square = self._make_obs("square")
        self.obs_var = self._make_obs("var")
        self.obs_eps = self._make_obs("eps")
        self.obs_add_eps = self._make_obs("add_eps")
        self.obs_inv_std = self._make_obs("inv_std")
        self.obs_norm = self._make_obs("norm")

        # Affine (weight only)
        self.obs_weight = self._make_obs("weight")
        self.obs_affine_mul = self._make_obs("affine_mul")
        self.obs_act_out = self._make_obs("act_out")

    def enable_calibration(self) -> None:
        """
        Switch to CALIB mode and immediately collect fixed range for weight.
        """
        super().enable_calibration()
        assert self.module.weight is not None
        self.obs_weight.collect(self.module.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 0) input
        x_q = self._fq(hidden_states, self.obs_act_in)

        # 1-3) variance = mean(x^2)
        s = x_q * x_q
        s_q = self._fq(s, self.obs_square)

        v = s_q.mean(dim=-1, keepdim=True)
        v_q = self._fq(v, self.obs_var)

        # 4) add eps
        eps_q = self._fq(self.eps, self.obs_eps)
        e = v_q + eps_q
        e_q = self._fq(e, self.obs_add_eps)

        # 5) inv std
        r = torch.rsqrt(e_q)
        r_q = self._fq(r, self.obs_inv_std)

        # 6) normalize
        n = x_q * r_q
        n_q = self._fq(n, self.obs_norm)

        # 7) affine: weight * n.to(input_dtype)
        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.obs_weight.fake_quant(w)  # type: ignore[assignment]

        y = n_q * w
        y = self._fq(y, self.obs_affine_mul)

        # 8) output
        return self._fq(y, self.obs_act_out)

    def _all_observers(self) -> Iterable:
        obs: Tuple = (
            self.obs_weight,
            self.obs_act_in,
            self.obs_square,
            self.obs_var,
            self.obs_eps,
            self.obs_add_eps,
            self.obs_inv_std,
            self.obs_norm,
            self.obs_affine_mul,
            self.obs_act_out,
        )
        return obs

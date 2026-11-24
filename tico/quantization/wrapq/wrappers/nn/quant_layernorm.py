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
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp

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
                self.obs_weight = self._make_per_tensor_obs("weight")
            if self.module.bias is not None:
                self.obs_bias = self._make_per_tensor_obs("bias")
            if self.module.weight is not None:
                self.obs_affine_mul = self._make_obs("affine_mul")
            if self.module.bias is not None:
                self.obs_affine_add = self._make_obs("affine_add")

    @staticmethod
    def _validate_legacy_epsilon(epsilon, configured_epsilon) -> None:
        """Accept an old buffer only when it represents the configured scalar."""
        if (
            not isinstance(epsilon, torch.Tensor)
            or epsilon.ndim != 0
            or not epsilon.is_floating_point()
            or not torch.isfinite(epsilon).item()
            or epsilon.item() != epsilon.new_tensor(configured_epsilon).item()
            or (configured_epsilon != 0 and epsilon.item() == 0)
        ):
            raise RuntimeError(
                "LayerNorm checkpoint has invalid or mismatched legacy epsilon; "
                "reprepare and recalibrate."
            )

    def __setstate__(self, state):
        if "eps" in state.get("_buffers", {}):
            self._validate_legacy_epsilon(
                state["_buffers"]["eps"], state["_modules"]["module"].eps
            )
            state = state.copy()
            state["_buffers"] = state["_buffers"].copy()
            del state["_buffers"]["eps"]
        super().__setstate__(state)

    def get_extra_state(self):
        state = super().get_extra_state()
        state["layernorm_eps"] = self.module.eps
        return state

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("layernorm_eps") != self.module.eps:
            raise RuntimeError(
                "LayerNorm checkpoint epsilon does not match the source module; "
                "reprepare and recalibrate."
            )
        super().set_extra_state(state)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if prefix + "eps" in state_dict:
            self._validate_legacy_epsilon(state_dict[prefix + "eps"], self.module.eps)
            # Consume the redundant buffer at the load boundary, keeping exact
            # frozen grids and the caller's state dictionary unchanged.
            state_dict = state_dict.copy()
            del state_dict[prefix + "eps"]
            extra_key = prefix + "_extra_state"
            if isinstance(state_dict.get(extra_key), dict):
                state_dict[extra_key] = state_dict[extra_key].copy()
                state_dict[extra_key].setdefault("layernorm_eps", self.module.eps)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _prepare_calibration(self) -> None:
        """
        Collect fixed affine parameters after the common calibration reset.
        """
        if self.module.elementwise_affine:
            if self.obs_weight is not None and self.module.weight is not None:
                self.obs_weight.collect(self.module.weight)
            if self.obs_bias is not None and self.module.bias is not None:
                self.obs_bias.collect(self.module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decomposed arithmetic would accept broadcasting that native LayerNorm
        # rejects. Check the module's shape contract before touching observers.
        normalized_shape = tuple(self.module.normalized_shape)
        norm_ndim = len(normalized_shape)
        if norm_ndim == 0:
            raise RuntimeError(
                "Expected normalized_shape to be at least 1-dimensional, i.e., "
                "containing at least one element, but got normalized_shape = []"
            )
        w, b = self.module.weight, self.module.bias
        for name, parameter in (("weight", w), ("bias", b)):
            if parameter is not None and tuple(parameter.shape) != normalized_shape:
                raise RuntimeError(
                    f"Expected {name} to be of same shape as normalized_shape, "
                    f"but got {name} of shape {list(parameter.shape)} "
                    f"and normalized_shape = {list(normalized_shape)}"
                )
        if x.dim() < norm_ndim or tuple(x.shape[-norm_ndim:]) != normalized_shape:
            expected = ", ".join(str(size) for size in normalized_shape)
            raise RuntimeError(
                f"Given normalized_shape={list(normalized_shape)}, "
                f"expected input with shape [*, {expected}], "
                f"but got input of size{list(x.shape)}"
            )
        parameters = tuple(p for p in (w, b) if p is not None)
        if any(p.dtype != x.dtype for p in parameters) and not (
            x.dtype in (torch.float16, torch.bfloat16)
            and all(p.dtype == torch.float32 for p in parameters)
        ):
            raise RuntimeError(
                "LayerNorm affine dtype must match the input dtype, or uniformly "
                "use float32 parameters with float16/bfloat16 input."
            )
        dims = tuple(range(x.dim() - norm_ndim, x.dim()))

        # 0) input
        x_q = self._fq(x, self.obs_act_in)
        # Native LayerNorm accumulates low-precision inputs in FP32. Keep the
        # input observer at the public boundary and all arithmetic sites wide.
        if x.dtype in (torch.float16, torch.bfloat16):
            x_q = x_q.float()

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
        # The source module owns epsilon as a Python scalar, so module.half()
        # cannot underflow it and double inputs do not inherit FP32 rounding.
        eps_q = self._fq(x_q.new_full((), self.module.eps), self.obs_eps)
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
            if self._mode is Mode.QUANT:
                if self.obs_weight is not None and w is not None:
                    w = self.obs_weight.fake_quant(w)  # type: ignore[assignment]
                if self.obs_bias is not None and b is not None:
                    b = self.obs_bias.fake_quant(b)  # type: ignore[assignment]
            y = n_q
            # 8a) n * w  (fake-quant the result of the mul)
            if w is not None:
                y = y * w.to(dtype=x_q.dtype)
                if self.obs_affine_mul is not None:
                    y = self._fq(y, self.obs_affine_mul)

            # 8b) (+ b)  (fake-quant the result of the add)
            if b is not None:
                y = y + b.to(dtype=x_q.dtype)
                if self.obs_affine_add is not None:
                    y = self._fq(y, self.obs_affine_add)
        else:
            y = n_q

        # 9) output activation
        return self._fq(y.to(dtype=x.dtype), self.obs_act_outs)

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

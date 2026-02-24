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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(nn.Conv3d)
class QuantConv3dDecomposed(QuantModuleBase):
    """
    Quantization wrapper for nn.Conv3d with decomposition to Conv2d.

    This class decomposes Conv3d into multiple Conv2d operations to ensure
    all computations remain quantized. The decomposition follows the slice + Conv2d + Add
    approach, avoiding graph passes that introduce floating-point operations.

    Quantization:
    - Per-channel weight quantization (asymmetric)
    - Per-tensor input activation quantization
    - Per-tensor output activation quantization
    - Per-tensor quantization for all intermediate tensors (input slices, conv2d outputs, accumulators)
    """

    def __init__(
        self,
        fp: nn.Conv3d,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        # Static observers (always exist)
        self.obs_weight = self._make_obs(
            "weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

        # Store original module
        self.module = fp

        # Dynamic observers (created lazily during first forward pass)
        self._input_slice_obs: Dict[int, ObserverBase] = {}  # Maps k (int) -> observer
        self._conv2d_obs: Dict[int, ObserverBase] = {}  # Maps k (int) -> observer
        self._acc_obs: Dict[int, ObserverBase] = {}  # Maps t_out (int) -> observer

        # Tracking for lazy observer creation
        self._dynamic_obs_calibrated = False

    def enable_calibration(self) -> None:
        """Enable calibration mode."""
        super().enable_calibration()

        # Collect weight statistics immediately (weights are static)
        self.obs_weight.collect(self.module.weight)

        # Reset dynamic observers for new calibration
        self._dynamic_obs_calibrated = False

    def _create_dynamic_observers(
        self,
        kT: int,
        T_out: int,
    ):
        """
        Create dynamic observers for intermediate quantization points.

        Args:
            kT: Kernel temporal dimension
            T_out: Number of output temporal positions
        """

        def create_observer(obs_name_prefix, obs_dictionary, dict_key):
            obs_name = f"{obs_name_prefix}{dict_key}"
            obs = self._make_obs(obs_name)
            obs_dictionary[dict_key] = obs
            self.add_module(obs_name, obs)
            # self.add_module(obs_name, obs) is required for torch.export() to properly access
            # the observer and its internal quantization parameters (cached_scale, cached_zp)
            # during graph construction. When torch.export() traces the model, it creates
            # 'get_attr' nodes to access module attributes. If observers are stored only in
            # dictionaries or via setattr(), torch.export() cannot create valid get_attr nodes
            # for the observer's cached_scale and cached_zp tensors, leading to warnings:
            #   "Attempted to insert a get_attr Node with no underlying reference"
            # By registering with add_module(), the observer becomes part of the module's
            # named_modules() tree, making both the observer AND its quantization parameters
            # accessible to the graph construction process. This ensures that the exported
            # graph can properly reference quantization parameters during Circle conversion.

        # Input slice observers (one for each temporal kernel position)
        for k in range(kT):
            create_observer(f"input_slice_k", self._input_slice_obs, k)

        # Conv2d output observers (one for each temporal kernel position)
        for k in range(kT):
            create_observer(f"conv2d_out_k", self._conv2d_obs, k)

        # Accumulator observers (one for each output temporal position)
        for t_out in range(T_out):
            create_observer(f"accumulator_t", self._acc_obs, t_out)

        self._dynamic_obs_calibrated = True

    def _parse_padding(self, padding) -> Tuple[int, int, int]:
        """Parse padding parameter to (temporal, height, width) tuple."""
        if isinstance(padding, str):
            if padding == "same":
                kT, kH, kW = self.module.kernel_size
                return kT // 2, kH // 2, kW // 2
            elif padding == "valid":
                return 0, 0, 0
            else:
                raise ValueError(f"Unsupported padding string: {padding}")
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 1:
                return padding[0], padding[0], padding[0]
            elif len(padding) == 3:
                return padding[0], padding[1], padding[2]
            else:
                raise ValueError(f"Unsupported padding format: {padding}")
        elif isinstance(padding, int):  # int
            return padding, padding, padding
        else:
            raise ValueError(f"Unsupported padding type: {type(padding)}")

    def _apply_temporal_padding(
        self,
        x: torch.Tensor,
        temporal_padding: int,
    ) -> torch.Tensor:
        """Apply temporal padding using zeros and cat."""
        if temporal_padding == 0:
            return x

        N, C_in, T_in, H_in, W_in = x.shape

        # Create zero padding tensors
        zero_pad = torch.zeros(
            N, C_in, temporal_padding, H_in, W_in, dtype=x.dtype, device=x.device
        )

        # Cat: [zeros, input, zeros]
        padded = torch.cat([zero_pad, x, zero_pad], dim=2)

        return padded

    def _get_padded_input_slice(
        self,
        padded_x: torch.Tensor,
        t_idx: int,
        k: int,
    ) -> torch.Tensor:
        """
        Get and quantize input slice at temporal position.

        Args:
            padded_x: Temporally padded input tensor (N, C_in, T_padded, H_in, W_in)
            t_idx: Temporal index to slice
            k: Kernel temporal position (for observer lookup)

        Returns:
            Quantized input slice (N, C_in, 1, H_in, W_in)
        """
        # Slice at temporal position
        input_slice = padded_x[:, :, t_idx : t_idx + 1, :, :]

        # Quantize input slice
        input_slice_q = self._fq(input_slice, self._input_slice_obs[k])

        return input_slice_q

    def _apply_conv2d_quantized(
        self,
        input_2d: torch.Tensor,
        weight_slice: torch.Tensor,
        bias: Optional[torch.Tensor],
        k: int,
        H_out: int,
        W_out: int,
        padding: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Apply quantized Conv2d operation.

        Args:
            input_2d: 2D input (N, C_in, H_in, W_in)
            weight_slice: 2D weight slice (C_out, C_in, kH, kW)
            bias: Optional bias tensor
            k: Kernel temporal position (for observer lookup)
            H_out: Output height
            W_out: Output width

        Returns:
            Quantized Conv2d output (N, C_out, H_out, W_out)
        """
        # Apply Conv2d
        conv_out = F.conv2d(
            input_2d,
            weight_slice,
            bias=None,  # Bias added after accumulation
            stride=(self.module.stride[1], self.module.stride[2]),
            padding=(padding[1], padding[2]),
            dilation=(self.module.dilation[1], self.module.dilation[2]),
            groups=self.module.groups,
        )

        # Quantize Conv2d output
        conv_out_q = self._fq(conv_out, self._conv2d_obs[k])

        return conv_out_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized Conv3d decomposition.

        Decomposes Conv3d into:
        1. Temporal padding (if needed)
        2. Slice input at each temporal kernel position
        3. Apply Conv2d to each slice
        4. Accumulate Conv2d results with quantization
        5. Add bias (if present)
        6. Stack temporal outputs

        Special case optimization:
        When kernel_size = input_size, stride = kernel_size,
        padding = 0, groups = 1, and dilation = 1 for all dimensions,
        the Conv3d operation reduces to matrix multiplication and is handled
        with a more efficient direct approach.

        All intermediate tensors are quantized to ensure integer-only computation.
        """
        N, C_in, T_in, H_in, W_in = x.shape
        C_out, C_in_weight, kT, kH, kW = self.module.weight.shape
        sT, sH, sW = self.module.stride
        dT, dH, dW = self.module.dilation
        groups = self.module.groups

        if C_in != C_in_weight:
            raise RuntimeError("Channels number mismatch")

        # Parse padding
        padding = self._parse_padding(self.module.padding)
        temporal_padding, h_padding, w_padding = padding

        # Quantize input activation
        x_q = self._fq(x, self.obs_act_in)

        # Get quantized weight
        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.obs_weight.fake_quant(w)

        # Check for special case:
        # kernel_size = input_size,
        # stride = kernel_size,
        # padding = 0,
        # no dilation
        # groups = 1
        is_special_case = (
            (kT, kH, kW) == (T_in, H_in, W_in)
            and (sT, sH, sW) == (kT, kH, kW)
            and (temporal_padding, h_padding, w_padding) == (0, 0, 0)
            and (dT, dH, dW) == (1, 1, 1)
            and groups == 1
        )

        # Special case: Conv3d reduces to matrix multiplication
        if is_special_case:
            # Reshape input: (N, C_in, T_in, H_in, W_in) -> (N, 1, 1, C_in*T_in*H_in*W_in)
            x_q = x_q.reshape(N, 1, 1, -1)

            # Reshape weights: (C_out, C_in, kT, kH, kW) -> (C_out, 1, 1, C_in*kT*kH*kW)
            w = w.reshape(C_out, 1, 1, -1)

            # Apply Conv2d directly
            if self.module.bias is not None:
                conv2d_result = F.conv2d(x_q, w, self.module.bias)
            else:
                conv2d_result = F.conv2d(x_q, w)

            # Reshape output: (1, C_out, N*C_in, 1) -> (N, C_out, 1, 1, 1)
            result = conv2d_result.reshape(N, C_out, 1, 1, 1)

            # Quantize output activation
            result_q = self._fq(result, self.obs_act_out)
            return result_q

        # Normal case: Conv3d is decomposed to multiple Conv2D and Add operations
        else:
            # Calculate output dimensions
            T_padded = T_in + 2 * temporal_padding
            T_out = (T_padded - dT * (kT - 1) - 1) // sT + 1
            H_out = (H_in + 2 * h_padding - dH * (kH - 1) - 1) // sH + 1
            W_out = (W_in + 2 * w_padding - dW * (kW - 1) - 1) // sW + 1

            # Create dynamic observers on first forward pass
            if not self._dynamic_obs_calibrated:
                if self._mode is Mode.QUANT:
                    raise RuntimeError(
                        "Trying to quantize without calibration. Need to calibrate first."
                    )
                self._create_dynamic_observers(kT, T_out)

            # Apply temporal padding
            padded_input = self._apply_temporal_padding(x_q, temporal_padding)

            # Temporal processing loop
            temporal_outputs = []
            for t_out in range(T_out):
                t_in = t_out * sT
                accumulator = None

                for k in range(kT):
                    t_idx = t_in + k * dT

                    # Handle dilation: mask out-of-bounds positions
                    if dT > 1 and t_idx >= T_padded:
                        # Skip this kernel position (out of bounds)
                        continue

                    # Get and quantize input slice
                    input_slice_q = self._get_padded_input_slice(padded_input, t_idx, k)

                    # Remove temporal dimension: (N, C_in, 1, H_in, W_in) → (N, C_in, H_in, W_in)
                    input_2d = input_slice_q.squeeze(2)

                    # Slice weight at temporal position k
                    weight_slice = w[:, :, k, :, :]  # (C_out, C_in, kH, kW)

                    # Apply quantized Conv2d
                    conv_out_q = self._apply_conv2d_quantized(
                        input_2d,
                        weight_slice,
                        self.module.bias,
                        k,
                        H_out,
                        W_out,
                        padding,
                    )

                    # Accumulate with quantization
                    if accumulator is None:
                        accumulator = conv_out_q
                    else:
                        accumulator = self._fq(
                            accumulator + conv_out_q, self._acc_obs[t_out]
                        )

                # Add bias if present
                if self.module.bias is not None:
                    bias_reshaped = self.module.bias.reshape(1, C_out, 1, 1)
                    accumulator = accumulator + bias_reshaped

                temporal_outputs.append(accumulator)

            # Stack temporal outputs
            unsqueezed = [t.unsqueeze(2) for t in temporal_outputs]  # type: ignore[union-attr]
            stacked = torch.cat(unsqueezed, dim=2)  # (N, C_out, T_out, H_out, W_out)

            # Quantize output activation
            stacked_q = self._fq(stacked, self.obs_act_out)

            return stacked_q

    def _all_observers(self):
        """Return all observers for this module."""
        # Static observers
        yield from (self.obs_weight, self.obs_act_in, self.obs_act_out)

        # Dynamic observers (if created)
        yield from self._input_slice_obs.values()
        yield from self._conv2d_obs.values()
        yield from self._acc_obs.values()

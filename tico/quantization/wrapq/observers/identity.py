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

"""
IdentityObserver: a "no-op" observer for FP-only modules.

Motivation
----------
Some layers should stay in full precision even when the rest of the model
is quantized.  Attaching an `IdentityObserver` satisfies the wrapper API
(`_update_stats()`, `compute_qparams()`, `fake_quant()`) without actually
performing any statistics gathering or fake-quantization.
"""
import torch

from tico.quantization.wrapq.observers.affine_base import AffineObserverBase


class IdentityObserver(AffineObserverBase):
    """
    Passthrough observer that NEVER alters the tensor.

    • `_update_stats()`   → does nothing
    • `compute_qparams()` → returns (1.0, 0) "dummy" q-params
    • `fake_quant()`      → returns `x` unchanged
    """

    def __init__(self, **kwargs):
        # Call parent so the usual fields (`dtype`, `qscheme`, …) exist,
        # but immediately disable any stateful behaviour.
        super().__init__(**kwargs)

        # Deactivate statistics collection permanently.
        self.enabled = False

        # Pre-cache sentinel q-params so wrapper code that blindly
        # accesses them won't crash.
        self._cached_scale = torch.tensor(1.0)
        self._cached_zp = torch.tensor(0, dtype=torch.int)

    def reset(self) -> None:  # (simple override – nothing to do)
        """No internal state to reset."""
        pass

    def _update_stats(self, x: torch.Tensor) -> None:
        """Skip statistic collection entirely."""
        return

    def compute_qparams(self):
        """
        Return the pre-cached (scale, zero_point) tuple.

        Keeping the signature identical to other observers allows uniform
        lifecycle management in wrapper code.
        """
        return self._cached_scale, self._cached_zp

    def fake_quant(self, x: torch.Tensor):
        """Identity mapping — leaves `x` in FP."""
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

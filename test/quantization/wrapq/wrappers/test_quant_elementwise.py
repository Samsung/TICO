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
Tests for the generic element-wise quant wrappers.

We verify for each wrapper:
  1.  It is discoverable via the registry & PTQWrapper factory
  2.  Mode transitions NO_QUANT → CALIB → QUANT
  3.  Quantized output differs from—but stays close to—FP
  4.  PTQConfig overrides propagate to observers
"""

import importlib.util
import inspect
import unittest
from functools import partial
from typing import Callable, List, Tuple, Type

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_elementwise import (
    QuantElementwise,
    QuantGELU,
    QuantGELUTanh,
    QuantReLU,
    QuantSigmoid,
    QuantTanh,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import lookup

ACTIVATIONS: List[
    Tuple[
        torch.nn.Module, Callable[[torch.Tensor], torch.Tensor], Type[QuantModuleBase]
    ]
] = [
    (torch.nn.Sigmoid(), torch.sigmoid, QuantSigmoid),
    (torch.nn.Tanh(), torch.tanh, QuantTanh),
    (torch.nn.ReLU(), torch.relu, QuantReLU),
    (torch.nn.GELU(), torch.nn.functional.gelu, QuantGELU),
]

if importlib.util.find_spec("transformers") is not None:
    import transformers

    ACTIVATIONS.append(
        (
            transformers.activations.GELUTanh(),
            partial(torch.nn.functional.gelu, approximate="tanh"),
            QuantGELUTanh,
        )
    )
else:
    print(f"\ntransformers not installed — skipping GELUTanh tests")


class TestElementwiseWrappers(unittest.TestCase):
    def _calibrate(self, qw, x):
        qw.enable_calibration()
        _ = qw(x)
        qw.freeze_qparams()

    # ------------------------------------------------------------------
    def test_registry_and_factory(self):
        for fp32_mod, _, quant_cls in ACTIVATIONS:
            self.assertIs(lookup(type(fp32_mod)), quant_cls)
            wrapper = PTQWrapper(fp32_mod)
            self.assertIsInstance(wrapper.wrapped, quant_cls)

    # ------------------------------------------------------------------
    def test_mode_and_forward_diff(self):
        torch.manual_seed(0)
        x = torch.randn(64, 8)

        for fp32_mod, func, _ in ACTIVATIONS:
            qw = PTQWrapper(fp32_mod)

            # default NO_QUANT
            self.assertIs(qw._mode, Mode.NO_QUANT)

            # CALIB -> QUANT
            self._calibrate(qw, x)
            self.assertIs(qw._mode, Mode.QUANT)

            with torch.no_grad():
                q_out = qw(x)
                fp_out = func(x)

            diff = (q_out - fp_out).abs().mean().item()
            self.assertGreater(diff, 0.0)
            self.assertLess(diff, 0.3)

    # ------------------------------------------------------------------
    def test_dtype_override(self):
        override = {
            "act_in": {"dtype": DType.uint(4)},
            "act_out": {"dtype": DType.uint(4)},
        }

        for fp32_mod, _, _ in ACTIVATIONS:
            cfg = PTQConfig(default_dtype=DType.uint(8), overrides=override)
            qw = PTQWrapper(fp32_mod, qcfg=cfg)
            wrapped = qw.wrapped  # QuantElementwise subclass

            self.assertEqual(wrapped.act_in_obs.dtype, DType.uint(4))
            self.assertEqual(wrapped.act_out_obs.dtype, DType.uint(4))

            # sanity: observers present & iterable
            obs_names = [
                name
                for name, _ in inspect.getmembers(
                    wrapped, lambda v: hasattr(v, "dtype")
                )
            ]
            self.assertIn("act_in_obs", obs_names)
            self.assertIn("act_out_obs", obs_names)

    def test_missing_FUNC_raises(self):
        with self.assertRaises(NotImplementedError):

            class BadQuant(QuantElementwise):
                """Forgot to define FUNC — should fail fast."""

                pass

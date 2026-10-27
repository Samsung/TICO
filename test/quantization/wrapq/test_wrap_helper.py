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

"""Tests for registered-wrapper failures and missing-wrapper traversal."""

import copy
import unittest
from typing import Optional

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
from tico.quantization.wrapq.wrappers import registry
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class UnsupportedModule(nn.Module):
    """A module with no wrapper registration."""

    def forward(self, value):
        """Pass through the input unchanged."""
        return value


class RejectedModule(UnsupportedModule):
    """An externally registered module whose variant is unsupported."""

    def __init__(self):
        """Track constructor attempts and the original rejection exception."""
        super().__init__()
        self.constructor_calls = 0
        self.rejection = NotImplementedError("registered wrapper rejects this variant")


class RejectingWrapper(QuantModuleBase):
    """Reject an unsupported variant during real wrapper construction."""

    def __init__(
        self,
        module: RejectedModule,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        module.constructor_calls += 1
        raise module.rejection

    def forward(self, value):
        """Provide the abstract wrapper interface."""
        return value

    def _all_observers(self):
        return ()


class RegisteredLinear(nn.Linear):
    """A distinct external module type with an explicitly registered wrapper."""

    def __init__(self):
        """Create a tiny Linear and track wrapper construction."""
        super().__init__(2, 2, bias=False)
        self.constructor_calls = 0


class ExternalQuantLinear(QuantLinear):
    """Use the real Linear wrapper while counting construction attempts."""

    def __init__(
        self,
        module: RegisteredLinear,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        module.constructor_calls += 1
        super().__init__(module, qcfg=qcfg, fp_name=fp_name)


class TestPTQWrapHelper(unittest.TestCase):
    """Exercise the public preparation path without mocking wrapper resolution."""

    def setUp(self):
        """Install temporary external registrations after core initialization."""
        registry.lookup(nn.Linear)
        self._wrappers = dict(registry._WRAPPERS)
        registry.register(RejectedModule)(RejectingWrapper)
        registry.register(RegisteredLinear)(ExternalQuantLinear)
        rng_state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng_state)
        torch.manual_seed(0)

    def tearDown(self):
        """Restore registrations for other tests in the same process."""
        registry._WRAPPERS.clear()
        registry._WRAPPERS.update(self._wrappers)

    def _assert_rejection(self, *, strict_wrap: bool, container: bool):
        module = RejectedModule()
        model = nn.Sequential(module) if container else module

        with self.assertRaises(NotImplementedError) as caught:
            prepare(model, PTQConfig(strict_wrap=strict_wrap))

        self.assertIs(caught.exception, module.rejection)
        self.assertEqual(module.constructor_calls, 1)
        self.assertFalse(hasattr(model, "tico_quantizer"))
        if isinstance(model, nn.Sequential):
            self.assertIs(model[0], module)

    def test_registered_root_rejection_is_preserved_in_strict_mode(self):
        """A registered root constructor's rejection retains its identity."""
        self._assert_rejection(strict_wrap=True, container=False)

    def test_registered_root_rejection_is_preserved_in_nonstrict_mode(self):
        """Non-strict wrapping cannot suppress a registered root's rejection."""
        self._assert_rejection(strict_wrap=False, container=False)

    def test_registered_child_rejection_is_preserved_in_strict_mode(self):
        """A container cannot relabel a registered child's constructor error."""
        self._assert_rejection(strict_wrap=True, container=True)

    def test_registered_child_rejection_is_preserved_in_nonstrict_mode(self):
        """A container cannot silently retry or skip a registered child."""
        self._assert_rejection(strict_wrap=False, container=True)

    def test_missing_registration_keeps_strict_error_and_cause(self):
        """Keep the existing missing-wrapper diagnostic at either boundary."""
        for container in (False, True):
            with self.subTest(container=container):
                module = UnsupportedModule()
                model = nn.Sequential(module) if container else module
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "PTQWrapHelper: no quantization wrapper for UnsupportedModule",
                ) as caught:
                    prepare(model, PTQConfig())
                cause = caught.exception.__cause__
                self.assertIsInstance(cause, NotImplementedError)
                self.assertEqual(str(cause), "No quant wrapper for UnsupportedModule")

    def test_missing_registration_keeps_nonstrict_module_identity(self):
        """Unknown modules still pass through non-strict preparation."""
        for container in (False, True):
            with self.subTest(container=container):
                module = UnsupportedModule()
                model = nn.Sequential(module) if container else module
                prepared = prepare(model, PTQConfig(strict_wrap=False))
                self.assertIs(prepared, model)
                self.assertTrue(hasattr(prepared, "tico_quantizer"))
                if container:
                    self.assertIs(prepared[0], module)

    def test_unregistered_subclass_does_not_inherit_parent_registration(self):
        """Registry lookup retains its exact-type policy without MRO fallback."""

        class UnregisteredLinear(RegisteredLinear):
            """Inherit behavior without adding a wrapper registration."""

        module = UnregisteredLinear()
        prepared = prepare(module, PTQConfig(strict_wrap=False))
        self.assertIs(prepared, module)
        self.assertEqual(module.constructor_calls, 0)

    def test_generic_recursion_keeps_scoped_config_and_original_name(self):
        """Traverse unknown parents while preserving external wrapper metadata."""
        module = RegisteredLinear()
        model = nn.Module()
        model.branch = nn.Sequential(module)
        config = PTQConfig(
            strict_wrap=False,
            overrides={"branch.0.act_in": {"dtype": DType.uint(4)}},
            model_args={"external": "metadata"},
        )

        wrapped = PTQWrapHelper(strict_wrap=False).wrap_supported(
            model, config, fp_name="original"
        )

        self.assertIs(wrapped, model)
        wrapper = model.branch[0]
        assert isinstance(wrapper, PTQWrapper)
        quant_linear = wrapper.wrapped
        assert isinstance(quant_linear, ExternalQuantLinear)
        self.assertIs(quant_linear.module, module)
        self.assertEqual(module.constructor_calls, 1)
        self.assertEqual(quant_linear.fp_name, "original.branch.0")
        self.assertIs(quant_linear.qcfg.activation, config.activation)
        self.assertIs(quant_linear.qcfg.model_args, config.model_args)
        self.assertEqual(quant_linear.obs_act_in.dtype, DType.uint(4))

    def test_successful_prepare_preserves_external_wrapper_and_quantization(self):
        """Prepare and convert a registered module with the direct-wrapper result."""
        module = RegisteredLinear()
        config = PTQConfig(overrides={"act_in": {"dtype": DType.uint(4)}})
        reference = PTQWrapper(copy.deepcopy(module), qcfg=config)
        reference.enable_calibration()
        sample = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

        prepared = prepare(module, config)
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, ExternalQuantLinear)
        self.assertIs(prepared.wrapped.qcfg, config)
        self.assertIs(prepared.wrapped.module, module)
        self.assertEqual(module.constructor_calls, 1)
        self.assertIs(prepared.wrapped._mode, Mode.CALIB)
        torch.testing.assert_close(prepared(sample), reference(sample), rtol=0, atol=0)

        quantized = convert(prepared)
        reference.freeze_qparams()
        self.assertIs(quantized.wrapped._mode, Mode.QUANT)
        torch.testing.assert_close(quantized(sample), reference(sample), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

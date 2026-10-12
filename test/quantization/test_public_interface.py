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

"""Lifecycle contracts of the public ``prepare`` / ``convert`` entry points."""

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import tico.quantization.quantizer_registry as registry_mod

import torch

from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.quantizer import GPTQQuantizer
from tico.quantization.config.base import BaseConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


@dataclass
class _CopyableConfig(BaseConfig):
    @property
    def name(self) -> str:
        return "_public_interface_copyable"


@dataclass
class _InplaceOnlyConfig(BaseConfig):
    @property
    def name(self) -> str:
        return "_public_interface_inplace_only"


class _MarkingQuantizer(BaseQuantizer):
    """Quantizer that tags the model it converts (no model-bound state)."""

    def prepare(self, model, args=None, kwargs=None):
        return model

    def convert(self, model):
        model.converted = True
        return model


class _InplaceOnlyQuantizer(_MarkingQuantizer):
    """Quantizer that declares it cannot convert a deep copy."""

    requires_inplace_convert = True


class PublicInterfaceConvertTest(unittest.TestCase):
    def setUp(self):
        self._registry_patch = patch.dict(registry_mod._REGISTRY, clear=False)
        self._registry_patch.start()
        register_quantizer(_CopyableConfig)(_MarkingQuantizer)
        register_quantizer(_InplaceOnlyConfig)(_InplaceOnlyQuantizer)

    def tearDown(self):
        self._registry_patch.stop()

    def test_default_quantizer_allows_copy_on_convert(self):
        """BaseQuantizer subclasses allow inplace=False unless they opt out."""
        self.assertFalse(BaseQuantizer.requires_inplace_convert)
        model = prepare(torch.nn.Linear(2, 2), _CopyableConfig())
        converted = convert(model, inplace=False)
        self.assertIsNot(converted, model)
        self.assertTrue(converted.converted)
        self.assertFalse(hasattr(model, "converted"))

    def test_inplace_only_quantizer_rejects_copy_on_convert(self):
        """A quantizer with requires_inplace_convert rejects inplace=False."""
        model = prepare(torch.nn.Linear(2, 2), _InplaceOnlyConfig())
        with self.assertRaisesRegex(RuntimeError, "_InplaceOnlyQuantizer.*in-place"):
            convert(model, inplace=False)
        # The quantizer attribute is consumed before the failure, exactly as
        # for the in-place path, so the caller cannot accidentally convert twice.
        self.assertFalse(hasattr(model, "tico_quantizer"))

    def test_builtin_gptq_quantizer_requires_inplace_convert(self):
        """The built-in GPTQ quantizer keeps its in-place-only convert contract."""
        self.assertTrue(GPTQQuantizer.requires_inplace_convert)


if __name__ == "__main__":
    unittest.main()

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
Unit-tests for the lightweight wrapper registry.

What is verified
----------------
1. `register` decorator adds the mapping and lookup returns it.
2. `try_register` succeeds when the target class exists.
3. `try_register` is a NO-OP when the module / class is absent.
4. Variant support:
   - multiple variants can coexist for a single fp class
   - lookup resolves exact variant
   - lookup falls back to "common" when exact variant is missing
"""

import sys
import types
import unittest

import torch.nn as nn

from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import lookup, register, try_register


# Dummy fp32 & quant modules for tests
class DummyFP(nn.Linear):  # inherit nn.Module for type-compat
    def __init__(self):
        super().__init__(4, 4)


class DummyQuant(QuantModuleBase):
    def forward(self, x):
        return x

    def _all_observers(self):
        return ()


@register(DummyFP, variant="prefill")
class QPrefill(DummyQuant):
    ...


@register(DummyFP, variant="decode")
class QDecode(DummyQuant):
    ...


@register(DummyFP)
class QCommon(DummyQuant):
    ...


class TestRegistry(unittest.TestCase):

    # 1) plain @register ------------------------------------------------
    def test_register_and_lookup(self):
        self.assertIs(lookup(DummyFP), QCommon)

    # 2) try_register when path exists ---------------------------------
    def test_try_register_success(self):
        # create a throw-away module with a class inside
        mod = types.ModuleType("tmp_mod")

        class TmpFP(nn.Linear):  # noqa: D401
            def __init__(self):
                super().__init__(2, 2)

        mod.TmpFP = TmpFP  # type: ignore[attr-defined]
        sys.modules["tmp_mod"] = mod  # inject into sys.modules

        @try_register("tmp_mod.TmpFP")
        class TmpQuant(DummyQuant):
            ...

        self.assertIs(lookup(TmpFP), TmpQuant)

        del sys.modules["tmp_mod"]  # clean up

    # 3) try_register when target missing --------------------------------
    def test_try_register_graceful_skip(self):
        path = "nonexistent.module.Foo"

        @try_register(path)
        class SkipQuant(DummyQuant):
            ...

        # lookup should fail (module missing) without raising
        self.assertIsNone(lookup(type("Fake", (), {})))

    # 4-1) Variant: exact match across multiple variants -------------------
    def test_variant_exact_match(self):
        self.assertIs(lookup(DummyFP, variant="prefill"), QPrefill)
        self.assertIs(lookup(DummyFP, variant="decode"), QDecode)
        self.assertIs(lookup(DummyFP, variant="common"), QCommon)

    # 4-2) Variant: fallback to common when requested variant missing -------
    def test_variant_fallback_to_common(self):
        # No decode/prefill registered: should still resolve to common
        self.assertIs(lookup(DummyFP, variant="dummy_1"), QCommon)
        self.assertIs(lookup(DummyFP, variant="dummy_2"), QCommon)

    # 4-3) try_register also supports variants -----------------------------
    def test_try_register_with_variant(self):
        mod = types.ModuleType("tmp_mod2")

        class TmpFP2(nn.Linear):
            def __init__(self):
                super().__init__(3, 3)

        mod.TmpFP2 = TmpFP2  # type: ignore[attr-defined]
        sys.modules["tmp_mod2"] = mod

        @try_register("tmp_mod2.TmpFP2", variant="decode")
        class TmpQuantDecode(DummyQuant):
            ...

        @try_register("tmp_mod2.TmpFP2", variant="common")
        class TmpQuantCommon(DummyQuant):
            ...

        self.assertIs(lookup(TmpFP2, variant="decode"), TmpQuantDecode)
        # asking for prefill should fall back to common if decode-only doesn't match
        self.assertIs(lookup(TmpFP2, variant="prefill"), TmpQuantCommon)

        del sys.modules["tmp_mod2"]

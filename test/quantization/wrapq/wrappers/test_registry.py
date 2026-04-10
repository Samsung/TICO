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

"""
Unit tests for the simplified wrapper registry.

What is verified
----------------
1. `register` adds a direct fp-module -> quant-wrapper mapping.
2. `try_register` succeeds when the target class exists.
3. `try_register` is a no-op when the target module or class is absent.
4. Duplicate registration is rejected for both `register` and `try_register`.
"""

import sys
import types
import unittest

import tico.quantization.wrapq.wrappers.registry as registry

import torch.nn as nn

from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class DummyQuant(QuantModuleBase):
    def forward(self, x):
        return x

    def _all_observers(self):
        return ()


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self._wrappers_backup = dict(registry._WRAPPERS)
        self._import_once_backup = registry._IMPORT_ONCE

        registry._WRAPPERS.clear()
        registry._IMPORT_ONCE = True

    def tearDown(self):
        registry._WRAPPERS.clear()
        registry._WRAPPERS.update(self._wrappers_backup)
        registry._IMPORT_ONCE = self._import_once_backup

        for mod_name in ("tmp_mod", "tmp_mod2", "tmp_mod3"):
            sys.modules.pop(mod_name, None)

    def test_register_and_lookup(self):
        class DummyFP(nn.Linear):
            def __init__(self):
                super().__init__(4, 4)

        @registry.register(DummyFP)
        class DummyQuantImpl(DummyQuant):
            pass

        self.assertIs(registry.lookup(DummyFP), DummyQuantImpl)

    def test_try_register_success(self):
        mod = types.ModuleType("tmp_mod")

        class TmpFP(nn.Linear):
            def __init__(self):
                super().__init__(2, 2)

        mod.TmpFP = TmpFP  # type: ignore[attr-defined]
        sys.modules["tmp_mod"] = mod

        @registry.try_register("tmp_mod.TmpFP")
        class TmpQuant(DummyQuant):
            pass

        self.assertIs(registry.lookup(TmpFP), TmpQuant)

    def test_try_register_graceful_skip(self):
        @registry.try_register("nonexistent.module.Foo", "tmp_mod.DoesNotExist")
        class SkipQuant(DummyQuant):
            pass

        class UnregisteredFP(nn.Linear):
            def __init__(self):
                super().__init__(3, 3)

        self.assertIsNone(registry.lookup(UnregisteredFP))

    def test_register_duplicate_raises(self):
        class DummyFP(nn.Linear):
            def __init__(self):
                super().__init__(4, 4)

        @registry.register(DummyFP)
        class FirstQuant(DummyQuant):
            pass

        with self.assertRaises(ValueError):

            @registry.register(DummyFP)
            class SecondQuant(DummyQuant):
                pass

        self.assertIs(registry.lookup(DummyFP), FirstQuant)

    def test_try_register_duplicate_raises(self):
        mod = types.ModuleType("tmp_mod2")

        class TmpFP2(nn.Linear):
            def __init__(self):
                super().__init__(5, 5)

        mod.TmpFP2 = TmpFP2  # type: ignore[attr-defined]
        sys.modules["tmp_mod2"] = mod

        @registry.try_register("tmp_mod2.TmpFP2")
        class FirstQuant(DummyQuant):
            pass

        with self.assertRaises(ValueError):

            @registry.try_register("tmp_mod2.TmpFP2")
            class SecondQuant(DummyQuant):
                pass

        self.assertIs(registry.lookup(TmpFP2), FirstQuant)

    def test_try_register_multiple_paths_registers_existing_target(self):
        mod = types.ModuleType("tmp_mod3")

        class TmpFP3(nn.Linear):
            def __init__(self):
                super().__init__(6, 6)

        mod.TmpFP3 = TmpFP3  # type: ignore[attr-defined]
        sys.modules["tmp_mod3"] = mod

        @registry.try_register(
            "nonexistent.module.Foo",
            "tmp_mod3.MissingClass",
            "tmp_mod3.TmpFP3",
        )
        class TmpQuant3(DummyQuant):
            pass

        self.assertIs(registry.lookup(TmpFP3), TmpQuant3)


if __name__ == "__main__":
    unittest.main()

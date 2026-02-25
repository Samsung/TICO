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

import unittest

import torch
from tico.utils.passes import PassBase


class PassTest(unittest.TestCase):
    """
    How to use?

    class DummyTest(PassTest):
        def test_pass(self):
            self.setup(DummyNet())
            self.run_pass(DummyPass())
    """

    initialized: bool = False

    def setup(self, mod: torch.nn.Module):
        assert hasattr(mod, "get_example_inputs")
        self.forward_args, self.forward_kwargs = mod.get_example_inputs()  # type: ignore[operator]

        with torch.no_grad():
            self.ep = torch.export.export(
                mod.eval(), self.forward_args, self.forward_kwargs
            )

        self.initialized = True

    def run_pass(self, test_pass: PassBase):  # type: ignore[override]
        assert self.initialized, "setup() must be called first"

        test_pass.call(self.ep)

    def exported_program(self):
        assert self.initialized, "setup() must be called first"
        return self.ep


class PassValueTest(PassTest):
    """
    Class for pass value test. This compares execution results before/after pass is applied

    How to use?

    class DummyTest(PassValueTest):
        def test_pass(self):
            self.setup(DummyNet())
            self.run_value_test(DummyPass())
    """

    def run_value_test(self):
        pass


class SinglePassValueTest(PassValueTest):
    def run_value_test(self, test_pass: PassBase):  # type: ignore[override]
        assert self.initialized, "setup() must be called first"

        # inference before pass
        ret_before = self.ep.module()(*self.forward_args, **self.forward_kwargs)
        test_pass.call(self.ep)

        # inference after pass
        ret_after = self.ep.module()(*self.forward_args, **self.forward_kwargs)

        self.assertEqual(ret_before.shape, ret_after.shape)
        self.assertTrue(torch.allclose(ret_before, ret_after, atol=1e-06))


class MultiPassValueTest(PassValueTest):
    def run_value_test(self, test_passes: list):  # type: ignore[override]
        assert self.initialized, "setup() must be called first"

        # inference before pass
        ret_before = self.ep.module()(*self.forward_args, **self.forward_kwargs)
        for p in test_passes:
            p.call(self.ep)

        # inference after pass
        ret_after = self.ep.module()(*self.forward_args, **self.forward_kwargs)

        self.assertEqual(ret_before.shape, ret_after.shape)
        self.assertTrue(torch.allclose(ret_before, ret_after, atol=1e-06))

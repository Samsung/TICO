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

"""Guard tests: the default Circle execution path must not touch ONE or onert.

These tests do not replace any result. They install detectors that make the
test fail if the default path tries to import ``onert`` or ``cffi``, spawn a
subprocess (``onecc``, ``circle2circle``, ...), or load a native library, and
then run real conversions and executions through the public APIs.
"""

from __future__ import annotations

import ctypes
import importlib.abc
import importlib.machinery
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import tico
import torch
from tico.quantization.evaluation.backend import BACKEND
from tico.quantization.evaluation.evaluate import evaluate
from tico.utils.model import CircleModel

from test.modules.op.add import SimpleAdd
from test.modules.op.conv2d import SimpleConv
from test.modules.op.linear import SimpleLinear

BLOCKED_MODULES = ("onert", "cffi")


class _BlockExternalRuntimeImports(importlib.abc.MetaPathFinder):
    """Fail any import of an external runtime package."""

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[override]
        root = fullname.split(".")[0]
        if root in BLOCKED_MODULES:
            raise AssertionError(
                f"The default Circle execution path imported {fullname!r}."
            )
        return None


def _forbid_subprocess(*args, **kwargs):
    raise AssertionError(
        f"The default Circle execution path spawned a subprocess: {args} {kwargs}"
    )


def _forbid_native_library(*args, **kwargs):
    raise AssertionError(
        f"The default Circle execution path loaded a native library: {args}"
    )


class ExternalRuntimeGuard:
    """Context manager installing every detector."""

    def __enter__(self):
        self._finder = _BlockExternalRuntimeImports()
        self._removed = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if name.split(".")[0] in BLOCKED_MODULES
        }
        sys.meta_path.insert(0, self._finder)
        self._patches = [
            patch.object(subprocess, "run", _forbid_subprocess),
            patch.object(subprocess, "Popen", _forbid_subprocess),
            patch.object(subprocess, "check_output", _forbid_subprocess),
            patch.object(subprocess, "check_call", _forbid_subprocess),
            patch.object(subprocess, "call", _forbid_subprocess),
            patch.object(ctypes, "CDLL", _forbid_native_library),
        ]
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, *exc_info):
        for item in reversed(self._patches):
            item.stop()
        sys.meta_path.remove(self._finder)
        sys.modules.update(self._removed)
        return False


class DefaultRuntimeIndependenceTest(unittest.TestCase):
    def test_default_runtime_name_is_reference(self):
        self.assertEqual(CircleModel(b"").runtime, "reference")

    def test_conversion_and_execution_do_not_use_external_runtimes(self):
        """Convert, save, reload, and execute through CircleModel under the guard."""

        with ExternalRuntimeGuard():
            for module_class in (SimpleAdd, SimpleLinear, SimpleConv):
                module = module_class().eval()
                args, kwargs = module.get_example_inputs()
                circle_model = tico.convert(module, args, kwargs)
                reloaded = CircleModel(circle_model.circle_binary)
                result = reloaded(*args, **kwargs)
                with torch.no_grad():
                    expected = module(*args, **kwargs).numpy()
                np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_quantized_evaluation_does_not_use_external_toolchain(self):
        """evaluate(BACKEND.CIRCLE) must not call onecc or import onert."""

        from tico.quantization import convert, prepare
        from tico.quantization.config.ptq import PTQConfig

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        prepared = prepare(model, PTQConfig(), inplace=True)
        with torch.inference_mode():
            prepared(torch.randn(2, 4))
        quantized = convert(prepared, inplace=True).eval()
        sample = torch.randn(2, 4)
        circle_model = tico.convert(quantized, (sample,))

        with ExternalRuntimeGuard():
            results = evaluate(
                quantized, circle_model, BACKEND.CIRCLE, [sample], mode="return"
            )
        assert results is not None
        self.assertIn("peir", results)

    def test_guard_detects_external_runtime_usage(self):
        """The guard itself must fail when an external runtime is requested."""

        from tico.interpreter.backends import run_circle

        module = SimpleAdd().eval()
        args, kwargs = module.get_example_inputs()
        circle_model = tico.convert(module, args, kwargs)
        with ExternalRuntimeGuard():
            with self.assertRaisesRegex(AssertionError, "imported 'onert'"):
                run_circle(circle_model.circle_binary, list(args), runtime="onert")
            with self.assertRaisesRegex(AssertionError, "spawned a subprocess"):
                subprocess.run(["onecc", "--version"])


if __name__ == "__main__":
    unittest.main()

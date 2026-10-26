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

"""Characterize the real staged Circle conversion paths without model downloads."""

import contextlib
import inspect
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from tico.circle import CircleDocument
from tico.circle.graph import as_indices
from tico.circle.runtime import CircleReferenceRuntime
from tico.quantization.recipes.export import gemma4, llama, qwen3_vl

_EXPORTERS = (llama, qwen3_vl, gemma4)
_CONVERSION_WARNING = "stage export warning"


class _ContextStage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Identity()
        self.eval_grad_enabled: bool | None = None

    def eval(self):
        # Observe the helper's context before tico.convert enters its own no_grad.
        self.eval_grad_enabled = torch.is_grad_enabled()
        warnings.warn(_CONVERSION_WARNING, UserWarning)
        return super().eval()

    def forward(self, x):
        return self.child(x) + (100 if self.training else 2)


class _StrictStage(torch.nn.Module):
    def forward(self, x, *, offset):
        # Strict export uses Dynamo; non-strict export must keep the other branch.
        strict_offset = 10 if torch.compiler.is_dynamo_compiling() else 0
        return x + offset + strict_offset


class _FailingStage(_ContextStage):
    def forward(self, x):
        raise RuntimeError("stage conversion failed")


class TestExportCircleStage(unittest.TestCase):
    def _load_and_run(self, path, inputs, expected):
        document = CircleDocument.load(path)
        self.assertEqual(document.source, path)
        document.verify()
        outputs = CircleReferenceRuntime(document).run(inputs).outputs
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].dtype, np.float32)
        # Addition of these small integers is exact in float32.
        np.testing.assert_array_equal(outputs[0], expected)
        return document

    def test_existing_signature_and_defaults(self):
        for exporter in _EXPORTERS:
            with self.subTest(exporter=exporter.__name__):
                parameters = inspect.signature(exporter._convert_and_save).parameters
                self.assertEqual(
                    tuple(parameters),
                    (
                        "module",
                        "example_inputs",
                        "save_path",
                        "kwargs",
                        "dynamic_shapes",
                        "strict",
                    ),
                )
                for name in ("module", "example_inputs", "save_path"):
                    self.assertEqual(
                        parameters[name].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD
                    )
                    self.assertIs(parameters[name].default, inspect.Parameter.empty)
                for name, default in (
                    ("kwargs", None),
                    ("dynamic_shapes", None),
                    ("strict", False),
                ):
                    self.assertEqual(
                        parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
                    )
                    self.assertIs(parameters[name].default, default)

    def test_default_export_preserves_context_and_input(self):
        for exporter in _EXPORTERS:
            for grad_enabled in (True, False):
                with self.subTest(
                    exporter=exporter.__name__, grad_enabled=grad_enabled
                ):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "stage.circle"
                        model = _ContextStage()
                        x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
                        original = x.clone()
                        log = io.StringIO()
                        with (
                            torch.set_grad_enabled(grad_enabled),
                            warnings.catch_warnings(record=True) as caught,
                        ):
                            warnings.simplefilter("always")
                            filters = list(warnings.filters)
                            with contextlib.redirect_stdout(log):
                                self.assertIsNone(
                                    exporter._convert_and_save(model, (x,), path)
                                )
                            self.assertEqual(torch.is_grad_enabled(), grad_enabled)
                            self.assertEqual(warnings.filters, filters)
                            self.assertNotIn(
                                _CONVERSION_WARNING, [str(w.message) for w in caught]
                            )
                            warnings.warn("after stage export", UserWarning)
                            self.assertEqual(
                                str(caught[-1].message), "after stage export"
                            )
                        self.assertFalse(model.training)
                        self.assertFalse(model.child.training)
                        self.assertIs(model.eval_grad_enabled, False)
                        self.assertTrue(torch.equal(x, original))
                        self.assertEqual(list(Path(directory).iterdir()), [path])
                        self.assertEqual(
                            log.getvalue(), f"Saving {path.name} to {path.resolve()}\n"
                        )
                        self._load_and_run(path, [x.numpy()], original.numpy() + 2)

    def test_kwargs_dynamic_shapes_and_strict_are_forwarded(self):
        for exporter in _EXPORTERS:
            for strict in (False, True):
                with self.subTest(exporter=exporter.__name__, strict=strict):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "dynamic.circle"
                        batch = torch.export.Dim("batch", min=1, max=4)
                        self.assertIsNone(
                            exporter._convert_and_save(
                                _StrictStage(),
                                (torch.zeros(2, 3),),
                                path,
                                kwargs={"offset": torch.ones(2, 3)},
                                dynamic_shapes={"x": {0: batch}, "offset": {0: batch}},
                                strict=strict,
                            )
                        )
                        # A different batch size requires the serialized dynamic ABI.
                        x = np.array(
                            [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32
                        )
                        offset = np.full((3, 3), 2, dtype=np.float32)
                        document = self._load_and_run(
                            path, [x, offset], x + offset + (10 if strict else 0)
                        )
                        graph = document.graph()
                        self.assertEqual(len(graph.inputs), 2)
                        for index in (*graph.inputs, *graph.outputs):
                            tensor = graph.subgraph.tensors[index]
                            self.assertEqual(as_indices(tensor.shapeSignature), [-1, 3])

    def test_omitted_strict_uses_non_strict_export(self):
        for exporter in _EXPORTERS:
            with self.subTest(exporter=exporter.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "default.circle"
                    x = torch.tensor([1.0, 2.0])
                    offset = torch.tensor([3.0, 4.0])
                    exporter._convert_and_save(
                        _StrictStage(), (x,), path, kwargs={"offset": offset}
                    )
                    self._load_and_run(path, [x.numpy(), offset.numpy()], [4.0, 6.0])

    def test_conversion_failure_propagates_and_restores_context(self):
        for exporter in _EXPORTERS:
            with self.subTest(exporter=exporter.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "failed.circle"
                    model = _FailingStage()
                    with (
                        torch.enable_grad(),
                        warnings.catch_warnings(record=True) as caught,
                    ):
                        warnings.simplefilter("always")
                        filters = list(warnings.filters)
                        with self.assertRaisesRegex(
                            RuntimeError, "stage conversion failed"
                        ):
                            exporter._convert_and_save(model, (torch.ones(2),), path)
                        self.assertTrue(torch.is_grad_enabled())
                        self.assertEqual(warnings.filters, filters)
                        self.assertNotIn(
                            _CONVERSION_WARNING, [str(w.message) for w in caught]
                        )
                    self.assertFalse(model.training)
                    self.assertIs(model.eval_grad_enabled, False)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_save_failure_propagates_without_creating_parent(self):
        for exporter in _EXPORTERS:
            with self.subTest(exporter=exporter.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "missing" / "stage.circle"
                    model = _ContextStage()
                    with torch.enable_grad(), warnings.catch_warnings():
                        warnings.simplefilter("error", UserWarning)
                        filters = list(warnings.filters)
                        with self.assertRaises(FileNotFoundError) as raised:
                            exporter._convert_and_save(model, (torch.ones(2),), path)
                        self.assertEqual(raised.exception.filename, str(path))
                        self.assertTrue(torch.is_grad_enabled())
                        self.assertEqual(warnings.filters, filters)
                    self.assertFalse(model.training)
                    self.assertIs(model.eval_grad_enabled, False)
                    self.assertFalse(path.parent.exists())

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

"""File conversion preserves the bytes API's graph and numerical behavior."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import tico
import tico.pt2_to_circle
import torch
from tico.circle import io
from tico.circle._schema import accessor_api_type
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleIOError
from tico.circle.export import optimize_document_for_export
from tico.config import CompileConfigV1
from tico.serialize import circle_binary, circle_serializer
from tico.utils.model import CircleModel

conversion = importlib.import_module("tico.utils.convert")


class _Affine(torch.nn.Module):
    bias: torch.Tensor

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8
        )
        self.register_buffer("bias", torch.tensor([0.25, -0.5, 1.0]))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias) * 1.0


class _KeywordAdd(torch.nn.Module):
    def forward(self, x, *, shift):
        return x + shift


class ConvertToFileTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        return _Affine().eval(), (torch.arange(8, dtype=torch.float32).reshape(2, 4),)

    @staticmethod
    def _convert(entry, module, args, path, config, *, to_file):
        if entry == "module":
            if to_file:
                return tico.convert_to_file(module, args, path, config=config)
            return tico.convert(module, args, config=config)
        ep = torch.export.export(module, args)
        if entry == "exported_program":
            if to_file:
                return tico.convert_from_exported_program_to_file(
                    ep, str(path), config=config
                )
            return tico.convert_from_exported_program(ep, config=config)
        pt2 = path.parent / "model.pt2"
        pt2.parent.mkdir(parents=True, exist_ok=True)
        torch.export.save(ep, pt2)
        if to_file:
            return tico.convert_from_pt2_to_file(pt2, path, config=config)
        return tico.convert_from_pt2(pt2, config=config)

    def _check_parity(self, *, appended):
        for entry in ("module", "exported_program", "pt2"):
            for optimize in (False, True):
                with self.subTest(entry=entry, optimize=optimize, appended=appended):
                    with tempfile.TemporaryDirectory() as temporary:
                        path = Path(temporary) / "nested" / "model.circle"
                        module, args = self._inputs()
                        original = {
                            name: tensor.clone()
                            for name, tensor in module.state_dict().items()
                        }
                        config = CompileConfigV1(circle_optimize=optimize)
                        expected = self._convert(
                            entry, module, args, path, config, to_file=False
                        )
                        self.assertIsInstance(expected.circle_binary, bytes)
                        budget = (
                            0 if appended else conversion._FILE_OUTPUT_INLINE_BUDGET
                        )
                        with patch.object(
                            conversion, "_FILE_OUTPUT_INLINE_BUDGET", budget
                        ):
                            result = self._convert(
                                entry, module, args, path, config, to_file=True
                            )
                        self.assertIsNone(result)
                        actual = path.read_bytes()
                        document = CircleDocument.from_bytes(actual)
                        document.verify()
                        if appended:
                            root = accessor_api_type("Model").GetRootAsModel(actual, 0)
                            self.assertTrue(
                                any(
                                    root.Buffers(index).Offset() > 0
                                    for index in range(root.BuffersLength())
                                )
                            )
                            self.assertEqual(
                                document.to_bytes(), expected.circle_binary
                            )
                        else:
                            self.assertEqual(actual, expected.circle_binary)
                        with torch.no_grad():
                            reference = module(*args).numpy()
                        np.testing.assert_allclose(
                            CircleModel.load(str(path))(*args),
                            reference,
                            rtol=1e-6,
                            atol=1e-6,
                        )
                        for name, tensor in module.state_dict().items():
                            torch.testing.assert_close(
                                tensor, original[name], rtol=0, atol=0
                            )

    def test_inline_file_matches_bytes_apis_with_and_without_o1(self):
        self._check_parity(appended=False)

    def test_appended_file_matches_bytes_apis_with_and_without_o1(self):
        self._check_parity(appended=True)

    def test_file_conversion_never_materializes_complete_intermediate_binaries(self):
        module, args = self._inputs()
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(
                patch.object(conversion, "_FILE_OUTPUT_INLINE_BUDGET", 0)
            )
            forbidden = (
                (conversion, "build_circle_bytes"),
                (circle_serializer, "build_circle_bytes"),
                (circle_serializer, "serialize_circle_model"),
                (circle_binary, "serialize_circle_model"),
                (io, "serialize_circle_model"),
                (circle_binary.CircleBinaryLayout, "to_bytes"),
                (CircleDocument, "from_bytes"),
                (CircleDocument, "to_bytes"),
            )
            for owner, name in forbidden:
                stack.enter_context(
                    patch.object(
                        owner,
                        name,
                        side_effect=AssertionError(f"File conversion called {name}"),
                    )
                )
            for entry in ("module", "exported_program", "pt2"):
                with self.subTest(entry=entry):
                    path = Path(temporary) / f"{entry}.circle"
                    self.assertIsNone(
                        self._convert(
                            entry,
                            module,
                            args,
                            path,
                            CompileConfigV1(),
                            to_file=True,
                        )
                    )
                    self.assertGreater(path.stat().st_size, 0)

    def test_o1_is_default_and_can_be_disabled(self):
        module, args = self._inputs()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.circle"
            with patch(
                "tico.circle.export.optimize_document_for_export",
                wraps=optimize_document_for_export,
            ) as optimize:
                tico.convert_to_file(module, args, path)
                optimize.assert_called_once()
            optimized = CircleDocument.load(path)
            with patch(
                "tico.circle.export.optimize_document_for_export",
                side_effect=AssertionError("O1 must be disabled"),
            ):
                tico.convert_to_file(
                    module, args, path, config=CompileConfigV1(circle_optimize=False)
                )
            unoptimized = CircleDocument.load(path)
            self.assertLess(
                len(optimized.model.subgraphs[0].operators),
                len(unoptimized.model.subgraphs[0].operators),
            )

    def test_keyword_inputs_dynamic_shapes_and_non_strict_export(self):
        module = _KeywordAdd().eval()
        x = torch.ones(2, 3)
        shift = torch.arange(3, dtype=torch.float32)
        batch = torch.export.Dim("batch", min=1, max=6)
        shapes = {"x": {0: batch}, "shift": None}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dynamic.circle"
            with patch.object(
                conversion, "export", wraps=torch.export.export
            ) as export:
                tico.convert_to_file(
                    module,
                    (x,),
                    path,
                    kwargs={"shift": shift},
                    dynamic_shapes=shapes,
                    strict=False,
                )
                self.assertIs(export.call_args.kwargs["strict"], False)
                self.assertEqual(export.call_args.kwargs["dynamic_shapes"], shapes)
            document = CircleDocument.load(path)
            document.verify()
            graph = document.model.subgraphs[0]
            tensor = graph.tensors[int(graph.inputs[0])]
            self.assertEqual(list(tensor.shapeSignature), [-1, 3])
            larger = torch.arange(15, dtype=torch.float32).reshape(5, 3)
            np.testing.assert_array_equal(
                CircleModel.load(str(path))(larger, shift=shift),
                module(larger, shift=shift).numpy(),
            )

    def test_pt2_cli_uses_file_conversion_and_preserves_numerical_output(self):
        module, args = self._inputs()
        config = CompileConfigV1(circle_optimize=False)
        with tempfile.TemporaryDirectory() as temporary:
            pt2 = Path(temporary) / "model.pt2"
            output = Path(temporary) / "model.circle"
            torch.export.save(torch.export.export(module, args), pt2)
            with (
                patch(
                    "tico.pt2_to_circle.convert_from_pt2_to_file",
                    wraps=tico.convert_from_pt2_to_file,
                ) as convert_file,
                patch.object(
                    conversion,
                    "build_circle_bytes",
                    side_effect=AssertionError("CLI must use file conversion"),
                ),
                patch.object(conversion, "_FILE_OUTPUT_INLINE_BUDGET", 0),
            ):
                tico.pt2_to_circle.convert(str(pt2), str(output), config=config)
                convert_file.assert_called_once_with(
                    str(pt2), str(output), config=config
                )
            CircleDocument.load(output).verify()
            with torch.no_grad():
                expected = module(*args).numpy()
            np.testing.assert_allclose(
                CircleModel.load(str(output))(*args), expected, rtol=1e-6, atol=1e-6
            )

    def test_existing_destination_survives_conversion_failures(self):
        def fail_write(layout, stream, **kwargs):
            stream.write(b"partial output")
            raise OSError("injected write failure")

        failures: tuple[
            tuple[AbstractContextManager[Any], type[Exception], str], ...
        ] = (
            (
                patch(
                    "tico.circle.export.optimize_document_for_export",
                    side_effect=RuntimeError("injected optimizer failure"),
                ),
                RuntimeError,
                "injected optimizer failure",
            ),
            (
                patch.object(
                    io,
                    "plan_circle_layout",
                    side_effect=circle_binary.CircleSerializationError(
                        "injected serialization failure"
                    ),
                ),
                CircleIOError,
                "injected serialization failure",
            ),
            (
                patch.object(circle_binary.CircleBinaryLayout, "write_to", fail_write),
                CircleIOError,
                "Failed to write Circle model",
            ),
        )
        for injected, error, message in failures:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "model.circle"
                path.write_bytes(b"existing destination")
                module, args = self._inputs()
                with injected, self.assertRaisesRegex(error, message):
                    tico.convert_to_file(module, args, path)
                self.assertEqual(path.read_bytes(), b"existing destination")
                self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()

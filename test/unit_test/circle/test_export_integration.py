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

"""Real serializer/default-O1 tests; require installed circle-schema and flatbuffers."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tico.circle.document import CircleDocument
from tico.circle.export import (
    _interface_snapshot,
    _operator_count,
    optimize_document_for_export,
    optimize_for_export,
)
from tico.config import CompileConfigV1


@unittest.skipUnless(
    importlib.util.find_spec("circle_schema") is not None
    and importlib.util.find_spec("flatbuffers") is not None,
    "Generated Circle schema and FlatBuffers runtime are required.",
)
class TestCircleExportIntegration(unittest.TestCase):
    @staticmethod
    def inputs():
        import torch

        class Tiny(torch.nn.Module):
            def forward(self, x, unused):
                return -(x * 1.0), x

        return Tiny().eval(), (torch.randn(2, 3), torch.randn(2, 3))

    def test_default_runs_o1_and_off_does_not(self):
        import tico

        model, args = self.inputs()
        with patch(
            "tico.circle.export.optimize_for_export", wraps=optimize_for_export
        ) as optimize:
            optimized = tico.convert(model, args)
            optimize.assert_called_once()
        with patch(
            "tico.circle.export.optimize_for_export",
            side_effect=AssertionError("OFF must bypass O1"),
        ):
            original = tico.convert(
                model, args, config=CompileConfigV1(circle_optimize=False)
            )
        before = CircleDocument.from_bytes(original.circle_binary)
        after = CircleDocument.from_bytes(optimized.circle_binary)
        after.verify()
        self.assertEqual(_interface_snapshot(after), _interface_snapshot(before))
        self.assertLessEqual(_operator_count(after), _operator_count(before))

    def test_real_binary_round_trip_and_idempotence(self):
        import tico

        model, args = self.inputs()
        original = tico.convert(
            model, args, config=CompileConfigV1(circle_optimize=False)
        )
        document = CircleDocument.from_bytes(original.circle_binary)
        round_trip = CircleDocument.from_bytes(document.to_bytes())
        self.assertEqual(_interface_snapshot(document), _interface_snapshot(round_trip))
        optimized = optimize_for_export(original.circle_binary)
        self.assertEqual(optimize_for_export(optimized), optimized)

    def test_document_default_o1_matches_bytes_and_saves_debug_without_binary_io(self):
        import tico

        model, args = self.inputs()
        original = tico.convert(
            model, args, config=CompileConfigV1(circle_optimize=False)
        )
        document = CircleDocument.from_bytes(original.circle_binary)
        expected = optimize_for_export(original.circle_binary)
        before = _interface_snapshot(document)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    CircleDocument,
                    "from_bytes",
                    side_effect=AssertionError("document O1 must not unpack a binary"),
                ),
                patch.object(
                    CircleDocument,
                    "to_bytes",
                    side_effect=AssertionError("document O1 must not join a binary"),
                ),
            ):
                optimize_document_for_export(document, dump_directory=directory)
            folder = next(Path(directory).iterdir())
            self.assertEqual((folder / "after.circle").read_bytes(), expected)
            original_dump = CircleDocument.load(folder / "before.circle")
            self.assertEqual(_interface_snapshot(original_dump), before)
            self.assertEqual(document.to_bytes(), expected)
            self.assertEqual(_interface_snapshot(document), before)
            report = json.loads((folder / "report.json").read_text())
            self.assertEqual(report["preset"], "o1")
            self.assertTrue(report["preserve_io"])
            self.assertEqual(report["bytes_after"], len(expected))
            self.assertEqual(
                report["bytes_before"], (folder / "before.circle").stat().st_size
            )
            self.assertEqual(report["operators_before"], _operator_count(original_dump))
            self.assertEqual(report["operators_after"], _operator_count(document))
            self.assertTrue(report["phases"][0]["executions"])
            self.assertTrue(
                all(
                    "diagnostics" in execution
                    for phase in report["phases"]
                    for execution in phase["executions"]
                )
            )
            self.assertEqual(
                {path.name for path in folder.iterdir()},
                {"before.circle", "after.circle", "report.json"},
            )

    def test_document_debug_original_survives_optimization_failure(self):
        import tico

        model, args = self.inputs()
        original = tico.convert(
            model, args, config=CompileConfigV1(circle_optimize=False)
        )
        document = CircleDocument.from_bytes(original.circle_binary)
        before = _interface_snapshot(document)
        with tempfile.TemporaryDirectory() as directory:
            with patch("tico.circle.export.create_o1_pipeline") as create:
                create.return_value.run.side_effect = RuntimeError(
                    "injected O1 failure"
                )
                with self.assertRaisesRegex(RuntimeError, "injected O1 failure"):
                    optimize_document_for_export(document, dump_directory=directory)
            folder = next(Path(directory).iterdir())
            restored = CircleDocument.load(folder / "before.circle")
            self.assertEqual(_interface_snapshot(restored), before)
            self.assertEqual(
                {path.name for path in folder.iterdir()}, {"before.circle"}
            )

    def test_save_writes_exact_already_optimized_bytes(self):
        import tico

        model, args = self.inputs()
        converted = tico.convert(model, args)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.circle"
            with patch(
                "tico.circle.export.optimize_for_export",
                side_effect=AssertionError("save must not optimize"),
            ):
                converted.save(path)
            self.assertEqual(path.read_bytes(), converted.circle_binary)

    def test_optimization_failure_propagates_through_convert(self):
        import tico

        model, args = self.inputs()
        with patch(
            "tico.circle.export.optimize_for_export",
            side_effect=RuntimeError("injected O1 failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected O1 failure"):
                tico.convert(model, args)


if __name__ == "__main__":
    unittest.main()

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

import copy
import gc
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.errors import CircleIOError
from tico.circle.operations import extract_by_operator_indices, PayloadOwnership
from tico.circle.runtime import CircleReferenceRuntime
from tico.serialize.circle_binary import plan_circle_layout

from test.support.circle.builder import CircleModelBuilder
from test.unit_test.circle.fixture import make_test_document


class CircleDocumentTest(unittest.TestCase):
    def test_clone_is_independent(self):
        document = make_test_document()

        clone = document.clone()
        clone.subgraph(0).name = "changed"

        self.assertEqual(document.subgraph(0).name, "primary")
        self.assertEqual(clone.subgraph(0).name, "changed")

    def test_clone_copies_payload_views_and_drops_the_mapping(self):
        document = make_test_document()
        base = np.arange(32, dtype=np.uint8)
        document.model.buffers[1].data = base[:8]
        mapping = Mock()
        mapped = CircleDocument(document.model, payload_mapping=mapping)

        clone = mapped.clone()
        deep = copy.deepcopy(mapped)

        for copied in (clone, deep):
            self.assertIsNone(copied.payload_mapping)
            payload = copied.model.buffers[1].data
            self.assertIsNot(payload, base)
            self.assertIsNone(payload.base)
            self.assertEqual(payload.tobytes(), bytes(range(8)))
        base[0] = 99
        self.assertEqual(clone.model.buffers[1].data[0], 0)
        mapping.release.assert_not_called()

    def test_release_payloads_and_context_manager_release_once(self):
        mapping = Mock()
        document = CircleDocument(make_test_document().model, payload_mapping=mapping)
        self.assertIs(document.payload_mapping, mapping)

        with document as entered:
            self.assertIs(entered, document)
        mapping.release.assert_called_once()
        self.assertIsNone(document.payload_mapping)
        document.release_payloads()
        mapping.release.assert_called_once()

        plain = make_test_document()
        plain.release_payloads()  # No mapping: a harmless no-op.
        self.assertIsNone(plain.payload_mapping)

    def test_non_atomic_save_over_mapped_source_is_refused_before_writing(self):
        mapping = Mock()
        mapping.path = "model.circle"
        mapping.same_file.side_effect = lambda path: str(path) == "model.circle"
        mapping._source = mapping
        document = CircleDocument(make_test_document().model, payload_mapping=mapping)

        with unittest.mock.patch("tico.circle.document.save_model") as save:
            with self.assertRaisesRegex(CircleIOError, "Refusing to overwrite"):
                document.save("model.circle", atomic=False)
            save.assert_not_called()
            document.save("model.circle")
            document.save("other.circle", atomic=False)
            document.save("-", atomic=False)
        self.assertEqual(save.call_count, 3)

    def test_subgraph_bounds_are_checked(self):
        document = make_test_document()

        with self.assertRaisesRegex(IndexError, "Subgraph index 2"):
            document.subgraph(2)

    def test_model_without_subgraphs_field_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "subgraphs"):
            CircleDocument(object())


class MappedPayloadSaveTest(unittest.TestCase):
    def _write_source(self, directory):
        builder = CircleModelBuilder()
        x = builder.input("x", [4096])
        weight = builder.const_f32("weight", np.arange(4096, dtype=np.float32))
        output = builder.add(x, weight, name="output")
        builder.set_outputs(output)
        path = Path(directory) / "source.circle"
        path.write_bytes(
            plan_circle_layout(builder.build().model, inline_budget=0).to_bytes()
        )
        return path

    def _borrow(self, document):
        return extract_by_operator_indices(
            document, (0,), payload_ownership=PayloadOwnership.BORROWED
        ).document

    def _assert_saved_values(self, path):
        loaded = CircleDocument.load(path)
        self.assertTrue(loaded.verify().ok)
        output = (
            CircleReferenceRuntime(loaded).run((np.zeros(4096, np.float32),)).outputs[0]
        )
        np.testing.assert_array_equal(output, np.arange(4096, dtype=np.float32))

    def _check_refused(self, scenario, alias):
        # Run in a child process: the original bug can truncate a live mmap.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("tico.circle.io._STREAMING_INLINE_BUDGET", 0),
        ):
            path = self._write_source(directory)
            original = path.read_bytes()
            document = CircleDocument.load_mapped(path)
            destination = path
            if alias != "original":
                destination = path.with_name("alias.circle")
                if alias == "hardlink":
                    os.link(path, destination)
                else:
                    destination.symlink_to(path)
            target = document
            if scenario == "released_then_borrowed":
                document.release_payloads()
                target = self._borrow(document)
            elif scenario.startswith("borrowed"):
                target = self._borrow(document)
                self.assertIsNone(target.payload_mapping)
                if scenario == "borrowed_twice":
                    target = self._borrow(target)
                elif scenario == "borrowed_source_deleted":
                    owner = weakref.ref(document)
                    document.release_payloads()
                    del document
                    gc.collect()
                    self.assertIsNone(owner())
            elif scenario == "released":
                document.release_payloads()
                document.release_payloads()
                self.assertIsNone(document.payload_mapping)
            elif scenario == "replaced":
                # The old inode is still mapped through destination's hard link.
                document.save(path)
                assert document.payload_mapping is not None
                self.assertFalse(document.payload_mapping.same_file(path))
                self.assertTrue(document.payload_mapping.same_file(destination))
                document.save(path, atomic=False)
                self._assert_saved_values(path)
            self.assertEqual(target.source, path)
            try:
                with self.assertRaisesRegex(CircleIOError, "Refusing to overwrite"):
                    target.save(destination, atomic=False)
            finally:
                # Verify the guard ran before any write, even when the exception differs.
                self.assertEqual(destination.read_bytes(), original)
            target.release_payloads()

    def _assert_refused_in_subprocess(self, scenario, alias="original"):
        script = (
            "import os\n"
            "if os.name == 'posix':\n"
            "    import resource\n"
            "    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
            "from test.unit_test.circle.test_document import MappedPayloadSaveTest\n"
            f"MappedPayloadSaveTest()._check_refused({scenario!r}, {alias!r})\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_direct_save_refuses_source_and_aliases(self):
        for alias in ("original", "hardlink", "symlink"):
            with self.subTest(alias=alias):
                self._assert_refused_in_subprocess("direct", alias)

    def test_borrowed_save_refuses_source_and_aliases(self):
        for alias in ("original", "hardlink", "symlink"):
            with self.subTest(alias=alias):
                self._assert_refused_in_subprocess("borrowed", alias)

    def test_released_save_refuses_source_and_aliases(self):
        for alias in ("original", "hardlink", "symlink"):
            with self.subTest(alias=alias):
                self._assert_refused_in_subprocess("released", alias)

    def test_borrowed_provenance_survives_source_release_and_collection(self):
        for scenario in (
            "borrowed_twice",
            "borrowed_source_deleted",
            "released_then_borrowed",
        ):
            with self.subTest(scenario=scenario):
                self._assert_refused_in_subprocess(scenario)

    @unittest.skipIf(os.name == "nt", "Windows cannot replace an open mapped file.")
    def test_atomic_replacement_keeps_old_inode_protected(self):
        self._assert_refused_in_subprocess("replaced", "hardlink")

    @unittest.skipIf(os.name == "nt", "Windows cannot replace an open mapped file.")
    def test_atomic_and_other_file_saves_preserve_values(self):
        for scenario in ("direct", "borrowed", "released"):
            with (
                self.subTest(scenario=scenario),
                tempfile.TemporaryDirectory() as directory,
                patch("tico.circle.io._STREAMING_INLINE_BUDGET", 0),
            ):
                path = self._write_source(directory)
                with CircleDocument.load_mapped(path) as document:
                    target = (
                        self._borrow(document) if scenario == "borrowed" else document
                    )
                    if scenario == "released":
                        document.release_payloads()
                    other = path.with_name("other.circle")
                    target.save(other, atomic=False)
                    self._assert_saved_values(other)
                    target.save(path)
                    self._assert_saved_values(path)

    def test_detached_copies_can_save_to_the_original_path(self):
        for kind in ("clone", "deepcopy", "extract"):
            with (
                self.subTest(kind=kind),
                tempfile.TemporaryDirectory() as directory,
                patch("tico.circle.io._STREAMING_INLINE_BUDGET", 0),
            ):
                path = self._write_source(directory)
                with CircleDocument.load_mapped(path) as document:
                    if kind == "clone":
                        target = document.clone()
                    elif kind == "deepcopy":
                        target = copy.deepcopy(document)
                    else:
                        target = extract_by_operator_indices(document, (0,)).document
                    self.assertEqual(target.source, path)
                    self.assertIsNone(target.payload_mapping)
                    document.release_payloads()
                    document.model.buffers[1].data = None
                    target.save(path, atomic=False)
                    self._assert_saved_values(path)

    def test_provenance_does_not_own_or_retain_the_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_source(directory)
            document = CircleDocument.load_mapped(path)
            owner = document.payload_mapping
            assert owner is not None
            mapping = weakref.ref(owner.buffer())
            borrowed = self._borrow(document)
            self.assertTrue(
                np.shares_memory(
                    document.model.buffers[1].data, borrowed.model.buffers[1].data
                )
            )
            borrowed.release_payloads()
            borrowed.release_payloads()
            self.assertFalse(owner.released)
            document.release_payloads()
            document.release_payloads()
            self.assertTrue(owner.released)
            self.assertFalse(owner.closed)
            document.model.buffers[1].data = None
            gc.collect()
            self.assertIsNotNone(mapping())
            borrowed.model.buffers[1].data = None
            gc.collect()
            self.assertTrue(owner.closed)
            self.assertIsNone(mapping())

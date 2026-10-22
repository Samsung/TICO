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

"""Positive and negative tests for the export-regression report comparator."""

import argparse
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tico.quantization.recipes.debug.export_regression import (
    _array_record,
    _first_differences,
    _host_contract,
    _json_value,
    _probes,
    _state_contract,
    _write_json,
    circle_contract,
    compare_arrays,
    compare_snapshots,
    main,
)


def _document(*, constant=1, qscale=0.5, internal_name="tmp", output_name="result"):
    """Build an Object-API-shaped fixture without pretending to serialize Circle."""
    quant = SimpleNamespace(scale=np.array([qscale], dtype=np.float32), zeroPoint=np.array([0]))
    tensors = [
        SimpleNamespace(name="input", buffer=0, shape=np.array([1]), type=0, quantization=None),
        SimpleNamespace(name=internal_name, buffer=1, shape=np.array([1]), type=0, quantization=quant),
        SimpleNamespace(name=output_name, buffer=0, shape=np.array([1]), type=0, quantization=None),
    ]
    opcode = SimpleNamespace(builtinCode=0, version=1, customCode=None)
    operator = SimpleNamespace(opcodeIndex=0, inputs=np.array([0, 1]), outputs=np.array([2]), builtinOptions=None, customOptions=None)
    graph = SimpleNamespace(tensors=tensors, operators=[operator], inputs=np.array([0]), outputs=np.array([2]))
    model = SimpleNamespace(
        version=3, subgraphs=[graph], signatureDefs=None, operatorCodes=[opcode],
        buffers=[SimpleNamespace(data=None), SimpleNamespace(data=np.array([constant], dtype=np.uint8))],
    )
    return SimpleNamespace(model=model)


def _phase(root, *, value=2.0):
    """Write a minimal complete snapshot for comparator tests."""
    (root / "records").mkdir(parents=True)
    snapshot = {
        "schema_version": 1,
        "stages": {"decoder.q.circle": "decoder"},
        "state_preserved": True, "host": {"host.npz": {"rope": {"sha256": "fixed"}}},
    }
    _write_json(root / "snapshot.json", snapshot)
    record = {
        "contract": circle_contract(_document()), "fx": [{"op": "call_function", "target": "aten.add.Tensor"}],
        "mode": "fake_quantize",
        "probes": [{"probe": "export_example", "torch_parity": [{"ok": True, "max_abs_error": 0.0}]}],
    }
    _write_json(root / "records/decoder.json", record)
    np.savez(root / "records/decoder.npz", p0_input_0=np.array([1.0], np.float32), p0_circle_0=np.array([value], np.float32), p0_torch_0=np.array([value], np.float32))


class TestArrayComparison(unittest.TestCase):
    def test_same_arrays_and_declared_tolerance(self):
        a = np.array([1.0], np.float32)
        b = np.array([1.00001], np.float32)
        self.assertTrue(compare_arrays(a, b, atol=1e-4, rtol=0)["ok"])
        self.assertFalse(compare_arrays(a, b, atol=0, rtol=0)["ok"])

    def test_dtype_and_shape_mismatches_are_not_cast_away(self):
        a = np.array([1.0], np.float32)
        for b in (a.astype(np.float64), a.reshape(1, 1)):
            self.assertFalse(compare_arrays(a, b, atol=1, rtol=1)["ok"])

    def test_nonfinite_results_never_pass(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            a = np.array([value], np.float32)
            self.assertFalse(compare_arrays(a, a, atol=0, rtol=0)["ok"])

    def test_integer_values_are_exact(self):
        a = np.array([1], np.int64)
        b = np.array([2], np.int64)
        self.assertFalse(compare_arrays(a, b, atol=100, rtol=100)["ok"])

    def test_scalar_and_empty_array_fingerprints(self):
        self.assertEqual(_array_record(np.array(1.0))["shape"], [])
        self.assertEqual(_array_record(np.array([]))["shape"], [0])
        self.assertEqual(_array_record(np.zeros((0, 2)))["shape"], [0, 2])
        with self.assertRaises(TypeError):
            _array_record(np.array([object()], dtype=object))

    def test_nested_qparams_include_tensor_bytes(self):
        left = {"qparams": SimpleNamespace(scale=torch.tensor([1.0]))}
        right = {"qparams": SimpleNamespace(scale=torch.tensor([2.0]))}
        self.assertNotEqual(_json_value(left), _json_value(right))
        self.assertIsInstance(_json_value(torch.ones(2, dtype=torch.bfloat16)), dict)


class TestCircleContract(unittest.TestCase):
    def test_internal_names_do_not_change_execution_contract(self):
        self.assertEqual(circle_contract(_document(internal_name="a")), circle_contract(_document(internal_name="b")))

    def test_constants_qparams_io_names_and_operator_versions_are_significant(self):
        reference = circle_contract(_document())
        for other in (_document(constant=2), _document(qscale=0.25), _document(output_name="renamed")):
            self.assertNotEqual(reference, circle_contract(other))
        other = _document()
        other.model.operatorCodes[0].version = 2
        self.assertNotEqual(reference, circle_contract(other))

    def test_buffer_reordering_is_ignored_but_topology_is_not(self):
        reference = _document()
        other = copy.deepcopy(reference)
        other.model.buffers.reverse()
        for tensor in other.model.subgraphs[0].tensors:
            tensor.buffer = 1 - tensor.buffer
        self.assertEqual(circle_contract(reference), circle_contract(other))
        other.model.subgraphs[0].operators[0].inputs = np.array([1, 0])
        self.assertNotEqual(circle_contract(reference), circle_contract(other))


class TestSnapshotComparison(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.left, self.right = root / "left", root / "right"
        _phase(self.left)
        _phase(self.right)

    def compare(self, **kwargs):
        return compare_snapshots(self.left, self.right, atol=1e-5, rtol=1e-4, **kwargs)

    def mutate_snapshot(self, key, value):
        path = self.right / "snapshot.json"
        data = json.loads(path.read_text())
        data[key] = value
        _write_json(path, data)

    def test_identical_complete_snapshots_pass(self):
        self.assertTrue(self.compare()["ok"])

    def rewrite_both_arrays(self, transform):
        """Damage both archives identically to catch false equality positives."""
        for root in (self.left, self.right):
            path = root / "records/decoder.npz"
            with np.load(path, allow_pickle=False) as values:
                arrays = {key: values[key] for key in values.files}
            transform(arrays)
            np.savez(path, **arrays)

    def rewrite_both_records(self, transform):
        """Apply a malformed-evidence fixture to each independent snapshot."""
        for root in (self.left, self.right):
            path = root / "records/decoder.json"
            data = json.loads(path.read_text())
            transform(data)
            _write_json(path, data)

    def test_missing_inputs_on_both_sides_fail(self):
        self.rewrite_both_arrays(lambda arrays: arrays.pop("p0_input_0"))
        result = self.compare()
        self.assertFalse(result["ok"])
        self.assertFalse(result["stages"]["decoder.q.circle"]["numerical_coverage_complete"])

    def test_missing_torch_outputs_and_empty_parity_fail(self):
        self.rewrite_both_arrays(lambda arrays: arrays.pop("p0_torch_0"))
        self.rewrite_both_records(lambda record: record["probes"][0].update(torch_parity=[]))
        self.assertFalse(self.compare()["ok"])
        self.assertFalse(self.compare(require_torch_parity=True)["ok"])

    def test_partial_multi_output_evidence_fails(self):
        def add_output(record):
            record["contract"]["subgraphs"][0]["outputs"].append(2)
            record["probes"][0]["torch_parity"].append({"ok": True, "max_abs_error": 0.0})

        self.rewrite_both_records(add_output)
        self.rewrite_both_arrays(lambda arrays: arrays.update(
            p0_circle_1=arrays["p0_circle_0"].copy(),
            p0_torch_1=arrays["p0_torch_0"].copy(),
        ))
        self.assertTrue(self.compare(require_torch_parity=True)["ok"])
        self.rewrite_both_arrays(lambda arrays: arrays.pop("p0_circle_1"))
        self.assertFalse(self.compare()["ok"])

    def test_shifted_output_index_does_not_count_as_coverage(self):
        self.rewrite_both_arrays(lambda arrays: arrays.update(p0_circle_9=arrays.pop("p0_circle_0")))
        self.assertFalse(self.compare()["ok"])

    def test_extra_unindexed_arrays_fail(self):
        self.rewrite_both_arrays(lambda arrays: arrays.update(unused=np.ones(1, np.float32)))
        self.assertFalse(self.compare()["ok"])

    def test_parity_record_count_must_match_graph_outputs(self):
        self.rewrite_both_records(lambda record: record["probes"][0]["torch_parity"].append({"ok": True}))
        self.assertFalse(self.compare()["ok"])

    def test_identical_shape_corruption_fails_against_graph_contract(self):
        self.rewrite_both_arrays(lambda arrays: arrays.update(p0_input_0=arrays["p0_input_0"].reshape(1, 1)))
        self.assertFalse(self.compare()["ok"])

    def test_stale_positive_parity_is_recomputed_from_actual_arrays(self):
        self.rewrite_both_arrays(lambda arrays: arrays.update(p0_torch_0=np.array([100.0], np.float32)))
        self.assertTrue(self.compare()["ok"])
        self.assertFalse(self.compare(require_torch_parity=True)["ok"])

    def test_matching_nonfinite_evidence_never_passes(self):
        for key in ("p0_input_0", "p0_circle_0", "p0_torch_0"):
            with self.subTest(key=key):
                self.rewrite_both_arrays(lambda arrays: arrays.update({key: np.array([np.inf], np.float32)}))
                self.assertFalse(self.compare()["ok"])
                self.rewrite_both_arrays(lambda arrays: arrays.update({key: np.array([1.0 if "input" in key else 2.0], np.float32)}))

    def test_constant_graph_with_no_inputs_is_valid(self):
        self.rewrite_both_records(lambda record: record["contract"]["subgraphs"][0].update(inputs=[]))
        self.rewrite_both_arrays(lambda arrays: arrays.pop("p0_input_0"))
        self.assertTrue(self.compare(require_torch_parity=True)["ok"])

    def test_no_graph_outputs_is_incomplete_not_vacuous_success(self):
        self.rewrite_both_records(lambda record: record["contract"]["subgraphs"][0].update(outputs=[]))
        self.assertFalse(self.compare()["ok"])

    def test_invalid_graph_boundary_index_fails(self):
        self.rewrite_both_records(lambda record: record["contract"]["subgraphs"][0].update(outputs=[999]))
        self.assertFalse(self.compare()["ok"])

    def test_state_mutation_fails(self):
        self.mutate_snapshot("state_preserved", False)
        self.assertFalse(self.compare()["ok"])

    def test_missing_stages_or_host_changes_fail(self):
        self.mutate_snapshot("stages", {})
        self.assertFalse(self.compare()["ok"])
        self.mutate_snapshot("stages", {"decoder.q.circle": "decoder"})
        self.mutate_snapshot("host", {})
        self.assertFalse(self.compare()["ok"])

    def test_missing_array_keys_and_nonfinite_values_fail(self):
        np.savez(self.right / "records/decoder.npz", p0_circle_0=np.array([np.nan], np.float32))
        result = self.compare()
        self.assertFalse(result["ok"])
        self.assertFalse(result["stages"]["decoder.q.circle"]["arrays"]["p0_circle_0"]["ok"])

    def test_empty_numerical_artifacts_do_not_pass(self):
        for root in (self.left, self.right):
            np.savez(root / "records/decoder.npz")
        self.assertFalse(self.compare()["ok"])

    def test_empty_probe_lists_do_not_pass(self):
        for root in (self.left, self.right):
            path = root / "records/decoder.json"
            record = json.loads(path.read_text())
            record["probes"] = []
            _write_json(path, record)
        self.assertFalse(self.compare()["ok"])

    def test_inputs_are_exact_even_when_output_tolerance_would_allow_change(self):
        np.savez(self.right / "records/decoder.npz", p0_input_0=np.array([1.000001], np.float32), p0_circle_0=np.array([2.0], np.float32), p0_torch_0=np.array([2.0], np.float32))
        self.assertFalse(self.compare()["ok"])

    def test_pytorch_parity_is_reported_and_can_be_required(self):
        path = self.right / "records/decoder.json"
        record = json.loads(path.read_text())
        record["probes"][0]["torch_parity"][0]["ok"] = False
        _write_json(path, record)
        self.assertTrue(self.compare()["ok"])
        self.assertFalse(self.compare(require_torch_parity=True)["ok"])

    def test_fx_change_is_diagnostic_when_final_circle_and_values_match(self):
        path = self.right / "records/decoder.json"
        record = json.loads(path.read_text())
        record["fx"] = []
        _write_json(path, record)
        result = self.compare()
        self.assertTrue(result["ok"])
        self.assertFalse(result["stages"]["decoder.q.circle"]["torch_export_ops_equal"])

    def test_cli_emits_report_and_nonzero_status_for_failure(self):
        self.mutate_snapshot("state_preserved", False)
        report = Path(self.directory.name) / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["compare", str(self.left), str(self.right), "--report", str(report)])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(report.read_text())["ok"])

    def test_invalid_tolerance_is_rejected(self):
        for value in ("nan", "inf", "-1"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(["compare", "a", "b", "--report", "unused", "--atol", value])

    def test_difference_locations_are_bounded(self):
        changes = _first_differences({str(i): 1 for i in range(50)}, {str(i): 2 for i in range(50)})
        self.assertEqual(len(changes), 20)


class TestProbeAndHostContracts(unittest.TestCase):
    def test_decode_probes_preserve_shape_and_leave_callers_inputs_untouched(self):
        hidden = torch.ones(1, 1, 4)
        mask = torch.tensor([[[0.0, -120.0, 0.0]]])
        kv = (torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4))
        probes = _probes((hidden,), {"attention_mask": mask, "past_key_value": kv})
        self.assertEqual([p[0] for p in probes], ["export_example", "decode_empty_past", "decode_full_past"])
        torch.testing.assert_close(mask, torch.tensor([[[0.0, -120.0, 0.0]]]))
        self.assertEqual(float(probes[1][2]["past_key_value"][0].sum()), 0.0)
        self.assertEqual(float(kv[0].sum()), 8.0)

    def test_host_npz_timestamp_is_not_part_of_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "host.npz", rope=np.ones((1, 2, 4), np.float32))
            torch.save({"weight": torch.ones(2), "scale": 0.25, "name": "embedding"}, root / "host.pt")
            first = _host_contract(root)
            np.savez(root / "host.npz", rope=np.ones((1, 2, 4), np.float32))
            second = _host_contract(root)
            self.assertEqual(first, second)
            np.savez(root / "host.npz", rope=np.zeros((1, 2, 4), np.float32))
            self.assertNotEqual(first, _host_contract(root))


class TestFrozenStateFingerprint(unittest.TestCase):
    def make_model(self):
        from tico.quantization.wrapq.observers.minmax import MinMaxObserver
        from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

        class Model(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.observer = MinMaxObserver(name="activation")

            def _all_observers(self):
                return (self.observer,)

            def forward(self, value):
                return self._fq(value * self.weight, self.observer)

        model = Model()
        model.enable_calibration()
        model(torch.tensor([-1.0, 0.0, 1.0]))
        model.freeze_qparams()
        return model

    def test_frozen_execution_does_not_change_fingerprint(self):
        model = self.make_model()
        initial = _state_contract(model)
        model(torch.linspace(-2, 2, 16))
        self.assertEqual(initial, _state_contract(model))

    def test_weight_and_qparam_changes_are_detected(self):
        model = self.make_model()
        initial = _state_contract(model)
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(initial, _state_contract(model))
        model = self.make_model()
        initial = _state_contract(model)
        model.observer._cached_scale.mul_(2)
        self.assertNotEqual(initial, _state_contract(model))

    def test_calibration_mode_is_rejected(self):
        model = self.make_model()
        model.enable_calibration()
        with self.assertRaisesRegex(ValueError, "Calibration"):
            _state_contract(model)



class TestRopeMigrationRouting(unittest.TestCase):
    """A requested correction is candidate-only and never hides old/new failures."""

    def _run_fixture(self, directory, *, migration, old_new_ok):
        from tico.quantization.recipes.debug.export_regression import _run

        root = Path(directory)
        options = argparse.Namespace(
            command="run", output=str(root / "results"), config=str(root / "config.yaml"),
            checkpoint=str(root / "model.pt"), seq_len=16, eager_seq_len=8,
            decode_steps=2, artifacts="auto", seed=42, threads=1, atol=1e-5,
            rtol=1e-4, eager_inputs=None, include_full_text=False, set=[],
            candidate_repo=str(root / "candidate"), baseline_repo=str(root / "baseline"),
            require_torch_parity=False, migrate_llama_rope=migration,
        )
        commands = []

        def worker(command, **kwargs):
            commands.append(command)
            target = Path(command[command.index("--output") + 1])
            _write_json(target / "environment.json", {"versions": {}, "checkpoint_sha256": "same"})
            _write_json(target / "quantization_state.json", {})
            if "--stress" in command:
                _write_json(target / "eager.json", {"state_preserved": True})
            return SimpleNamespace(returncode=0)

        owner = "tico.quantization.recipes.debug.export_regression"
        with patch(owner + ".subprocess.run", side_effect=worker), patch(
            owner + ".compare_snapshots", side_effect=[{"ok": old_new_ok}, {"ok": True}]
        ), contextlib.redirect_stdout(io.StringIO()):
            code = _run(options)
        summary = json.loads((root / "results/summary.json").read_text())
        return code, commands, summary

    def test_only_candidate_receives_explicit_migration_flag(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                code, commands, summary = self._run_fixture(directory, migration=enabled, old_new_ok=True)
                self.assertEqual(code, 0)
                self.assertNotIn("--migrate-llama-rope", commands[0])
                self.assertEqual("--migrate-llama-rope" in commands[1], enabled)
                self.assertEqual(summary["llama_rope_migration_requested"], enabled)

    def test_intentional_rope_difference_is_still_a_reported_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            code, _, summary = self._run_fixture(directory, migration=True, old_new_ok=False)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(summary["candidate_before_vs_after_eager"]["ok"])
        self.assertIn("never waived", summary["intentional_change_note"])

if __name__ == "__main__":
    unittest.main()

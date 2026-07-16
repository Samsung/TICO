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

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tico.quantization.analysis.weight_sparsity import (
    format_layer_weight_sparsity_table,
    format_weight_sparsity_table,
    measure_layer_weight_sparsity,
    measure_weight_sparsity,
    measure_weight_sparsity_report,
    WeightSparsityError,
    write_layer_weight_sparsity_csv,
    write_weight_sparsity_csv,
)


class _FakeDType:
    def __init__(self, bits: int, signed: bool = False):
        self.bits = bits
        self.signed = signed

    @property
    def qmin(self) -> int:
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    @property
    def qmax(self) -> int:
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    def __str__(self) -> str:
        prefix = "int" if self.signed else "uint"
        return f"{prefix}{self.bits}"


class _FakeObserver(torch.nn.Module):
    def __init__(
        self,
        *,
        scale,
        zero_point,
        bits: int = 4,
        signed: bool = False,
        channel_axis: int | None = 0,
    ):
        super().__init__()
        self.dtype = _FakeDType(bits, signed=signed)
        self.channel_axis = channel_axis
        self.register_buffer(
            "_cached_scale", torch.as_tensor(scale, dtype=torch.float32)
        )
        self.register_buffer(
            "_cached_zp", torch.as_tensor(zero_point, dtype=torch.int32)
        )


class IdentityObserver(_FakeObserver):
    pass


class _WeightOwner(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())


class _QuantLeaf(torch.nn.Module):
    def __init__(
        self,
        fp_name: str,
        weight: torch.Tensor,
        observer: torch.nn.Module,
        *,
        shared_parameter: torch.nn.Parameter | None = None,
        mode: str = "QUANT",
    ):
        super().__init__()
        self.fp_name = fp_name
        self._mode = SimpleNamespace(name=mode)
        self.module = _WeightOwner(weight)
        if shared_parameter is not None:
            self.module.weight = shared_parameter
        self.observer = observer

    def get_observer(self, name: str, *, recurse: bool = True):
        if name == "weight" and not recurse:
            return self.observer
        return None


class _Model(torch.nn.Module):
    def __init__(self, *leaves: _QuantLeaf):
        super().__init__()
        self.leaves = torch.nn.ModuleList(leaves)


class TestWeightSparsity(unittest.TestCase):
    def test_semantic_zero_uses_zero_point(self):
        leaf = _QuantLeaf(
            "model.layers.0.self_attn.q_proj",
            torch.tensor([[-0.49, 0.49, 0.51, -0.51]]),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )

        rows = measure_weight_sparsity(_Model(leaf), "llama")
        q_proj = next(row for row in rows if row.scope == "Attention / q_proj")

        self.assertEqual(q_proj.qdtype, "uint4")
        self.assertEqual(q_proj.numel, 4)
        self.assertEqual(q_proj.zero_count, 2)
        self.assertAlmostEqual(q_proj.sparsity_percent, 50.0)

    def test_rounding_matches_torch_fake_quantization(self):
        leaf = _QuantLeaf(
            "model.layers.0.self_attn.k_proj",
            torch.tensor([[0.5, -0.5, 1.5, -1.5]]),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )

        rows = measure_weight_sparsity(_Model(leaf), "llama")
        k_proj = next(row for row in rows if row.scope == "Attention / k_proj")

        reference = torch.fake_quantize_per_channel_affine(
            leaf.module.weight.detach(),
            scale=torch.tensor([1.0]),
            zero_point=torch.tensor([7], dtype=torch.int32),
            axis=0,
            quant_min=0,
            quant_max=15,
        )
        self.assertEqual(k_proj.zero_count, int(torch.count_nonzero(reference == 0)))
        self.assertEqual(k_proj.zero_count, 2)

    def test_per_channel_zero_points_are_broadcast(self):
        leaf = _QuantLeaf(
            "model.layers.0.mlp.up_proj",
            torch.tensor([[0.4, 0.6], [0.9, 1.1]]),
            _FakeObserver(
                scale=[1.0, 2.0],
                zero_point=[5, 9],
                channel_axis=0,
            ),
        )

        rows = measure_weight_sparsity(_Model(leaf), "llama", max_chunk_elements=2)
        up_proj = next(row for row in rows if row.scope == "MLP / up_proj")

        self.assertEqual(up_proj.zero_count, 2)
        self.assertEqual(up_proj.numel, 4)
        self.assertAlmostEqual(up_proj.sparsity_percent, 50.0)

    def test_scope_aggregation_is_weighted_by_numel(self):
        dense_zero = _QuantLeaf(
            "model.layers.0.self_attn.q_proj",
            torch.tensor([[0.0, 0.0]]),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )
        no_zero = _QuantLeaf(
            "model.layers.1.self_attn.q_proj",
            torch.ones(1, 8),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )

        rows = measure_weight_sparsity(_Model(dense_zero, no_zero), "llama")
        q_proj = next(row for row in rows if row.scope == "Attention / q_proj")

        self.assertEqual(q_proj.zero_count, 2)
        self.assertEqual(q_proj.numel, 10)
        self.assertAlmostEqual(q_proj.sparsity_percent, 20.0)

    def test_identity_weight_observer_is_skipped(self):
        quantized = _QuantLeaf(
            "model.layers.0.mlp.down_proj",
            torch.zeros(1, 2),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )
        floating_point = _QuantLeaf(
            "lm_head",
            torch.zeros(1, 4),
            IdentityObserver(scale=[1.0], zero_point=[0], channel_axis=0),
        )

        rows = measure_weight_sparsity(_Model(quantized, floating_point), "llama")
        scopes = {row.scope for row in rows}

        self.assertIn("MLP / down_proj", scopes)
        self.assertNotIn("LM head", scopes)

    def test_shared_weight_with_identical_qparams_is_deduplicated(self):
        shared = torch.nn.Parameter(torch.zeros(1, 4))
        embedding = _QuantLeaf(
            "model.embed_tokens",
            torch.empty(1, 4),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
            shared_parameter=shared,
        )
        lm_head = _QuantLeaf(
            "lm_head",
            torch.empty(1, 4),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
            shared_parameter=shared,
        )

        rows = measure_weight_sparsity(_Model(embedding, lm_head), "llama")
        all_weights = rows[0]

        self.assertEqual(all_weights.numel, 4)
        self.assertEqual(all_weights.zero_count, 4)
        self.assertIn("Token embedding", {row.scope for row in rows})
        self.assertIn("LM head", {row.scope for row in rows})

    def test_qwen_projection_scopes_are_reported(self):
        paths = [
            "visual.patch_embed.proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.0.attn.proj",
            "model.visual.blocks.0.mlp.linear_fc1",
            "model.visual.blocks.0.mlp.linear_fc2",
            "model.visual.merger.linear_fc1",
            "model.visual.merger.linear_fc2",
            "model.visual.deepstack_merger_list.0.linear_fc1",
            "model.visual.deepstack_merger_list.0.linear_fc2",
            "model.language_model.embed_tokens",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.self_attn.v_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "lm_head",
        ]
        leaves = [
            _QuantLeaf(
                path,
                torch.zeros(1, 1),
                _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
            )
            for path in paths
        ]

        rows = measure_weight_sparsity(_Model(*leaves), "qwen3_vl")
        scopes = {row.scope for row in rows}

        expected = {
            "Vision / patch_embed.proj",
            "Vision attention / qkv",
            "Vision attention / proj",
            "Vision MLP / linear_fc1",
            "Vision MLP / linear_fc2",
            "Vision merger / linear_fc1",
            "Vision merger / linear_fc2",
            "Deepstack merger / linear_fc1",
            "Deepstack merger / linear_fc2",
            "Text / token embedding",
            "Text attention / q_proj",
            "Text attention / k_proj",
            "Text attention / v_proj",
            "Text attention / o_proj",
            "Text MLP / gate_proj",
            "Text MLP / up_proj",
            "Text MLP / down_proj",
            "LM head",
        }
        self.assertTrue(expected.issubset(scopes))

    def test_llama_layer_report_contains_totals_and_projection_rows(self):
        layer_zero = _QuantLeaf(
            "model.layers.0.self_attn.q_proj",
            torch.zeros(1, 2),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )
        layer_nonzero = _QuantLeaf(
            "model.layers.1.self_attn.q_proj",
            torch.ones(1, 4),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )
        layer_mlp = _QuantLeaf(
            "model.layers.0.mlp.up_proj",
            torch.tensor([[0.0, 1.0]]),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )

        report = measure_weight_sparsity_report(
            _Model(layer_zero, layer_nonzero, layer_mlp),
            "llama",
        )
        rows = list(report.layer_rows)

        layer_names = list(dict.fromkeys(row.layer for row in rows))
        self.assertEqual(layer_names, ["model.layers.0", "model.layers.1"])

        layer0_total = next(
            row
            for row in rows
            if row.layer == "model.layers.0" and row.scope == "All quantized weights"
        )
        layer0_q = next(
            row
            for row in rows
            if row.layer == "model.layers.0" and row.scope == "Attention / q_proj"
        )
        layer0_up = next(
            row
            for row in rows
            if row.layer == "model.layers.0" and row.scope == "MLP / up_proj"
        )
        layer1_total = next(
            row
            for row in rows
            if row.layer == "model.layers.1" and row.scope == "All quantized weights"
        )

        self.assertEqual((layer0_total.zero_count, layer0_total.numel), (3, 4))
        self.assertAlmostEqual(layer0_total.sparsity_percent, 75.0)
        self.assertEqual((layer0_q.zero_count, layer0_q.numel), (2, 2))
        self.assertEqual((layer0_up.zero_count, layer0_up.numel), (1, 2))
        self.assertEqual((layer1_total.zero_count, layer1_total.numel), (0, 4))

    def test_qwen_layer_report_uses_architecture_order(self):
        paths = [
            "model.language_model.layers.10.self_attn.q_proj",
            "model.visual.deepstack_merger_list.1.linear_fc1",
            "model.visual.blocks.2.attn.qkv",
            "model.language_model.layers.2.self_attn.q_proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.patch_embed.proj",
            "model.visual.merger.linear_fc1",
            "model.language_model.embed_tokens",
            "model.language_model.norm",
            "lm_head",
        ]
        leaves = [
            _QuantLeaf(
                path,
                torch.zeros(1, 1),
                _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
            )
            for path in paths
        ]

        rows = measure_layer_weight_sparsity(_Model(*leaves), "qwen3_vl")
        layers = list(dict.fromkeys(row.layer for row in rows))

        self.assertEqual(
            layers,
            [
                "model.visual.patch_embed",
                "model.visual.blocks.0",
                "model.visual.blocks.2",
                "model.visual.merger",
                "model.visual.deepstack_merger_list.1",
                "model.language_model.embed_tokens",
                "model.language_model.layers.2",
                "model.language_model.layers.10",
                "model.language_model.norm",
                "lm_head",
            ],
        )

    def test_layer_totals_can_be_disabled(self):
        leaf = _QuantLeaf(
            "model.layers.0.self_attn.o_proj",
            torch.zeros(1, 2),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )

        rows = measure_layer_weight_sparsity(
            _Model(leaf),
            "llama",
            include_layer_totals=False,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].layer, "model.layers.0")
        self.assertEqual(rows[0].scope, "Attention / o_proj")

    def test_layer_output_contains_requested_columns(self):
        leaf = _QuantLeaf(
            "model.layers.0.mlp.gate_proj",
            torch.zeros(1, 2),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )
        rows = measure_layer_weight_sparsity(_Model(leaf), "llama")

        markdown = format_layer_weight_sparsity_table(rows, precision=2)
        self.assertIn(
            "| Layer | Scope | Qdtype | Sparsity (%) |",
            markdown,
        )
        self.assertNotIn("Qscheme", markdown)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_layer_weight_sparsity_csv(
                rows,
                Path(tmpdir) / "report_by_layer.csv",
            )
            header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "layer,scope,qdtype,sparsity_percent")

    def test_non_quant_mode_is_rejected(self):
        leaf = _QuantLeaf(
            "model.layers.0.self_attn.q_proj",
            torch.zeros(1, 1),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
            mode="CALIB",
        )

        with self.assertRaises(WeightSparsityError):
            measure_weight_sparsity(_Model(leaf), "llama")

    def test_output_contains_only_requested_columns(self):
        leaf = _QuantLeaf(
            "model.layers.0.mlp.gate_proj",
            torch.zeros(1, 2),
            _FakeObserver(scale=[1.0], zero_point=[7], channel_axis=0),
        )
        rows = measure_weight_sparsity(_Model(leaf), "llama")

        markdown = format_weight_sparsity_table(rows, precision=2)
        self.assertIn("| Scope | Qdtype | Sparsity (%) |", markdown)
        self.assertNotIn("Qscheme", markdown)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_weight_sparsity_csv(rows, Path(tmpdir) / "report.csv")
            header = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "scope,qdtype,sparsity_percent")


if __name__ == "__main__":
    unittest.main()

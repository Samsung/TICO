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

"""Static input ownership, numerical provenance, and trace isolation tests."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tico.quantization.wrapq.utils.static_export import (
    prepared_static_export,
    require_static_decoder_inputs,
    validate_static_length,
)
from tico.quantization.wrapq.wrappers.llama.static_inputs import (
    capture_rope_state,
    LlamaStaticInputs,
    prepare_llama_static_inputs,
)
from tico.quantization.wrapq.wrappers.qwen_vl.static_inputs import (
    prepare_qwen3_vl_static_text_inputs,
    Qwen3VLStaticTextInputs,
)
from torch import nn


class _LlamaRotary(nn.Module):
    """Expose a deterministic rotary source without a Transformers dependency."""

    inv_freq: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.01]), persistent=False)
        self.inv_freq = self.get_buffer("inv_freq")
        self.attention_scaling = 1.0
        self.calls = 0

    def forward(self, hidden, positions):
        self.calls += 1
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq
        phases = torch.cat((frequencies, frequencies), dim=-1)
        return (
            phases.cos() * self.attention_scaling,
            phases.sin() * self.attention_scaling,
        )


class _LlamaOwner(nn.Module):
    """Represent only the static-input protocol, not a language model."""

    _static_export_inputs: LlamaStaticInputs | None
    _export_rotary_emb: _LlamaRotary
    _export_rope_contract_version: int
    _export_rope_attention_scaling: float
    _mode: SimpleNamespace

    def __init__(self, *, rope="hf", source=None, capacity=131072):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            head_dim=4,
            max_position_embeddings=capacity,
            rope_theta=10000.0,
        )
        self.qcfg = SimpleNamespace(attention_mask_fill_value=-120.0)
        self.attn_options = SimpleNamespace(rope=rope)
        capture_rope_state(
            self, self.config, source if source is not None else _LlamaRotary()
        )

    def _build_static_export_inputs(self, length):
        return prepare_llama_static_inputs(self, length)


class _MutableRotary(nn.Module):
    """Expose mutation to test snapshot isolation without HF dependencies."""

    inv_freq: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0, 0.1]))
        self.calls = 0

    def forward(self, hidden, positions):
        self.calls += 1
        frequencies = positions[0].to(hidden.dtype).unsqueeze(-1) * self.inv_freq
        phases = torch.cat((frequencies, frequencies), dim=-1)
        return phases.cos(), phases.sin()


class _QwenOwner(nn.Module):
    """Implement the fixed-position owner protocol for snapshot tests."""

    _static_export_inputs: Qwen3VLStaticTextInputs | None
    rotary_emb: _MutableRotary
    _export_rotary_emb: _MutableRotary

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(max_position_embeddings=65536)
        self.qcfg = SimpleNamespace(attention_mask_fill_value=-120.0)
        self.rotary_emb = _MutableRotary()
        self._export_rotary_emb = copy.deepcopy(self.rotary_emb)

    @staticmethod
    def _get_visual_start_idx(qcfg):
        return 0

    @staticmethod
    def _get_vision_grid_thw(qcfg):
        return (1, 2, 2)

    @staticmethod
    def _get_spatial_merge_size(qcfg):
        return 2

    @staticmethod
    def _compute_3d_position_ids(*, device, seq_len, **kwargs):
        return torch.arange(seq_len, device=device).view(1, 1, -1).expand(3, 1, -1)

    def _build_static_export_inputs(self, length):
        return prepare_qwen3_vl_static_text_inputs(self, length)


class TestStaticInputContext(unittest.TestCase):
    def test_explicit_length_validation(self):
        for value in (None, True, 2.5, "4"):
            with self.subTest(value=value), self.assertRaises(TypeError):
                validate_static_length(value, 8)
        for value in (-1, 0, 9):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_static_length(value, 8)
        self.assertEqual(validate_static_length(8, 8), 8)

    def test_temporary_buffers_preserve_weight_identity_and_state_dict(self):
        owner = _LlamaOwner()
        weight = owner.weight
        original_keys = tuple(owner.state_dict())
        with prepared_static_export(owner, 8):
            assert owner._static_export_inputs is not None
            self.assertEqual(owner._static_export_inputs.cos.shape, (1, 8, 4))
            self.assertEqual(
                owner._static_export_inputs.attention_mask.shape, (1, 1, 8, 8)
            )
            self.assertIs(owner.weight, weight)
            self.assertEqual(tuple(owner.state_dict()), original_keys)
        self.assertFalse(hasattr(owner, "_static_export_inputs"))
        self.assertIs(owner.weight, weight)

    def test_cleanup_after_trace_failure(self):
        owner = _LlamaOwner()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with prepared_static_export(owner, 4):
                raise RuntimeError("synthetic trace failure")
        self.assertFalse(hasattr(owner, "_static_export_inputs"))

    def test_cleanup_after_later_owner_preparation_failure(self):
        model = nn.ModuleList([_LlamaOwner(capacity=16), _LlamaOwner(capacity=2)])
        with self.assertRaises(ValueError):
            with prepared_static_export(model, 4):
                self.fail("Invalid profile must not enter the context")
        for owner in model:
            self.assertFalse(hasattr(owner, "_static_export_inputs"))

    def test_nested_context_does_not_remove_outer_inputs(self):
        owner = _LlamaOwner()
        with prepared_static_export(owner, 4):
            inputs = owner._static_export_inputs
            with self.assertRaisesRegex(ValueError, "Nested"):
                with prepared_static_export(owner, 4):
                    pass
            self.assertIs(owner._static_export_inputs, inputs)

    def test_preexisting_none_attribute_is_restored(self):
        for registered in (False, True):
            owner = _LlamaOwner()
            if registered:
                owner.add_module("_static_export_inputs", None)
            else:
                owner._static_export_inputs = None
            with prepared_static_export(owner, 4):
                self.assertIsInstance(owner._static_export_inputs, nn.Module)
            self.assertIsNone(owner._static_export_inputs)
            self.assertEqual("_static_export_inputs" in owner._modules, registered)

    def test_calibration_rejected_before_preparation(self):
        owner = _LlamaOwner()
        owner._mode = SimpleNamespace(name="CALIB")
        with self.assertRaisesRegex(ValueError, "calibration"):
            with prepared_static_export(owner, 4):
                pass
        self.assertFalse(hasattr(owner, "_static_export_inputs"))

    def test_required_host_inputs_do_not_silently_fall_back(self):
        mask = torch.zeros(1, 4, 4)
        positions = (torch.ones(1, 4, 4), torch.zeros(1, 4, 4))
        require_static_decoder_inputs(mask, positions)
        # Existing boolean-mask support remains the responsibility of each wrapper.
        require_static_decoder_inputs(mask.bool(), positions)
        for bad_mask, bad_rope in (
            (None, positions),
            (mask, None),
            (mask, (positions[0],)),
        ):
            with self.assertRaises(ValueError):
                require_static_decoder_inputs(bad_mask, bad_rope)


class TestLlamaStaticInputs(unittest.TestCase):
    def test_default_frequencies_and_sine_conventions(self):
        for convention in ("hf", "pre_negated_sin"):
            owner = _LlamaOwner(rope=convention)
            actual = prepare_llama_static_inputs(owner, 6)
            frequencies = torch.outer(
                torch.arange(6).float(), torch.tensor([1.0, 0.01])
            )
            phases = torch.cat((frequencies, frequencies), dim=-1)
            expected_sin = phases.sin()
            if convention == "pre_negated_sin":
                expected_sin[:, :2] *= -1
            torch.testing.assert_close(
                actual.cos, phases.cos().unsqueeze(0), atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual.sin, expected_sin.unsqueeze(0), atol=0, rtol=0
            )
            expected_mask = torch.full((6, 6), -120.0).triu(1)
            torch.testing.assert_close(
                actual.attention_mask[0, 0], expected_mask, atol=0, rtol=0
            )

    def test_source_frequency_and_scaling_are_snapshotted(self):
        source = _LlamaRotary()
        source.attention_scaling = 1.25
        owner = _LlamaOwner(source=source)
        before = prepare_llama_static_inputs(owner, 8)
        source.inv_freq.mul_(10)
        source.attention_scaling = 9.0
        after = prepare_llama_static_inputs(owner, 8)
        torch.testing.assert_close(before.cos, after.cos, atol=0, rtol=0)
        torch.testing.assert_close(before.sin, after.sin, atol=0, rtol=0)
        self.assertEqual(source.calls, 0)
        self.assertEqual(float(before.cos[0, 0, 0]), 1.25)

    def test_stale_legacy_tables_never_override_the_canonical_snapshot(self):
        owner = _LlamaOwner()
        expected = prepare_llama_static_inputs(owner, 4)
        owner.rope_cos_template = torch.full((1, 8, 4), 2.0)
        owner.rope_sin_template = torch.full((1, 8, 4), 3.0)
        actual = prepare_llama_static_inputs(owner, 4)
        torch.testing.assert_close(actual.cos, expected.cos, atol=0, rtol=0)
        torch.testing.assert_close(actual.sin, expected.sin, atol=0, rtol=0)

    def test_legacy_and_intermediate_checkpoints_require_explicit_migration(self):
        owner = _LlamaOwner()
        del owner._export_rotary_emb
        for with_tables in (False, True):
            if with_tables:
                owner.rope_cos_template = torch.ones(1, 8, 4)
                owner.rope_sin_template = torch.ones(1, 8, 4)
            with self.assertRaisesRegex(ValueError, "migrate_llama_rope_state"):
                prepare_llama_static_inputs(owner, 4)
        owner = _LlamaOwner()
        owner._export_rope_contract_version = 1
        with self.assertRaisesRegex(ValueError, "unified RoPE"):
            prepare_llama_static_inputs(owner, 4)

    def test_missing_source_uses_the_complete_config_factory(self):
        owner = _LlamaOwner()
        owner.config.rope_scaling = {"rope_type": "llama3", "factor": 32.0}
        source = _LlamaRotary()
        source.inv_freq /= 4
        with patch(
            "tico.quantization.wrapq.wrappers.llama.static_inputs.make_rotary_embedding",
            return_value=source,
        ) as factory:
            capture_rope_state(owner, owner.config)
        factory.assert_called_once_with(owner.config, None)
        torch.testing.assert_close(owner._export_rotary_emb.inv_freq, source.inv_freq)

    def test_a_fresh_copy_is_used_for_each_length_dependent_preparation(self):
        class DynamicRotary(_LlamaRotary):
            def forward(self, hidden, positions):
                self.inv_freq /= float(positions.shape[-1])
                return super().forward(hidden, positions)

        source = DynamicRotary()
        owner = _LlamaOwner(source=source)
        before = prepare_llama_static_inputs(owner, 4)
        prepare_llama_static_inputs(owner, 16)
        after = prepare_llama_static_inputs(owner, 4)
        torch.testing.assert_close(before.cos, after.cos, atol=0, rtol=0)
        torch.testing.assert_close(before.sin, after.sin, atol=0, rtol=0)
        self.assertEqual(owner._export_rotary_emb.calls, 0)
        self.assertEqual(source.calls, 0)

    def test_rebinding_removes_obsolete_tables_and_frequency_fallbacks(self):
        owner = _LlamaOwner()
        owner.register_buffer("rope_cos_template", torch.ones(1, 8, 4))
        owner.register_buffer("rope_sin_template", torch.zeros(1, 8, 4))
        owner.register_buffer("_export_rope_inv_freq", torch.ones(2))
        owner._export_rope_attention_scaling = 1.0
        capture_rope_state(owner, owner.config, _LlamaRotary())
        for name in (
            "rope_cos_template",
            "rope_sin_template",
            "_export_rope_inv_freq",
            "_export_rope_attention_scaling",
        ):
            self.assertFalse(hasattr(owner, name), name)

    def test_no_native_context_allocation(self):
        owner = _LlamaOwner(capacity=1048576)
        self.assertEqual(sum(t.numel() for t in owner.buffers()), 2)
        actual = prepare_llama_static_inputs(owner, 8)
        self.assertEqual(sum(t.numel() for t in actual.buffers()), 128)

    def test_slice_boundaries(self):
        inputs = prepare_llama_static_inputs(_LlamaOwner(), 8)
        cos, sin = inputs.rope(7, 1, device="cpu", dtype=torch.float32)
        self.assertEqual(cos.shape, sin.shape)
        self.assertEqual(inputs.mask(1, 8, 7, device="cpu").shape, (1, 1, 8))
        for start, length in ((8, 1), (-1, 1), (0, 0)):
            with self.assertRaises(ValueError):
                inputs.rope(start, length, device="cpu", dtype=torch.float32)
        with self.assertRaises(ValueError):
            inputs.mask(1, 9, 0, device="cpu")

    def test_prefill_and_decode_capacity_boundaries_at_two_profiles(self):
        for capacity in (16, 32):
            inputs = prepare_llama_static_inputs(
                _LlamaOwner(capacity=capacity), capacity
            )
            for valid in (1, capacity // 2, capacity - 1, capacity):
                with self.subTest(capacity=capacity, valid=valid):
                    cos, sin = inputs.rope(0, valid, device="cpu", dtype=torch.float32)
                    self.assertEqual(cos.shape, (1, valid, 4))
                    self.assertEqual(sin.shape, cos.shape)
                    mask = inputs.mask(valid, valid, 0, device="cpu")
                    expected = torch.full((valid, valid), -120.0).triu(1)
                    torch.testing.assert_close(mask[0], expected, atol=0, rtol=0)
            for past in (0, capacity // 2, capacity - 1):
                with self.subTest(capacity=capacity, past=past):
                    cos, _ = inputs.rope(past, 1, device="cpu", dtype=torch.float32)
                    self.assertEqual(cos.shape, (1, 1, 4))
                    mask = inputs.mask(1, past + 1, past, device="cpu")
                    torch.testing.assert_close(
                        mask, torch.zeros_like(mask), atol=0, rtol=0
                    )
            with self.assertRaisesRegex(ValueError, "exceeds"):
                inputs.rope(capacity, 1, device="cpu", dtype=torch.float32)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                inputs.mask(capacity + 1, capacity + 1, 0, device="cpu")

    def test_torch_export_graph_contains_no_table_generation(self):
        class Consumer(nn.Module):
            inputs: LlamaStaticInputs

            def __init__(self, inputs: LlamaStaticInputs):
                super().__init__()
                self.inputs = inputs

            def forward(self, hidden):
                cos, sin = self.inputs.rope(
                    0, 4, device=hidden.device, dtype=hidden.dtype
                )
                mask = self.inputs.mask(4, 4, 0, device=hidden.device)
                return hidden * cos + sin + mask

        consumer = Consumer(prepare_llama_static_inputs(_LlamaOwner(), 4))
        hidden = torch.randn(1, 4, 4)
        exported = torch.export.export(consumer, (hidden,), strict=False)
        torch.testing.assert_close(
            exported.module()(hidden), consumer(hidden), atol=0, rtol=0
        )
        targets = [
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        ]
        for forbidden in (
            "aten.sin.",
            "aten.cos.",
            "aten.arange.",
            "aten.triu.",
            "aten.outer.",
        ):
            self.assertFalse(any(forbidden in target for target in targets), targets)


class TestQwenStaticInputs(unittest.TestCase):
    def test_snapshot_not_live_rotary_is_executed(self):
        owner = _QwenOwner()
        before = prepare_qwen3_vl_static_text_inputs(owner, 8)
        owner.rotary_emb.inv_freq.mul_(100)
        owner.rotary_emb.calls = 10
        after = prepare_qwen3_vl_static_text_inputs(owner, 8)
        torch.testing.assert_close(before.cos, after.cos, atol=0, rtol=0)
        torch.testing.assert_close(before.sin, after.sin, atol=0, rtol=0)
        self.assertEqual(owner.rotary_emb.calls, 10)
        self.assertEqual(owner._export_rotary_emb.calls, 0)
        self.assertEqual(before.position_ids.shape, (3, 1, 8))

    def test_legacy_tables_and_bounds(self):
        owner = _QwenOwner()
        del owner._export_rotary_emb
        with self.assertRaisesRegex(ValueError, "requires valid legacy rotary tables"):
            prepare_qwen3_vl_static_text_inputs(owner, 4)
        owner.position_ids_template = torch.arange(8).view(1, 1, -1).expand(3, 1, -1)
        owner.cos_template = torch.full((1, 8, 4), 2.0)
        owner.sin_template = torch.full((1, 8, 4), 3.0)
        actual = prepare_qwen3_vl_static_text_inputs(owner, 4)
        self.assertEqual(actual.cos.shape, (1, 4, 4))
        self.assertEqual(float(actual.cos[0, 0, 0]), 2.0)
        with self.assertRaisesRegex(ValueError, "cover"):
            prepare_qwen3_vl_static_text_inputs(owner, 9)

    def test_snapshot_storage_does_not_scale_with_native_capacity(self):
        owner = _QwenOwner()
        self.assertEqual(sum(t.numel() for t in owner.buffers()), 4)
        with prepared_static_export(owner, 8):
            assert owner._static_export_inputs is not None
            self.assertEqual(owner._static_export_inputs.attention_mask.shape[-1], 8)
        self.assertFalse(hasattr(owner, "_static_export_inputs"))


class TestDecoderExportBoundaries(unittest.TestCase):
    def test_llama_and_qwen_adapters_do_not_change_eager_return_policy(self):
        from tico.quantization.wrapq.wrappers.llama.export_adapters import (
            LlamaDecoderLayerDecodeExportAdapter,
            LlamaDecoderLayerPrefillExportAdapter,
        )
        from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
            Qwen3VLTextDecoderLayerDecodeExportAdapter,
            Qwen3VLTextDecoderLayerPrefillExportAdapter,
        )

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.return_type = "tensor"

            def forward(
                self, hidden_states, *, return_type=None, use_cache=False, **kwargs
            ):
                hidden = hidden_states + 1
                key = hidden.unsqueeze(1)
                return (hidden, (key, key)) if return_type == "tuple" else hidden

        hidden = torch.zeros(1, 1, 4)
        mask = torch.zeros(1, 1, 4)
        positions = (torch.ones(1, 1, 4), torch.zeros(1, 1, 4))
        past = (torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4))
        for cls in (
            LlamaDecoderLayerPrefillExportAdapter,
            LlamaDecoderLayerDecodeExportAdapter,
            Qwen3VLTextDecoderLayerPrefillExportAdapter,
            Qwen3VLTextDecoderLayerDecodeExportAdapter,
        ):
            with self.subTest(adapter=cls.__name__):
                layer = Layer()
                adapter = cls(layer, return_kv=True)
                self.assertEqual(layer.return_type, "tensor")
                kwargs = {"attention_mask": mask, "position_embeddings": positions}
                if "DecodeExportAdapter" in cls.__name__:
                    key = (
                        "past_key_values"
                        if cls.__name__.startswith("Qwen")
                        else "past_key_value"
                    )
                    kwargs[key] = past
                output = adapter(hidden, **kwargs)
                self.assertEqual(len(output), 3)
                self.assertEqual(layer.return_type, "tensor")
                self.assertIsInstance(layer(hidden), torch.Tensor)
                program = torch.export.export(
                    adapter, (hidden,), kwargs=kwargs, strict=False
                )
                for actual, expected in zip(program.module()(hidden, **kwargs), output):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, "explicit attention_mask"):
                    adapter(hidden, **{**kwargs, "attention_mask": None})

    def test_gemma4_preserves_shared_kv_and_boolean_mask_contract(self):
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4TextDecoderLayerDecodeExportAdapter,
        )

        class Consumer(nn.Module):
            def forward(
                self, hidden_states, *, shared_key_value, static_export, **kwargs
            ):
                if not static_export:
                    raise AssertionError("Static path was not selected")
                return hidden_states + shared_key_value[0][:, 0, -1:]

        hidden = torch.zeros(1, 1, 4)
        shared = (torch.ones(1, 1, 8, 4), torch.ones(1, 1, 8, 4))
        adapter = Gemma4TextDecoderLayerDecodeExportAdapter(Consumer(), return_kv=True)
        kwargs = {
            "attention_mask": torch.ones(1, 1, 8, dtype=torch.bool),
            "position_embeddings": (torch.ones(1, 1, 4), torch.zeros(1, 1, 4)),
            "shared_key_value": shared,
        }
        output = adapter(hidden, **kwargs)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, hidden.shape)
        self.assertEqual(shared[0].shape[-2], 8)
        program = torch.export.export(adapter, (hidden,), kwargs=kwargs, strict=False)
        torch.testing.assert_close(
            program.module()(hidden, **kwargs), output, atol=0, rtol=0
        )

    def test_full_model_owner_suppresses_duplicate_layer_tables(self):
        owner = _LlamaOwner()
        owner.layers = nn.ModuleList([_LlamaOwner(), _LlamaOwner()])
        with prepared_static_export(owner, 4):
            self.assertTrue(hasattr(owner, "_static_export_inputs"))
            for layer in owner.layers:
                self.assertFalse(hasattr(layer, "_static_export_inputs"))


if __name__ == "__main__":
    unittest.main()

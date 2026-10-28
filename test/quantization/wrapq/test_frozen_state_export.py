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

"""Frozen-state persistence must preserve real Circle contracts and execution."""

import copy
import io
import unittest
from typing import cast

import numpy as np
import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.version import has_transformers_for


def _export(module, args, kwargs=None):
    """Exercise the public conversion path with its default Circle O1 enabled."""
    with torch.no_grad():
        converted = tico.convert(module.eval(), args, kwargs=kwargs, strict=False)
    document = CircleDocument.from_bytes(converted.circle_binary)
    document.verify()
    return CircleReferenceRuntime(document)


def _tensor_contract(tensor):
    return tensor.shape, tensor.shape_signature, tensor.dtype, tensor.quantization


class TestFrozenStateCircleExport(unittest.TestCase):
    """Check exact ABI, constant payloads and numerics, independently of FX metadata."""

    @staticmethod
    def _linear(dtype):
        module = torch.nn.Linear(2, 2, bias=False).eval()
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[0.25, 0.5], [-0.25, 1.0]]))
        return prepare(
            module,
            PTQConfig(activation=affine(dtype), weight=affine(dtype)),
        )

    def _assert_same_artifact(self, left, right):
        """Buffer names/ordering may change; every serialized tensor contract may not."""
        self.assertEqual(left.program.inputs, right.program.inputs)
        self.assertEqual(left.program.outputs, right.program.outputs)
        self.assertEqual(
            [tensor.name for tensor in (*left.input_tensors, *left.output_tensors)],
            [tensor.name for tensor in (*right.input_tensors, *right.output_tensors)],
        )
        self.assertEqual(
            [_tensor_contract(tensor) for tensor in left.program.tensors],
            [_tensor_contract(tensor) for tensor in right.program.tensors],
        )
        self.assertEqual(
            [(op.name, op.inputs, op.outputs) for op in left.program.operators],
            [(op.name, op.inputs, op.outputs) for op in right.program.operators],
        )
        for source, target in zip(left.program.tensors, right.program.tensors):
            self.assertEqual(source.is_constant, target.is_constant)
            if source.is_constant:
                np.testing.assert_array_equal(source.constant, target.constant)

    def test_missing_calibration_cannot_create_an_artifact(self):
        """Regression: main used to emit a valid UINT8 FC with two floor scales."""
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                prepared = self._linear(DType.uint(8))
                with self.assertRaisesRegex(RuntimeError, "act_in|calibrat|statistics"):
                    convert(prepared, inplace=inplace)
                self.assertTrue(hasattr(prepared, "tico_quantizer"))

    def test_linear_state_restore_preserves_grid_constants_and_reference_output(self):
        sample = torch.tensor([[0.125, 0.375]])
        # These are the serialized baseline grids for the deterministic calibration
        # below, including the signed activation and asymmetric weight distinction.
        expected = {
            "uint8": ((0.007843137718737125, 127), (0.007352941203862429, 85)),
            "int16": ((3.0518509447574615e-05, 0), (3.814813680946827e-05, 0)),
        }
        for dtype in (DType.uint(8), DType.int(16)):
            with self.subTest(dtype=dtype):
                prepared = self._linear(dtype)
                prepared(torch.tensor([[-1.0, 1.0], [0.5, -0.5]]))
                prepared(sample)
                source = convert(prepared).eval()
                before = _export(source, (sample,))
                self.assertEqual(
                    [op.name for op in before.program.operators], ["FULLY_CONNECTED"]
                )
                name = "int16" if dtype.signed else "uint8"
                for tensor, grid in zip(
                    (*before.input_tensors, *before.output_tensors), expected[name]
                ):
                    self.assertEqual(tensor.dtype, np.dtype(name))
                    self.assertEqual(tensor.shape, (1, 2))
                    self.assertEqual(tensor.quantization.scale, (grid[0],))
                    self.assertEqual(tensor.quantization.zero_point, (grid[1],))
                weight = next(t for t in before.program.tensors if t.is_constant)
                self.assertEqual(weight.shape, (2, 2))
                self.assertEqual(weight.dtype, np.dtype(name))
                self.assertEqual(len(weight.quantization.scale), 2)
                self.assertEqual(weight.quantization.quantized_dimension, 0)

                restored = self._linear(dtype)
                restored.load_state_dict(copy.deepcopy(source.state_dict()))
                self.assertIs(restored.wrapped._mode, Mode.QUANT)
                after = _export(restored, (sample,))
                self._assert_same_artifact(before, after)
                original = before.run(
                    [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs[0]
                reloaded = after.run(
                    [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs[0]
                np.testing.assert_array_equal(original, reloaded)
                np.testing.assert_array_equal(original, source(sample).detach().numpy())
                np.testing.assert_array_equal(
                    original, restored(sample).detach().numpy()
                )

                stream = io.BytesIO()
                torch.save(source, stream)
                stream.seek(0)
                whole = torch.load(stream, weights_only=False)
                self._assert_same_artifact(before, _export(whole, (sample,)))

    def test_imported_weight_grids_survive_subtype_reload_and_circle_export(self):
        """GPTQ-style imported grids cannot be reconstructed from weight ranges."""
        torch.manual_seed(71)
        sample = torch.tensor([[0.125, 0.375]])
        for observer_type in (MinMaxObserver, EMAObserver, PercentileObserver):
            with self.subTest(observer=observer_type.__name__):
                per_tensor = observer_type is PercentileObserver
                scales = torch.tensor(0.3) if per_tensor else torch.tensor([0.3, 0.2])
                zero_points = torch.tensor(
                    7 if per_tensor else [7, 9], dtype=torch.int32
                )
                weight_spec = affine(
                    DType.uint(8),
                    observer=observer_type,
                    qscheme=(
                        QScheme.PER_TENSOR_ASYMM
                        if per_tensor
                        else QScheme.PER_CHANNEL_ASYMM
                    ),
                    channel_axis=None if per_tensor else 0,
                )
                config = PTQConfig(
                    activation=affine(DType.uint(8)),
                    weight=weight_spec,
                    overrides={"weight": weight_spec},
                )
                source = prepare(torch.nn.Linear(2, 2, bias=False).eval(), config)
                source.wrapped.obs_weight.load_qparams(scales, zero_points, lock=True)
                source(torch.tensor([[-1.0, 1.0], [0.5, -0.5]]))
                source = convert(source).eval()
                source_state = source.wrapped.obs_weight.require_frozen()
                self.assertEqual(source_state.provenance, "imported")
                self.assertTrue(source_state.locked)
                before = _export(source, (sample,))
                restored = prepare(torch.nn.Linear(2, 2, bias=False).eval(), config)
                restored.load_state_dict(copy.deepcopy(source.state_dict()))
                target_state = restored.wrapped.obs_weight.require_frozen()
                torch.testing.assert_close(target_state.scale, scales, rtol=0, atol=0)
                torch.testing.assert_close(
                    target_state.zero_point, zero_points, rtol=0, atol=0
                )
                self.assertTrue(target_state.locked)
                after = _export(restored, (sample,))
                self._assert_same_artifact(before, after)
                weight = next(t for t in after.program.tensors if t.is_constant)
                self.assertEqual(
                    weight.quantization.scale,
                    tuple(float(x) for x in scales.reshape(-1)),
                )
                self.assertEqual(
                    weight.quantization.zero_point, (7,) if per_tensor else (7, 9)
                )
                np.testing.assert_array_equal(
                    before.run(
                        [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                    ).outputs[0],
                    after.run(
                        [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                    ).outputs[0],
                )

    def _assert_snapshots_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for name in left:
            source, target = left[name], right[name]
            self.assertEqual(source is None, target is None, name)
            if source is None:
                continue
            self.assertEqual(source.policy, target.policy, name)
            self.assertEqual(source.provenance, target.provenance, name)
            self.assertEqual(source.locked, target.locked, name)
            for field in ("scale", "zero_point"):
                a, b = getattr(source, field), getattr(target, field)
                if a is None:
                    self.assertIsNone(b)
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def _assert_reference_parity(self, module, args, runtime):
        leaves, structure = torch.utils._pytree.tree_flatten(args)
        self.assertEqual(len(leaves), len(runtime.input_tensors))
        arrays = [
            (
                fake_quantize_array(value.detach().numpy(), tensor)
                if tensor.is_quantized_integer
                else value.detach().numpy().copy()
            )
            for value, tensor in zip(leaves, runtime.input_tensors)
        ]
        encoded_args = torch.utils._pytree.tree_unflatten(
            [torch.from_numpy(value.copy()) for value in arrays], structure
        )
        with torch.no_grad():
            expected = module(*encoded_args)
        outputs = runtime.run(arrays, mode=ExecutionMode.FAKE_QUANTIZE).outputs
        self.assertEqual(len(outputs), len(expected))
        for index, (actual, wanted) in enumerate(zip(outputs, expected)):
            self.assertTrue(np.isfinite(actual).all())
            error = np.abs(actual - wanted.detach().numpy())
            print(
                f"decoder output{index}: max_abs={float(error.max())}, mean_abs={float(error.mean())}"
            )
            if index == 0:
                # The baseline has four half-tie hidden differences in prefill:
                # PyTorch FQ rounds to even; Circle rounds away from zero. Bound
                # this existing reference error by one serialized output step,
                # as in CircleExecutorTest. Cache deltas remain exact, and the
                # restored Circle outputs are compared exactly below.
                output_step = runtime.output_tensors[index].quantization.scale[0]
                np.testing.assert_allclose(
                    actual, wanted.detach().numpy(), rtol=0, atol=output_step
                )
            else:
                np.testing.assert_array_equal(actual, wanted.detach().numpy())
        return outputs

    @staticmethod
    def _snapshots(module):
        return {
            name: observer.frozen_state
            for name, observer in module.named_modules()
            if isinstance(observer, ObserverBase)
        }

    def _attention_roundtrip(self, family):
        from test.quantization.wrapq.wrappers.llama.test_kv_cache_circle import (
            TestLlamaKVCacheCircle,
        )
        from test.quantization.wrapq.wrappers.qwen_vl.test_kv_cache_circle import (
            TestQwenKVCacheCircle,
        )

        fixture_class = (
            TestLlamaKVCacheCircle if family == "llama" else TestQwenKVCacheCircle
        )
        source_fixture = fixture_class()
        source_fixture.setUp()
        source_fixture._make_attention(DType.uint(8))
        source = source_fixture.attention
        snapshots = self._snapshots(source)
        restored_fixture = fixture_class()
        restored_fixture.setUp()
        restored_fixture._make_attention(DType.uint(8))
        restored = restored_fixture.attention
        restored.enable_calibration()
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
        self._assert_snapshots_equal(snapshots, self._snapshots(restored))

        prefill_args = source_fixture._prefill_args()
        source_prefill = source.as_export_module("prefill", return_kv=True)
        restored_prefill = restored.as_export_module("prefill", return_kv=True)
        before = _export(source_prefill, prefill_args)
        after = _export(restored_prefill, prefill_args)
        self._assert_same_artifact(before, after)
        outputs = source_fixture._assert_parity(
            source_prefill, prefill_args, before, label=family + " frozen prefill"
        )
        restored_fixture._assert_parity(
            restored_prefill, prefill_args, after, label=family + " restored prefill"
        )
        past = tuple(torch.from_numpy(value.copy()) for value in outputs[-2:])
        cosine, sine = source_fixture._rope(3)
        decode_args = (
            source_fixture._hidden(1) + 0.31,
            (cosine[:, -1:], sine[:, -1:]),
            source_fixture._mask(1, 3),
            past,
        )
        source_decode = source.as_export_module("decode", return_kv=True)
        restored_decode = restored.as_export_module("decode", return_kv=True)
        before_decode = _export(source_decode, decode_args)
        after_decode = _export(restored_decode, decode_args)
        self._assert_same_artifact(before_decode, after_decode)
        source_fixture._assert_decode_topology(before_decode)
        source_fixture._assert_parity(
            source_decode, decode_args, before_decode, label=family + " frozen decode"
        )
        restored_fixture._assert_parity(
            restored_decode,
            decode_args,
            after_decode,
            label=family + " restored decode",
        )
        for output, input_index in zip(
            before.output_tensors[-2:], before_decode.program.inputs[-2:]
        ):
            self.assertEqual(
                _tensor_contract(output),
                _tensor_contract(before_decode.program.tensor(input_index)),
            )
        self._assert_snapshots_equal(snapshots, self._snapshots(source))
        self._assert_snapshots_equal(snapshots, self._snapshots(restored))

    @unittest.skipUnless(
        has_transformers_for("llama"), "Llama Transformers unavailable"
    )
    def test_llama_prefill_decode_survive_frozen_state_restore(self):
        self._attention_roundtrip("llama")

    @unittest.skipUnless(
        has_transformers_for("qwen3-vl"), "Qwen3-VL Transformers unavailable"
    )
    def test_qwen_prefill_decode_survive_frozen_state_restore(self):
        self._attention_roundtrip("qwen")

    @unittest.skipUnless(
        has_transformers_for("llama"), "Llama Transformers unavailable"
    )
    def test_decoder_layer_prefill_and_chained_decode_survive_restore(self):
        from tico.quantization.wrapq.wrappers.llama.quant_attention import (
            QuantLlamaAttention,
        )
        from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
            QuantLlamaDecoderLayer,
        )
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        torch.manual_seed(20261027)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            num_hidden_layers=1,
            max_position_embeddings=8,
            attention_bias=False,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        qconfig = PTQConfig(model_args={"profile": "npu_export"})
        source = QuantLlamaDecoderLayer(
            LlamaDecoderLayer(config, layer_idx=0).eval(), qcfg=qconfig, layer_idx=0
        ).eval()
        source.enable_calibration()
        hidden = torch.linspace(-0.75, 0.75, 24).reshape(1, 3, 8)
        rope = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))
        prefill_args = (hidden, torch.zeros(1, 3, 3), rope)
        prefill_module = source.as_export_module("prefill", return_kv=True)
        with torch.no_grad():
            calibrated = prefill_module(*prefill_args)
            source.as_export_module("decode", return_kv=True)(
                hidden[:, :1],
                torch.zeros(1, 1, 4),
                (rope[0][:, :1], rope[1][:, :1]),
                calibrated[-2:],
            )
        for observer in source.modules():
            if isinstance(observer, AffineObserverBase):
                observer.load_qparams(
                    torch.full_like(observer.min_val, 0.03125),
                    torch.full_like(observer.min_val, 128, dtype=torch.int32),
                )
        attention = cast(QuantLlamaAttention, source.self_attn.wrapped)
        for storage_observer in (attention.obs_new_k, attention.obs_new_v):
            cast(AffineObserverBase, storage_observer).load_qparams(
                torch.tensor(0.09375), torch.tensor(113, dtype=torch.int32)
            )
        for compute_observer in (
            attention.obs_present_key,
            attention.obs_present_value,
        ):
            cast(AffineObserverBase, compute_observer).load_qparams(
                torch.tensor(0.21875), torch.tensor(91, dtype=torch.int32)
            )
        source.freeze_qparams()
        frozen = self._snapshots(source)
        restored = QuantLlamaDecoderLayer(
            LlamaDecoderLayer(config, layer_idx=0).eval(), qcfg=qconfig, layer_idx=0
        ).eval()
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
        self._assert_snapshots_equal(frozen, self._snapshots(restored))
        before = _export(prefill_module, prefill_args)
        restored_prefill = restored.as_export_module("prefill", return_kv=True)
        after = _export(restored_prefill, prefill_args)
        self._assert_same_artifact(before, after)
        outputs = self._assert_reference_parity(prefill_module, prefill_args, before)
        restored_outputs = self._assert_reference_parity(
            restored_prefill, prefill_args, after
        )
        for original, reloaded in zip(outputs, restored_outputs):
            np.testing.assert_array_equal(original, reloaded)
        past = tuple(torch.from_numpy(value.copy()) for value in outputs[-2:])
        decode_args = (
            hidden[:, :1],
            torch.zeros(1, 1, 4),
            (rope[0][:, :1], rope[1][:, :1]),
            past,
        )
        source_decode = source.as_export_module("decode", return_kv=True)
        restored_decode = restored.as_export_module("decode", return_kv=True)
        before_decode = _export(source_decode, decode_args)
        after_decode = _export(restored_decode, decode_args)
        self._assert_same_artifact(before_decode, after_decode)
        original_decode = self._assert_reference_parity(
            source_decode, decode_args, before_decode
        )
        restored_decode_outputs = self._assert_reference_parity(
            restored_decode, decode_args, after_decode
        )
        for original, reloaded in zip(original_decode, restored_decode_outputs):
            np.testing.assert_array_equal(original, reloaded)
        for produced, consumed in zip(
            before.output_tensors[-2:], before_decode.input_tensors[-2:]
        ):
            self.assertEqual(_tensor_contract(produced), _tensor_contract(consumed))
            self.assertEqual(produced.quantization.scale, (0.09375,))
            self.assertEqual(produced.quantization.zero_point, (113,))
        self.assertNotIn(
            "DEQUANTIZE", [op.name for op in before_decode.program.operators]
        )
        self.assertTrue(
            any(op.name == "QUANTIZE" for op in before_decode.program.operators)
        )
        self._assert_snapshots_equal(frozen, self._snapshots(source))
        self._assert_snapshots_equal(frozen, self._snapshots(restored))

    @unittest.skipUnless(
        has_transformers_for("gemma4"), "Gemma4 Transformers unavailable"
    )
    def test_gemma_prefill_decode_survive_frozen_state_restore(self):
        from test.quantization.wrapq.wrappers.gemma4.test_kv_cache_circle import (
            _args,
            TestGemma4KVCacheCircle as Fixture,
        )

        fixture = Fixture()
        fixture.setUp()
        source = fixture._attention()
        fixture._freeze(source)
        snapshots = self._snapshots(source)
        restored = fixture._attention()
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
        self._assert_snapshots_equal(snapshots, self._snapshots(restored))
        prefill_args = _args(2, 2)
        source_prefill = source.as_export_module("prefill")
        restored_prefill = restored.as_export_module("prefill")
        before = _export(source_prefill, prefill_args)
        after = _export(restored_prefill, prefill_args)
        self._assert_same_artifact(before, after)
        outputs = fixture._assert_parity(
            source_prefill, prefill_args, before, label="Gemma frozen prefill"
        )
        fixture._assert_parity(
            restored_prefill, prefill_args, after, label="Gemma restored prefill"
        )
        past = tuple(torch.from_numpy(value.copy()) for value in outputs[-2:])
        decode_args = _args(1, 3)
        kwargs = {"past_key_value": past}
        source_decode = source.as_export_module("decode")
        restored_decode = restored.as_export_module("decode")
        before_decode = _export(source_decode, decode_args, kwargs)
        after_decode = _export(restored_decode, decode_args, kwargs)
        self._assert_same_artifact(before_decode, after_decode)
        fixture._assert_storage_conversion(
            before_decode, source.obs_new_k, source.obs_present_key
        )
        for module, runtime in (
            (source_decode, before_decode),
            (restored_decode, after_decode),
        ):
            fixture._assert_parity(
                module, decode_args, runtime, kwargs=kwargs, label="Gemma frozen decode"
            )
        for output, input_index in zip(
            before.output_tensors[-2:], before_decode.program.inputs[-2:]
        ):
            self.assertEqual(
                _tensor_contract(output),
                _tensor_contract(before_decode.program.tensor(input_index)),
            )
        self._assert_snapshots_equal(snapshots, self._snapshots(source))
        self._assert_snapshots_equal(snapshots, self._snapshots(restored))


if __name__ == "__main__":
    unittest.main()

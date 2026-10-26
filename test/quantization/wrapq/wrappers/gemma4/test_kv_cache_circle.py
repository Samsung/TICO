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

"""Serialized Gemma4 cache storage/compute contracts and CPU reference parity."""

import unittest

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)

try:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    _HAS_GEMMA4 = True
except ImportError:
    _HAS_GEMMA4 = False


def _config(layer_type="full_attention"):
    """Keep a full-attention final layer, as required by Transformers."""
    sliding = layer_type == "sliding_attention"
    layer_types = (
        ["sliding_attention", "full_attention"] * 2
        if sliding
        else ["full_attention", "full_attention"]
    )
    config = Gemma4TextConfig.from_dict(
        dict(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=len(layer_types),
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=4,
            global_head_dim=4,
            max_position_embeddings=32,
            layer_types=layer_types,
            sliding_window=4,
            hidden_size_per_layer_input=0,
            attention_k_eq_v=False,
            num_kv_shared_layers=2 if sliding else 1,
            enable_moe_block=False,
            attention_dropout=0.0,
            allow_global_per_layer_attribute_access=True,
        )
    )
    config._attn_implementation = "eager"
    return config


def _args(query_length, key_length):
    """Avoid random inputs and keep the static capacity at 32 tokens."""
    hidden = torch.linspace(0.25, 1.0, query_length * 8).reshape(1, query_length, 8)
    return (
        hidden,
        torch.zeros(1, 1, query_length, key_length),
        (torch.ones(1, query_length, 4), torch.zeros(1, query_length, 4)),
    )


def _load(observer, scale, zero_point):
    """Install intentional frozen mismatch qparams for a boundary unit test."""
    observer.load_qparams(
        torch.full_like(observer.min_val, scale),
        torch.full_like(observer.min_val, zero_point, dtype=torch.int32),
    )


class _CacheBranchExport(torch.nn.Module):
    """Expose the actual wrapper cache branch without unrelated attention math."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, new_key, new_value, past_key, past_value):
        attention = self.attention
        delta = (
            attention._fq(new_key, attention.obs_new_k),
            attention._fq(new_value, attention.obs_new_v),
        )
        compute, _storage, _cache_object = attention._build_present_key_value(
            new_key_value=delta,
            current_key_value=(new_key, new_value),
            past_key_value=(past_key, past_value),
        )
        return *compute, *delta


@unittest.skipUnless(
    _HAS_GEMMA4, "required Transformers Gemma4 modules are not installed"
)
class TestGemma4KVCacheCircle(unittest.TestCase):
    """Export complete attention modules and inspect their serialized ABI."""

    def setUp(self):
        torch.manual_seed(317)

    @staticmethod
    def _attention(
        layer_index=0,
        *,
        signed_storage=False,
        compute_policy=None,
        layer_type="full_attention",
    ):
        storage_dtype = DType.int(16) if signed_storage else DType.uint(8)
        config = PTQConfig(
            overrides={
                "new_k": affine(storage_dtype),
                "new_v": affine(storage_dtype),
                "present_key": compute_policy or affine(DType.uint(8)),
                "present_value": compute_policy or affine(DType.uint(8)),
            },
            model_args={"text": {"max_seq": 32}},
        )
        return QuantGemma4TextAttention(
            Gemma4TextAttention(
                _config(layer_type),
                layer_idx=layer_index * 2
                if layer_type == "sliding_attention"
                else layer_index,
            ).eval(),
            qcfg=config,
        ).eval()

    @staticmethod
    def _freeze(attention, *, shared_key_value=None):
        """Collect real shapes, then intentionally select different frozen grids.

        Recipe calibration coverage belongs to the adapter integration test;
        this fixture exercises serialization of known, incompatible grids.
        """
        attention.enable_calibration()
        args = _args(32, 32)
        with torch.no_grad():
            attention(
                args[0],
                args[2],
                attention_mask=args[1],
                shared_key_value=shared_key_value,
                use_cache=True,
            )
        for observer in attention.modules():
            if isinstance(observer, AffineObserverBase):
                _load(observer, 0.03125, 128)
        for observer in (attention.obs_new_k, attention.obs_new_v):
            if observer.dtype == DType.int(16):
                _load(observer, 0.09375, 0)
            else:
                _load(observer, 0.09375, 113)
        for observer in (attention.obs_present_key, attention.obs_present_value):
            _load(observer, 0.21875, 0 if observer.dtype == DType.int(16) else 91)
        attention.freeze_qparams()

    @staticmethod
    def _convert(module, args, kwargs=None, *, optimize=True):
        with torch.no_grad():
            converted = tico.convert(
                module.eval(),
                args,
                kwargs=kwargs,
                strict=False,
                config=tico.CompileConfigV1(circle_optimize=optimize),
            )
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify()
        return CircleReferenceRuntime(document)

    def _assert_storage(self, tensor, observer):
        self.assertEqual(
            tensor.dtype,
            np.dtype(np.int16 if observer.dtype == DType.int(16) else np.uint8),
        )
        self.assertIsNotNone(tensor.quantization)
        scale, zero_point = observer.compute_qparams()
        self.assertEqual(tensor.quantization.scale, (float(scale),))
        self.assertEqual(tensor.quantization.zero_point, (int(zero_point),))
        self.assertEqual(tensor.quantization.quantized_dimension, 0)

    @staticmethod
    def _qparams(tensor):
        quantization = tensor.quantization
        if quantization is None or not quantization.scale:
            return None
        return quantization.scale, quantization.zero_point

    def _assert_storage_conversion(self, runtime, storage_observer, compute_observer):
        """The storage boundary reaches C through exactly one Quantize."""
        program = runtime.program
        compute_scale, compute_zp = compute_observer.compute_qparams()
        for index in program.inputs[-2:]:
            self._assert_storage(program.tensor(index), storage_observer)
            users = [op for op in program.operators if index in op.inputs]
            self.assertEqual(len(users), 1)
            self.assertEqual(users[0].name, "QUANTIZE")
            converted = program.tensor(users[0].outputs[0])
            self.assertEqual(
                self._qparams(converted),
                ((float(compute_scale),), (int(compute_zp),)),
            )
            self.assertEqual(converted.dtype, np.dtype(np.uint8))

    def _assert_cache_concats(self, runtime, *, key_length, compute_observer):
        program = runtime.program
        concats = [
            op
            for op in program.operators
            if op.name == "CONCATENATION"
            and program.tensor(op.outputs[0]).shape == (1, 1, key_length, 4)
        ]
        self.assertEqual(len(concats), 2)
        scale, zero_point = compute_observer.compute_qparams()
        expected = ((float(scale),), (int(zero_point),))
        for operator in concats:
            tensors = [program.tensor(i) for i in (*operator.inputs, *operator.outputs)]
            self.assertEqual({self._qparams(tensor) for tensor in tensors}, {expected})
            self.assertEqual(len({tensor.dtype for tensor in tensors}), 1)
        return concats

    def _assert_independent_current_branch(self, runtime, concats):
        """Current C and the storage delta share a source before storage rounding."""
        program = runtime.program
        producers = {
            output: operator
            for operator in program.operators
            for output in operator.outputs
        }
        for concat, delta_index in zip(concats, program.outputs[-2:]):
            compute_quantize = producers[concat.inputs[1]]
            storage_quantize = producers[delta_index]
            self.assertEqual(compute_quantize.name, "QUANTIZE")
            self.assertEqual(storage_quantize.name, "QUANTIZE")
            self.assertEqual(compute_quantize.inputs, storage_quantize.inputs)
            self.assertNotEqual(compute_quantize.outputs, storage_quantize.outputs)

    def _assert_no_identity_quantizes(self, runtime):
        for operator in runtime.program.operators:
            if operator.name != "QUANTIZE":
                continue
            source = runtime.program.tensor(operator.inputs[0])
            target = runtime.program.tensor(operator.outputs[0])
            self.assertNotEqual(
                (source.dtype, self._qparams(source)),
                (target.dtype, self._qparams(target)),
                "Redundant identity Quantize remains in serialized Circle",
            )

    def _assert_parity(
        self,
        module,
        args,
        runtime,
        *,
        kwargs=None,
        label,
        allow_hidden_rounding_tie=False,
    ):
        """Use the same dequantized ABI values in eager and Circle execution."""
        leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs or {}))
        self.assertEqual(len(leaves), len(runtime.program.inputs))
        encoded = []
        for value, index in zip(leaves, runtime.program.inputs):
            tensor = runtime.program.tensor(index)
            array = value.detach().numpy()
            if tensor.is_quantized_integer:
                array = fake_quantize_array(array, tensor)
            encoded.append(np.array(array, copy=True))
        eager_args, eager_kwargs = torch.utils._pytree.tree_unflatten(
            [torch.from_numpy(value.copy()) for value in encoded], spec
        )
        with torch.no_grad():
            eager = module(*eager_args, **eager_kwargs)
        expected, _ = torch.utils._pytree.tree_flatten(eager)
        result = runtime.run(encoded, mode=ExecutionMode.FAKE_QUANTIZE)
        self.assertEqual(len(result.outputs), len(expected))
        for index, (actual, wanted) in enumerate(zip(result.outputs, expected)):
            error = np.abs(actual - wanted.detach().numpy())
            metrics = {
                "max_abs": float(error.max()),
                "mean_abs": float(error.mean()),
                "differing": int(np.count_nonzero(error)),
            }
            print(f"{label} output{index}: {metrics}")
            if index == 0 and allow_hidden_rounding_tie and np.any(error):
                # This equal-grid UINT8 fixture reaches an attention reduction
                # half tie: Circle rounds away from zero, whereas PyTorch FQ
                # rounds to even. Only one hidden element may differ, by exactly
                # one output-grid step. Every cache value remains bit-exact.
                tensor = runtime.program.tensor(runtime.program.outputs[index])
                output_step = tensor.quantization.scale[0]
                self.assertEqual(metrics["differing"], 1, metrics)
                self.assertEqual(metrics["max_abs"], output_step, metrics)
            else:
                np.testing.assert_array_equal(
                    actual, wanted.detach().numpy(), err_msg=f"{label}: {metrics}"
                )
        return result.outputs

    def _exercise_attention(self, *, signed_storage=False, equal=False):
        dtype = DType.int(16) if signed_storage else DType.uint(8)
        attention = self._attention(
            signed_storage=signed_storage,
            compute_policy=affine(dtype) if equal else None,
        )
        self._freeze(attention)
        if equal:
            for observer in (
                attention.obs_present_key,
                attention.obs_present_value,
            ):
                _load(observer, 0.09375, 0 if signed_storage else 113)
        prefill_module = attention.as_export_module("prefill")
        decode_module = attention.as_export_module("decode")
        for optimize in (False, True):
            label = f"{dtype} S=C={equal} O1={optimize}"
            with self.subTest(dtype=dtype, equal=equal, optimize=optimize):
                prefill_args = _args(2, 2)
                prefill = self._convert(prefill_module, prefill_args, optimize=optimize)
                for index, observer in zip(
                    prefill.program.outputs[-2:], attention.cache_storage_observers()
                ):
                    self._assert_storage(prefill.program.tensor(index), observer)
                outputs = self._assert_parity(
                    prefill_module, prefill_args, prefill, label=label + " prefill"
                )
                # Feed serialized storage outputs into both decode graphs unchanged.
                past = tuple(torch.from_numpy(value.copy()) for value in outputs[-2:])
                for step in range(2):
                    key_length = past[0].shape[-2] + 1
                    args = _args(1, key_length)
                    kwargs = {"past_key_value": past}
                    decode = self._convert(
                        decode_module, args, kwargs, optimize=optimize
                    )
                    for input_index, output_index, observer in zip(
                        decode.program.inputs[-2:],
                        decode.program.outputs[-2:],
                        attention.cache_storage_observers(),
                    ):
                        self._assert_storage(
                            decode.program.tensor(input_index), observer
                        )
                        self._assert_storage(
                            decode.program.tensor(output_index), observer
                        )
                    concats = self._assert_cache_concats(
                        decode,
                        key_length=key_length,
                        compute_observer=attention.obs_present_key,
                    )
                    if not equal:
                        self._assert_storage_conversion(
                            decode, attention.obs_new_k, attention.obs_present_key
                        )
                        self._assert_independent_current_branch(decode, concats)
                    self._assert_no_identity_quantizes(decode)
                    outputs = self._assert_parity(
                        decode_module,
                        args,
                        decode,
                        kwargs=kwargs,
                        label=label + f" decode{step + 1}",
                        allow_hidden_rounding_tie=equal and not signed_storage,
                    )
                    past = tuple(
                        torch.cat((previous, torch.from_numpy(delta.copy())), dim=-2)
                        for previous, delta in zip(past, outputs[-2:])
                    )

    def test_uint8_prefill_and_two_decodes_have_independent_storage_and_compute(self):
        self._exercise_attention()

    def test_int16_prefill_and_two_decodes_have_independent_storage_and_compute(self):
        self._exercise_attention(signed_storage=True)

    def test_equal_domains_remove_identity_quantizes(self):
        for signed_storage in (False, True):
            self._exercise_attention(signed_storage=signed_storage, equal=True)

    def test_raw_current_branch_avoids_storage_double_rounding(self):
        """1/8 rounds directly to 7/32, but via S=3/32 it rounds to zero."""
        for signed_storage in (False, True):
            with self.subTest(signed_storage=signed_storage):
                attention = self._attention(signed_storage=signed_storage)
                self._freeze(attention)
                module = _CacheBranchExport(attention).eval()
                past_key = torch.tensor([-3, -1, 1, 3, -5, 5, -7, 7]).reshape(
                    1, 1, 2, 4
                )
                past_key = past_key.float() * 0.09375
                past_value = -past_key
                current = torch.full((1, 1, 1, 4), 0.125)
                direct = torch.full_like(current, 0.21875)
                stored = torch.full_like(current, 0.09375)
                double_rounded = attention.obs_present_key.fake_quant(stored)
                self.assertFalse(torch.equal(direct, double_rounded))
                for step in range(2):
                    args = (current, -current, past_key, past_value)
                    originals = tuple(value.clone() for value in args)
                    with torch.no_grad():
                        eager = module(*args)
                    expected = tuple(
                        torch.cat((torch.round(past / 0.21875) * 0.21875, new), dim=-2)
                        for past, new in ((past_key, direct), (past_value, -direct))
                    ) + (stored, -stored)
                    for actual, wanted in zip(eager, expected):
                        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                    runtime = self._convert(module, args)
                    self._assert_storage_conversion(
                        runtime, attention.obs_new_k, attention.obs_present_key
                    )
                    concats = self._assert_cache_concats(
                        runtime,
                        key_length=past_key.shape[-2] + 1,
                        compute_observer=attention.obs_present_key,
                    )
                    self._assert_independent_current_branch(runtime, concats)
                    self._assert_parity(
                        module,
                        args,
                        runtime,
                        label=f"raw current INT16={signed_storage} decode{step + 1}",
                    )
                    for actual, original in zip(args, originals):
                        torch.testing.assert_close(actual, original, rtol=0, atol=0)
                    past_key = torch.cat((past_key, eager[2]), dim=-2)
                    past_value = torch.cat((past_value, eager[3]), dim=-2)

    def test_per_channel_compute_export_is_rejected(self):
        attention = self._attention(
            compute_policy=affine(
                DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=1
            )
        )
        with self.assertRaisesRegex(ValueError, "per-tensor affine"):
            self._freeze(attention)
            attention.as_export_module("decode")

    def test_shared_consumer_prefill_and_decode_use_producer_storage(self):
        for layer_type in ("full_attention", "sliding_attention"):
            for signed_storage in (False, True):
                with self.subTest(layer_type=layer_type, signed_storage=signed_storage):
                    producer = self._attention(
                        signed_storage=signed_storage, layer_type=layer_type
                    )
                    self._freeze(producer)
                    consumer = self._attention(
                        layer_index=1,
                        signed_storage=not signed_storage,
                        layer_type=layer_type,
                    )
                    consumer.bind_cache_source(producer)
                    shared_key = torch.linspace(-1.0, 1.0, 32 * 4).reshape(1, 1, 32, 4)
                    self._freeze(consumer, shared_key_value=(shared_key, -shared_key))
                    shared = (shared_key[..., :3, :], -shared_key[..., :3, :])
                    for mode, query_length in (("prefill", 3), ("decode", 1)):
                        module = consumer.as_export_module(mode, cache_source=producer)
                        args = _args(query_length, 3)
                        kwargs = {"shared_key_value": shared}
                        runtime = self._convert(module, args, kwargs)
                        self.assertEqual(len(runtime.program.outputs), 1)
                        self._assert_storage_conversion(
                            runtime, producer.obs_new_k, consumer.obs_present_key
                        )
                        self._assert_parity(
                            module,
                            args,
                            runtime,
                            kwargs=kwargs,
                            label=f"shared {layer_type} INT16={signed_storage} {mode}",
                        )

    def test_external_float_input_has_explicit_compute_quantize(self):
        consumer = self._attention(layer_index=1)
        consumer.bind_external_float_cache()
        shared_key = torch.linspace(-0.875, 0.875, 32 * 4).reshape(1, 1, 32, 4)
        self._freeze(consumer, shared_key_value=(shared_key, -shared_key))
        shared = (shared_key[..., :3, :], -shared_key[..., :3, :])
        for mode, query_length in (("prefill", 3), ("decode", 1)):
            with self.subTest(mode=mode):
                module = consumer.as_export_module(mode)
                args = _args(query_length, 3)
                kwargs = {"shared_key_value": shared}
                runtime = self._convert(module, args, kwargs)
                self.assertEqual(len(runtime.program.outputs), 1)
                for index in runtime.program.inputs[-2:]:
                    tensor = runtime.program.tensor(index)
                    self.assertEqual(tensor.dtype, np.dtype(np.float32))
                    self.assertIsNone(self._qparams(tensor))
                    users = [
                        op for op in runtime.program.operators if index in op.inputs
                    ]
                    self.assertEqual(len(users), 1)
                    self.assertEqual(users[0].name, "QUANTIZE")
                    converted = runtime.program.tensor(users[0].outputs[0])
                    self.assertEqual(converted.dtype, np.dtype(np.uint8))
                    self.assertEqual(self._qparams(converted), ((0.21875,), (91,)))
                self._assert_parity(
                    module,
                    args,
                    runtime,
                    kwargs=kwargs,
                    label=f"external FP {mode}",
                )


if __name__ == "__main__":
    unittest.main()

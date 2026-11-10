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
from tico.circle.runtime import CircleReferenceRuntime
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme

from test.support.quantization.gemma_kv_cache import (
    args as _args,
    assert_gemma_parity,
    assert_storage,
    assert_storage_conversion,
    freeze_attention,
    load as _load,
    make_gemma_attention,
    tensor_qparams,
)

try:
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    _HAS_GEMMA4 = True
except ImportError:
    _HAS_GEMMA4 = False


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
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(317)

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
            self.assertEqual({tensor_qparams(tensor) for tensor in tensors}, {expected})
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
                (source.dtype, tensor_qparams(source)),
                (target.dtype, tensor_qparams(target)),
                "Redundant identity Quantize remains in serialized Circle",
            )

    def _exercise_attention(self, *, signed_storage=False, equal=False):
        dtype = DType.int(16) if signed_storage else DType.uint(8)
        attention = make_gemma_attention(
            signed_storage=signed_storage,
            compute_policy=affine(dtype) if equal else None,
        )
        freeze_attention(attention)
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
                    assert_storage(self, prefill.program.tensor(index), observer)
                outputs = assert_gemma_parity(
                    self,
                    prefill_module,
                    prefill_args,
                    prefill,
                    label=label + " prefill",
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
                        assert_storage(
                            self, decode.program.tensor(input_index), observer
                        )
                        assert_storage(
                            self, decode.program.tensor(output_index), observer
                        )
                    concats = self._assert_cache_concats(
                        decode,
                        key_length=key_length,
                        compute_observer=attention.obs_present_key,
                    )
                    if not equal:
                        assert_storage_conversion(
                            self, decode, attention.obs_new_k, attention.obs_present_key
                        )
                        self._assert_independent_current_branch(decode, concats)
                    self._assert_no_identity_quantizes(decode)
                    outputs = assert_gemma_parity(
                        self,
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

    def test_current_input_owns_storage_domain_and_requantizes_for_compute(self):
        """The first marker annotates the input ABI; S-to-C stays explicit."""
        for signed_storage in (False, True):
            with self.subTest(signed_storage=signed_storage):
                attention = make_gemma_attention(signed_storage=signed_storage)
                freeze_attention(attention)
                module = _CacheBranchExport(attention).eval()
                past_key = torch.tensor([-3, -1, 1, 3, -5, 5, -7, 7]).reshape(
                    1, 1, 2, 4
                )
                past_key = past_key.float() * 0.09375
                past_value = -past_key
                current = torch.full((1, 1, 1, 4), 0.125)
                direct = torch.full_like(current, 0.21875)
                stored = torch.full_like(current, 0.09375)
                # Eager wrapper semantics retain independent raw-to-C and raw-to-S
                # branches. This boundary-only export has no upstream current
                # domain, so its first marker instead defines the serialized S ABI.
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
                    assert_storage_conversion(
                        self, runtime, attention.obs_new_k, attention.obs_present_key
                    )
                    concats = self._assert_cache_concats(
                        runtime,
                        key_length=past_key.shape[-2] + 1,
                        compute_observer=attention.obs_present_key,
                    )
                    program = runtime.program
                    for input_index, delta_index, concat, observer in zip(
                        program.inputs[:2],
                        program.outputs[-2:],
                        concats,
                        attention.cache_storage_observers(),
                    ):
                        self.assertEqual(delta_index, input_index)
                        assert_storage(self, program.tensor(input_index), observer)
                        users = [
                            op for op in program.operators if input_index in op.inputs
                        ]
                        self.assertEqual(len(users), 1)
                        self.assertEqual(users[0].name, "QUANTIZE")
                        self.assertEqual(users[0].outputs, (concat.inputs[1],))
                    # This minimal graph contains only cache data tensors. There
                    # are no index, shape, or control tensors requiring exclusion.
                    for tensor in program.tensors:
                        self.assertIsNotNone(tensor.quantization)
                        self.assertNotEqual(tensor.dtype, np.dtype(np.float32))
                    self._assert_no_identity_quantizes(runtime)
                    # Compare eager and Circle with the encoded S input values,
                    # as required by the serialized ABI, rather than raw current.
                    assert_gemma_parity(
                        self,
                        module,
                        args,
                        runtime,
                        label=f"folded current S INT16={signed_storage} decode{step + 1}",
                    )
                    for actual, original in zip(args, originals):
                        torch.testing.assert_close(actual, original, rtol=0, atol=0)
                    past_key = torch.cat((past_key, eager[2]), dim=-2)
                    past_value = torch.cat((past_value, eager[3]), dim=-2)

    def test_per_channel_compute_export_is_rejected(self):
        attention = make_gemma_attention(
            compute_policy=affine(
                DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=1
            )
        )
        with self.assertRaisesRegex(ValueError, "per-tensor affine"):
            freeze_attention(attention)
            attention.as_export_module("decode")

    def test_shared_consumer_prefill_and_decode_use_producer_storage(self):
        for layer_type in ("full_attention", "sliding_attention"):
            for signed_storage in (False, True):
                with self.subTest(layer_type=layer_type, signed_storage=signed_storage):
                    producer = make_gemma_attention(
                        signed_storage=signed_storage, layer_type=layer_type
                    )
                    freeze_attention(producer)
                    consumer = make_gemma_attention(
                        layer_index=1,
                        signed_storage=not signed_storage,
                        layer_type=layer_type,
                    )
                    consumer.bind_cache_source(producer)
                    shared_key = torch.linspace(-1.0, 1.0, 32 * 4).reshape(1, 1, 32, 4)
                    freeze_attention(
                        consumer, shared_key_value=(shared_key, -shared_key)
                    )
                    shared = (shared_key[..., :3, :], -shared_key[..., :3, :])
                    for mode, query_length in (("prefill", 3), ("decode", 1)):
                        module = consumer.as_export_module(mode, cache_source=producer)
                        args = _args(query_length, 3)
                        kwargs = {"shared_key_value": shared}
                        runtime = self._convert(module, args, kwargs)
                        self.assertEqual(len(runtime.program.outputs), 1)
                        assert_storage_conversion(
                            self, runtime, producer.obs_new_k, consumer.obs_present_key
                        )
                        assert_gemma_parity(
                            self,
                            module,
                            args,
                            runtime,
                            kwargs=kwargs,
                            label=f"shared {layer_type} INT16={signed_storage} {mode}",
                        )

    def test_external_float_policy_uses_folded_compute_input_grid(self):
        """Ordinary folding places C on the serialized external-cache inputs."""
        consumer = make_gemma_attention(layer_index=1)
        consumer.bind_external_float_cache()
        shared_key = torch.linspace(-0.875, 0.875, 32 * 4).reshape(1, 1, 32, 4)
        freeze_attention(consumer, shared_key_value=(shared_key, -shared_key))
        shared = (shared_key[..., :3, :], -shared_key[..., :3, :])
        for mode, query_length in (("prefill", 3), ("decode", 1)):
            with self.subTest(mode=mode):
                module = consumer.as_export_module(mode)
                args = _args(query_length, 3)
                kwargs = {"shared_key_value": shared}
                runtime = self._convert(module, args, kwargs)
                self.assertEqual(len(runtime.program.outputs), 1)
                for index, observer in zip(
                    runtime.program.inputs[-2:],
                    (consumer.obs_present_key, consumer.obs_present_value),
                ):
                    tensor = runtime.program.tensor(index)
                    assert_storage(self, tensor, observer)
                    users = [
                        op for op in runtime.program.operators if index in op.inputs
                    ]
                    self.assertTrue(users)
                    self.assertNotIn("QUANTIZE", {op.name for op in users})
                assert_gemma_parity(
                    self,
                    module,
                    args,
                    runtime,
                    kwargs=kwargs,
                    label=f"external FP policy, folded C input {mode}",
                )


if __name__ == "__main__":
    unittest.main()

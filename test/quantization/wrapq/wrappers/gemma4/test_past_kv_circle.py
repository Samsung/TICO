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

"""Serialized affine storage/read contracts, independent of MX support."""

import unittest

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
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


def _config():
    """Use two small layers so the second consumes the first's storage KV."""
    config = Gemma4TextConfig.from_dict(
        dict(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=4,
            global_head_dim=4,
            max_position_embeddings=32,
            layer_types=["full_attention", "full_attention"],
            hidden_size_per_layer_input=0,
            attention_k_eq_v=False,
            num_kv_shared_layers=1,
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


class _CacheReadExport(torch.nn.Module):
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
            new_key_value=delta, past_key_value=(past_key, past_value)
        )
        return *compute, *delta


@unittest.skipUnless(
    _HAS_GEMMA4, "required Transformers Gemma4 modules are not installed"
)
class TestGemma4PastKVCircle(unittest.TestCase):
    """Export complete attention modules and inspect their serialized ABI."""

    def setUp(self):
        torch.manual_seed(317)

    @staticmethod
    def _attention(layer_index=0, *, signed_storage=False, read_policy=None):
        storage_dtype = DType.int(16) if signed_storage else DType.uint(8)
        config = PTQConfig(
            overrides={
                "new_k": affine(storage_dtype),
                "new_v": affine(storage_dtype),
                "past_key": read_policy or affine(DType.uint(8)),
                "past_value": read_policy or affine(DType.uint(8)),
                "present_key": affine(DType.uint(8)),
                "present_value": affine(DType.uint(8)),
            },
            model_args={"text": {"max_seq": 32}},
        )
        return QuantGemma4TextAttention(
            Gemma4TextAttention(_config(), layer_idx=layer_index).eval(), qcfg=config
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
                _load(observer, 0.0009765625, 0)
            else:
                _load(observer, 0.0625, 113)
        for observer in (attention.obs_past_key, attention.obs_past_value):
            _load(observer, 0.1875, 127)
        for observer in (attention.obs_present_key, attention.obs_present_value):
            _load(observer, 0.3125, 91)
        attention.freeze_qparams()

    @staticmethod
    def _convert(module, args, kwargs=None):
        with torch.no_grad():
            converted = tico.convert(module.eval(), args, kwargs=kwargs, strict=False)
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

    def _assert_read_conversion(self, runtime, storage_observer):
        """Storage/read and read/compute conversions must survive folding."""
        quantizes = [op for op in runtime.program.operators if op.name == "QUANTIZE"]
        pairs = [
            (
                runtime.program.tensor(op.inputs[0]),
                runtime.program.tensor(op.outputs[0]),
            )
            for op in quantizes
        ]
        storage_scale = float(storage_observer.compute_qparams()[0])
        self.assertTrue(
            any(
                left.quantization is not None
                and right.quantization is not None
                and left.quantization.scale == (storage_scale,)
                and right.quantization.scale == (0.1875,)
                and right.quantization.zero_point == (127,)
                and right.dtype == np.dtype(np.uint8)
                for left, right in pairs
            ),
            "the serialized cache input must be converted from storage to the read grid",
        )
        self.assertTrue(
            any(
                left.quantization is not None
                and right.quantization is not None
                and left.quantization.scale == (0.1875,)
                and right.quantization.scale == (0.3125,)
                and right.quantization.zero_point == (91,)
                and right.dtype == np.dtype(np.uint8)
                for left, right in pairs
            ),
            "the read grid must be explicitly converted to the concat compute grid",
        )

    def _assert_cache_concats(self, runtime):
        concats = [
            op
            for op in runtime.program.operators
            if op.name == "CONCATENATION"
            and runtime.program.tensor(op.outputs[0]).shape == (1, 1, 32, 4)
        ]
        self.assertEqual(len(concats), 2)
        for operator in concats:
            tensors = [
                runtime.program.tensor(i) for i in (*operator.inputs, *operator.outputs)
            ]
            self.assertEqual({tensor.dtype for tensor in tensors}, {np.dtype(np.uint8)})
            self.assertEqual(
                {tensor.quantization.scale for tensor in tensors}, {(0.3125,)}
            )
            self.assertEqual(
                {tensor.quantization.zero_point for tensor in tensors}, {(91,)}
            )

    def test_prefill_decode_delta_storage_and_independent_read_grids(self):
        """Same-dtype scale/zp mismatch must retain the original delta ABI."""
        attention = self._attention()
        self._freeze(attention)
        prefill_args = _args(32, 32)
        prefill = self._convert(attention.as_export_module("prefill"), prefill_args)
        for index, observer in zip(
            prefill.program.outputs[1:], attention.cache_storage_observers()
        ):
            self._assert_storage(prefill.program.tensor(index), observer)

        with torch.no_grad():
            _, new_key, new_value = attention.as_export_module("prefill")(*prefill_args)
        past = (new_key[:, :, :31], new_value[:, :, :31])
        decode = self._convert(
            attention.as_export_module("decode"),
            _args(1, 32),
            {"past_key_value": past},
        )
        for input_index, output_index, observer in zip(
            decode.program.inputs[-2:],
            decode.program.outputs[1:],
            attention.cache_storage_observers(),
        ):
            self._assert_storage(decode.program.tensor(input_index), observer)
            self._assert_storage(decode.program.tensor(output_index), observer)
        self._assert_read_conversion(decode, attention.obs_new_k)
        self._assert_cache_concats(decode)

    def test_signed_storage_uses_affine_requantize_without_mx(self):
        """Main's INT16/UINT8 conversion retains the signed cache ABI."""
        attention = self._attention(signed_storage=True)
        self._freeze(attention)
        past = torch.linspace(-1.0, 1.0, 31 * 4).reshape(1, 1, 31, 4)
        runtime = self._convert(
            attention.as_export_module("decode"),
            _args(1, 32),
            {"past_key_value": (past, -past)},
        )
        for input_index, output_index, observer in zip(
            runtime.program.inputs[-2:],
            runtime.program.outputs[1:],
            attention.cache_storage_observers(),
        ):
            self._assert_storage(runtime.program.tensor(input_index), observer)
            self._assert_storage(runtime.program.tensor(output_index), observer)
        self._assert_read_conversion(runtime, attention.obs_new_k)
        self._assert_cache_concats(runtime)

    def test_cache_branch_reference_values_and_delta_storage(self):
        """Serialized affine conversions match eager and preserve stored values."""
        attention = self._attention()
        self._freeze(attention)
        module = _CacheReadExport(attention).eval()
        # Storage values are exact multiples of 1/16; division by the read
        # grid (3/16) and compute grid (5/16) never lands on a rounding tie.
        past_key = torch.tensor([-11, -7, 1, 5, -5, -1, 7, 11], dtype=torch.float32)
        past_key = past_key.reshape(1, 1, 2, 4) * 0.0625
        past_value = -past_key
        current = (
            torch.tensor([1, 3, 7, 9], dtype=torch.float32).reshape(1, 1, 1, 4) * 0.0625
        )

        def grid(value, scale, zero_point):
            return (
                torch.round(value / scale + zero_point).clamp(0, 255) - zero_point
            ) * scale

        for step in range(2):
            with self.subTest(step=step):
                args = (current, -current, past_key, past_value)
                originals = tuple(value.clone() for value in args)
                with torch.no_grad():
                    eager = module(*args)
                expected = tuple(
                    torch.cat(
                        (
                            grid(
                                grid(grid(past, 0.0625, 113), 0.1875, 127), 0.3125, 91
                            ),
                            grid(grid(new, 0.0625, 113), 0.3125, 91),
                        ),
                        dim=-2,
                    )
                    for new, past in ((current, past_key), (-current, past_value))
                ) + (grid(current, 0.0625, 113), grid(-current, 0.0625, 113))
                for got, want in zip(eager, expected):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
                runtime = self._convert(module, args)
                actual = runtime.run(
                    [value.numpy() for value in args], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs
                for got, want in zip(actual, expected):
                    np.testing.assert_array_equal(got, want.numpy())
                for value, original in zip(args, originals):
                    torch.testing.assert_close(value, original, rtol=0, atol=0)
                self.assertFalse(torch.equal(eager[0][..., :-1, :], past_key))
                # The runtime stores the unchanged old grid plus the original
                # producer delta, rather than either attention compute tensor.
                past_key = torch.cat((past_key, eager[2]), dim=-2)
                past_value = torch.cat((past_value, eager[3]), dim=-2)

    def test_per_channel_cache_read_export_is_rejected(self):
        """Unsupported cache axis semantics fail before Circle serialization."""
        attention = self._attention(
            read_policy=affine(
                DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=1
            )
        )
        with self.assertRaisesRegex(ValueError, "per-tensor affine"):
            self._freeze(attention)
            attention.as_export_module("decode")

    def test_shared_consumer_input_uses_producer_storage_qparams(self):
        """A consumer's independent read policy cannot redefine producer storage."""
        producer = self._attention()
        self._freeze(producer)
        consumer = self._attention(layer_index=1, signed_storage=True)
        consumer.bind_cache_source(producer)
        shared_key = torch.linspace(-1.0, 1.0, 32 * 4).reshape(1, 1, 32, 4)
        self._freeze(consumer, shared_key_value=(shared_key, -shared_key))
        for mode, query_length in (("prefill", 32), ("decode", 1)):
            with self.subTest(mode=mode):
                runtime = self._convert(
                    consumer.as_export_module(mode, cache_source=producer),
                    _args(query_length, 32),
                    {"shared_key_value": (shared_key, -shared_key)},
                )
                self.assertEqual(len(runtime.program.outputs), 1)
                for index, observer in zip(
                    runtime.program.inputs[-2:], producer.cache_storage_observers()
                ):
                    self._assert_storage(runtime.program.tensor(index), observer)
                self._assert_read_conversion(runtime, producer.obs_new_k)


if __name__ == "__main__":
    unittest.main()

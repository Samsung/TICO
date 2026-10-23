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

"""Real cached calibration and static export for INT16 storage / MX attention."""

import copy
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from circle_schema import circle
from torch.utils._pytree import tree_leaves

from test.quantization.recipes.optional_dependency_stubs import (
    install_optional_dependency_stubs,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_for_conditional_generation import (
    _make_vision_config,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import (
    _has_gemma4,
    _make_text_config,
)

install_optional_dependency_stubs()

import tico
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.circle.runtime.errors import UnsupportedCircleOperatorError
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.utils.mx.mx_ops import quantize_mx


class _StorageAndCompute(torch.nn.Module):
    """Expose the wrapper's actual storage/read/compute branches for a counterexample."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, current_key, current_value, past_key, past_value):
        attention = self.attention
        delta = (
            attention._fq(current_key, attention.obs_new_k),
            attention._fq(current_value, attention.obs_new_v),
        )
        compute, storage, _ = attention._build_present_key_value(
            new_key_value=delta, past_key_value=(past_key, past_value)
        )
        return compute[1], delta[1], storage[1]


@unittest.skipUnless(
    _has_gemma4(), "required transformers Gemma4 modules are not installed"
)
class TestGemma4MXAttentionIntegration(unittest.TestCase):
    """No downloads: real prepared tiny model, adapter decode, and Circle bytes."""

    @staticmethod
    def _prepared(layout):
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )

        torch.manual_seed(1077)
        text_config = _make_text_config(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            max_position_embeddings=32,
            num_kv_shared_layers=1,
            rope_parameters={
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0}
            },
        )
        config = Gemma4Config.from_dict(
            {
                "text_config": text_config,
                "vision_config": _make_vision_config(),
                "audio_config": None,
                "image_token_id": 128,
                "video_token_id": 129,
                "audio_token_id": 130,
            }
        )
        model = Gemma4ForConditionalGeneration(config).eval()
        overrides = {}
        for index in range(2):
            prefix = f"model.language_model.layers.{index}.self_attn."
            overrides.update(
                {
                    prefix + "new_k": affine(DType.int(16)),
                    prefix + "new_v": affine(DType.int(16)),
                    prefix + "past_key": affine(DType.uint(8)),
                    prefix + "past_value": affine(DType.uint(8)),
                    prefix + "present_key": affine(DType.int(16)),
                    prefix + "present_value": mx("int8", axis=-2),
                    prefix + "softmax": mx("int8", axis=-1),
                    prefix + "attn_weights": mx("int8", axis=-1),
                    prefix + "attn_out": affine(DType.int(16)),
                }
            )
        qcfg = PTQConfig(
            activation=affine(DType.int(16)),
            weight=affine(DType.int(16)),
            overrides=overrides,
            model_args={
                "text": {"max_seq": 32},
                "vision": {"num_visual_tokens": 4},
                "attention": {"layout": layout, "rope": "hf"},
            },
        )
        return model, prepare(model, qcfg)

    @staticmethod
    def _attentions(prepared):
        text = prepared.wrapped.model.wrapped.language_model.wrapped
        return tuple(layer.wrapped.self_attn.wrapped for layer in text.layers)

    def _calibrate(self, layout):
        model, prepared = self._prepared(layout)
        owner, consumer = self._attentions(prepared)
        self.assertIsInstance(owner.obs_present_value, MXObserver)
        self.assertEqual(owner.obs_present_value.axis, -2)
        calls = []
        observed: dict[str, list[torch.Tensor]] = {"owner": [], "consumer": []}

        def before_forward(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            calls.append(
                (
                    kwargs["input_ids"].shape[1],
                    0 if cache is None else int(cache.get_seq_length()),
                )
            )
            self.assertIs(module._mode, Mode.CALIB)
            self.assertTrue(kwargs["use_cache"])

        def collect_spy(observer, name):
            original = observer.collect

            def collect(value):
                observed[name].append(value.detach().clone())
                original(value)

            return mock.patch.object(observer, "collect", side_effect=collect)

        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "calibration": {"seq_len": 32, "decode_steps": 2},
                "runtime": {"show_progress": False},
            },
            adapter=adapter,
            model=model,
            calibration_inputs=[{"input_ids": torch.tensor([[1, 3, 5, 7]])}],
        )
        handle = prepared.register_forward_pre_hook(before_forward, with_kwargs=True)
        try:
            with ExitStack() as stack:
                stack.enter_context(collect_spy(owner.obs_past_value, "owner"))
                stack.enter_context(collect_spy(consumer.obs_past_value, "consumer"))
                adapter.calibrate_prepared_model(ctx, prepared, {})
        finally:
            handle.remove()
        self.assertEqual(calls, [(4, 0), (1, 4), (1, 5)])
        self.assertEqual([value.shape[-2] for value in observed["owner"]], [4, 5])
        self.assertEqual([value.shape[-2] for value in observed["consumer"]], [4, 5, 6])
        quantized = convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)
        for attention in self._attentions(quantized):
            self.assertTrue(attention.obs_past_value.has_qparams)
            self.assertIsNot(
                attention.obs_past_value, attention.cache_storage_observers()[1]
            )
        return quantized

    @staticmethod
    def _args(query_length, key_length):
        return (
            torch.linspace(-0.75, 0.75, query_length * 32).reshape(1, query_length, 32),
            torch.zeros(1, 1, query_length, key_length),
            (torch.ones(1, query_length, 32), torch.zeros(1, query_length, 32)),
        )

    def _export(self, module, args, kwargs=None):
        kwargs = kwargs or {}
        with torch.no_grad():
            eager = module(*args, **kwargs)
            ep = torch.export.export(module.eval(), args, kwargs=kwargs, strict=False)
            actual = ep.module()(*args, **kwargs)
            for got, want in zip(tree_leaves(actual), tree_leaves(eager), strict=True):
                torch.testing.assert_close(got, want, atol=0, rtol=0)
            converted = tico.convert_from_exported_program(ep)
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify()
        # The built-in reference has no MX value codec. Exercise the real artifact
        # and preserve its explicit failure instead of substituting float kernels.
        with self.assertRaisesRegex(UnsupportedCircleOperatorError, "no value codec"):
            CircleReferenceRuntime(document)
        return document.model, eager

    @staticmethod
    def _builtin(model, operator):
        code = model.operatorCodes[operator.opcodeIndex]
        return code.builtinCode if code.builtinCode else code.deprecatedBuiltinCode

    def _assert_storage(self, tensor, observer):
        self.assertEqual(tensor.type, circle.TensorType.TensorType.INT16)
        scale, zero_point = observer.compute_qparams()
        self.assertEqual(tuple(tensor.quantization.scale), (float(scale),))
        self.assertEqual(tuple(tensor.quantization.zeroPoint), (int(zero_point),))
        self.assertEqual(tensor.quantization.quantizedDimension, 0)

    def _assert_mx_attention(self, model):
        graph = model.subgraphs[0]
        types = circle.TensorType.TensorType
        builtins = circle.BuiltinOperator.BuiltinOperator
        mx_matmuls = []
        for operator in graph.operators:
            builtin = self._builtin(model, operator)
            inputs = [graph.tensors[index] for index in operator.inputs if index >= 0]
            output = graph.tensors[operator.outputs[0]]
            if builtin == builtins.CONCATENATION:
                self.assertNotIn(
                    types.MXINT8, [value.type for value in inputs] + [output.type]
                )
            if builtin == builtins.BATCH_MATMUL and inputs[0].type == types.MXINT8:
                left, right = inputs[:2]
                self.assertEqual(right.type, types.MXINT8)
                self.assertEqual(output.type, types.INT16)
                self.assertEqual(
                    left.quantization.quantizedDimension, len(left.shape) - 1
                )
                self.assertEqual(
                    right.quantization.quantizedDimension, len(right.shape) - 2
                )
                self.assertEqual(left.shape[-1], 32)
                self.assertEqual(right.shape[-2], 32)
                mx_matmuls.append(operator)
        self.assertEqual(len(mx_matmuls), 1)

    def test_real_cached_calibration_decode_chain_and_shared_storage_in_both_layouts(
        self,
    ):
        caches = []
        for layout in ("unrolled", "batched"):
            with self.subTest(layout=layout):
                model = self._calibrate(layout)
                owner, consumer = self._attentions(model)
                self.assertIs(consumer.cache_storage_observers()[1], owner.obs_new_v)
                self.assertIsNot(consumer.obs_past_value, owner.obs_past_value)
                read_lengths: dict[str, list[int]] = {"owner": [], "consumer": []}

                def read_spy(observer, name):
                    original = observer.fake_quant

                    def fake_quant(value):
                        read_lengths[name].append(value.shape[-2])
                        return original(value)

                    return mock.patch.object(
                        observer, "fake_quant", side_effect=fake_quant
                    )

                with ExitStack() as stack, torch.no_grad():
                    stack.enter_context(read_spy(owner.obs_past_value, "owner"))
                    stack.enter_context(read_spy(consumer.obs_past_value, "consumer"))
                    result = model(
                        input_ids=torch.tensor([[1, 3, 5, 7]]),
                        use_cache=True,
                        return_dict=True,
                    )
                    before = result.past_key_values.layers[0].values.clone()
                    for token, length in ((9, 5), (11, 6)):
                        result = model(
                            input_ids=torch.tensor([[token]]),
                            past_key_values=result.past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                        self.assertEqual(
                            int(result.past_key_values.get_seq_length()), length
                        )
                        torch.testing.assert_close(
                            result.past_key_values.layers[0].values[..., :4, :],
                            before,
                            atol=0,
                            rtol=0,
                        )
                        self.assertTrue(bool(torch.isfinite(result.logits).all()))
                self.assertEqual(read_lengths["owner"], [4, 5])
                self.assertEqual(read_lengths["consumer"], [4, 5, 6])
                caches.append(result.past_key_values.layers[0].values)
                state = copy.deepcopy(model.state_dict())
                _, restored = self._prepared(layout)
                restored.load_state_dict(state)
                restored_owner, restored_consumer = self._attentions(restored)
                self.assertIs(
                    restored_consumer.cache_storage_observers()[1],
                    restored_owner.obs_new_v,
                )
                for before_attention, after_attention in zip(
                    (owner, consumer), (restored_owner, restored_consumer), strict=True
                ):
                    self.assertEqual(
                        after_attention.obs_present_value.elem_format, "int8"
                    )
                    self.assertEqual(after_attention.obs_present_value.axis, -2)
                    self.assertEqual(
                        after_attention.get_extra_state()["policies"],
                        before_attention.get_extra_state()["policies"],
                    )
                    for actual, expected in zip(
                        after_attention.obs_past_value.compute_qparams(),
                        before_attention.obs_past_value.compute_qparams(),
                        strict=True,
                    ):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                restored = convert(restored, inplace=True)
                self.assertIs(restored._mode, Mode.QUANT)
                with torch.no_grad():
                    restored_result = restored(
                        input_ids=torch.tensor([[1, 3, 5, 7]]),
                        use_cache=True,
                        return_dict=True,
                    )
                    for token in (9, 11):
                        restored_result = restored(
                            input_ids=torch.tensor([[token]]),
                            past_key_values=restored_result.past_key_values,
                            use_cache=True,
                            return_dict=True,
                        )
                torch.testing.assert_close(
                    restored_result.logits, result.logits, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    restored_result.past_key_values.layers[0].values,
                    result.past_key_values.layers[0].values,
                    atol=0,
                    rtol=0,
                )
        torch.testing.assert_close(caches[0], caches[1], atol=0, rtol=0)

    def test_calibrated_prefill_and_31_plus_1_decode_circle_storage_abi(self):
        for layout in ("unrolled", "batched"):
            with self.subTest(layout=layout):
                model = self._calibrate(layout)
                owner, consumer = self._attentions(model)
                prefill, outputs = self._export(
                    owner.as_export_module("prefill"), self._args(32, 32)
                )
                self._assert_mx_attention(prefill)
                graph = prefill.subgraphs[0]
                for index, observer in zip(
                    graph.outputs[1:], owner.cache_storage_observers(), strict=True
                ):
                    self._assert_storage(graph.tensors[index], observer)
                past = tuple(value[..., :31, :].contiguous() for value in outputs[1:])
                originals = tuple(value.clone() for value in past)
                decode, _ = self._export(
                    owner.as_export_module("decode"),
                    self._args(1, 32),
                    {"past_key_value": past},
                )
                self._assert_mx_attention(decode)
                graph = decode.subgraphs[0]
                for input_index, output_index, observer in zip(
                    graph.inputs[-2:],
                    graph.outputs[1:],
                    owner.cache_storage_observers(),
                    strict=True,
                ):
                    self._assert_storage(graph.tensors[input_index], observer)
                    self._assert_storage(graph.tensors[output_index], observer)
                for value, original in zip(past, originals, strict=True):
                    torch.testing.assert_close(value, original, atol=0, rtol=0)
                for mode, query_length in (("prefill", 32), ("decode", 1)):
                    shared, _ = self._export(
                        consumer.as_export_module(mode, cache_source=owner),
                        self._args(query_length, 32),
                        {"shared_key_value": outputs[1:]},
                    )
                    self._assert_mx_attention(shared)
                    graph = shared.subgraphs[0]
                    self.assertEqual(len(graph.outputs), 1)
                    for index, observer in zip(
                        graph.inputs[-2:], owner.cache_storage_observers(), strict=True
                    ):
                        self._assert_storage(graph.tensors[index], observer)

    def test_independent_read_grid_then_full_mx_block_preserves_storage(self):
        model = self._calibrate("unrolled")
        attention = self._attentions(model)[0]
        for observer in attention.cache_storage_observers():
            observer.load_qparams(
                torch.tensor(0.125), torch.tensor(0, dtype=torch.int32)
            )
        for observer in (attention.obs_past_key, attention.obs_past_value):
            observer.load_qparams(
                torch.tensor(0.5), torch.tensor(127, dtype=torch.int32)
            )
        past = torch.full((1, 1, 31, 32), 0.375)
        # The old read grid would clip 100 and 64.375 to 64, and round .375
        # to .5. None of those transformations belongs on the current delta.
        current = torch.tensor([100.0] * 8 + [64.375] * 8 + [0.375] * 16).reshape(
            1, 1, 1, 32
        )
        args = (current, current, past, past)
        originals = tuple(value.clone() for value in args)
        module = _StorageAndCompute(attention).eval()
        read_calls = []
        compute_inputs = []
        read_quantize = attention.obs_past_value.fake_quant
        compute_quantize = attention.obs_present_value.fake_quant

        def record_read(value):
            read_calls.append(value.detach().clone())
            return read_quantize(value)

        def record_compute(value):
            compute_inputs.append(value.detach().clone())
            return compute_quantize(value)

        with mock.patch.object(
            attention.obs_past_value, "fake_quant", side_effect=record_read
        ), mock.patch.object(
            attention.obs_present_value, "fake_quant", side_effect=record_compute
        ), torch.no_grad():
            eager = module(*args)
        self.assertEqual(len(read_calls), 1)
        torch.testing.assert_close(read_calls[0], past, atol=0, rtol=0)
        self.assertEqual(len(compute_inputs), 1)
        self.assertEqual(tuple(compute_inputs[0].shape), (1, 1, 32, 32))
        torch.testing.assert_close(
            compute_inputs[0][..., -1:, :], current, atol=0, rtol=0
        )
        circle_model, outputs = self._export(module, args)
        compute, delta, storage = outputs
        for actual, expected_output in zip(outputs, eager, strict=True):
            torch.testing.assert_close(actual, expected_output, atol=0, rtol=0)
        read_past = torch.full_like(past, 0.5)
        read_full = torch.cat((read_past, current), dim=-2)
        torch.testing.assert_close(compute_inputs[0], read_full, atol=0, rtol=0)
        expected = quantize_mx(read_full, elem_format="int8", axis=-2)
        incorrect = torch.cat(
            (
                quantize_mx(read_past, elem_format="int8", axis=-2),
                quantize_mx(current, elem_format="int8", axis=-2),
            ),
            dim=-2,
        )
        torch.testing.assert_close(compute, expected, atol=0, rtol=0)
        self.assertFalse(
            torch.equal(expected, incorrect),
            "the outlier must distinguish full-block quantization from per-part MX",
        )
        bypass_read = quantize_mx(
            torch.cat((past, current), dim=-2), elem_format="int8", axis=-2
        )
        self.assertFalse(
            torch.equal(expected, bypass_read),
            "the independent affine read must affect the MX result",
        )
        current_on_read_grid = (
            torch.round(current / 0.5 + 127).clamp(0, 255) - 127
        ) * 0.5
        wrong_current = quantize_mx(
            torch.cat((read_past, current_on_read_grid), dim=-2),
            elem_format="int8",
            axis=-2,
        )
        self.assertFalse(
            torch.equal(expected, wrong_current),
            "current values must keep their producer storage grid before full MX",
        )
        torch.testing.assert_close(
            compute[..., -1:, :8], current[..., :8], atol=0, rtol=0
        )
        torch.testing.assert_close(delta, current, atol=0, rtol=0)
        torch.testing.assert_close(
            storage, torch.cat((past, current), dim=-2), atol=0, rtol=0
        )
        self.assertFalse(torch.equal(compute[..., :31, :], past))
        for actual, original in zip(args, originals, strict=True):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)
        scale, zero_point = attention.obs_past_value.compute_qparams()
        self.assertEqual(float(scale), 0.5)
        self.assertEqual(int(zero_point), 127)
        self.assertIsNot(attention.obs_past_value, attention.obs_new_v)
        graph = circle_model.subgraphs[0]
        conversions = [
            (graph.tensors[operator.inputs[0]], graph.tensors[operator.outputs[0]])
            for operator in graph.operators
            if self._builtin(circle_model, operator)
            == circle.BuiltinOperator.BuiltinOperator.QUANTIZE
        ]
        self.assertTrue(
            any(
                before.type == circle.TensorType.TensorType.INT16
                and tuple(before.quantization.scale) == (0.125,)
                and after.type == circle.TensorType.TensorType.UINT8
                and tuple(after.quantization.scale) == (0.5,)
                and tuple(after.quantization.zeroPoint) == (127,)
                for before, after in conversions
            )
        )
        self.assertFalse(
            any(
                after.type == circle.TensorType.TensorType.UINT8
                and int(after.shape[-2]) == 1
                for _, after in conversions
            ),
            "the current delta must never be requantized to the past-read grid",
        )
        mx_quantizes = [
            operator
            for operator in graph.operators
            if self._builtin(circle_model, operator)
            == circle.BuiltinOperator.BuiltinOperator.QUANTIZE
            and graph.tensors[operator.outputs[0]].type
            == circle.TensorType.TensorType.MXINT8
        ]
        self.assertEqual(len(mx_quantizes), 1)
        mx_quantize = mx_quantizes[0]
        concat = next(
            operator
            for operator in graph.operators
            if mx_quantize.inputs[0] in operator.outputs
        )
        self.assertEqual(
            self._builtin(circle_model, concat),
            circle.BuiltinOperator.BuiltinOperator.CONCATENATION,
        )
        self.assertEqual(
            [int(graph.tensors[index].shape[-2]) for index in concat.inputs], [31, 1]
        )
        for index in (*concat.inputs, *concat.outputs):
            tensor = graph.tensors[index]
            self.assertEqual(tensor.type, circle.TensorType.TensorType.FLOAT32)
            self.assertIsNone(tensor.quantization)
        producers = {
            int(index): operator
            for operator in graph.operators
            for index in operator.outputs
        }
        for index, dtype, scale, zero_point in zip(
            concat.inputs,
            (circle.TensorType.TensorType.UINT8, circle.TensorType.TensorType.INT16),
            (0.5, 0.125),
            (127, 0),
            strict=True,
        ):
            dequantize = producers[int(index)]
            self.assertEqual(
                self._builtin(circle_model, dequantize),
                circle.BuiltinOperator.BuiltinOperator.DEQUANTIZE,
            )
            source = graph.tensors[dequantize.inputs[0]]
            self.assertEqual(source.type, dtype)
            self.assertEqual(tuple(source.quantization.scale), (scale,))
            self.assertEqual(tuple(source.quantization.zeroPoint), (zero_point,))
        output = graph.tensors[mx_quantize.outputs[0]]
        self.assertEqual(list(output.shape), [1, 1, 32, 32])
        self.assertEqual(output.quantization.quantizedDimension, 2)


if __name__ == "__main__":
    unittest.main()

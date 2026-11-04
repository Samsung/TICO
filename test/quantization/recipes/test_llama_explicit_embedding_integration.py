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

"""Real, locally initialized Llama PTQ and saved CPU token-embedding stages."""

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tico.circle import CircleDocument
from tico.circle.runtime import (
    CircleReferenceRuntime,
    ExecutionMode,
    UnsupportedCircleOperatorError,
)
from tico.circle.runtime.quantization import dequantize_array, quantize_array
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.export.circle import convert_and_save_circle
from tico.quantization.recipes.export.llama import (
    export_lm_head,
    export_token_embedding,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import FrozenObserverState, ObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaTokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm import (
    QuantLlamaForCausalLM,
)
from tico.quantization.wrapq.wrappers.llama.static_inputs import (
    prepare_llama_static_inputs,
)
from tico.quantization.wrapq.wrappers.nn.quant_embedding import QuantEmbedding
from tico.utils.model import CircleModel


def _make_quantized_llama(signed):
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        config = LlamaConfig.from_dict(
            dict(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                vocab_size=7,
                max_position_embeddings=8,
                tie_word_embeddings=True,
                attention_dropout=0.0,
                attn_implementation="eager",
                use_cache=False,
                pad_token_id=None,
            )
        )
        source = LlamaForCausalLM(config).eval()
        with torch.no_grad():
            row = torch.arange(7)[:, None]
            source.model.embed_tokens.weight.copy_(
                (torch.arange(8)[None, :] - 3) * (row + 1) / 32 + (row - 3) / 64
            )
        qconfig = PTQConfig(
            activation=affine(
                DType.int(16) if signed else DType.uint(8),
                qscheme=QScheme.PER_TENSOR_SYMM if signed else QScheme.PER_TENSOR_ASYMM,
            ),
            weight=affine(DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM),
            model_args={"profile": "npu_export"},
        )
        prepared = prepare(source, qconfig)
        with torch.no_grad():
            prepared(torch.tensor([[0, 6, 3, 1, 6, 2, 4, 5]]), use_cache=False)
        return convert(prepared).eval()


def _snapshot(model):
    return {
        name: observer.require_frozen()
        for name, observer in model.named_modules()
        if isinstance(observer, ObserverBase) and observer.frozen_state is not None
    }


def _native_oracle(codes, weight_grid, output_grid, ids, dtype):
    """FP32 DQ and half-away Q, independent of runtime arithmetic helpers."""
    row = codes[ids].astype(np.float32)
    row -= np.asarray(weight_grid.zero_point, np.float32)[ids, None]
    row *= np.asarray(weight_grid.scale, np.float32)[ids, None]
    scaled = row / np.float32(output_grid.scale[0])
    # Compute round using the exact value of the FP32 quotient in float64, so
    # addition of .5 cannot spuriously turn a neighboring FP32 value into a tie.
    rounded = np.copysign(np.floor(np.abs(scaled.astype(np.float64)) + 0.5), scaled)
    limits = np.iinfo(dtype)
    return np.clip(rounded + output_grid.zero_point[0], limits.min, limits.max).astype(
        dtype
    )


@unittest.skipUnless(has_transformers_for("llama"), "requires local Llama transformers")
class LlamaExplicitEmbeddingIntegrationTest(unittest.TestCase):
    def _assert_snapshot(self, before, model):
        after = _snapshot(model)
        self.assertEqual(before.keys(), after.keys())
        for name, original in before.items():
            current = after[name]
            self.assertIsInstance(current, FrozenObserverState)
            for field in fields(original):
                left, right = getattr(original, field.name), getattr(
                    current, field.name
                )
                with self.subTest(observer=name, field=field.name):
                    if isinstance(left, torch.Tensor):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    elif field.name == "clipping_bounds" and left is not None:
                        for x, y in zip(left, right):
                            torch.testing.assert_close(x, y, rtol=0, atol=0)
                    else:
                        self.assertEqual(left, right)

    def test_real_adapter_native_codes_frozen_grids_and_selected_memory(self):
        for signed in (False, True):
            with (
                self.subTest(activation="int16" if signed else "uint8"),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = _make_quantized_llama(signed)
                llama = model.wrapped
                self.assertIsInstance(llama, QuantLlamaForCausalLM)
                embedding = llama.model.wrapped.embed_tokens.wrapped
                self.assertIsInstance(embedding, QuantEmbedding)
                head = llama.lm_head.wrapped
                self.assertIs(embedding.module.weight, head.module.weight)
                before_weight = embedding.module.weight.detach().clone()
                before_states = _snapshot(model)
                sample = torch.linspace(-0.7, 0.9, 8).reshape(1, 1, 8)
                with torch.no_grad():
                    before_head = llama.lm_head(sample).detach().clone()
                adapter = LlamaTokenEmbeddingExportAdapter(llama)
                self.assertIs(adapter.embed_tokens, llama.model.wrapped.embed_tokens)
                self.assertIsNone(adapter.rotate_embedding)
                export_token_embedding(llama, 8, Path(directory))
                path = Path(directory) / "token_embedding.q.circle"

                from tico.circle.runtime import executor, rowwise_gather
                from tico.circle.runtime.kernels import quantize

                dq_calls, selected_shapes = [], []
                original_dq, original_take = executor.dequantize_array, np.take

                def checked_dq(value, tensor):
                    dq_calls.append(value.shape)
                    self.assertNotEqual(value.shape, (7, 8))
                    return original_dq(value, tensor)

                def checked_take(value, ids, *args, **kwargs):
                    selected = original_take(value, ids, *args, **kwargs)
                    if value.shape == (7, 8):
                        self.assertEqual(value.dtype, np.uint8)
                        selected_shapes.append(selected.shape)
                    return selected

                with (
                    patch.object(executor, "dequantize_array", side_effect=checked_dq),
                    patch.object(quantize, "dequantize_array", side_effect=checked_dq),
                    patch.object(rowwise_gather.np, "take", side_effect=checked_take),
                ):
                    document = CircleDocument.load(path)
                    self.assertTrue(document.verify().ok)
                    runtime = CircleReferenceRuntime(document)
                    self.assertEqual(dq_calls, [])  # Loading creates no float table.
                    program = runtime.program
                    self.assertEqual(
                        [op.name for op in program.operators],
                        ["DEQUANTIZE", "GATHER", "QUANTIZE"],
                    )
                    dq, gather, quantizer = program.operators
                    table = program.tensor(dq.inputs[0])
                    assert table.quantization is not None
                    assert table.constant is not None
                    self.assertEqual(table.dtype, np.dtype(np.uint8))
                    self.assertEqual(table.shape, (7, 8))
                    self.assertEqual(table.quantization.quantized_dimension, 0)
                    self.assertEqual(len(set(table.quantization.scale)), 7)
                    self.assertEqual(len(set(table.quantization.zero_point)), 7)
                    self.assertFalse(program.tensor(gather.inputs[1]).is_quantized)
                    for index in (dq.outputs[0], gather.outputs[0]):
                        self.assertEqual(
                            program.tensor(index).dtype, np.dtype(np.float32)
                        )
                        self.assertFalse(program.tensor(index).is_quantized)
                    output = program.tensor(quantizer.outputs[0])
                    assert output.quantization is not None
                    self.assertEqual(
                        output.dtype, np.dtype(np.int16 if signed else np.uint8)
                    )
                    self.assertEqual(output.shape_signature, (1, -1, 8))
                    self.assertEqual(len(runtime.rowwise_gather_plans), 1)
                    self.assertEqual(
                        [
                            t.dtype
                            for t in program.tensors
                            if t.is_constant and t.shape == (7, 8)
                        ],
                        [np.dtype(np.uint8)],
                    )
                    weight_grid, activation_grid = (
                        table.quantization,
                        output.quantization,
                    )
                    for grid, observer in (
                        (weight_grid, embedding.weight_obs),
                        (activation_grid, embedding.act_out_obs),
                    ):
                        state = observer.require_frozen()
                        np.testing.assert_array_equal(
                            grid.scale, state.scale.reshape(-1).tolist()
                        )
                        np.testing.assert_array_equal(
                            grid.zero_point, state.zero_point.reshape(-1).tolist()
                        )
                    original_codes = table.constant.copy()
                    # Reuse the declared export weight encoding (FP32 inverse
                    # scale multiplication), not a division-based re-quantizer
                    # with different behavior at FP32 half-neighbors.
                    expected_codes = (
                        torch.ops.quantized_decomposed.quantize_per_channel.default(
                            before_weight,
                            torch.tensor(weight_grid.scale),
                            torch.tensor(weight_grid.zero_point, dtype=torch.int32),
                            0,
                            0,
                            255,
                            torch.uint8,
                        ).numpy()
                    )
                    np.testing.assert_array_equal(original_codes, expected_codes)
                    for token_ids in ([[0]], [[6]], [[6, 0, 3, 6, 1, 0]]):
                        ids = np.asarray(token_ids, dtype=np.int64)
                        actual = runtime.run(
                            (ids,), mode=ExecutionMode.NATIVE, trace=True
                        )
                        expected = _native_oracle(
                            original_codes,
                            weight_grid,
                            activation_grid,
                            ids,
                            output.dtype,
                        )
                        np.testing.assert_array_equal(actual.outputs[0], expected)
                        self.assertEqual(actual.outputs[0].dtype, output.dtype)
                        assert actual.tensor_values is not None
                        self.assertNotIn(dq.outputs[0], actual.tensor_values)
                    self.assertEqual(selected_shapes, [(1, 1, 8), (1, 1, 8), (1, 6, 8)])
                    self.assertEqual(dq_calls, [])
                    self.assertEqual(runtime._dequantized_constants, {})
                # CircleModel's current reference API returns raw NATIVE integer
                # arrays too; dequantization is explicit, not hidden in this API.
                ids = np.array([[6, 0, 3, 6, 1, 0]], dtype=np.int64)
                convenience = CircleModel.load(str(path), runtime="reference")(
                    torch.from_numpy(ids)
                )
                np.testing.assert_array_equal(convenience, expected)
                self.assertEqual(convenience.dtype, output.dtype)
                fake = runtime.run((ids,), mode=ExecutionMode.FAKE_QUANTIZE).outputs[0]
                np.testing.assert_array_equal(fake, dequantize_array(expected, output))
                self.assertEqual(fake.dtype, np.float32)
                np.testing.assert_array_equal(table.constant, original_codes)
                torch.testing.assert_close(
                    embedding.module.weight, before_weight, rtol=0, atol=0
                )
                # Separately packaged tied head uses the same codes/grid. This
                # does not imply physical buffer sharing between Circle files.
                export_lm_head(llama, Path(directory))
                head_runtime = CircleReferenceRuntime(
                    CircleDocument.load(Path(directory) / "lm_head.q.circle")
                )
                fc = next(
                    op
                    for op in head_runtime.program.operators
                    if op.name == "FULLY_CONNECTED"
                )
                head_table = head_runtime.program.tensor(fc.inputs[1])
                np.testing.assert_array_equal(head_table.constant, original_codes)
                self.assertEqual(head_table.quantization, weight_grid)
                torch.testing.assert_close(
                    embedding.module.weight, before_weight, rtol=0, atol=0
                )
                with torch.no_grad():
                    torch.testing.assert_close(
                        llama.lm_head(sample), before_head, rtol=0, atol=0
                    )
                self._assert_snapshot(before_states, model)

    def test_first_decoder_handoff_is_explicit_fake_quant_reference(self):
        model = _make_quantized_llama(True)
        llama = model.wrapped
        layer = llama.model.wrapped.layers[0].wrapped
        before = _snapshot(model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_token_embedding(llama, 8, root)
            embedding_runtime = CircleReferenceRuntime(
                CircleDocument.load(root / "token_embedding.q.circle")
            )
            ids = np.array([[6, 0, 3, 6, 1, 0, 2, 4]], dtype=np.int64)
            raw = embedding_runtime.run((ids,), mode=ExecutionMode.NATIVE).outputs[0]
            hidden = dequantize_array(raw, embedding_runtime.output_tensors[0])
            prepared = prepare_llama_static_inputs(layer, 8)
            mask = prepared.mask(8, 8, 0, device=torch.device("cpu"))
            position = prepared.rope(
                0, 8, device=torch.device("cpu"), dtype=torch.float32
            )
            destination = root / "decoder.circle"
            convert_and_save_circle(
                layer.as_export_module("prefill", return_kv=False),
                (torch.from_numpy(hidden),),
                destination,
                kwargs={"attention_mask": mask, "position_embeddings": position},
            )
            decoder = CircleReferenceRuntime(CircleDocument.load(destination))
            boundary = decoder.input_tensors[0]
            producer = embedding_runtime.output_tensors[0]
            assert boundary.quantization is not None
            self.assertEqual(boundary.shape, raw.shape)
            self.assertEqual(boundary.dtype, producer.dtype)
            self.assertEqual(boundary.quantization, producer.quantization)
            consumer = layer.input_layernorm.wrapped.obs_act_in.require_frozen()
            self.assertEqual(
                boundary.quantization.scale, tuple(consumer.scale.reshape(-1).tolist())
            )
            self.assertEqual(
                boundary.quantization.zero_point,
                tuple(consumer.zero_point.reshape(-1).tolist()),
            )
            # This actual recipe's decoder contains quantized arithmetic unsupported
            # by NATIVE. Its reference bundle deliberately executes FAKE_QUANTIZE.
            other = [mask.numpy(), position[0].numpy(), position[1].numpy()]
            values = [hidden, *other]
            result = decoder.run(values, mode=ExecutionMode.FAKE_QUANTIZE).outputs[0]
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, hidden.shape)
            self.assertTrue(np.isfinite(result).all())
            native_values = [
                raw,
                *[
                    (
                        quantize_array(value, tensor)
                        if tensor.is_quantized_integer
                        else value
                    )
                    for value, tensor in zip(other, decoder.input_tensors[1:])
                ],
            ]
            with self.assertRaisesRegex(
                UnsupportedCircleOperatorError, "backend-defined integer arithmetic"
            ):
                decoder.run(native_values, mode=ExecutionMode.NATIVE)
        self._assert_snapshot(before, model)

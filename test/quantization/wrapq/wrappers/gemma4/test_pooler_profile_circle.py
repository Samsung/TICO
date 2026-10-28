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

"""Serialized pooler profiles preserve ownership and requested execution policy."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.wrapq.lifecycle import validate_frozen_model
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_pooler import (
    QuantGemma4VisionPooler,
)
from torch import nn

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.quantization.wrapq.wrappers.gemma4.test_quant_vision_model import (
    _has_gemma4_vision,
    _make_vision_config,
    _vision_position_ids,
)


class _PoolerMetadata(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 8
        self.root_hidden_size = 8**0.5


def _pooler(enabled):
    source = QuantGemma4VisionPooler(
        _PoolerMetadata(), qcfg=make_affine_ptq_config()
    ).eval()
    hidden = torch.linspace(-1, 1, 128).reshape(1, 16, 8)
    positions = _vision_position_ids(1, 16)
    source.enable_calibration()
    source(hidden, positions, torch.zeros(1, 16, dtype=torch.bool), 4)
    source.freeze_qparams()
    if not enabled:
        source.obs_pool_weight.disable_fake_quant()
    return source, hidden, positions


class TestPoolerProfileCircle(unittest.TestCase):
    def _assert_grid(self, tensor, observer, shape):
        state = observer.require_frozen()
        assert state.scale is not None and state.zero_point is not None
        self.assertEqual(tensor.dtype, np.dtype(np.uint8))
        self.assertEqual(tensor.shape, shape)
        assert tensor.quantization is not None
        self.assertEqual(tensor.quantization.scale, (float(state.scale),))
        self.assertEqual(tensor.quantization.zero_point, (int(state.zero_point),))

    def test_enabled_and_disabled_matrix_contracts_and_reference_outputs(self):
        for enabled in (True, False):
            source, hidden, positions = _pooler(enabled)
            before = _state_contract(source)
            for length in (4, 16):
                with self.subTest(fake_quant_enabled=enabled, output_length=length):
                    adapter = source.as_export_module(
                        output_length=length, pixel_position_ids=positions
                    ).eval()
                    self.assertEqual(
                        adapter.wrapped.obs_pool_weight.require_frozen().fake_quant_enabled,
                        enabled,
                    )
                    with torch.no_grad():
                        converted = tico.convert(adapter, (hidden,))
                    with TemporaryDirectory() as directory:
                        path = Path(directory) / "pooler.circle"
                        path.write_bytes(converted.circle_binary)
                        document = CircleDocument.load(path)
                    self.assertFalse(document.verify().issues)
                    runtime = CircleReferenceRuntime(document)
                    program = runtime.program
                    self.assertEqual(
                        [operator.name for operator in program.operators],
                        ["BATCH_MATMUL", "MUL"],
                    )
                    self._assert_grid(
                        program.tensor(program.inputs[0]), source.obs_act_in, (1, 16, 8)
                    )
                    self._assert_grid(
                        program.tensor(program.outputs[0]),
                        source.obs_pool_out,
                        (1, length, 8),
                    )
                    matrix = next(
                        tensor
                        for tensor in program.tensors
                        if tensor.is_constant and tensor.shape == (1, length, 16)
                    )
                    if enabled:
                        self._assert_grid(
                            matrix, adapter.wrapped.obs_pool_weight, (1, length, 16)
                        )
                        np.testing.assert_array_equal(
                            np.unique(cast(np.ndarray, matrix.constant)),
                            np.asarray([0, 255]),
                        )
                    else:
                        self.assertEqual(matrix.dtype, np.dtype(np.float32))
                        self.assertIsNone(matrix.quantization)
                        np.testing.assert_array_equal(
                            matrix.constant, adapter.wrapped.pool_weights.numpy()
                        )
                    encoded = fake_quantize_array(
                        hidden.numpy(), program.tensor(program.inputs[0])
                    )
                    actual = runtime.run(
                        [encoded], mode=ExecutionMode.FAKE_QUANTIZE
                    ).outputs[0]
                    with torch.no_grad():
                        expected = adapter(torch.from_numpy(encoded.copy())).numpy()
                    np.testing.assert_array_equal(actual, expected)
                    # The source and profile remain admitted together after real export.
                    root = nn.Module()
                    root.source = source
                    root.profile = adapter
                    validate_frozen_model(root)
                    self.assertEqual(before, _state_contract(source))

    @unittest.skipUnless(
        _has_gemma4_vision(),
        "required transformers Gemma4 vision modules are not installed",
    )
    def test_vision_conversion_and_equivalent_profile_checkpoint_remain_valid(self):
        from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
            QuantGemma4VisionModel,
        )
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        config = _make_vision_config(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            standardize=False,
        )
        positions = _vision_position_ids(1, 16)
        values = torch.linspace(-1, 1, 16 * 3 * config.patch_size**2).reshape(1, 16, -1)

        def make(enabled):
            torch.manual_seed(1097)
            model = QuantGemma4VisionModel(
                Gemma4VisionModel(config).eval(), qcfg=make_affine_ptq_config()
            ).eval()
            model.enable_calibration()
            with torch.no_grad():
                model(
                    pixel_values=values, pixel_position_ids=positions, return_dict=True
                )
            model.freeze_qparams()
            if not enabled:
                cast(
                    QuantGemma4VisionPooler, model.pooler.wrapped
                ).obs_pool_weight.disable_fake_quant()
            return model

        for enabled in (True, False):
            with self.subTest(fake_quant_enabled=enabled):
                model = make(enabled)
                source_state = _state_contract(model.pooler.wrapped)
                adapter = model.as_export_module(pixel_position_ids=positions).eval()
                validate_frozen_model(model)
                with torch.no_grad():
                    converted = tico.convert(adapter, (values,))
                document = CircleDocument.from_bytes(converted.circle_binary)
                self.assertFalse(document.verify().issues)
                runtime = CircleReferenceRuntime(document)
                program = runtime.program
                self.assertEqual(len(program.inputs), 1)
                self.assertEqual(len(program.outputs), 1)
                self._assert_grid(
                    program.tensor(program.inputs[0]),
                    model.patch_embedder.wrapped.obs_act_in,
                    tuple(values.shape),
                )
                self._assert_grid(
                    program.tensor(program.outputs[0]),
                    model.obs_last_hidden_state,
                    (4, config.hidden_size),
                )
                matrix = next(
                    tensor
                    for tensor in program.tensors
                    if tensor.is_constant and tensor.shape == (1, 4, 16)
                )
                if enabled:
                    self._assert_grid(
                        matrix, model.pooler_export.wrapped.obs_pool_weight, (1, 4, 16)
                    )
                else:
                    self.assertEqual(matrix.dtype, np.dtype(np.float32))
                    self.assertIsNone(matrix.quantization)
                validate_frozen_model(model)
                model.freeze_qparams()
                self.assertEqual(source_state, _state_contract(model.pooler.wrapped))
                state = copy.deepcopy(model.state_dict())
                receiver = make(enabled)
                restored_adapter = receiver.as_export_module(
                    pixel_position_ids=positions
                ).eval()
                receiver.load_state_dict(state, strict=True)
                validate_frozen_model(receiver)
                self.assertEqual(_state_contract(model), _state_contract(receiver))
                encoded = fake_quantize_array(
                    values.numpy(), program.tensor(program.inputs[0])
                )
                with torch.no_grad():
                    expected = adapter(
                        torch.from_numpy(encoded.copy())
                    ).last_hidden_state
                    restored = restored_adapter(
                        torch.from_numpy(encoded.copy())
                    ).last_hidden_state
                torch.testing.assert_close(expected, restored, rtol=0, atol=0)
                # Full-tower reference rounding has its own quantized semantics.
                # Preserve its exact result across the checkpoint independently.
                with torch.no_grad():
                    restored_circle = tico.convert(restored_adapter, (values,))
                restored_runtime = CircleReferenceRuntime(
                    CircleDocument.from_bytes(restored_circle.circle_binary)
                )
                np.testing.assert_array_equal(
                    runtime.run([encoded], mode=ExecutionMode.FAKE_QUANTIZE).outputs[0],
                    restored_runtime.run(
                        [encoded], mode=ExecutionMode.FAKE_QUANTIZE
                    ).outputs[0],
                )

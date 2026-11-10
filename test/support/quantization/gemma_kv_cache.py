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

"""Gemma4 cache fixtures and ABI checks shared with frozen-state export.

Optional family imports occur only when the attention factory is called.
"""

import numpy as np
import torch
from tico.circle.runtime import ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase


def tiny_config(layer_type="full_attention"):
    """Keep a full-attention final layer, as required by Transformers."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

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


def args(query_length, key_length):
    """Avoid random inputs and keep the static capacity at 32 tokens."""
    hidden = torch.linspace(0.25, 1.0, query_length * 8).reshape(1, query_length, 8)
    return (
        hidden,
        torch.zeros(1, 1, query_length, key_length),
        (torch.ones(1, query_length, 4), torch.zeros(1, query_length, 4)),
    )


def load(observer, scale, zero_point):
    """Install intentional frozen mismatch qparams for a boundary unit test."""
    observer.load_qparams(
        torch.full_like(observer.min_val, scale),
        torch.full_like(observer.min_val, zero_point, dtype=torch.int32),
    )


def make_gemma_attention(
    layer_index=0,
    *,
    signed_storage=False,
    compute_policy=None,
    layer_type="full_attention",
):
    """Create a fresh attention, consuming the caller's scoped RNG."""
    from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
        QuantGemma4TextAttention,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

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
            tiny_config(layer_type),
            layer_idx=(
                layer_index * 2 if layer_type == "sliding_attention" else layer_index
            ),
        ).eval(),
        qcfg=config,
    ).eval()


def freeze_attention(attention, *, shared_key_value=None):
    """Collect real shapes, then intentionally select different frozen grids.

    Recipe calibration coverage belongs to the adapter integration test;
    this fixture exercises serialization of known, incompatible grids.
    """
    attention.enable_calibration()
    sample_args = args(32, 32)
    with torch.no_grad():
        attention(
            sample_args[0],
            sample_args[2],
            attention_mask=sample_args[1],
            shared_key_value=shared_key_value,
            use_cache=True,
        )
    for _, observer in attention.named_observers():
        if isinstance(observer, AffineObserverBase):
            load(observer, 0.03125, 0 if observer.qscheme.is_symmetric() else 128)
    storage_observers = (
        ()
        if attention.is_kv_shared_layer
        else (attention.obs_new_k, attention.obs_new_v)
    )
    for observer in storage_observers:
        if observer.dtype == DType.int(16):
            load(observer, 0.09375, 0)
        else:
            load(observer, 0.09375, 113)
    for observer in (attention.obs_present_key, attention.obs_present_value):
        load(observer, 0.21875, 0 if observer.dtype == DType.int(16) else 91)
    attention.freeze_qparams()


def assert_storage(case, tensor, observer):
    case.assertEqual(
        tensor.dtype,
        np.dtype(np.int16 if observer.dtype == DType.int(16) else np.uint8),
    )
    case.assertIsNotNone(tensor.quantization)
    scale, zero_point = observer.compute_qparams()
    case.assertEqual(tensor.quantization.scale, (float(scale),))
    case.assertEqual(tensor.quantization.zero_point, (int(zero_point),))
    case.assertEqual(tensor.quantization.quantized_dimension, 0)


def tensor_qparams(tensor):
    quantization = tensor.quantization
    if quantization is None or not quantization.scale:
        return None
    return quantization.scale, quantization.zero_point


def assert_storage_conversion(case, runtime, storage_observer, compute_observer):
    """The storage boundary reaches C through exactly one Quantize."""
    program = runtime.program
    compute_scale, compute_zp = compute_observer.compute_qparams()
    for index in program.inputs[-2:]:
        assert_storage(case, program.tensor(index), storage_observer)
        users = [op for op in program.operators if index in op.inputs]
        case.assertEqual(len(users), 1)
        case.assertEqual(users[0].name, "QUANTIZE")
        converted = program.tensor(users[0].outputs[0])
        case.assertEqual(
            tensor_qparams(converted),
            ((float(compute_scale),), (int(compute_zp),)),
        )
        case.assertEqual(converted.dtype, np.dtype(np.uint8))


def assert_gemma_parity(
    case,
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
    case.assertEqual(len(leaves), len(runtime.program.inputs))
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
    case.assertEqual(len(result.outputs), len(expected))
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
            case.assertEqual(metrics["differing"], 1, metrics)
            case.assertEqual(metrics["max_abs"], output_step, metrics)
        else:
            np.testing.assert_array_equal(
                actual, wanted.detach().numpy(), err_msg=f"{label}: {metrics}"
            )
    return result.outputs

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

"""Tiny Qwen3-VL attention, cache input builders and serialized domain assertions."""

import torch
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.support.quantization.kv_cache import (
    assert_value_preserving_qparams,
    CacheDomains,
    find_named_tensor,
    hidden,
    load_qparam,
    producer_map,
    tensor_qparam,
)


def make_qwen_attention(dtype, *, layout="unrolled", compute_dtype=None):
    """Return a fresh eval/QUANT attention and explicit storage/compute grids.

    Consume the caller's RNG so repeated calls preserve model creation order.
    Callers scope/seeding the RNG own determinism; no module or observer is cached.
    """
    from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
        QuantQwen3VLTextAttention,
    )

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    config = Qwen3VLTextConfig.from_dict(
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 8,
        }
    )
    config._attn_implementation = "eager"

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    compute_dtype = compute_dtype or dtype
    domains = CacheDomains.affine(dtype, compute_dtype)
    qcfg = make_affine_ptq_config(
        dtype=dtype,
        overrides={
            "present_key": affine(compute_dtype),
            "present_value": affine(compute_dtype),
        },
        model_args={"attention": {"layout": layout}},
    )
    attention = QuantQwen3VLTextAttention(
        Qwen3VLTextAttention(config, layer_idx=0).eval(),
        qcfg=qcfg,
    ).eval()
    calibrate_and_freeze(attention)
    load_controlled_qparams(attention, domains)
    return attention, domains


def rope(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
    frequencies = torch.tensor([0.07, 0.13, 0.07, 0.13]).reshape(1, 1, 4)
    angles = positions * frequencies
    cosine = angles.cos()
    sine = angles.sin()
    return cosine, sine


def mask(query_length: int, key_length: int) -> torch.Tensor:
    return torch.zeros(1, 1, query_length, key_length)


def prefill_args():
    return hidden(2), rope(2), mask(2, 2)


def calibrate_and_freeze(attention) -> None:
    attention.enable_calibration()
    with torch.no_grad():
        # Prefill with cache output disabled must establish both S and C.
        attention(
            hidden(2),
            rope(2),
            attention_mask=mask(2, 2),
            use_cache=False,
        )
    attention.freeze_qparams()


def load_controlled_qparams(attention, domains) -> None:
    for observer in (attention.obs_new_k, attention.obs_new_v):
        load_qparam(observer, *domains.storage_qparam)
    for observer in (
        attention.obs_present_key,
        attention.obs_present_value,
    ):
        load_qparam(observer, *domains.compute_qparam)

    # Keep unrelated RoPE split/concatenation boundaries on the norm's
    # grid. Independently calibrated slice grids exercise the generic
    # split quantization lowering, rather than the KV storage contract.
    for kind in ("q", "k"):
        norm = getattr(attention, kind + "_norm").wrapped
        load_qparam(norm.obs_act_out, *domains.producer_qparam)
        for suffix in ("x1", "x2", "neg", "cat"):
            load_qparam(
                getattr(attention, f"obs_{kind}_{suffix}"),
                *domains.producer_qparam,
            )
    load_qparam(attention.obs_k_rot, *domains.producer_qparam)
    # The value path has no RoPE observers, but its layout bracket owns the
    # projection-output grid as well; pin it so the view chain keeps the
    # controlled domain.
    load_qparam(attention.obs_v_view, *domains.producer_qparam)
    value_projection = attention.v_proj.wrapped
    if not isinstance(value_projection, QuantLinear):
        raise AssertionError("value projection must be QuantLinear")
    load_qparam(value_projection.obs_act_out, *domains.producer_qparam)


def assert_qwen_decode_topology(case, runtime, domains) -> None:
    program = runtime.program
    producers = producer_map(program)

    past_key = find_named_tensor(program, program.inputs, "past_key_values_0")
    past_value = find_named_tensor(program, program.inputs, "past_key_values_1")
    case.assertEqual(tensor_qparam(past_key), domains.storage_qparam)
    case.assertEqual(tensor_qparam(past_value), domains.storage_qparam)
    for tensor in (past_key, past_value):
        case.assertEqual(tensor.dtype, domains.numpy_dtype)

    delta_key_index, delta_value_index = program.outputs[-2:]
    delta_key = program.tensor(delta_key_index)
    delta_value = program.tensor(delta_value_index)
    case.assertEqual(tensor_qparam(delta_key), domains.storage_qparam)
    case.assertEqual(tensor_qparam(delta_value), domains.storage_qparam)
    case.assertEqual(delta_key.dtype, domains.numpy_dtype)
    case.assertEqual(delta_value.dtype, domains.numpy_dtype)

    main_concats = []
    for operator in program.operators:
        if operator.name != "CONCATENATION" or len(operator.inputs) != 2:
            continue
        inputs = [program.tensor(index) for index in operator.inputs]
        output = program.tensor(operator.outputs[0])
        input_shapes = {tuple(tensor.shape) for tensor in inputs}
        if input_shapes == {(1, 1, 2, 4), (1, 1, 1, 4)} and tuple(output.shape) == (
            1,
            1,
            3,
            4,
        ):
            main_concats.append(operator)

    case.assertEqual(len(main_concats), 2)
    for concat in main_concats:
        for tensor_index in (*concat.inputs, concat.outputs[0]):
            case.assertEqual(
                program.tensor(tensor_index).dtype, domains.compute_numpy_dtype
            )
            case.assertEqual(
                tensor_qparam(program.tensor(tensor_index)),
                domains.compute_qparam,
            )

        for tensor_index in concat.inputs:
            _, producer = producers[tensor_index]
            case.assertEqual(producer.name, "QUANTIZE")
            source = program.tensor(producer.inputs[0])
            target = program.tensor(producer.outputs[0])
            case.assertEqual(tensor_qparam(target), domains.compute_qparam)
            sequence_length = tuple(target.shape)[-2]
            expected_source = (
                domains.storage_qparam
                if sequence_length == 2
                else domains.producer_qparam
            )
            case.assertEqual(tensor_qparam(source), expected_source)

    for output_index in (delta_key_index, delta_value_index):
        output = program.tensor(output_index)
        _, producer = producers[output_index]
        case.assertEqual(producer.name, "QUANTIZE")
        source = program.tensor(producer.inputs[0])
        case.assertEqual(tensor_qparam(source), domains.producer_qparam)
        case.assertEqual(tensor_qparam(output), domains.storage_qparam)

    assert_value_preserving_qparams(case, program)

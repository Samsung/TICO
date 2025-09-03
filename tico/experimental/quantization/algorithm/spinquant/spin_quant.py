# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the same directory of this source file.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import Iterable

import torch
from tqdm import tqdm

from tico.experimental.quantization.algorithm.spinquant.utils import (
    get_orthogonal_matrix,
)


@torch.no_grad()
def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: Iterable[torch.nn.Linear]
) -> None:
    for linear in linear_layers:
        # Calculate new weight and bias
        linear.weight.data = linear.weight.data * layernorm.weight.data

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data + torch.matmul(
                linear.weight.data, layernorm.bias
            )


@torch.no_grad()
def fuse_layernorm(
    model: torch.nn.Module,
) -> None:
    layers = [layer for layer in model.model.layers]

    for layer in layers:
        # Self attention
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        # MLP
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )

        layer.post_attention_layernorm.weight.data = torch.ones_like(
            layer.post_attention_layernorm.weight.data
        )
        layer.input_layernorm.weight.data = torch.ones_like(
            layer.input_layernorm.weight.data
        )

    # LM head
    fuse_ln_linear(model.model.norm, [model.lm_head])


@torch.no_grad()
def rotate_model(model: torch.nn.Module, mode: str = "hadamard") -> None:
    # TODO: Use pre-trained rotation matrix
    r1 = get_orthogonal_matrix(model.config.hidden_size, mode=mode)

    # Rotate embedding with r1 matrix
    for embedding in [model.model.embed_tokens]:
        embedding.weight.data = torch.matmul(embedding.weight.data, r1)

    # Rotate head with r1 matrix
    model.lm_head.weight.data = torch.matmul(model.lm_head.weight.data, r1)

    num_heads = model.config.num_attention_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads

    for layer in tqdm(model.model.layers, unit="layer", desc="Rotating"):
        # Rotate W_q, W_k, and W_v of attention inputs with r1 matrix
        for attn_input in [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ]:
            attn_input.weight.data = torch.matmul(attn_input.weight.data, r1)

        # Rotate W_o of attetion output with r1 matrix
        attn_output = layer.self_attn.o_proj

        attn_output.weight.data = torch.matmul(r1.T, attn_output.weight.data)
        if attn_output.bias is not None:
            attn_output.bias.data = torch.matmul(r1.T, attn_output.bias.data)

        # Rotate W_up and W_gate of mlp inputs with r1 matrix
        for mlp_input in [layer.mlp.up_proj, layer.mlp.gate_proj]:
            mlp_input.weight.data = torch.matmul(mlp_input.weight.data, r1)

        # Rotate W_down of mlp output with r1 matrix
        mlp_output = layer.mlp.down_proj

        mlp_output.weight.data = torch.matmul(r1.T, mlp_output.weight.data)
        if mlp_output.bias is not None:
            mlp_output.bias.data = torch.matmul(r1.T, mlp_output.bias.data)

        # TODO: Use pre-trained rotation matrix
        r2 = get_orthogonal_matrix(head_dim, mode=mode)

        # Rotate W_o of attention input and W_o of attention output with r2 matrix
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj

        weight_v = v_proj.weight.data.T
        v_shape = weight_v.shape
        temp = weight_v.reshape(-1, v_shape[-1] // head_dim, head_dim)
        temp = temp @ r2
        weight_v = temp.reshape(v_shape).T

        weight_o = o_proj.weight.data
        o_shape = weight_o.shape
        temp = weight_o.reshape(-1, o_shape[-1] // head_dim, head_dim)
        temp = temp @ r2
        weight_o = temp.reshape(o_shape)


@torch.no_grad()
def apply_spin(
    model: torch.nn.Module,
    mode: str = "hadamard",
):
    """
    Rotates the model's weights using rotation matrix according to the mode.
    Random hadamard matrix is used by default.

    Parameters
    -----------
        model
            A torch module whose weights will be rotated.
        mode
            The mode for deciding rotation matrix.
    """

    dtype = model.dtype

    model.to(torch.float64)

    # Fuse the linear operations which are in the Layernorm into the adjcent linear operation.
    fuse_layernorm(model)

    # Roate the model's weights
    # TODO: Use pre-trained rotation matrices
    rotate_model(model, mode)

    model.to(dtype)

    return model

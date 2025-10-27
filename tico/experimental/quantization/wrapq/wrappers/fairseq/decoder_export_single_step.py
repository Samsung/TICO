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
#
# -----------------------------------------------------------------------------
# This file includes modifications based on fairseq
#  (https://github.com/facebookresearch/fairseq), originally licensed under
# the MIT License. See the LICENSE file in the fairseq repository for details.
# -----------------------------------------------------------------------------

"""
Q) Why the name "SingleStep"?

Fairseq's decoder already advances one token at a time during generation,
but the default path is "stateful" and "shape-polymorphic": it owns and
mutates K/V caches internally, prefix lengths and triangular masks grow with
the step, and beam reordering updates hidden module state. That's friendly
for eager execution, but hostile to `torch.export` and many accelerator
backends.

This export wrapper makes the per-token call truly "single-step" in the
export sense: "stateless" and "fixed-shape" so every invocation has the
exact same graph.

Key invariants
--------------
• "Stateless": K/V caches come in as explicit inputs and go out as outputs.
  The module does not store or mutate hidden state.
• "Static shapes": Query is always [B, 1, C]; encoder features and masks
  have fixed, predeclared sizes; K/V slots use fixed capacity (unused tail
  is simply masked/ignored).
• "External control": Step indexing, cache slot management (append/roll),
  and beam reordering are handled outside the module.
• "Prebuilt additive masks": Self-attention masks are provided by the
  caller (0 for valid, large negative sentinel, e.g. -120, for masked),
  avoiding data-dependent control flow.

In short: still step-wise like fairseq, but restructured for export—no
internal state, no data-dependent shapes, no dynamic control flow.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

import tico

# ----- 1) Export wrapper module -------------------------------------------
class DecoderExportSingleStep(nn.Module):
    """
    Export-only single-step decoder module.

    Inputs (example shapes; B=1, H=8, Dh=64, C=512, S=64, Tprev=63):
      - prev_x:               [B, 1, C]          embedded decoder input for the current step
      - enc_x:                [S, B, C]          encoder hidden states (fixed-length export input)
      - enc_pad_additive:     [B, 1, S]          additive float key_padding_mask for enc-dec attn (0 for keep, -120 for pad)
      - self_attn_mask:       [B, 1, S]          additive float mask for decoder self-attn at this step; pass zeros if unused
      - prev_self_k_0..L-1:   [B, H, Tprev, Dh]  cached self-attn K per layer
      - prev_self_v_0..L-1:   [B, H, Tprev, Dh]  cached self-attn V per layer

    Outputs:
      - x_out:                [B, 1, C]          new decoder features at the current step
      - new_k_0..L-1:         [H, B, Dh]         per-layer new K (single-timestep; time dim squeezed)
      - new_v_0..L-1:         [H, B, Dh]         per-layer new V (single-timestep; time dim squeezed)

    Notes:
      • We keep masks/additive semantics externally to avoid any mask-building inside the graph.
      • We reshape the new K/V from [B,H,1,Dh] -> [H,B,Dh] to match the requested output spec (8,1,64).
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        # Cache common meta for assertions
        self.num_layers = len(getattr(decoder, "layers"))
        # Infer heads/head_dim from the wrapped self_attn of layer 0
        any_layer = getattr(decoder.layers[0], "wrapped", decoder.layers[0])  # type: ignore[index]
        mha = getattr(any_layer, "self_attn", None)
        assert mha is not None, "Decoder layer must expose self_attn"
        self.num_heads = int(mha.num_heads)
        self.head_dim = int(mha.head_dim)
        # Embed dim (C)
        self.embed_dim = int(getattr(decoder, "embed_dim"))

    def forward(
        self,
        prev_x: torch.Tensor,  # [B,1,C]
        enc_x: torch.Tensor,  # [S,B,C]
        enc_pad_additive: torch.Tensor,  # [B,1,S]
        *kv_args: torch.Tensor,  # prev_k_0..L-1, prev_v_0..L-1 (total 2L tensors)
        self_attn_mask: torch.Tensor,  # [B,1,S] (or zeros)
    ):
        L = self.num_layers
        H = self.num_heads
        Dh = self.head_dim
        B, one, C = prev_x.shape
        S, B2, C2 = enc_x.shape
        assert (
            one == 1 and C == self.embed_dim and B == B2 and C2 == C
        ), "Shape mismatch in prev_x/enc_x"
        assert len(kv_args) == 2 * L, f"Expected {2*L} KV tensors, got {len(kv_args)}"

        # Unpack previous self-attn caches
        prev_k_list: List[torch.Tensor] = list()  # each [B,H,Tprev,Dh]
        prev_v_list: List[torch.Tensor] = list()  # each [B,H,Tprev,Dh]
        for i in range(L):
            prev_k_list.append(kv_args[2 * i])
            prev_v_list.append(kv_args[2 * i + 1])
        for i in range(L):
            assert (
                prev_k_list[i].dim() == 4 and prev_v_list[i].dim() == 4
            ), "KV must be [B,H,Tprev,Dh]"
            assert (
                prev_k_list[i].shape[0] == B
                and prev_k_list[i].shape[1] == H
                and prev_k_list[i].shape[3] == Dh
            )

        # Call decoder's external single-step path
        # Returns:
        #   x_step: [B,1,C]
        #   newk/newv: lists of length L, each [B*H,1,Dh]
        x_step, newk_list, newv_list = self.decoder.forward_external_step(  # type: ignore[operator]
            prev_output_x=prev_x,
            encoder_out_x=enc_x,
            encoder_padding_mask=enc_pad_additive,
            self_attn_mask=self_attn_mask,
            prev_self_k_list=prev_k_list,
            prev_self_v_list=prev_v_list,
        )

        out_tensors: List[torch.Tensor] = [
            x_step
        ]  # first output is the new decoder features
        for i in range(L):
            nk = newk_list[i]  # [B*H, Tnew, Dh]
            nv = newv_list[i]  # [B*H, Tnew, Dh]
            out_tensors.append(nk)
            out_tensors.append(nv)

        # Return tuple: (x_step, new_k_0, new_v_0, new_k_1, new_v_1, ..., new_k_{L-1}, new_v_{L-1})
        return tuple(out_tensors)


# ----- 2) Example inputs (B=1, S=64, H=8, Dh=64, C=512, L=4) ---------------
def make_example_inputs(*, L=4, B=1, S=64, H=8, Dh=64, C=512, Tprev=63, device="cpu"):
    """
    Build example tensors that match the export I/O spec.
    Shapes follow the request:
      prev_x:             [1,1,512]
      enc_x:              [64,1,512]
      enc_pad_additive:   [1,1,64]    (additive float; zeros -> keep)
      prev_k_i / prev_v_i (for i in 0..L-1): [1,8,63,64]
      self_attn_mask:     [1,1,64]    (additive float; zeros -> keep)
    """
    g = torch.Generator(device=device).manual_seed(0)

    prev_x = torch.randn(B, 1, C, device=device, dtype=torch.float32, generator=g)
    enc_x = torch.randn(S, B, C, device=device, dtype=torch.float32, generator=g)

    # Additive masks (0 for allowed, -120 for masked)
    enc_pad_additive = torch.full((B, 1, S), float(-120), device=device)
    self_attn_mask = torch.full((B, 1, S), float(-120), device=device)
    enc_pad_additive[0, :27] = 0  # 27 is a random example.
    self_attn_mask[0, :27] = 0  # 27 is a random example.

    # Previous self-attn caches for each layer
    prev_k_list = []
    prev_v_list = []
    for _ in range(L):
        prev_k = torch.randn(
            B, H, Tprev, Dh, device=device, dtype=torch.float32, generator=g
        )
        prev_v = torch.randn(
            B, H, Tprev, Dh, device=device, dtype=torch.float32, generator=g
        )
        prev_k_list.append(prev_k)
        prev_v_list.append(prev_v)

    # Pack inputs as the export function will expect:
    # (prev_x, enc_x, enc_pad_additive, self_attn_mask, prev_k_0..L-1, prev_v_0..L-1)
    example_args: Tuple[torch.Tensor, ...] = (
        prev_x,
        enc_x,
        enc_pad_additive,
        *prev_k_list,
        *prev_v_list,
    )
    example_kwargs = {"self_attn_mask": self_attn_mask}
    return example_args, example_kwargs


# ----- 3) Export driver -----------------------------------------------------
def export_decoder_single_step(translator, *, save_path="decoder_step_export.circle"):
    """
    Wrap the QuantFairseqDecoder into the export-friendly single-step module
    and export with torch.export.export using example inputs.
    """
    # Grab the wrapped decoder
    dec = translator.models[
        0
    ].decoder  # assumed QuantFairseqDecoder with forward_external_step
    # Build export wrapper
    wrapper = DecoderExportSingleStep(decoder=dec).eval()

    # Example inputs (L inferred from wrapper/decoder)
    L = wrapper.num_layers
    H = wrapper.num_heads
    Dh = wrapper.head_dim
    C = wrapper.embed_dim
    example_inputs, example_kwargs = make_example_inputs(L=L, H=H, Dh=Dh, C=C)

    # Export circle (no dynamism assumed; shapes are fixed for export)
    cm = tico.convert(
        wrapper,
        args=example_inputs,
        kwargs=example_kwargs,
        strict=True,  # fail if something cannot be captured
    )

    # Save .pte
    cm.save(save_path)
    print(f"Saved decoder single-step export to: {save_path}")

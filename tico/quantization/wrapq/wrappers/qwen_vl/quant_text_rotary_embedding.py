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

from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding",
)
class QuantQwen3VLTextRotaryEmbedding(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLTextRotaryEmbedding module.

    This module generates MRoPE (multimodal rotary positional embeddings) cos/sin values
    for text attention. All floating-point computations are quantized.
    """

    def __init__(
        self,
        fp_rope: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        assert hasattr(fp_rope, "config")
        assert hasattr(fp_rope, "inv_freq")
        assert hasattr(fp_rope, "mrope_section")
        assert hasattr(fp_rope, "attention_scaling")

        self.config = fp_rope.config
        self.mrope_section = fp_rope.mrope_section
        self.attention_scaling = fp_rope.attention_scaling

        # Copy the inv_freq buffer to the wrapper
        self.register_buffer("inv_freq", fp_rope.inv_freq.clone())

        # Observers for all intermediate tensor values
        mk = self._make_obs
        self.obs_inv_freq = mk("inv_freq")  # Constant buffer
        self.obs_freqs = mk("freqs")  # After matrix multiplication
        self.obs_freqs_mrope = mk("freqs_mrope")  # After MRoPE
        self.obs_emb = mk("emb")  # After concatenation
        self.obs_cos = mk("cos")  # Final cosine output
        self.obs_sin = mk("sin")  # Final sine output

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Forward pass with fake quantization.

        Args:
            x: Input tensor (used only for device/dtype)
            position_ids: Position identifiers (batch_size, seq_len)

        Returns:
            (cos, sin): Tuple of rotary embeddings
        """
        # Expand position_ids for MRoPE
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Expand inv_freq for batched computation
        # Shape: (3, batch_size, head_dim//2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        inv_freq_expanded = self._fq(inv_freq_expanded, self.obs_inv_freq)

        # Reshape position_ids for matrix multiplication
        # Shape: (3, batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # Compute frequencies via matrix multiplication
        # Shape: (3, batch_size, seq_len, head_dim//2)
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )

        # Force float32 for precision
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        freqs = self._fq(freqs, self.obs_freqs)

        # Apply interleaved MRoPE
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        freqs = self._fq(freqs, self.obs_freqs_mrope)

        # Concatenate frequencies (duplicate for sin/cos)
        # Shape: (batch_size, seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = self._fq(emb, self.obs_emb)

        # Compute cos and sin
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        # Quantize final outputs
        cos = self._fq(cos, self.obs_cos)
        sin = self._fq(sin, self.obs_sin)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """
        Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.

        Args:
            freqs: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)

        Returns:
            freqs_t: (bs, seq_len, head_dim // 2)

        Design Note:
            This implementation is using slice_copy, index_select, and cat
            to avoid yet unsupported slice_scatter with step=3 operation and
            to avoid unsupported in-place operator index_put.default.
        """
        # Start with T dimension (will keep some, replace some)
        freqs_t_base = freqs[0]

        # For each dimension (H, W), extract frequency bands to be interleaved
        h_w_bands = []

        for dim, offset in enumerate((1, 2), start=1):  # H, W dimensions
            length = mrope_section[dim] * 3
            indices = torch.arange(offset, length, 3, device=freqs.device)

            # Select frequency bands from H/W dimensions
            # freqs[dim] has shape (bs, seq_len, head_dim//2)
            # index_select on last dim: (bs, seq_len, num_selected)
            freqs_bands = freqs[dim].index_select(dim=-1, index=indices)
            h_w_bands.append(freqs_bands)

        # Now we need to build the interleaved output
        # Original T dimension has indices 0-63
        # We want to replace specific indices with H/W bands

        # The interleaving pattern: T0, H1, W2, T3, T4, H5, W6, T7, ...
        # Where T, H, W bands follow the pattern from mrope_section

        # Build the output by slicing and concatenating
        # Strategy: Slice T dimension into chunks, insert H/W bands, concatenate

        chunks = []
        pos = 0

        # Total length in the last dimension
        total_len = freqs_t_base.shape[-1]

        for i in range(total_len):
            # Determine which dimension this position belongs to
            # Pattern: T, H, W, T, T, H, W, T, ...
            mod = i % 3

            if mod == 0:
                # T dimension position - take from T
                # Slice just this index from T
                chunk = freqs_t_base[..., i : i + 1]
                chunks.append(chunk)
            elif mod == 1:
                # H dimension position - take from H
                # Calculate which band this is
                band_idx = (i - 1) // 3
                if band_idx < h_w_bands[0].shape[-1]:
                    chunk = h_w_bands[0][..., band_idx : band_idx + 1]
                    chunks.append(chunk)
                else:
                    # Fallback to T if out of bounds
                    chunk = freqs_t_base[..., i : i + 1]
                    chunks.append(chunk)
            else:  # mod == 2
                # W dimension position - take from W
                band_idx = (i - 2) // 3
                if band_idx < h_w_bands[1].shape[-1]:
                    chunk = h_w_bands[1][..., band_idx : band_idx + 1]
                    chunks.append(chunk)
                else:
                    # Fallback to T if out of bounds
                    chunk = freqs_t_base[..., i : i + 1]
                    chunks.append(chunk)

        # Concatenate all chunks
        freqs_t = torch.cat(chunks, dim=-1)

        return freqs_t

    def _all_observers(self):
        """Yield all observers."""
        yield from (
            self.obs_inv_freq,
            self.obs_freqs,
            self.obs_freqs_mrope,
            self.obs_emb,
            self.obs_cos,
            self.obs_sin,
        )

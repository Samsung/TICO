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

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4VisionEncoderPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionEncoder")
class QuantGemma4VisionEncoder(QuantModuleBase):
    """PTQ wrapper for the Gemma4 vision encoder with fixed ``seq_len`` support.

    The wrapper provides two forward paths:

    * **``forward``** — dynamic evaluation path used during calibration and
      accuracy evaluation.  It computes ``position_embeddings`` via
      ``Gemma4VisionRotaryEmbedding`` and the bidirectional attention mask
      internally, matching the Hugging Face ``Gemma4VisionEncoder`` API.

    * **``forward_export``** — static export path used for ``torch.export``
      tracing.  It reads pre-computed ``position_embeddings`` and
      ``attention_mask`` from registered buffers, avoiding any dynamic
      shape-dependent computation inside the graph.
    """

    def __init__(
        self,
        fp_encoder: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_encoder
        self.config = fp_encoder.config
        self.rotary_emb = fp_encoder.rotary_emb
        self.layers = nn.ModuleList(
            [
                PTQWrapper(
                    layer,
                    qcfg=qcfg.child("layers").child(str(i)) if qcfg else None,
                    fp_name=join_name(fp_name, f"layers.{i}"),
                )
                for i, layer in enumerate(fp_encoder.layers)
            ]
        )

        mk = self._make_obs
        self.obs_act_in = mk("act_in")
        self.obs_attention_mask = mk("attention_mask")
        self.obs_position_cos = mk("position_embeddings_cos")
        self.obs_position_sin = mk("position_embeddings_sin")
        self.obs_encoder_out = mk("encoder_out")

        # Register RoPE tables for `forward`
        self._register_rope_tables()

    def _register_rope_tables(self) -> None:
        """Precompute sin/cos lookup tables for ALL possible position ID values.

        position_embedding_size defines the vocabulary size for position IDs
        (max position value = position_embedding_size - 1).
        This replaces the dynamic matmul+cos/sin computation in forward()
        with a simple gather from precomputed tables.

        Called from ``__init__`` so tables are available during calibration.
        """
        inv_freq = self.rotary_emb.inv_freq  # [spatial_dim // 2]
        attention_scaling = self.rotary_emb.attention_scaling

        max_pos = self.config.position_embedding_size  # e.g. 10240
        position_indices = torch.arange(
            max_pos, dtype=torch.float, device=inv_freq.device
        )  # [0..max_pos-1]
        # freq_table[pos, i] = inv_freq[i] * pos  — equivalent to the matmul in original forward
        freq_table = torch.outer(
            position_indices, inv_freq
        )  # [max_pos, spatial_dim//2]
        # Concat to match the full embedding dim per spatial dim (same as original: cat(freqs, freqs))
        emb_table = torch.cat(
            (freq_table, freq_table), dim=-1
        )  # [max_pos, spatial_dim]
        cos_table = emb_table.cos() * attention_scaling  # [max_pos, spatial_dim]
        sin_table = emb_table.sin() * attention_scaling  # [max_pos, spatial_dim]
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def _gather_position_embeddings(
        self,
        pixel_position_ids: torch.LongTensor,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather position embeddings from precomputed lookup tables.

        Equivalent to ``Gemma4VisionRotaryEmbedding.forward()`` but uses
        precomputed ``cos_table``/``sin_table`` instead of dynamic matmul.

        Args:
            pixel_position_ids: Patch positions as (x, y) coordinates,
                shaped ``(B, S, 2)``.  Padding patches have (-1, -1).
            dtype: Target dtype for the output tensors.

        Returns:
            ``(cos, sin)`` tuple each shaped ``(B, S, head_dim)``.
        """
        all_cos, all_sin = [], []
        for i in range(2):
            # Clamp negative positions (padding markers) to 0 for safe gather
            dim_pos = pixel_position_ids[:, :, i].clamp(min=0)  # [B, S]
            all_cos.append(self.cos_table[dim_pos])
            all_sin.append(self.sin_table[dim_pos])
        cos = torch.cat(all_cos, dim=-1).to(dtype=dtype)  # [B, S, head_dim]
        sin = torch.cat(all_sin, dim=-1).to(dtype=dtype)  # [B, S, head_dim]
        return (cos, sin)

    def _make_bidirectional_mask(
        self,
        pixel_position_ids: torch.LongTensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create a 4D additive bidirectional attention mask with padding support.

        Equivalent to ``create_bidirectional_mask`` from HuggingFace
        ``masking_utils``, but computed directly from ``pixel_position_ids``
        where padding patches are marked with ``(-1, -1)``.

        The mask is additive: ``0.0`` for valid↔valid pairs,
        ``attention_invalid_logits_value`` (e.g. ``-1e9``) for any pair
        involving a padding position.

        Args:
            pixel_position_ids: Patch positions as (x, y) coordinates,
                shaped ``(B, S, 2)``.  Padding patches have (-1, -1).
            dtype: Target dtype for the output tensors.

        Returns:
            Additive attention mask shaped ``(B, 1, S, S)``.
        """
        # Valid patches have position_ids >= 0; padding has -1.
        # Use float arithmetic instead of boolean ops (& / ~).
        valid = (pixel_position_ids[:, :, 0] >= 0).to(dtype)  # (B, S) 1.0 or 0.0
        # Pairwise: both query and key must be valid → 1.0, else 0.0
        mask_2d = valid.unsqueeze(2) * valid.unsqueeze(1)  # (B, S, S)
        # Convert to additive mask: 0.0 for valid, fill_value for invalid.
        fill_value = float(self.qcfg.attention_mask_fill_value)
        attention_mask = (1.0 - mask_2d) * fill_value  # (B, S, S)
        return attention_mask.unsqueeze(1)  # (B, 1, S, S)

    # ------------------------------------------------------------------
    # Dynamic evaluation forward
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the vision encoder with dynamic RoPE and mask computation.

        This path mirrors the Hugging Face ``Gemma4VisionEncoder`` forward and
        is used during calibration and accuracy evaluation.  It computes
        ``position_embeddings`` as ``Gemma4VisionRotaryEmbedding`` and the
        bidirectional attention mask internally.

        Args:
            inputs_embeds: Input patch embeddings shaped ``(B, S, hidden_size)``.
            attention_mask: Optional 2-D attention mask shaped ``(B, S)``.
            pixel_position_ids: Optional 2-D pixel position ids shaped
                ``(B, S, 2)``.  Hugging Face name for the same tensor.
            **kwargs: Additional keyword arguments forwarded to each encoder
                layer for Hugging Face API compatibility.

        Returns:
            Output hidden states shaped ``(B, S, hidden_size)``.
        """
        # Compute position_embeddings dynamically from lookup tables using
        # the actual pixel_position_ids for this batch.
        cos, sin = self._gather_position_embeddings(
            pixel_position_ids, dtype=inputs_embeds.dtype
        )

        # Observe position embeddings
        cos = self._fq(cos, self.obs_position_cos)
        sin = self._fq(sin, self.obs_position_sin)
        position_embeddings = (cos, sin)

        # Compute bidirectional attention mask with padding support.
        # Convert 2D boolean mask (True=valid) from the HF API to a
        # 4D additive mask that the encoder layers expect.
        # Use float arithmetic to avoid unsupported bitwise ops.
        if attention_mask.ndim == 2:
            valid = attention_mask.to(inputs_embeds.dtype)  # (B, S) 1.0 or 0.0
            mask_2d = valid.unsqueeze(2) * valid.unsqueeze(1)  # (B, S, S)
            fill_value = float(self.qcfg.attention_mask_fill_value)
            attention_mask = ((1.0 - mask_2d) * fill_value).unsqueeze(1)  # (B, 1, S, S)

        # Observe attention mask
        attention_mask = self._fq(attention_mask, self.obs_attention_mask)

        hidden_states = self._fq(inputs_embeds, self.obs_act_in)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
                **kwargs,
            )

        return self._fq(hidden_states, self.obs_encoder_out)

    # ------------------------------------------------------------------
    # Static export forward
    # ------------------------------------------------------------------

    def forward_export(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run the vision encoder using pre-computed static templates.

        This path uses ``position_embeddings`` and ``attention_mask`` from
        pre-computed templates that were materialised by
        ``as_export_module``.  It contains no dynamic shape-dependent
        computation and is safe for ``torch.export`` tracing.

        Args:
            inputs_embeds: Input patch embeddings shaped ``(1, S, hidden_size)``.

        Returns:
            Output hidden states shaped ``(1, S, hidden_size)``.
        """
        seq_len = inputs_embeds.shape[1]

        # precomputed position embeddings.
        cos = self.position_embeddings_cos_template
        sin = self.position_embeddings_sin_template
        # Fake-quantize position embeddings.
        cos = self.obs_position_cos.fake_quant(cos)
        sin = self.obs_position_sin.fake_quant(sin)
        position_embeddings = (cos, sin)

        # precomputed mask template.
        attention_mask = self.attention_mask_template
        # Fake-quantize attention mask.
        attention_mask = self.obs_attention_mask.fake_quant(attention_mask)

        hidden_states = self._fq(inputs_embeds, self.obs_act_in)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        return self._fq(hidden_states, self.obs_encoder_out)

    # ------------------------------------------------------------------
    # Export adapter
    # ------------------------------------------------------------------

    def as_export_module(
        self,
        mode: ExportMode = "prefill",
        *,
        pixel_position_ids: torch.LongTensor,
    ) -> nn.Module:
        """Return a static export adapter for the requested execution mode.

        ``forward_export`` uses position embeddings and attention mask from
        pre-computed templates (``position_embeddings_cos_template`` /
        ``sin_template`` and ``attention_mask_template``).

        Args:
            mode: Export mode (only ``"prefill"`` is supported).
            pixel_position_ids: Optional 2-D pixel position ids shaped
                ``(1, S, 2)``.

        Returns:
            A ``Gemma4VisionEncoderPrefillExportAdapter`` wrapping this module.
        """
        if mode != "prefill":
            raise ValueError(f"Unsupported Gemma4 vision encoder export mode: {mode!r}")

        # Assert QUANT mode
        assert self._mode is Mode.QUANT, "Must be in QUANT mode for export"

        # Make sure that all observers are calibrated
        for obs in self._all_observers():
            assert obs.has_qparams, f"Observer {obs.name} has not been calibrated"

        cos, sin = self._gather_position_embeddings(pixel_position_ids)
        self.register_buffer("position_embeddings_cos_template", cos)
        self.register_buffer("position_embeddings_sin_template", sin)

        attention_mask = self._make_bidirectional_mask(
            pixel_position_ids,
            dtype=pixel_position_ids.dtype,
        )
        self.register_buffer("attention_mask_template", attention_mask)

        return Gemma4VisionEncoderPrefillExportAdapter(self)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
            self.obs_act_in,
            self.obs_attention_mask,
            self.obs_position_cos,
            self.obs_position_sin,
            self.obs_encoder_out,
        )

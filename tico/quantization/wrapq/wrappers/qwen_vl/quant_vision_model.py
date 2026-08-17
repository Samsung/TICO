# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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

from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.qwen_vl.vision_profile import (
    Qwen3VLVisionGridInput,
    Qwen3VLVisionProfile,
)
from tico.quantization.wrapq.wrappers.registry import try_register
from tico.utils.compat.transformers import qwen3_vl_has_deepstack_model_output


@try_register("transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel")
class QuantQwen3VLVisionModel(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLVisionModel module.

    The wrapper owns quantized model computation and observer state, but no
    deployment grid. Eager calibration and evaluation derive metadata from the
    runtime ``grid_thw`` argument, while fixed-grid export adapters materialize
    their own profile-specific tensors.
    """

    has_deepstack_model_output: bool = qwen3_vl_has_deepstack_model_output()

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model

        cfg = fp_model.config
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.hidden_size = cfg.hidden_size
        self.num_position_embeddings = cfg.num_position_embeddings
        self.num_grid_per_side = int(cfg.num_position_embeddings**0.5)
        self.deepstack_visual_indexes = cfg.deepstack_visual_indexes

        # Precompute rotary frequency table for RoPE
        self.dim = (
            fp_model.rotary_pos_emb.dim
            if hasattr(fp_model.rotary_pos_emb, "dim")
            else (cfg.hidden_size // cfg.num_heads) // 2
        )
        self.theta = (
            fp_model.rotary_pos_emb.theta
            if hasattr(fp_model.rotary_pos_emb, "theta")
            else 10000.0
        )
        inv_freq = self._precompute_rope_inv_freq(dim=self.dim, theta=self.theta)
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        # Wrap patch embedder
        self.patch_embed = PTQWrapper(
            fp_model.patch_embed,
            qcfg=qcfg.child("patch_embed") if qcfg else None,
            fp_name=join_name(fp_name, "patch_embed"),
        )

        # Wrap transformer blocks
        self.blocks = nn.ModuleList()
        blocks_cfg = qcfg.child("blocks") if qcfg else None
        for i, blk in enumerate(fp_model.blocks):
            self.blocks.append(
                PTQWrapper(
                    blk,
                    qcfg=blocks_cfg.child(str(i)) if blocks_cfg else None,
                    fp_name=join_name(fp_name, f"blocks.{i}"),
                )
            )

        # Wrap merger
        self.merger = PTQWrapper(
            fp_model.merger,
            qcfg=qcfg.child("merger") if qcfg else None,
            fp_name=join_name(fp_name, "merger"),
        )

        # Wrap deepstack merger list
        self.deepstack_merger_list = nn.ModuleList()
        deepstack_merger_cfg = qcfg.child("deepstack_merger_list") if qcfg else None
        for i, merger in enumerate(fp_model.deepstack_merger_list):
            self.deepstack_merger_list.append(
                PTQWrapper(
                    merger,
                    qcfg=deepstack_merger_cfg.child(str(i))
                    if deepstack_merger_cfg
                    else None,
                    fp_name=join_name(fp_name, f"deepstack_merger_list.{i}"),
                )
            )

        # --- Observers for intermediate tensors --------------------------------
        mk = self._make_obs

        # Position embedding observers
        self.obs_pos_embeds = mk("pos_embed")
        self.obs_pos_add = mk("pos_add")

        # RoPE observers
        self.obs_rope_cos = mk("rope_cos")
        self.obs_rope_sin = mk("rope_sin")

    @staticmethod
    def _precompute_rope_inv_freq(dim: int, theta: float) -> torch.Tensor:
        """Precompute rotary frequency table for RoPE."""
        # Compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq

    @staticmethod
    def _precompute_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute cumulative sequence lengths for concrete vision grids."""
        # Compute cumulative sequence lengths
        from torch.nn import functional as F

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        return cu_seqlens

    @staticmethod
    def _precompute_rope_position_embeddings(
        merge_size: int, rope_inv_freq: torch.Tensor, grid_thw: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cosine and sine tensors for concrete vision grids."""
        seq_len = int(torch.prod(grid_thw, dim=1).sum().item())
        rotary_pos_emb = QuantQwen3VLVisionModel._rot_pos_emb(
            merge_size, rope_inv_freq, grid_thw
        )
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rot_pos_emb(
        merge_size: int, rope_inv_freq: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Compute rotary position embeddings from grid dimensions."""
        max_hw = int(grid_thw[:, 1:].max().item())

        # Create frequency table up to max_hw
        freq_table = QuantQwen3VLVisionModel._create_freq_table(
            seqlen=max_hw, rope_inv_freq=rope_inv_freq
        )
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # Compute full-resolution positions
            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    @staticmethod
    def _create_freq_table(seqlen: int, rope_inv_freq: torch.Tensor) -> torch.Tensor:
        """Create rotary frequency table."""
        seq = torch.arange(
            seqlen, device=rope_inv_freq.device, dtype=rope_inv_freq.dtype
        )
        freqs = torch.outer(seq, rope_inv_freq)
        return freqs

    @staticmethod
    def _fast_pos_embed_interpolate(
        merge_size: int,
        num_grid_per_side: int,
        pos_embedder: nn.Module,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Compute interpolated position embeddings."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = pos_embedder.weight.device

        idx_list: List[Any] = [[] for _ in range(4)]
        weight_list: List[Any] = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * num_grid_per_side
            base_h_ceil = h_idxs_ceil * num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=pos_embedder.weight.dtype, device=device
        )
        pos_embeds = pos_embedder(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )

        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(
                    t, h // merge_size, merge_size, w // merge_size, merge_size, -1
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        *,
        pos_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        attention_split_sizes: Optional[tuple[int, ...]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Run the shared vision tensor path with explicit metadata.

        Dynamic eager execution and fixed-profile export differ only in how
        position embeddings, rotary embeddings, cumulative sequence lengths,
        and attention split sizes are produced. Both paths delegate the actual
        patch, transformer-block, and merger computation to this method.

        Args:
            hidden_states: Flattened patches with shape ``(seq_len, patch_dim)``
                or the static batch-one ABI ``(1, seq_len, patch_dim)``.
            pos_embeds: Spatial position embeddings for every patch.
            position_embeddings: RoPE cosine and sine tensors.
            cu_seqlens: Cumulative sequence lengths for image or frame chunks.
            attention_split_sizes: Optional static Python split sizes used by
                export to avoid deriving Python integers from tensors.
            **kwargs: Additional keyword arguments forwarded to vision blocks.

        Returns:
            The version-compatible Qwen3-VL vision output.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = pos_embeds.to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        pos_embeds = self._fq(pos_embeds, self.obs_pos_embeds)
        hidden_states = hidden_states + pos_embeds
        hidden_states = self._fq(hidden_states, self.obs_pos_add)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)

        cos, sin = position_embeddings
        cos = cos.to(dtype=hidden_states.dtype, device=hidden_states.device)
        sin = sin.to(dtype=hidden_states.dtype, device=hidden_states.device)
        position_embeddings = (
            self._fq(cos, self.obs_rope_cos),
            self._fq(sin, self.obs_rope_sin),
        )

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attention_split_sizes=attention_split_sizes,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        if self.has_deepstack_model_output:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                BaseModelOutputWithDeepstackFeatures,
            )

            return BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=hidden_states,
                pooler_output=merged_hidden_states,
                deepstack_features=deepstack_feature_lists,
            )
        return merged_hidden_states, deepstack_feature_lists

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        attention_split_sizes: Optional[tuple[int, ...]] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Run the dynamic eager/PTQ vision path.

        Metadata is computed from the actual ``grid_thw`` argument on every
        call. This path is used for calibration, evaluation, and ordinary eager
        execution. Static NPU export must use :meth:`as_export_module` instead.

        Args:
            hidden_states: Flattened patches with shape ``(seq_len, patch_dim)``
                or the static batch-one ABI ``(1, seq_len, patch_dim)``.
            grid_thw: Image or video grids with shape ``(num_items, 3)``.
            attention_split_sizes: Optional eager split sizes supplied by a
                caller that already knows the image or frame boundaries.
            **kwargs: Additional keyword arguments forwarded to vision blocks.

        Returns:
            The version-compatible Qwen3-VL vision output.
        """
        pos_embeds = QuantQwen3VLVisionModel._fast_pos_embed_interpolate(
            merge_size=self.spatial_merge_size,
            num_grid_per_side=self.num_grid_per_side,
            pos_embedder=self.module.pos_embed,
            grid_thw=grid_thw,
        )

        inv_freq = self.rope_inv_freq.to(hidden_states.device)
        cos, sin = QuantQwen3VLVisionModel._precompute_rope_position_embeddings(
            merge_size=self.spatial_merge_size,
            rope_inv_freq=inv_freq,
            grid_thw=grid_thw,
        )
        cu_seqlens = QuantQwen3VLVisionModel._precompute_cu_seqlens(grid_thw)

        return self._forward_impl(
            hidden_states,
            pos_embeds=pos_embeds,
            position_embeddings=(cos, sin),
            cu_seqlens=cu_seqlens,
            attention_split_sizes=attention_split_sizes,
            **kwargs,
        )

    def forward_export(
        self,
        hidden_states: torch.Tensor,
        *,
        pos_embeds: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        attention_split_sizes: tuple[int, ...],
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        """Run the fixed-profile tensor path with adapter-owned metadata.

        The static adapter materializes every grid-dependent tensor before graph
        capture and passes those tensors explicitly. The quantization wrapper
        therefore owns only model computation and observers, while each adapter
        owns one deployment profile.

        Args:
            hidden_states: Flattened patches for the adapter's fixed grid.
            pos_embeds: Precomputed spatial position embeddings.
            position_embeddings: Precomputed RoPE cosine and sine tensors.
            cu_seqlens: Precomputed cumulative sequence lengths.
            attention_split_sizes: Static Python split sizes derived from the
                adapter's fixed grid.
            **kwargs: Additional keyword arguments forwarded to vision blocks.

        Returns:
            The version-compatible Qwen3-VL vision output.
        """
        return self._forward_impl(
            hidden_states,
            pos_embeds=pos_embeds,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            attention_split_sizes=attention_split_sizes,
            **kwargs,
        )

    def as_export_module(
        self,
        mode: ExportMode = "prefill",
        *,
        grid_thw: (Qwen3VLVisionProfile | Qwen3VLVisionGridInput | None) = None,
    ) -> nn.Module:
        """Return a static vision adapter for one explicit THW profile.

        Args:
            mode: Export mode. Qwen3-VL vision supports prefill only.
            grid_thw: Fixed ``(temporal, height, width)`` profile materialized
                by the returned adapter. Ordinary eager/PTQ execution does not
                require this value.

        Returns:
            A fixed-profile vision prefill adapter.
        """
        if mode != "prefill":
            raise ValueError(f"Unsupported Qwen3-VL vision export mode: {mode!r}")
        if grid_thw is None:
            raise ValueError(
                "Qwen3-VL vision export requires an explicit grid_thw profile."
            )
        if self._mode not in (Mode.NO_QUANT, Mode.QUANT):
            raise RuntimeError(
                "Qwen3-VL vision export requires NO_QUANT or QUANT mode, "
                f"got {self._mode}."
            )
        if self._mode is Mode.QUANT:
            for observer in self._all_observers():
                assert (
                    observer.has_qparams
                ), f"Observer {observer.name} has not been calibrated"

        from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
            Qwen3VLVisionPrefillExportAdapter,
        )

        return Qwen3VLVisionPrefillExportAdapter(self, grid_thw=grid_thw)

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module."""
        # Local observers
        yield from (
            self.obs_pos_embeds,
            self.obs_pos_add,
            self.obs_rope_cos,
            self.obs_rope_sin,
        )

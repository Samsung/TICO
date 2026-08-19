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

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register

ExportMode = str


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionPooler")
class QuantGemma4VisionPooler(QuantModuleBase):
    """PTQ wrapper for Gemma4 vision pooler.

    The pooler performs spatial pooling of vision patch tokens and scales the
    result by ``sqrt(hidden_size)`` in float32.  The wrapper observes both the
    input and the pooled output so that activation quantization ranges are
    calibrated on both sides of the pooler.

    TODO: Replace the delegation to ``self.module`` with a decomposed static
    implementation once the exact Gemma4 pooler fields are finalized.  The
    current delegation uses dynamic operations (``F.one_hot``, ``torch.div``)
    that are not ``torch.export``-friendly.
    """

    def __init__(
        self,
        fp_pooler: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        self.module = fp_pooler
        assert type(fp_pooler.hidden_size) is int
        self.hidden_size: int = fp_pooler.hidden_size
        assert type(fp_pooler.root_hidden_size) is float
        self.register_buffer(
            "root_hidden_size", torch.tensor(fp_pooler.root_hidden_size)
        )

        self.obs_act_in = self._make_obs("act_in")
        self.obs_pool_in = self._make_obs("pool_in")
        self.obs_pool_out = self._make_obs("pool_out")
        self.obs_pool_weight = self._make_obs("pool_weight")
        self.obs_root_hidden_size = self._make_obs("root_hidden_size")
        self.obs_pool_matmul_out = self._make_obs("pool_matmul_out")

    def _all_observers(self) -> Iterable[ObserverBase]:
        return (
            self.obs_act_in,
            self.obs_pool_in,
            self.obs_pool_weight,
            self.obs_root_hidden_size,
            self.obs_pool_matmul_out,
            self.obs_pool_out,
        )

    @staticmethod
    def _build_pool_weights(
        *,
        seq_len: int,
        output_length: int,
        pixel_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the pooling weight matrix and output mask.

        This replaces the dynamic ``F.one_hot`` + ``torch.div`` computation in
        ``Gemma4VisionPooler._avg_pool_by_positions`` with a one-time
        precomputation that produces the same result.

        Args:
            seq_len: Number of input patches (``S``).
            output_length: Number of output soft tokens (``V``).
            pixel_position_ids: Patch position ids ``(1, S, 2)``.

        Returns:
            Tuple of ``(weights, mask)`` where:
            - ``weights`` has shape ``(1, V, S)`` — averaging weight matrix
            - ``mask`` has shape ``(1, V)`` — valid output token mask
        """
        k = int((seq_len // output_length) ** 0.5)
        k_squared = k * k
        if k_squared * output_length != seq_len:
            raise ValueError(
                f"Cannot pool {seq_len} patches to {output_length} soft tokens: "
                f"{k=}^2 times {output_length=} must equal {seq_len}."
            )

        # Clamp padding positions (which are -1) to 0 so they don't break
        # the index computation.  Padding patches have zero hidden states so
        # they contribute nothing to the average.
        clamped = pixel_position_ids.clamp(min=0)

        # Compute image width in patches (max_x + 1).
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1  # (1, 1)

        # Compute which output slot each input patch maps to.
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")  # (1, S, 2)
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]  # (1, S)

        # Build the weight matrix
        weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared

        # Transpose for matmul: (1, V, S) so that weights @ hidden_states = (1, V, D)
        weights = weights.transpose(1, 2)  # (1, V, S)

        # Compute output mask: True where at least one patch contributes.
        mask = torch.logical_not((weights == 0).all(dim=2))  # (1, V)

        return weights, mask

    @classmethod
    def _build_export_pool_weights(
        cls,
        *,
        seq_len: int,
        output_length: int,
        pixel_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Build a padding-baked pooling matrix for one static profile.

        ``padding_positions`` is completely determined by the construction-time
        position profile.  Instead of exporting a Boolean mask and applying
        ``masked_fill`` at runtime, this helper zeros the corresponding columns
        of the pooling matrix before tracing.  For any input hidden states,

        ``W_baked @ hidden == W @ hidden.masked_fill(padding, 0)``.

        The full pooling output keeps ``output_length`` rows so standalone
        pooler export preserves its eager pooled-feature shape.  The returned
        integer records how many output rows are valid; the full vision-model
        export uses it to remove the static invalid suffix without a Boolean
        output mask.
        """
        if (
            pixel_position_ids.dim() != 3
            or pixel_position_ids.shape[0] != 1
            or pixel_position_ids.shape[-1] != 2
        ):
            raise ValueError(
                "Gemma4 VisionPooler export requires pixel_position_ids with "
                "shape (1, num_patches, 2), got "
                f"{tuple(pixel_position_ids.shape)}."
            )

        weights, _ = cls._build_pool_weights(
            seq_len=seq_len,
            output_length=output_length,
            pixel_position_ids=pixel_position_ids,
        )
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        weights = weights.masked_fill(padding_positions.unsqueeze(1), 0.0)

        valid_outputs = torch.logical_not((weights == 0).all(dim=2))
        valid_row = valid_outputs[0]
        num_valid_outputs = int(valid_row.sum().item())
        if num_valid_outputs <= 0:
            raise ValueError(
                "Gemma4 VisionPooler export profile contains no valid pooled "
                "output positions."
            )

        # Static Gemma4 profiles use a row-major rectangular patch grid followed
        # by padding.  Their valid pooled rows must therefore form one prefix,
        # which can later be represented by Slice instead of a Boolean mask or
        # an integer Gather input.
        expected_valid_row = (
            torch.arange(output_length, device=valid_row.device) < num_valid_outputs
        )
        if not torch.equal(valid_row, expected_valid_row):
            valid_indices = torch.nonzero(valid_row, as_tuple=True)[0].tolist()
            raise ValueError(
                "Gemma4 VisionPooler static export requires valid pooled "
                "positions to form a contiguous prefix, got indices "
                f"{valid_indices}."
            )

        return weights, num_valid_outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ):
        """Run the pooler and observe pooled visual features.

        Args:
            hidden_states: Vision encoder output with shape ``(B, S, D)``.
            pixel_position_ids: Patch position ids with shape ``(B, S, 2)``.
            padding_positions: Boolean padding mask with shape ``(B, S)``.
            output_length: Number of soft tokens to produce.
        """
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."
            )

        pool_weights, pool_mask = self._build_pool_weights(
            seq_len=pixel_position_ids.shape[1],
            output_length=output_length,
            pixel_position_ids=pixel_position_ids,
        )

        hidden_states = self._fq(hidden_states, self.obs_act_in)

        # Step 1: Zero out padding positions.
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0)

        # Step 2: Fake-quantize input (collects stats in CALIB, applies Q-DQ in QUANT).
        hidden_states = self._fq(hidden_states, self.obs_pool_in)

        # Step 3: Fake-quantize pool_weights (weight quantization for matmul).
        pool_weights_q = self._fq(pool_weights, self.obs_pool_weight)

        # Step 4: Spatial pooling via precomputed weight matrix.
        # pool_weights_q: (1, V, S), hidden_states.float(): (1, S, D)
        pooled: torch.Tensor = pool_weights_q @ hidden_states.float()  # (1, V, D)

        # Step 5: Fake-quantize matmul output (intermediate activation).
        pooled = self._fq(pooled, self.obs_pool_matmul_out)

        # Step 6: Scale by sqrt(hidden_size) in float32.
        root_hidden_size = self._fq(self.root_hidden_size, self.obs_root_hidden_size)
        pooled = pooled * root_hidden_size

        # Step 7: Fake-quantize final output (collects stats in CALIB, applies Q-DQ in QUANT).
        pooled = self._fq(pooled, self.obs_pool_out)

        # Step 8: Apply precomputed output mask.
        updated_padding = pool_mask.expand(pooled.shape[0], -1)  # (1, V)

        return pooled, updated_padding

    def forward_export(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decomposed static pooler forward.

        Construction-time specialization has already baked padded input columns
        into ``pool_weights``.  The runtime graph therefore contains neither a
        Boolean padding input nor a ``masked_fill`` operation.

        The returned tensor retains all fixed ``output_length`` rows.  Invalid
        suffix rows are zero and are stripped by ``QuantGemma4VisionModel``
        using the construction-time ``num_valid_outputs`` integer.

        Args:
            hidden_states: Vision encoder output ``(1, S, D)``.

        Returns:
            Pooled features with shape ``(1, V, D)`` in float32.
        """
        hidden_states = self._fq(hidden_states, self.obs_act_in)

        # Step 1: Fake-quantize input (collects stats in CALIB, applies Q-DQ in QUANT).
        hidden_states = self._fq(hidden_states, self.obs_pool_in)

        # Step 2: Fake-quantize padding-baked pool weights.
        pool_weights_q = self._fq(self.pool_weights, self.obs_pool_weight)

        # Step 3: Spatial pooling via precomputed weight matrix.
        # pool_weights_q: (1, V, S), hidden_states.float(): (1, S, D)
        pooled = pool_weights_q @ hidden_states.float()  # (1, V, D)

        # Step 4: Fake-quantize matmul output (intermediate activation).
        pooled = self._fq(pooled, self.obs_pool_matmul_out)

        # Step 5: Scale by sqrt(hidden_size) in float32.
        root_hidden_size = self._fq(self.root_hidden_size, self.obs_root_hidden_size)
        pooled = pooled * root_hidden_size

        # Step 6: Fake-quantize final output.
        pooled = self._fq(pooled, self.obs_pool_out)

        return pooled

    def as_export_module(
        self,
        mode: ExportMode = "prefill",
        *,
        output_length: int,
        pixel_position_ids: torch.Tensor,
    ) -> nn.Module:
        if mode != "prefill":
            raise ValueError(f"Unsupported Gemma4 VisionPooler export mode: {mode!r}")

        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionPoolerPrefillExportAdapter,
        )

        if self._mode not in (Mode.NO_QUANT, Mode.QUANT):
            raise RuntimeError(
                "Gemma4 VisionPooler export requires NO_QUANT or QUANT mode, "
                f"got {self._mode}."
            )

        if self._mode is Mode.QUANT:
            # Make sure that all observers are calibrated.
            for obs in self._all_observers():
                if isinstance(obs, AffineObserverBase):
                    assert obs.has_qparams

        # Precompute a numeric pooling matrix with padding already baked in.
        weights, num_valid_outputs = self._build_export_pool_weights(
            seq_len=pixel_position_ids.shape[1],
            output_length=output_length,
            pixel_position_ids=pixel_position_ids,
        )
        self.register_buffer("pool_weights", weights)
        self.num_valid_outputs = num_valid_outputs

        if self._mode is Mode.QUANT:
            # Collect statistics about generated pool weights and compute qparams.
            obs_pool_weight_enabled: bool = self.obs_pool_weight.enabled
            self.obs_pool_weight.enabled = True
            self.obs_pool_weight.reset()
            self.obs_pool_weight.collect(self.pool_weights)
            self.obs_pool_weight.compute_qparams()
            self.obs_pool_weight.enabled = obs_pool_weight_enabled

        return Gemma4VisionPoolerPrefillExportAdapter(self)

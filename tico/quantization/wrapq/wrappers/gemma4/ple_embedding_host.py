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

"""Host-side ``.pt`` artifact for the Gemma4 PLE token-identity lookup.

The dense Gemma4 ``embed_tokens_per_layer`` table has
``vocab_size_per_layer_input x num_hidden_layers x hidden_size_per_layer_input``
elements (35 x 256 columns for E2B, 42 x 256 for E4B). At 8-bit weights,
the E4B table alone is 2.625 GiB. The recipe keeps these large tables in the
shared host ``ple_embedding`` torch artifact under its existing ``auto``
policy. The complete table is loaded in memory; this is not streaming or
out-of-core lookup.

The artifact stores the complete stage contract, not just a weight tensor:
the (integer) table with its weight qparams, ``embed_scale``, the packed
reshape geometry, and every activation observer that
``Gemma4PLEEmbeddingExportAdapter`` applies in ``QUANT`` mode. The host module
``Gemma4PLEEmbeddingHostTable`` replays the same operations so its output is
bit-identical to the export adapter (and therefore to the Circle graph).
"""

from pathlib import Path
from typing import Any, cast, Mapping, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.wrappers.gemma4.embedding_scale_fusion import (
    FusedGemma4PLEEmbedding,
)

if TYPE_CHECKING:
    from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
        Gemma4PLEEmbeddingExportAdapter,
    )
    from tico.quantization.wrapq.wrappers.gemma4.quant_text_scaled_word_embedding import (
        QuantGemma4TextScaledWordEmbedding,
    )


PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1
# Version 2 has the same tensor payload but omits the embed_scale operation.
# Old readers reject it instead of silently replaying the unfused contract.
PLE_EMBEDDING_FUSED_ARTIFACT_SCHEMA_VERSION = 2
PLE_EMBEDDING_ARTIFACT_STAGE = "ple_embedding"

# flatbuffers builders cannot grow beyond 2 GiB. Circle stores the packed table
# as one constant buffer, so this is the hard cap for a single-file export.
CIRCLE_FLATBUFFER_LIMIT_BYTES = 2**31

# Observers applied by QuantGemma4TextScaledWordEmbedding and the PLE adapter,
# in execution order.
_ACTIVATION_OBSERVER_KEYS = (
    "embedding",
    "embed_scale",
    "act_out",
    "per_layer_token_inputs",
)


def _unwrap_embedding(adapter: nn.Module) -> "QuantGemma4TextScaledWordEmbedding":
    """Return the QuantGemma4TextScaledWordEmbedding behind the PLE adapter."""
    wrapper = adapter.embed_tokens_per_layer
    quant_embedding = getattr(wrapper, "wrapped", wrapper)
    if not hasattr(quant_embedding, "module") or not hasattr(
        quant_embedding, "obs_weight"
    ):
        raise TypeError(
            "Gemma4 PLE embedding artifact requires a "
            "QuantGemma4TextScaledWordEmbedding wrapper."
        )
    return cast("QuantGemma4TextScaledWordEmbedding", quant_embedding)


def _require_affine(observer: Any, name: str) -> AffineObserverBase:
    """Return an affine observer with frozen qparams or raise."""
    if not isinstance(observer, AffineObserverBase):
        raise TypeError(
            f"Gemma4 PLE embedding artifact requires an affine observer for "
            f"{name!r}, got {type(observer).__name__}."
        )
    observer.require_frozen()
    return observer


def _require_activation(observer: Any, name: str) -> ObserverBase:
    """Admit frozen affine grids or the explicit stateless identity policy."""
    if not isinstance(observer, (AffineObserverBase, IdentityObserver)):
        raise TypeError(
            f"Gemma4 PLE embedding artifact requires an affine/identity observer for {name!r}, "
            f"got {type(observer).__name__}."
        )
    observer.require_frozen()
    return observer


def _observer_payload(observer: ObserverBase) -> dict[str, Any]:
    """Serialize a frozen grid, or a legacy-reader-compatible identity bypass."""
    state = observer.require_frozen()
    if isinstance(observer, IdentityObserver) or not observer.fake_quant_enabled:
        # Schema 1/2 readers return immediately when FQ is disabled. Empty
        # qparams explicitly avoid inventing an affine grid for identity sites.
        return {
            "kind": "identity" if isinstance(observer, IdentityObserver) else "bypass",
            "scale": torch.empty(0),
            "zero_point": torch.empty(0, dtype=torch.int),
            "quant_min": 0,
            "quant_max": 0,
            "channel_axis": None,
            "dtype": "identity",
            "qscheme": "identity",
            "fake_quant_enabled": False,
        }
    assert isinstance(observer, AffineObserverBase)
    if state.scale is None or state.zero_point is None:
        raise RuntimeError(
            "Quantized Gemma4 PLE artifact requires an affine frozen grid; "
            "use the raw float artifact or enable quantization, reprepare and recalibrate."
        )
    return {
        "scale": state.scale.detach().cpu(),
        "zero_point": state.zero_point.detach().cpu().to(torch.int),
        "quant_min": int(observer.dtype.qmin),
        "quant_max": int(observer.dtype.qmax),
        "channel_axis": (
            None if observer.channel_axis is None else int(observer.channel_axis)
        ),
        "dtype": str(observer.dtype),
        "qscheme": str(observer.qscheme),
        "fake_quant_enabled": bool(observer.fake_quant_enabled),
    }


def _apply_observer_payload(
    x: torch.Tensor, payload: Mapping[str, Any]
) -> torch.Tensor:
    """Replay ``AffineObserverBase.fake_quant`` from a serialized payload."""
    if not payload["fake_quant_enabled"]:
        return x
    scale = payload["scale"].to(x.device)
    zero_point = payload["zero_point"].to(x.device, dtype=torch.int)
    if payload["channel_axis"] is None:
        return torch.fake_quantize_per_tensor_affine(
            x,
            scale=scale,
            zero_point=zero_point,
            quant_min=int(payload["quant_min"]),
            quant_max=int(payload["quant_max"]),
        )
    return torch.fake_quantize_per_channel_affine(
        x,
        scale=scale,
        zero_point=zero_point,
        axis=int(payload["channel_axis"]),
        quant_min=int(payload["quant_min"]),
        quant_max=int(payload["quant_max"]),
    )


def resolve_gemma4_ple_layer_range(
    num_hidden_layers: int, layer_start: int = 0, layer_end: Optional[int] = None
) -> tuple[int, int]:
    """Validate a nonempty half-open range in original decoder layer indices."""
    end = num_hidden_layers if layer_end is None else layer_end
    if (
        not isinstance(layer_start, int)
        or not isinstance(end, int)
        or not (0 <= layer_start < end <= num_hidden_layers)
    ):
        raise ValueError(
            f"Gemma4 PLE layer range [{layer_start}, {end}) must be within "
            f"[0, {num_hidden_layers}) and nonempty."
        )
    return layer_start, end


def validate_gemma4_ple_embedding_slicing(
    embedding_wrapper: nn.Module, token_observer: Any, *, quantized: bool
) -> None:
    """Prove that selected lookup preserves every frozen producer boundary.

    Activations must be pointwise scalar affine or identity. Weight grids are
    independent: per-vocabulary-row and per-column affine grids can be selected
    exactly without estimating new qparams.
    """
    embedding = getattr(embedding_wrapper, "wrapped", embedding_wrapper)
    table: Any = getattr(embedding, "module", embedding)
    if getattr(table, "max_norm", None) is not None:
        raise ValueError("Gemma4 selective PLE lookup does not support max_norm.")
    if not quantized:
        return
    embedding = cast("QuantGemma4TextScaledWordEmbedding", embedding)
    embedding.require_entry_available()
    if getattr(embedding, "_mode", None) is not Mode.QUANT:
        raise RuntimeError(
            "Gemma4 selective PLE lookup requires the frozen embedding QUANT mode."
        )
    activations = {
        "embedding": embedding.obs_embedding,
        "act_out": embedding.obs_act_out,
        "per_layer_token_inputs": token_observer,
    }
    if not isinstance(table, FusedGemma4PLEEmbedding):
        activations["embed_scale"] = embedding.obs_embed_scale
    for name, observer in activations.items():
        if not isinstance(observer, (AffineObserverBase, IdentityObserver)) or (
            isinstance(observer, AffineObserverBase)
            and observer.channel_axis is not None
        ):
            raise ValueError(
                "Gemma4 selective PLE lookup requires frozen per-tensor affine/identity "
                f"activation policy at {name}; got {type(observer).__name__}."
            )
        observer.require_frozen()
    weight_observer = embedding.obs_weight
    if not isinstance(weight_observer, AffineObserverBase):
        raise ValueError(
            "Gemma4 selective PLE lookup supports affine weight policy only."
        )
    weight_state = weight_observer.require_frozen()
    if weight_state.scale is None or weight_state.zero_point is None:
        raise ValueError(
            "Gemma4 selective PLE lookup requires an enabled affine weight grid for the shared host table."
        )
    if isinstance(weight_observer, AffineObserverBase):
        if weight_observer.channel_axis not in (None, 0, 1, -1, -2):
            raise ValueError(
                "Gemma4 selective PLE lookup weight channel axis must be 0 or 1."
            )
        weight_observer.validate_execution(table.weight.device)


def lookup_gemma4_selected_token_inputs(
    input_ids: torch.Tensor,
    *,
    embed_tokens_per_layer: nn.Module,
    num_hidden_layers: int,
    hidden_size_per_layer_input: int,
    token_inputs_observer: Any,
    quantized: bool,
    layer_start: int,
    layer_end: int,
) -> torch.Tensor:
    """Gather only requested rows/columns, then replay the original lookup ABI.

    No full floating table, packed S*L*P activation, or phase-specific table is
    constructed. Repeated token IDs select their original row grid repeatedly.
    """
    validate_gemma4_ple_embedding_slicing(
        embed_tokens_per_layer, token_inputs_observer, quantized=quantized
    )
    start, end = resolve_gemma4_ple_layer_range(
        num_hidden_layers, layer_start, layer_end
    )
    embedding = cast(
        "QuantGemma4TextScaledWordEmbedding",
        getattr(embed_tokens_per_layer, "wrapped", embed_tokens_per_layer),
    )
    table: Any = getattr(embedding, "module", embedding)
    column_start = start * hidden_size_per_layer_input
    column_end = end * hidden_size_per_layer_input
    width = column_end - column_start
    # Slice is a view; advanced indexing allocates only selected rows/columns.
    selected = F.embedding(
        input_ids,
        table.weight[:, column_start:column_end],
        padding_idx=table.padding_idx,
    ).reshape(-1, width)
    if quantized:
        observer = cast(AffineObserverBase, embedding.obs_weight)
        if observer.channel_axis is None:
            selected = observer.fake_quant(selected)
        else:
            state = observer.require_frozen()
            assert state.scale is not None and state.zero_point is not None
            axis = int(observer.channel_axis) % 2
            if axis == 0:
                indices = input_ids.reshape(-1).to(state.scale.device)
                scale = state.scale[indices].to(selected.device)
                zp = state.zero_point[indices].to(selected.device, dtype=torch.int)
            else:
                scale = state.scale[column_start:column_end].to(selected.device)
                zp = state.zero_point[column_start:column_end].to(
                    selected.device, dtype=torch.int
                )
            selected = torch.fake_quantize_per_channel_affine(
                selected,
                scale=scale,
                zero_point=zp,
                axis=axis,
                quant_min=int(observer.dtype.qmin),
                quant_max=int(observer.dtype.qmax),
            )
    result = selected.reshape(*input_ids.shape, width)
    if quantized:
        result = embedding.obs_embedding.fake_quant(result)
    if not isinstance(table, FusedGemma4PLEEmbedding):
        scale = table.embed_scale
        if quantized:
            scale = embedding.obs_embed_scale.fake_quant(scale)
        result = result * scale.to(dtype=result.dtype, device=result.device)
    if quantized:
        result = embedding.obs_act_out.fake_quant(result)
    result = result.reshape(*input_ids.shape, end - start, hidden_size_per_layer_input)
    return token_inputs_observer.fake_quant(result) if quantized else result


def _weight_storage_bytes_per_element(
    quantized: bool, embedding: "QuantGemma4TextScaledWordEmbedding"
) -> float:
    """Return the bytes one table element occupies in a Circle constant buffer."""
    if not quantized:
        return float(embedding.module.weight.element_size())
    bits = int(embedding.obs_weight.dtype.bits)
    if bits <= 4:
        # Circle packs 4-bit weights two per byte.
        return 0.5
    return float((bits + 7) // 8)


def estimate_gemma4_ple_embedding_circle_bytes(adapter: nn.Module) -> int:
    """Estimate the Circle constant-buffer size of the PLE lookup table.

    The estimate covers only the packed weight table, which dominates the
    artifact. Callers compare it against ``CIRCLE_FLATBUFFER_LIMIT_BYTES`` to
    decide whether the stage can be serialized as one Circle file.
    """
    embedding = _unwrap_embedding(adapter)
    numel = int(embedding.module.weight.numel())
    return int(
        numel
        * _weight_storage_bytes_per_element(
            cast("Gemma4PLEEmbeddingExportAdapter", adapter).quantized, embedding
        )
    )


def _int_storage_dtype(bits: int, signed: bool) -> torch.dtype:
    """Return the torch dtype used to store an integer table of ``bits``."""
    if bits <= 8:
        return torch.int8 if signed else torch.uint8
    if bits <= 16 and signed:
        return torch.int16
    return torch.int32


def _quantized_weight_payload(
    embedding: "QuantGemma4TextScaledWordEmbedding",
) -> dict[str, Any]:
    """Return the integer table and qparams of the frozen weight observer.

    The integer values are recovered from the fake-quantized weight so that
    dequantization ``(q - zp) * scale`` reproduces ``fake_quant(weight)``
    bit-exactly with the same float32 operations as the fake-quant kernel.
    """
    observer = _require_affine(embedding.obs_weight, "weight")
    weight = embedding.module.weight.detach()
    fake_quantized = observer.fake_quant(weight).cpu()
    state = observer.require_frozen()
    if state.scale is None or state.zero_point is None:
        raise RuntimeError(
            "Quantized Gemma4 PLE artifact requires an affine frozen grid; "
            "use the raw float artifact or enable quantization, reprepare and recalibrate."
        )
    scale = state.scale.detach().cpu()
    zero_point = state.zero_point.detach().cpu().to(torch.int)
    channel_axis = observer.channel_axis

    if channel_axis is None:
        scale_view = scale
        zp_view = zero_point
    else:
        view_shape = [1] * weight.dim()
        view_shape[int(channel_axis)] = -1
        scale_view = scale.reshape(view_shape)
        zp_view = zero_point.reshape(view_shape)

    qmin, qmax = int(observer.dtype.qmin), int(observer.dtype.qmax)
    int_weight = torch.clamp(
        torch.round(fake_quantized.to(torch.float32) / scale_view.to(torch.float32))
        + zp_view.to(torch.float32),
        qmin,
        qmax,
    ).to(_int_storage_dtype(int(observer.dtype.bits), bool(observer.dtype.signed)))

    return {
        "weight_int": int_weight,
        "weight_scale": scale,
        "weight_zero_point": zero_point,
        "weight_channel_axis": (None if channel_axis is None else int(channel_axis)),
        "weight_dtype": str(observer.dtype),
        "weight_qscheme": str(observer.qscheme),
        "weight_float_dtype": str(weight.dtype).removeprefix("torch."),
    }


def build_gemma4_ple_embedding_artifact(adapter: nn.Module) -> dict[str, Any]:
    """Return the host ``ple_embedding`` payload for a PLE embedding adapter."""
    adapter = cast("Gemma4PLEEmbeddingExportAdapter", adapter)
    adapter.require_available()
    if adapter.layer_start != 0 or adapter.layer_end != adapter.num_hidden_layers:
        raise ValueError(
            "Save one shared full PLE table; layer-range artifacts would duplicate the lookup payload."
        )
    embedding = _unwrap_embedding(adapter)
    module = embedding.module
    quantized = bool(adapter.quantized)

    artifact: dict[str, Any] = {
        "schema_version": (
            PLE_EMBEDDING_FUSED_ARTIFACT_SCHEMA_VERSION
            if isinstance(module, FusedGemma4PLEEmbedding)
            else PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION
        ),
        "stage": PLE_EMBEDDING_ARTIFACT_STAGE,
        "quantized": quantized,
        "num_hidden_layers": int(adapter.num_hidden_layers),
        "hidden_size_per_layer_input": int(adapter.hidden_size_per_layer_input),
        "vocab_size_per_layer_input": int(module.weight.shape[0]),
        "padding_idx": (
            None if module.padding_idx is None else int(module.padding_idx)
        ),
        "embed_scale": module.embed_scale.detach().cpu().clone(),
    }
    if not quantized:
        artifact["weight"] = module.weight.detach().cpu().clone()
        return artifact

    artifact.update(_quantized_weight_payload(embedding))
    artifact["observers"] = {
        "embedding": _observer_payload(
            _require_activation(embedding.obs_embedding, "embedding")
        ),
        "embed_scale": _observer_payload(
            _require_activation(embedding.obs_embed_scale, "embed_scale")
        ),
        "act_out": _observer_payload(
            _require_activation(embedding.obs_act_out, "act_out")
        ),
        "per_layer_token_inputs": _observer_payload(
            _require_activation(
                adapter.per_layer_token_inputs_observer, "per_layer_token_inputs"
            )
        ),
    }
    return artifact


def save_gemma4_ple_embedding_artifact(adapter: nn.Module, path: str | Path) -> Path:
    """Save the host ``ple_embedding`` artifact as a ``.pt`` file."""
    path = Path(path)
    artifact = build_gemma4_ple_embedding_artifact(adapter)
    print(f"Saving {path.name} to {path.resolve()}")
    torch.save(artifact, path)
    return path


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject payloads that are not a supported PLE embedding artifact."""
    stage = artifact.get("stage")
    if stage != PLE_EMBEDDING_ARTIFACT_STAGE:
        raise ValueError(
            f"Expected a {PLE_EMBEDDING_ARTIFACT_STAGE!r} artifact, got {stage!r}."
        )
    version = artifact.get("schema_version")
    if version not in (
        PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        PLE_EMBEDDING_FUSED_ARTIFACT_SCHEMA_VERSION,
    ):
        raise ValueError(
            "Unsupported Gemma4 PLE embedding artifact schema_version "
            f"{version!r}; expected 1 (unfused) or 2 (pre-quantization fused)."
        )
    if version == PLE_EMBEDDING_FUSED_ARTIFACT_SCHEMA_VERSION:
        scale = artifact["embed_scale"]
        if scale.numel() != 1 or float(scale) != 1.0:
            raise ValueError("Fused PLE embedding artifacts must have embed_scale=1.")
    if artifact.get("quantized"):
        missing = [
            key for key in _ACTIVATION_OBSERVER_KEYS if key not in artifact["observers"]
        ]
        if missing:
            raise ValueError(
                f"Quantized PLE embedding artifact is missing observers {missing}."
            )


class Gemma4PLEEmbeddingHostTable(nn.Module):
    """Host implementation of the ``ple_embedding`` stage from a ``.pt`` artifact.

    Input contract:
        ``input_ids`` has shape ``(1, S)`` for any ``S >= 1``.

    Output contract:
        ``per_layer_token_inputs`` has shape
        ``(1, S, num_hidden_layers, hidden_size_per_layer_input)`` and equals
        ``Gemma4PLEEmbeddingExportAdapter(input_ids)`` for the same model.
    """

    embed_scale: torch.Tensor
    weight_int: torch.Tensor
    weight_scale: torch.Tensor
    weight_zero_point: torch.Tensor
    weight: torch.Tensor

    def __init__(self, artifact: Mapping[str, Any]):
        super().__init__()
        _validate_artifact(artifact)
        self.quantized = bool(artifact["quantized"])
        self.embed_scale_fused = (
            artifact["schema_version"] == PLE_EMBEDDING_FUSED_ARTIFACT_SCHEMA_VERSION
        )
        self.num_hidden_layers = int(artifact["num_hidden_layers"])
        self.hidden_size_per_layer_input = int(artifact["hidden_size_per_layer_input"])
        self.vocab_size_per_layer_input = int(artifact["vocab_size_per_layer_input"])
        self.padding_idx: Optional[int] = artifact["padding_idx"]
        self.register_buffer("embed_scale", artifact["embed_scale"].clone())

        if not self.quantized:
            self.register_buffer("weight", artifact["weight"])
            self.observers: dict[str, dict[str, Any]] = {}
            return

        self.register_buffer("weight_int", artifact["weight_int"])
        self.register_buffer("weight_scale", artifact["weight_scale"].clone())
        self.register_buffer(
            "weight_zero_point", artifact["weight_zero_point"].to(torch.int).clone()
        )
        self.weight_channel_axis: Optional[int] = artifact["weight_channel_axis"]
        self.weight_float_dtype = getattr(torch, artifact["weight_float_dtype"])
        self.observers = {
            key: {
                **payload,
                "scale": payload["scale"].clone(),
                "zero_point": payload["zero_point"].clone(),
            }
            for key, payload in artifact["observers"].items()
        }

    @classmethod
    def from_artifact(
        cls,
        artifact: str | Path | Mapping[str, Any],
        *,
        map_location: Any = "cpu",
    ) -> "Gemma4PLEEmbeddingHostTable":
        """Load a saved ``ple_embedding`` artifact or wrap an in-memory payload."""
        payload: Mapping[str, Any]
        if isinstance(artifact, (str, Path)):
            payload = torch.load(artifact, map_location=map_location, weights_only=True)
        else:
            payload = artifact
        return cls(payload)

    def _dequantized_weight(self) -> torch.Tensor:
        """Return the table exactly as ``fake_quant(weight)`` produced it."""
        weight = self.weight_int.to(torch.float32)
        zero_point = self.weight_zero_point.to(torch.float32)
        scale = self.weight_scale.to(torch.float32)
        if self.weight_channel_axis is not None:
            view_shape = [1] * weight.dim()
            view_shape[int(self.weight_channel_axis)] = -1
            scale = scale.reshape(view_shape)
            zero_point = zero_point.reshape(view_shape)
        return ((weight - zero_point) * scale).to(self.weight_float_dtype)

    def _fq(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """Apply one replayed observer in QUANT mode only."""
        if not self.quantized:
            return x
        return _apply_observer_payload(x, self.observers[key])

    def _selected_weight(
        self, input_ids: torch.Tensor, column_start: int, column_end: int
    ) -> torch.Tensor:
        """Select original storage before dequantizing, including row qparams."""
        if not self.quantized:
            return F.embedding(
                input_ids,
                self.weight[:, column_start:column_end],
                padding_idx=self.padding_idx,
            )
        selected = F.embedding(
            input_ids,
            self.weight_int[:, column_start:column_end],
            padding_idx=self.padding_idx,
        ).to(torch.float32)
        scale = self.weight_scale
        zp = self.weight_zero_point
        if self.weight_channel_axis is not None:
            axis = int(self.weight_channel_axis) % 2
            if axis == 0:
                scale = scale[input_ids].unsqueeze(-1)
                zp = zp[input_ids].unsqueeze(-1)
            else:
                scale = scale[column_start:column_end]
                zp = zp[column_start:column_end]
        return ((selected - zp.to(torch.float32)) * scale.to(torch.float32)).to(
            self.weight_float_dtype
        )

    def validate_layer_range(self) -> None:
        """Reject unsupported selected activation grids before doing work."""
        if not self.quantized:
            return
        for key, payload in self.observers.items():
            if key == "embed_scale" and self.embed_scale_fused:
                continue
            if payload["channel_axis"] is not None:
                raise ValueError(
                    "Gemma4 selective PLE host lookup requires frozen per-tensor "
                    f"affine activation policy at {key}."
                )
        if self.weight_channel_axis not in (None, 0, 1, -1, -2):
            raise ValueError(
                "Gemma4 selective PLE host weight channel axis must be 0 or 1."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        layer_start: int = 0,
        layer_end: Optional[int] = None,
    ) -> torch.Tensor:
        """Return token PLE for a range using one shared table and frozen grids."""
        start, end = resolve_gemma4_ple_layer_range(
            self.num_hidden_layers, layer_start, layer_end
        )
        selected = start != 0 or end != self.num_hidden_layers
        if selected:
            self.validate_layer_range()
        hidden_states = self._selected_weight(
            input_ids,
            start * self.hidden_size_per_layer_input,
            end * self.hidden_size_per_layer_input,
        )
        hidden_states = self._fq(hidden_states, "embedding")
        if not getattr(self, "embed_scale_fused", False):
            scale = self._fq(self.embed_scale, "embed_scale")
            hidden_states = hidden_states * scale.to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        hidden_states = self._fq(hidden_states, "act_out")
        hidden_states = hidden_states.reshape(
            *input_ids.shape[:-1],
            -1,
            end - start,
            self.hidden_size_per_layer_input,
        )
        return self._fq(hidden_states, "per_layer_token_inputs")

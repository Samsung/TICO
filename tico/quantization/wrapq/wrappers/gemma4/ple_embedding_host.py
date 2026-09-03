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

The Gemma4 E2B ``embed_tokens_per_layer`` table has
``vocab_size_per_layer_input x num_hidden_layers x hidden_size_per_layer_input``
elements (2.35 B for E2B). That exceeds the 2 GiB flatbuffer limit of a Circle
file in float32 and even at 8-bit weights, so the shared ``ple_embedding``
stage can alternatively be saved as a torch artifact that a host runtime loads
directly.

The artifact stores the complete stage contract, not just a weight tensor:
the (integer) table with its weight qparams, ``embed_scale``, the packed
reshape geometry, and every activation observer that
``Gemma4PLEEmbeddingExportAdapter`` applies in ``QUANT`` mode. The host module
``Gemma4PLEEmbeddingHostTable`` replays the same operations so its output is
bit-identical to the export adapter (and therefore to the Circle graph).
"""

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.wrapq.observers.affine_base import AffineObserverBase

PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1
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


def _unwrap_embedding(adapter: nn.Module) -> nn.Module:
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
    return quant_embedding


def _require_affine(observer: Any, name: str) -> AffineObserverBase:
    """Return an affine observer with frozen qparams or raise."""
    if not isinstance(observer, AffineObserverBase):
        raise TypeError(
            f"Gemma4 PLE embedding artifact requires an affine observer for "
            f"{name!r}, got {type(observer).__name__}."
        )
    if not observer.has_qparams:
        raise RuntimeError(
            f"Gemma4 PLE embedding observer {name!r} has no frozen qparams. "
            "Convert the model before saving the artifact."
        )
    return observer


def _observer_payload(observer: AffineObserverBase) -> dict[str, Any]:
    """Serialize the frozen fake-quant contract of one affine observer."""
    return {
        "scale": observer._cached_scale.detach().cpu().clone(),
        "zero_point": observer._cached_zp.detach().cpu().to(torch.int).clone(),
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


def _weight_storage_bytes_per_element(quantized: bool, embedding: nn.Module) -> float:
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
    return int(numel * _weight_storage_bytes_per_element(adapter.quantized, embedding))


def _int_storage_dtype(bits: int, signed: bool) -> torch.dtype:
    """Return the torch dtype used to store an integer table of ``bits``."""
    if bits <= 8:
        return torch.int8 if signed else torch.uint8
    if bits <= 16 and signed:
        return torch.int16
    return torch.int32


def _quantized_weight_payload(embedding: nn.Module) -> dict[str, Any]:
    """Return the integer table and qparams of the frozen weight observer.

    The integer values are recovered from the fake-quantized weight so that
    dequantization ``(q - zp) * scale`` reproduces ``fake_quant(weight)``
    bit-exactly with the same float32 operations as the fake-quant kernel.
    """
    observer = _require_affine(embedding.obs_weight, "weight")
    weight = embedding.module.weight.detach()
    fake_quantized = observer.fake_quant(weight).cpu()
    scale = observer._cached_scale.detach().cpu()
    zero_point = observer._cached_zp.detach().cpu().to(torch.int)
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
    embedding = _unwrap_embedding(adapter)
    module = embedding.module
    quantized = bool(adapter.quantized)

    artifact: dict[str, Any] = {
        "schema_version": PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION,
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
            _require_affine(embedding.obs_embedding, "embedding")
        ),
        "embed_scale": _observer_payload(
            _require_affine(embedding.obs_embed_scale, "embed_scale")
        ),
        "act_out": _observer_payload(_require_affine(embedding.obs_act_out, "act_out")),
        "per_layer_token_inputs": _observer_payload(
            _require_affine(
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
    if version != PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Gemma4 PLE embedding artifact schema_version "
            f"{version!r}; expected {PLE_EMBEDDING_ARTIFACT_SCHEMA_VERSION}."
        )
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

    def __init__(self, artifact: Mapping[str, Any]):
        super().__init__()
        _validate_artifact(artifact)
        self.quantized = bool(artifact["quantized"])
        self.num_hidden_layers = int(artifact["num_hidden_layers"])
        self.hidden_size_per_layer_input = int(artifact["hidden_size_per_layer_input"])
        self.vocab_size_per_layer_input = int(artifact["vocab_size_per_layer_input"])
        self.padding_idx: Optional[int] = artifact["padding_idx"]
        self.register_buffer("embed_scale", artifact["embed_scale"].clone())

        if not self.quantized:
            self.register_buffer("weight", artifact["weight"].clone())
            self.observers: dict[str, dict[str, Any]] = {}
            return

        self.register_buffer("weight_int", artifact["weight_int"].clone())
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return observed token-identity PLE with shape ``(1, S, L, P)``."""
        weight = self._dequantized_weight() if self.quantized else self.weight
        hidden_states = F.embedding(input_ids, weight, padding_idx=self.padding_idx)
        hidden_states = self._fq(hidden_states, "embedding")
        scale = self._fq(self.embed_scale, "embed_scale")
        hidden_states = hidden_states * scale.to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        hidden_states = self._fq(hidden_states, "act_out")
        hidden_states = hidden_states.reshape(
            *input_ids.shape[:-1],
            -1,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        return self._fq(hidden_states, "per_layer_token_inputs")

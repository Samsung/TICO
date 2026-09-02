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

"""Static Circle export for the Gemma4 assistant draft-one core.

Artifacts:
  - ``gemma4_assistant_core.q.circle``: the batch=1, q_len=1 NPU core graph.
  - ``gemma4_assistant_sparse_head.pt``: host-side ordered sparse LM-head
    weights and integer metadata.
  - ``gemma4_assistant_manifest.json``: the runtime execution contract.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4_assistant.export_adapters import (
    Gemma4AssistantCoreExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_for_causal_lm import (
    QuantGemma4AssistantForCausalLM,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.sparse_head import (
    SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
    canonicalize_gemma4_assistant_static_inputs,
    GEMMA4_ASSISTANT_CORE_INPUT_NAMES,
    GEMMA4_ASSISTANT_CORE_OUTPUT_NAMES,
    Gemma4AssistantStaticShapeConfig,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    assistant_layer_type_head_dim,
    assistant_shared_kv_num_heads,
    extract_assistant_text_config,
)


GEMMA4_ASSISTANT_MANIFEST_SCHEMA_VERSION = 1

CORE_CIRCLE_ARTIFACT = "gemma4_assistant_core.q.circle"
SPARSE_HEAD_ARTIFACT = "gemma4_assistant_sparse_head.pt"
MANIFEST_ARTIFACT = "gemma4_assistant_manifest.json"


def resolve_assistant_quant_wrapper(
    model: torch.nn.Module,
) -> QuantGemma4AssistantForCausalLM:
    """Return the converted assistant quant wrapper from a checkpoint model."""
    wrapped = getattr(model, "wrapped", model)
    if not isinstance(wrapped, QuantGemma4AssistantForCausalLM):
        raise TypeError(
            "Gemma4 assistant export requires a QuantGemma4AssistantForCausalLM "
            f"(optionally inside a PTQWrapper), got {type(model).__name__}."
        )
    if wrapped._mode is not Mode.QUANT:
        raise RuntimeError(
            "Gemma4 assistant export requires a converted (QUANT) model. "
            "Run the PTQ pipeline (prepare → calibrate → convert) first."
        )
    return wrapped


def build_assistant_core_example_inputs(
    assistant: QuantGemma4AssistantForCausalLM,
    shape: Gemma4AssistantStaticShapeConfig,
) -> tuple[torch.Tensor, ...]:
    """Build canonicalized fixed-shape tracing inputs for the core graph."""
    text_config = extract_assistant_text_config(assistant.config)
    kv_heads = assistant_shared_kv_num_heads(text_config)
    full_head_dim = assistant_layer_type_head_dim(text_config, "full_attention")
    sliding_head_dim = assistant_layer_type_head_dim(text_config, "sliding_attention")
    full_valid = shape.full_kv_length
    sliding_valid = min(shape.sliding_kv_length, full_valid)

    static = canonicalize_gemma4_assistant_static_inputs(
        inputs_embeds=torch.randn(1, 1, 2 * assistant.backbone_hidden_size),
        position_ids=torch.tensor([[full_valid - 1]]),
        shared_kv_states={
            "full_attention": (
                torch.randn(1, kv_heads, full_valid, full_head_dim),
                torch.randn(1, kv_heads, full_valid, full_head_dim),
            ),
            "sliding_attention": (
                torch.randn(1, kv_heads, sliding_valid, sliding_head_dim),
                torch.randn(1, kv_heads, sliding_valid, sliding_head_dim),
            ),
        },
        shape=shape,
        model_or_config=assistant.config,
        rotary_emb=assistant.model.rotary_emb,
        mask_fill_value=float(assistant.qcfg.attention_mask_fill_value),
    )
    return static.as_tuple()


def export_gemma4_assistant_core_circle(
    assistant: QuantGemma4AssistantForCausalLM,
    shape: Gemma4AssistantStaticShapeConfig,
    output_dir: Path,
    *,
    strict: bool = False,
) -> Path:
    """Export the converted assistant draft-one core to one Circle graph."""
    import tico
    from tico.utils.utils import SuppressWarning

    adapter = Gemma4AssistantCoreExportAdapter(assistant).eval()
    example_inputs = build_assistant_core_example_inputs(assistant, shape)

    save_path = output_dir / CORE_CIRCLE_ARTIFACT
    print(f"Saving {save_path.name} to {save_path.resolve()}")
    with torch.no_grad(), SuppressWarning(UserWarning, ".*"):
        circle_model = tico.convert(adapter, example_inputs, strict=strict)
    circle_model.save(save_path)
    return save_path


def _quantized_lm_head_payload(
    assistant: QuantGemma4AssistantForCausalLM,
) -> dict[str, Any]:
    """Extract the integer tied LM-head weight and its qparams."""
    quant_linear = assistant.lm_head.wrapped
    observer = quant_linear.obs_weight
    if not getattr(observer, "has_qparams", False):
        raise RuntimeError(
            "The assistant lm_head weight observer has no qparams. "
            "Convert the model before exporting the sparse-head artifact."
        )
    weight = quant_linear.module.weight.detach().cpu()
    scale = observer._cached_scale.detach().cpu()
    zero_point = observer._cached_zp.detach().cpu()
    channel_axis = observer.channel_axis
    qmin, qmax = observer.dtype.qmin, observer.dtype.qmax

    if channel_axis is None:
        scale_view = scale
        zp_view = zero_point
    else:
        view_shape = [1] * weight.dim()
        view_shape[int(channel_axis)] = -1
        scale_view = scale.reshape(view_shape)
        zp_view = zero_point.reshape(view_shape)

    int_weight = torch.clamp(torch.round(weight / scale_view) + zp_view, qmin, qmax).to(
        torch.int32
    )

    return {
        "lm_head_weight_int": int_weight,
        "lm_head_weight_scale": scale,
        "lm_head_weight_zero_point": zero_point,
        "lm_head_weight_channel_axis": channel_axis,
        "lm_head_weight_dtype": str(observer.dtype),
        "lm_head_weight_qscheme": str(observer.qscheme),
    }


def export_gemma4_assistant_sparse_head(
    assistant: QuantGemma4AssistantForCausalLM,
    output_dir: Path,
) -> Path:
    """Save the host sparse-head artifact next to the core Circle graph."""
    masked_embedding = assistant.masked_embedding
    if masked_embedding is None:
        raise ValueError(
            "The assistant does not use ordered embeddings; there is no "
            "sparse-head artifact to export."
        )

    artifact: dict[str, Any] = {
        "schema_version": SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION,
        # token_ordering is integer metadata and is intentionally unquantized.
        "token_ordering": masked_embedding.token_ordering.detach().cpu().long(),
        "num_centroids": masked_embedding.num_centroids,
        "centroid_top_k": masked_embedding.centroid_top_k,
        "vocab_per_centroid": masked_embedding.vocab_per_centroid,
        "vocab_size": masked_embedding.vocab_size,
        "hidden_size": masked_embedding.hidden_size,
        # lm_head.weight is tied to model.embed_tokens.weight; only the single
        # quantized source of truth is stored.
        "tied_to_embedding": bool(
            getattr(assistant.config, "tie_word_embeddings", False)
        ),
    }
    artifact.update(_quantized_lm_head_payload(assistant))

    path = output_dir / SPARSE_HEAD_ARTIFACT
    torch.save(artifact, path)
    print(f"Saved sparse-head artifact to {path.resolve()}")
    return path


def _tensor_contract(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Return a JSON-safe tensor contract entry."""
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _token_ordering_sha256(token_ordering: torch.Tensor) -> str:
    """Return a stable checksum of the integer token ordering."""
    data = token_ordering.detach().cpu().long().contiguous()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def write_gemma4_assistant_manifest(
    assistant: QuantGemma4AssistantForCausalLM,
    shape: Gemma4AssistantStaticShapeConfig,
    output_dir: Path,
    *,
    source_model: str,
    quantization_profile: Mapping[str, Any],
) -> Path:
    """Write the JSON execution contract for the exported assistant."""
    import transformers

    text_config = extract_assistant_text_config(assistant.config)
    example_inputs = build_assistant_core_example_inputs(assistant, shape)
    kv_heads = assistant_shared_kv_num_heads(text_config)
    masked_embedding = assistant.masked_embedding

    output_shapes = {
        "projected_state": [1, 1, assistant.backbone_hidden_size],
        "assistant_hidden": [1, 1, assistant.hidden_size],
        "centroid_logits": [
            1,
            1,
            masked_embedding.num_centroids if masked_embedding else 0,
        ],
    }

    manifest: dict[str, Any] = {
        "schema_version": GEMMA4_ASSISTANT_MANIFEST_SCHEMA_VERSION,
        "source_model": source_model,
        "transformers_version": transformers.__version__,
        "assistant_config": {
            "backbone_hidden_size": assistant.backbone_hidden_size,
            "hidden_size": assistant.hidden_size,
            "vocab_size": assistant.vocab_size,
            "num_hidden_layers": int(text_config.num_hidden_layers),
            "num_attention_heads": int(text_config.num_attention_heads),
            "num_key_value_heads": kv_heads,
            "head_dim": int(text_config.head_dim),
            "global_head_dim": int(
                getattr(text_config, "global_head_dim", 0) or text_config.head_dim
            ),
            "layer_types": list(text_config.layer_types),
            "sliding_window": int(getattr(text_config, "sliding_window", 0) or 0),
            "use_ordered_embeddings": assistant.use_ordered_embeddings,
            "num_centroids": (
                masked_embedding.num_centroids if masked_embedding else None
            ),
            "centroid_intermediate_top_k": (
                masked_embedding.centroid_top_k if masked_embedding else None
            ),
            "vocab_per_centroid": (
                masked_embedding.vocab_per_centroid if masked_embedding else None
            ),
        },
        "static_shape": {
            "batch_size": shape.batch_size,
            "query_length": shape.query_length,
            "full_kv_length": shape.full_kv_length,
            "sliding_kv_length": shape.sliding_kv_length,
        },
        "attention_mask_fill_value": float(assistant.qcfg.attention_mask_fill_value),
        "inputs": [
            _tensor_contract(name, tensor)
            for name, tensor in zip(GEMMA4_ASSISTANT_CORE_INPUT_NAMES, example_inputs)
        ],
        "outputs": [
            {"name": name, "shape": output_shapes[name], "dtype": "float32"}
            for name in GEMMA4_ASSISTANT_CORE_OUTPUT_NAMES
        ],
        "quantization_profile": dict(quantization_profile),
        "sparse_head": {
            "execution_location": "host",
            "weight_artifact": SPARSE_HEAD_ARTIFACT,
            "token_ordering_sha256": (
                _token_ordering_sha256(masked_embedding.token_ordering)
                if masked_embedding
                else None
            ),
            "tied_weight_invariant": (
                "lm_head.weight is tied to model.embed_tokens.weight and is "
                "stored once in the sparse-head artifact"
            ),
        },
        "core_artifact": CORE_CIRCLE_ARTIFACT,
    }

    path = output_dir / MANIFEST_ARTIFACT
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved manifest to {path.resolve()}")
    return path

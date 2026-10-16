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

"""Self-contained host context and manifest for staged LLaMA Circle export."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tico.circle import CircleDocument
from tico.circle._schema import decode_text


_INPUT_ROLES = {
    "input_ids": "input_ids",
    "hidden_states": "hidden_states",
    "attention_mask": "attention_mask",
    "position_embeddings_0": "rope_cos",
    "position_embeddings_1": "rope_sin",
    "past_key_value_0": "past_key",
    "past_key_value_1": "past_value",
}
_LAYER_INPUTS = frozenset({"hidden_states", "attention_mask", "rope_cos", "rope_sin"})


def _token_id(value: Any, *, name: str, vocab_size: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"LLaMA Circle bundle {name} must be an integer or null.")
    if not 0 <= value < vocab_size:
        raise ValueError(f"LLaMA Circle bundle {name} is outside the vocabulary.")
    return value


def build_llama_bundle_context(
    *,
    qmodel: Any,
    max_seq_len: int,
    artifact_tag: str,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Capture the exact prepared-layer host inputs before writing artifacts."""
    config = qmodel.config
    dimensions = {
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads,
    }
    for name, value in dimensions.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"LLaMA Circle bundle {name} must be a positive integer.")

    layers = qmodel.model.wrapped.layers
    if not layers:
        raise ValueError("LLaMA Circle bundle requires at least one decoder layer.")
    host_context: dict[str, np.ndarray] = {}
    mask_fill_values = []
    for index, layer in enumerate(layers):
        wrapped = layer.wrapped
        for key in ("rope_cos", "rope_sin"):
            template = getattr(wrapped, f"{key}_template", None)
            if (
                not isinstance(template, torch.Tensor)
                or template.ndim != 3
                or template.shape[0] != 1
                or template.shape[1] < max_seq_len
                or template.shape[2] != dimensions["head_dim"]
            ):
                raise ValueError(
                    f"LLaMA Circle bundle layer {index} {key}_template must "
                    f"cover shape (1, {max_seq_len}, {dimensions['head_dim']})."
                )
            value = template[:, :max_seq_len, :].detach().cpu().float().numpy().copy()
            if not np.isfinite(value).all():
                raise ValueError(
                    f"LLaMA Circle bundle layer {index} {key} contains non-finite values."
                )
            host_context[f"{key}_{index}"] = value
        with np.errstate(over="ignore"):
            mask_fill = np.float32(float(wrapped.qcfg.attention_mask_fill_value))
        if not np.isfinite(mask_fill) or mask_fill >= 0:
            raise ValueError(
                "LLaMA Circle bundle mask fill values must be finite and negative."
            )
        mask_fill_values.append(mask_fill)
    host_context["mask_fill_values"] = np.asarray(mask_fill_values, dtype=np.float32)

    vocab_size = dimensions["vocab_size"]
    pad_id = getattr(config, "pad_token_id", None)
    eos_id = getattr(config, "eos_token_id", None)
    generation_config = getattr(qmodel, "generation_config", None)
    generation_pad_id = getattr(generation_config, "pad_token_id", None)
    generation_eos_id = getattr(generation_config, "eos_token_id", None)
    if generation_pad_id is not None:
        pad_id = generation_pad_id
    if generation_eos_id is not None:
        eos_id = generation_eos_id
    if tokenizer is not None:
        if not callable(getattr(tokenizer, "save_pretrained", None)):
            raise TypeError(
                "LLaMA Circle bundle tokenizer must provide save_pretrained()."
            )
        if pad_id is None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
        if eos_id is None:
            eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        if not eos_id or any(value is None for value in eos_id):
            raise ValueError(
                "LLaMA Circle bundle eos_token_id list must contain integers."
            )
        eos_id = [
            _token_id(value, name="eos_token_id", vocab_size=vocab_size)
            for value in eos_id
        ]
    else:
        eos_id = _token_id(eos_id, name="eos_token_id", vocab_size=vocab_size)
    metadata = {
        "schema_version": 1,
        "model_family": "llama",
        "execution_mode": "native" if artifact_tag == "f32" else "fake_quantize",
        "max_seq_len": max_seq_len,
        **dimensions,
        "pad_token_id": _token_id(pad_id, name="pad_token_id", vocab_size=vocab_size),
        "eos_token_id": eos_id,
        "host_context": "llama_host.npz",
        "tokenizer": "tokenizer" if tokenizer is not None else None,
    }
    return metadata, host_context


def _stage_contract(
    path: Path, *, expected_inputs: frozenset[str], outputs: list[str]
) -> dict[str, Any]:
    """Read USER INPUT order from the saved Circle instead of guessing flattening."""
    with CircleDocument.load_mapped(path) as document:
        document.verify()
        graph = document.graph()
        inputs = []
        for tensor_index in graph.inputs:
            name = decode_text(graph.subgraph.tensors[tensor_index].name).split("::")[
                -1
            ]
            if name not in _INPUT_ROLES:
                raise ValueError(
                    f"Unsupported LLaMA Circle bundle input {name!r} in {path.name}."
                )
            inputs.append(_INPUT_ROLES[name])
        if len(inputs) != len(expected_inputs) or set(inputs) != expected_inputs:
            raise ValueError(
                f"LLaMA Circle bundle {path.name} inputs {inputs!r} do not "
                f"match required roles {sorted(expected_inputs)!r}."
            )
        if len(graph.outputs) != len(outputs):
            raise ValueError(
                f"LLaMA Circle bundle {path.name} requires {len(outputs)} outputs, "
                f"got {len(graph.outputs)}."
            )
    return {"file": path.name, "inputs": inputs, "outputs": outputs}


def _save_tokenizer(tokenizer: Any, output_dir: Path) -> None:
    """Replace the dedicated tokenizer directory only after a fresh save succeeds."""
    with tempfile.TemporaryDirectory(
        dir=output_dir, prefix=".llama-tokenizer-"
    ) as temporary:
        saved = Path(temporary) / "tokenizer"
        saved.mkdir()
        tokenizer.save_pretrained(saved)
        destination = output_dir / "tokenizer"
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        saved.replace(destination)


def write_llama_bundle(
    *,
    metadata: dict[str, Any],
    host_context: dict[str, np.ndarray],
    output_dir: Path,
    artifact_tag: str,
    tokenizer: Any,
) -> Path:
    """Verify saved stages and publish the manifest after all bundle data exists."""
    layer_outputs = ["hidden_states", "new_key", "new_value"]
    layers = []
    for index in range(len(host_context["mask_fill_values"])):
        layers.append(
            {
                "prefill": _stage_contract(
                    output_dir / f"decoder_layer_prefill_{index}.{artifact_tag}.circle",
                    expected_inputs=_LAYER_INPUTS,
                    outputs=layer_outputs,
                ),
                "decode": _stage_contract(
                    output_dir / f"decoder_layer_decode_{index}.{artifact_tag}.circle",
                    expected_inputs=_LAYER_INPUTS | {"past_key", "past_value"},
                    outputs=layer_outputs,
                ),
            }
        )
    manifest = {
        **metadata,
        "stages": {
            "token_embedding": _stage_contract(
                output_dir / f"token_embedding.{artifact_tag}.circle",
                expected_inputs=frozenset({"input_ids"}),
                outputs=["hidden_states"],
            ),
            "lm_head": _stage_contract(
                output_dir / f"lm_head.{artifact_tag}.circle",
                expected_inputs=frozenset({"hidden_states"}),
                outputs=["logits"],
            ),
            "layers": layers,
        },
    }
    archive_values: dict[str, Any] = dict(host_context)
    np.savez(output_dir / metadata["host_context"], **archive_values)
    if tokenizer is not None:
        _save_tokenizer(tokenizer, output_dir)
    path = output_dir / "llama_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path

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

"""CPU reference generation from saved LLaMA Circle stages and host context.

No original model, checkpoint, Transformers configuration, or tokenizer is needed
for token-ID inference. Quantized bundles use the Circle runtime's explicit
fake-quantize semantics, rather than claiming bit-exact integer-backend execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode, RuntimeTensor


_INPUT_ROLES = {
    "input_ids": "input_ids",
    "hidden_states": "hidden_states",
    "attention_mask": "attention_mask",
    "position_embeddings_0": "rope_cos",
    "position_embeddings_1": "rope_sin",
    "past_key_value_0": "past_key",
    "past_key_value_1": "past_value",
}


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _bundle_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a relative bundle path.")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{name} must stay inside the bundle directory.")
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def _check_contract(
    tensor: RuntimeTensor,
    shape: tuple[int, ...],
    dtype: np.dtype,
    mode: ExecutionMode,
    name: str,
    *,
    dynamic_sequence: bool = False,
) -> None:
    logical_dtype = (
        np.dtype(np.float32)
        if mode is ExecutionMode.FAKE_QUANTIZE and tensor.is_quantized_integer
        else tensor.dtype
    )
    if logical_dtype != dtype:
        raise ValueError(
            f"{name} has incompatible dtype {logical_dtype}; expected {dtype}."
        )
    signature = tensor.shape_signature or tensor.shape
    if len(signature) != len(shape):
        raise ValueError(
            f"{name} has incompatible shape {signature}; expected {shape}."
        )
    for axis, (actual, expected) in enumerate(zip(signature, shape)):
        if dynamic_sequence and axis == 1:
            if actual != -1:
                raise ValueError(f"{name} must have a dynamic sequence dimension.")
        elif actual != expected:
            raise ValueError(
                f"{name} has incompatible shape {signature}; expected {shape}."
            )


@dataclass
class _Stage:
    runtime: CircleReferenceRuntime
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    mode: ExecutionMode

    @classmethod
    def load(
        cls,
        root: Path,
        spec: Any,
        *,
        inputs: dict[str, tuple[tuple[int, ...], np.dtype]],
        outputs: dict[str, tuple[tuple[int, ...], np.dtype]],
        mode: ExecutionMode,
        name: str,
        dynamic_sequence: bool = False,
    ) -> _Stage:
        if not isinstance(spec, dict):
            raise ValueError(f"Missing or invalid stage {name}.")
        input_roles = spec.get("inputs")
        output_roles = spec.get("outputs")
        if (
            not isinstance(input_roles, list)
            or not all(isinstance(role, str) for role in input_roles)
            or len(input_roles) != len(inputs)
            or set(input_roles) != set(inputs)
            or output_roles != list(outputs)
        ):
            raise ValueError(f"{name} has incompatible input/output roles.")
        path = _bundle_path(root, spec.get("file"), f"{name} Circle artifact")
        runtime = CircleReferenceRuntime(CircleDocument.load(path))
        if len(runtime.input_tensors) != len(inputs) or len(
            runtime.output_tensors
        ) != len(outputs):
            raise ValueError(f"{name} has incompatible Circle input/output counts.")
        if mode is ExecutionMode.NATIVE and any(
            tensor.is_quantized_integer for tensor in runtime.program.tensors
        ):
            raise ValueError(f"{name} contains quantized tensors in a native bundle.")
        for role, tensor in zip(input_roles, runtime.input_tensors):
            source_name = tensor.name.rsplit("::", 1)[-1]
            if _INPUT_ROLES.get(source_name) != role:
                raise ValueError(
                    f"{name} input {tensor.name!r} does not match role {role!r}."
                )
            shape, dtype = inputs[role]
            _check_contract(
                tensor,
                shape,
                dtype,
                mode,
                f"{name}.{role}",
                dynamic_sequence=dynamic_sequence,
            )
        for role, tensor in zip(outputs, runtime.output_tensors):
            shape, dtype = outputs[role]
            _check_contract(
                tensor,
                shape,
                dtype,
                mode,
                f"{name}.{role}",
                dynamic_sequence=dynamic_sequence,
            )
        return cls(runtime, tuple(input_roles), tuple(outputs), mode)

    def run(self, **values: np.ndarray) -> dict[str, np.ndarray]:
        result = self.runtime.run(
            [values[role] for role in self.inputs], mode=self.mode
        )
        return dict(zip(self.outputs, result.outputs))


class LlamaCircleRuntime:
    """Load a ``circle_bundle`` and run batch-one, greedy LLaMA generation.

    ``prefill`` takes compact, unpadded INT64 token IDs and pads them to the
    exported static shape internally. The host owns masks, saved RoPE selection,
    and external KV caches. All neural-network computation runs saved Circle
    stages. This is a correctness reference, not a performance runtime.
    """

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir).resolve()
        manifest_path = self.bundle_dir / "llama_manifest.json"
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise ValueError("LLaMA bundle manifest must be a JSON object.")
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            raise ValueError("Unsupported LLaMA bundle schema_version; expected 1.")
        if manifest.get("model_family") != "llama":
            raise ValueError("Bundle model_family must be 'llama'.")
        self.execution_mode = manifest.get("execution_mode")
        modes = {
            "native": ExecutionMode.NATIVE,
            "fake_quantize": ExecutionMode.FAKE_QUANTIZE,
        }
        if not isinstance(self.execution_mode, str) or self.execution_mode not in modes:
            raise ValueError("Unsupported LLaMA bundle execution_mode.")
        mode = modes[self.execution_mode]
        self.max_seq_len = _positive_int(manifest.get("max_seq_len"), "max_seq_len")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2 for paired generation.")
        self.hidden_size = _positive_int(manifest.get("hidden_size"), "hidden_size")
        self.vocab_size = _positive_int(manifest.get("vocab_size"), "vocab_size")
        self.num_key_value_heads = _positive_int(
            manifest.get("num_key_value_heads"), "num_key_value_heads"
        )
        self.head_dim = _positive_int(manifest.get("head_dim"), "head_dim")
        self.pad_token_id = manifest.get("pad_token_id")
        if self.pad_token_id is not None:
            self._validate_token_id(self.pad_token_id, "pad_token_id")
        self.eos_token_id = manifest.get("eos_token_id")
        self._eos_ids(self.eos_token_id)
        tokenizer = manifest.get("tokenizer")
        self.tokenizer_dir = (
            None
            if tokenizer is None
            else _bundle_path(self.bundle_dir, tokenizer, "tokenizer")
        )
        if self.tokenizer_dir is not None and not self.tokenizer_dir.is_dir():
            raise ValueError("Bundle tokenizer must refer to a directory.")
        stages = manifest.get("stages")
        if not isinstance(stages, dict):
            raise ValueError("Bundle stages must be an object.")
        layers = stages.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("Bundle must contain at least one decoder layer.")
        self.num_hidden_layers = len(layers)
        self._load_host_context(manifest.get("host_context"))
        f32, i64 = np.dtype(np.float32), np.dtype(np.int64)
        seq, hidden = self.max_seq_len, self.hidden_size
        self._embedding = _Stage.load(
            self.bundle_dir,
            stages.get("token_embedding"),
            inputs={"input_ids": ((1, seq), i64)},
            outputs={"hidden_states": ((1, seq, hidden), f32)},
            mode=mode,
            name="token_embedding",
            dynamic_sequence=True,
        )
        self._lm_head = _Stage.load(
            self.bundle_dir,
            stages.get("lm_head"),
            inputs={"hidden_states": ((1, 1, hidden), f32)},
            outputs={"logits": ((1, 1, self.vocab_size), f32)},
            mode=mode,
            name="lm_head",
        )
        self._prefill_layers: list[_Stage] = []
        self._decode_layers: list[_Stage] = []
        cache_shape = (1, self.num_key_value_heads, seq - 1, self.head_dim)
        for index, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"Invalid decoder layer {index}.")
            for phase, query_len, destination in (
                ("prefill", seq, self._prefill_layers),
                ("decode", 1, self._decode_layers),
            ):
                inputs: dict[str, tuple[tuple[int, ...], np.dtype]] = {
                    "hidden_states": ((1, query_len, hidden), f32),
                    "attention_mask": ((1, query_len, seq), f32),
                    "rope_cos": ((1, query_len, self.head_dim), f32),
                    "rope_sin": ((1, query_len, self.head_dim), f32),
                }
                if phase == "decode":
                    inputs.update(
                        past_key=(cache_shape, f32), past_value=(cache_shape, f32)
                    )
                kv_shape = (1, self.num_key_value_heads, query_len, self.head_dim)
                destination.append(
                    _Stage.load(
                        self.bundle_dir,
                        layer.get(phase),
                        inputs=inputs,
                        outputs={
                            "hidden_states": ((1, query_len, hidden), f32),
                            "new_key": (kv_shape, f32),
                            "new_value": (kv_shape, f32),
                        },
                        mode=mode,
                        name=f"layer_{index}.{phase}",
                    )
                )
        self.reset_cache()

    def _load_host_context(self, context: Any) -> None:
        path = _bundle_path(self.bundle_dir, context, "host_context")
        shape = (1, self.max_seq_len, self.head_dim)
        self._rope: list[tuple[np.ndarray, np.ndarray]] = []
        with np.load(path, allow_pickle=False) as arrays:
            for index in range(self.num_hidden_layers):
                pair = []
                for kind in ("cos", "sin"):
                    key = f"rope_{kind}_{index}"
                    if key not in arrays:
                        raise ValueError(f"Missing host context array {key}.")
                    value = np.asarray(arrays[key])
                    if (
                        value.dtype != np.float32
                        or value.shape != shape
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(f"Invalid host context {key} shape or dtype.")
                    pair.append(value)
                self._rope.append((pair[0], pair[1]))
            if "mask_fill_values" not in arrays:
                raise ValueError("Missing host context mask_fill_values.")
            self._mask_fill = np.asarray(arrays["mask_fill_values"])
            if (
                self._mask_fill.shape != (self.num_hidden_layers,)
                or self._mask_fill.dtype != np.float32
                or not np.isfinite(self._mask_fill).all()
                or not (self._mask_fill < 0).all()
            ):
                raise ValueError("Invalid host context mask_fill_values.")

    def _validate_token_id(self, value: Any, name: str) -> None:
        if type(value) is not int or not 0 <= value < self.vocab_size:
            raise ValueError(f"{name} must be a token ID in [0, {self.vocab_size}).")

    def _eos_ids(self, value: int | Sequence[int] | None) -> set[int]:
        if value is None:
            return set()
        values = [value] if type(value) is int else value
        if not isinstance(values, (list, tuple)):
            raise ValueError("eos_token_id must be an integer or sequence of integers.")
        for token_id in values:
            self._validate_token_id(token_id, "eos_token_id")
        return set(values)

    def _token_ids(self, input_ids: Any) -> np.ndarray:
        values = np.asarray(input_ids)
        if values.dtype != np.int64:
            raise TypeError("input_ids must have dtype int64.")
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] < 1:
            raise ValueError("input_ids must have shape (1, sequence_length >= 1).")
        if (values < 0).any() or (values >= self.vocab_size).any():
            raise ValueError("input_ids contains a token ID outside the vocabulary.")
        return values

    def reset_cache(self) -> None:
        """Start a new sequence; decoded Circle constants remain cached."""
        self.past_len = 0
        self._layer_caches: list[tuple[np.ndarray, np.ndarray]] = []

    def prefill(self, input_ids: Any) -> np.ndarray:
        """Reset and process one compact prompt; return next-token logits (1, V)."""
        tokens = self._token_ids(input_ids)
        length = tokens.shape[1]
        if length > self.max_seq_len:
            raise ValueError("Prompt exceeds the exported max_seq_len.")
        self.reset_cache()
        padded = np.full(
            (1, self.max_seq_len),
            self.pad_token_id if self.pad_token_id is not None else 0,
            dtype=np.int64,
        )
        padded[:, :length] = tokens
        hidden = self._embedding.run(input_ids=padded)["hidden_states"]
        positions: np.ndarray = np.arange(self.max_seq_len)
        visible = (positions[None, :] <= positions[:, None]) & (
            positions[None, :] < length
        )
        caches = []
        cache_shape = (
            1,
            self.num_key_value_heads,
            self.max_seq_len - 1,
            self.head_dim,
        )
        for index, stage in enumerate(self._prefill_layers):
            cos, sin = self._rope[index]
            mask: np.ndarray = np.where(visible, np.float32(0), self._mask_fill[index])
            output = stage.run(
                hidden_states=hidden,
                attention_mask=mask[None, ...],
                rope_cos=cos,
                rope_sin=sin,
            )
            hidden = output["hidden_states"]
            keys = np.zeros(cache_shape, dtype=np.float32)
            values = np.zeros_like(keys)
            cached = min(length, self.max_seq_len - 1)
            keys[:, :, :cached, :] = output["new_key"][:, :, :cached, :]
            values[:, :, :cached, :] = output["new_value"][:, :, :cached, :]
            caches.append((keys, values))
        logits = self._lm_head.run(hidden_states=hidden[:, length - 1 : length, :])[
            "logits"
        ][:, 0, :]
        self._layer_caches = caches
        self.past_len = length
        return logits

    def decode_one(self, input_ids: Any) -> np.ndarray:
        """Consume one token, commit its KV deltas, and return next-token logits."""
        tokens = self._token_ids(input_ids)
        if tokens.shape != (1, 1):
            raise ValueError("decode_one input_ids must have shape (1, 1).")
        if not self._layer_caches:
            raise ValueError("Call prefill() before decode_one().")
        if self.past_len >= self.max_seq_len:
            raise ValueError("The exported KV-cache capacity is full.")
        hidden = self._embedding.run(input_ids=tokens)["hidden_states"]
        visible: np.ndarray = np.less(np.arange(self.max_seq_len), self.past_len)
        visible[-1] = True
        deltas = []
        for index, stage in enumerate(self._decode_layers):
            cos, sin = self._rope[index]
            keys, values = self._layer_caches[index]
            mask: np.ndarray = np.where(visible, np.float32(0), self._mask_fill[index])
            output = stage.run(
                hidden_states=hidden,
                attention_mask=mask[None, None, :],
                rope_cos=cos[:, self.past_len : self.past_len + 1, :],
                rope_sin=sin[:, self.past_len : self.past_len + 1, :],
                past_key=keys,
                past_value=values,
            )
            hidden = output["hidden_states"]
            deltas.append((output["new_key"], output["new_value"]))
        logits = self._lm_head.run(hidden_states=hidden)["logits"][:, 0, :]
        # Commit only after every stage succeeds, so a failed decode is retryable.
        # The terminal step can score the full context using S-1 past slots.
        # Its delta needs no storage because no subsequent decode can fit.
        if self.past_len < self.max_seq_len - 1:
            for (keys, values), (new_key, new_value) in zip(self._layer_caches, deltas):
                keys[:, :, self.past_len : self.past_len + 1, :] = new_key
                values[:, :, self.past_len : self.past_len + 1, :] = new_value
        self.past_len += 1
        return logits

    def generate(
        self,
        input_ids: Any,
        max_new_tokens: int,
        eos_token_id: int | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Return prompt plus greedy tokens, stopping at EOS or the token budget.

        Requests exceeding the static context are rejected before computation;
        prompts are never truncated. ``None`` uses the bundle's EOS metadata.
        A zero-token request returns a copy of the prompt without changing KV state.
        """
        tokens = self._token_ids(input_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if tokens.shape[1] + max_new_tokens > self.max_seq_len:
            raise ValueError("Prompt plus max_new_tokens exceeds exported max_seq_len.")
        eos_ids = self._eos_ids(
            self.eos_token_id if eos_token_id is None else eos_token_id
        )
        if max_new_tokens == 0:
            return tokens.copy()
        logits = self.prefill(tokens)
        generated = tokens.copy()
        for step in range(max_new_tokens):
            next_token = np.asarray(np.argmax(logits, axis=-1), dtype=np.int64).reshape(
                1, 1
            )
            generated = np.concatenate((generated, next_token), axis=1)
            if int(next_token[0, 0]) in eos_ids or step + 1 == max_new_tokens:
                break
            logits = self.decode_one(next_token)
        return generated

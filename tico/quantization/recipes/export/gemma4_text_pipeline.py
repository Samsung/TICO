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

"""Saved Gemma4 decoder stages and shared-consumer finalize aliases.

This is a decoder-input interface. The caller supplies fused embeddings, the
existing full-prefill PLE projection, masks and RoPE; tokenizer/vision/PLE host
preparation remains with its existing owner.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.wrapq.wrappers.gemma4.utils import gemma4_decode_capacity
from tico.utils.signature import ModelInputSpec


def gemma4_text_stage_inputs(
    layer: dict[str, Any],
    *,
    query_length: int,
    capacity: int,
    hidden_size: int,
    ple_dim: int,
    decode: bool,
) -> dict[str, list[int]]:
    """Return the semantic, static input shapes of a saved decoder stage."""
    inputs = {
        "hidden_states": [1, query_length, hidden_size],
        "attention_mask": (
            [1, query_length, capacity] if decode else [1, 1, query_length, capacity]
        ),
        "position_embeddings_0": [1, query_length, layer["head_dim"]],
        "position_embeddings_1": [1, query_length, layer["head_dim"]],
    }
    if ple_dim:
        inputs["per_layer_input"] = [1, query_length, ple_dim]
    if layer["is_shared"] or decode:
        name = "shared_key_value" if layer["is_shared"] else "past_key_value"
        cache_len = capacity if layer["is_shared"] else capacity - 1
        shape = [1, layer["num_kv_heads"], cache_len, layer["head_dim"]]
        inputs[f"{name}_0"] = shape
        inputs[f"{name}_1"] = shape
    return inputs


class Gemma4CircleTextStage(nn.Module):
    """Execute a validated saved stage using the explicit reference backend."""

    def __init__(
        self,
        path: Path,
        *,
        input_shapes: dict[str, list[int]],
        output_shapes: list[list[int]],
        execution_mode: ExecutionMode,
    ):
        super().__init__()
        binary = path.read_bytes()
        self.input_spec = ModelInputSpec(binary)
        self.runtime = CircleReferenceRuntime(binary)
        self.execution_mode = execution_mode
        actual_inputs = {
            name: list(shape)
            for name, shape in zip(self.input_spec.names, self.input_spec.shapes)
        }
        if actual_inputs != input_shapes:
            raise ValueError(
                f"Gemma4 stage {path.name} has incompatible input contracts."
            )
        if any(
            shape is not None and -1 in shape
            for shape in self.input_spec.shape_signatures
        ):
            raise ValueError(f"Gemma4 stage {path.name} must have static input shapes.")
        if [list(t.shape) for t in self.runtime.output_tensors] != output_shapes:
            raise ValueError(
                f"Gemma4 stage {path.name} has incompatible output contracts."
            )
        if (
            execution_mode is ExecutionMode.NATIVE
            and self.runtime.has_quantized_activations()
        ):
            raise ValueError(
                f"Gemma4 stage {path.name} requires fake_quantize execution."
            )
        for tensor in self.runtime.input_tensors + self.runtime.output_tensors:
            if self.runtime.effective_dtype(tensor, execution_mode) != np.dtype(
                np.float32
            ):
                raise ValueError(
                    f"Gemma4 stage {path.name} requires FP32 semantic boundaries."
                )
        self.runtime.prepare()

    def forward(self, *args, **kwargs):
        """Bind the original adapter ABI without casting or reshaping tensors."""
        # Quantized Circle storage types describe integer grids; fake-quantize
        # execution deliberately consumes/returns their FP32 semantic values.
        bound = self.input_spec.bind(args, kwargs, check=False)
        self.input_spec.check_shapes(bound)
        if any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            for value in bound
        ):
            raise TypeError("Gemma4 saved text stages require FP32 semantic inputs.")
        if any(value.device.type != "cpu" for value in bound):
            raise ValueError("Gemma4 saved text stages require CPU reference inputs.")
        outputs = self.runtime.run(bound, mode=self.execution_mode).outputs
        tensors = tuple(
            torch.from_numpy(np.array(value, copy=True)) for value in outputs
        )
        return tensors[0] if len(tensors) == 1 else tensors


@dataclass(frozen=True)
class Gemma4TextPipeline:
    """Loaded decoder artifacts; finalize stages alias the decode consumers."""

    prefill_layers: tuple[nn.Module, ...]
    decode_layers: tuple[nn.Module, ...]
    finalize_layers: tuple[nn.Module, ...]
    lm_head: nn.Module
    suffix_start: int
    producer_indices: dict[int, int]
    max_seq_len: int
    layers: tuple[dict[str, Any], ...]
    finalize_supported: bool
    finalize_unsupported_reason: str | None
    hidden_size: int
    ple_dim: int
    rope_convention: str
    decode_kv_policy: str
    decode_capacities: tuple[int, ...]


def load_gemma4_text_pipeline(
    directory: str | Path,
    *,
    expected_max_seq_len: int | None = None,
    expected_num_layers: int | None = None,
    expected_decode_kv_policy: str | None = None,
) -> Gemma4TextPipeline:
    """Load compact or legacy-full stages, validating shape and frozen grids."""
    root = Path(directory)
    manifest_path = root / "text_pipeline.json"
    if not manifest_path.is_file():
        raise ValueError(
            "Gemma4 text_pipeline.json is missing; re-export to use saved text stages. "
            "Legacy artifacts remain usable through the existing full-prefill path."
        )
    manifest = json.loads(manifest_path.read_text())
    version = manifest.get("schema_version")
    if version not in (1, 2) or manifest.get("model_family") != "gemma4":
        raise ValueError("Unsupported Gemma4 text pipeline schema.")
    policy = "full_capacity_v1" if version == 1 else manifest.get("decode_kv_policy")
    if (
        not isinstance(policy, str)
        or (version == 1 and manifest.get("decode_kv_policy", policy) != policy)
        or (version == 2 and policy not in {"compact_sliding_v1", "full_capacity_v1"})
    ):
        raise ValueError("Unsupported Gemma4 decode KV policy for pipeline schema.")
    if expected_decode_kv_policy is not None and policy != expected_decode_kv_policy:
        raise ValueError("Gemma4 decode KV policy does not match the runtime.")
    expected_finalize = f"shared_consumer_decode_alias_v{version}"
    if manifest.get("finalize_contract") != expected_finalize:
        raise ValueError("Unsupported Gemma4 finalize contract.")
    if type(manifest.get("finalize_supported")) is not bool:
        raise ValueError("Gemma4 finalize capability must be explicit.")
    if manifest.get("rope_convention") != "pre_negated_sin":
        raise ValueError("Gemma4 saved text stages require pre_negated_sin RoPE.")
    capacity = manifest["max_seq_len"]
    layers = manifest["layers"]
    boundary = manifest["suffix_start"]
    hidden, ple_dim = manifest["hidden_size"], manifest["ple_dim"]
    if (
        type(capacity) is not int
        or capacity < 2
        or not layers
        or type(boundary) is not int
        or not 0 <= boundary <= len(layers)
        or type(hidden) is not int
        or hidden < 1
        or type(ple_dim) is not int
        or ple_dim < 0
    ):
        raise ValueError("Invalid Gemma4 text pipeline dimensions.")
    if expected_max_seq_len is not None and capacity != expected_max_seq_len:
        raise ValueError("Gemma4 text pipeline capacity does not match the runtime.")
    if expected_num_layers is not None and len(layers) != expected_num_layers:
        raise ValueError("Gemma4 text pipeline layer count does not match the runtime.")
    actual_boundary = max(
        (index + 1 for index, layer in enumerate(layers) if not layer["is_shared"]),
        default=0,
    )
    if boundary != actual_boundary:
        raise ValueError(
            "Gemma4 finalize boundary must identify the trailing shared-consumer suffix."
        )
    mode = ExecutionMode(manifest["execution_mode"])
    if mode not in (ExecutionMode.NATIVE, ExecutionMode.FAKE_QUANTIZE):
        raise ValueError("Unsupported Gemma4 text pipeline execution mode.")
    producers: dict[int, int] = {}
    sources: dict[str, int] = {}
    prefills: list[Gemma4CircleTextStage] = []
    decodes: list[Gemma4CircleTextStage] = []
    decode_capacities: list[int] = []

    def stage(name, inputs, outputs) -> Gemma4CircleTextStage:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".circle")
        ):
            raise ValueError("Gemma4 stage artifact must be a local .circle filename.")
        return Gemma4CircleTextStage(
            root / name,
            input_shapes=inputs,
            output_shapes=outputs,
            execution_mode=mode,
        )

    for index, layer in enumerate(layers):
        if layer["layer_index"] != index:
            raise ValueError("Gemma4 text pipeline layers must be in decoder order.")
        producer = layer["producer_index"]
        if layer["layer_type"] not in {"full_attention", "sliding_attention"} or any(
            type(layer[key]) is not int or layer[key] < 1
            for key in ("num_kv_heads", "head_dim")
        ):
            raise ValueError("Invalid Gemma4 layer attention geometry.")
        window = layer.get("sliding_window")
        if (
            layer["layer_type"] == "sliding_attention"
            and (type(window) is not int or window < 1)
        ) or (layer["layer_type"] == "full_attention" and window is not None):
            raise ValueError("Invalid Gemma4 layer sliding-window geometry.")
        decode_capacity = gemma4_decode_capacity(
            capacity, window, compact=policy == "compact_sliding_v1"
        )
        if version == 2 and (
            type(layer.get("decode_capacity")) is not int
            or layer["decode_capacity"] != decode_capacity
        ):
            raise ValueError("Gemma4 layer decode capacity does not match its policy.")
        if version == 1 and layer.get("decode_capacity", capacity) != capacity:
            raise ValueError("Legacy Gemma4 layer decode capacity must remain full.")
        decode_capacities.append(decode_capacity)
        if layer["is_shared"]:
            if type(producer) is not int or not 0 <= producer < index:
                raise ValueError("Invalid Gemma4 shared-KV producer index.")
            source = layers[producer]
            if (
                source["is_shared"]
                or not source["store_full_length_kv"]
                or any(
                    source[key] != layer[key]
                    for key in (
                        "layer_type",
                        "num_kv_heads",
                        "head_dim",
                        "sliding_window",
                    )
                )
            ):
                raise ValueError("Incompatible Gemma4 shared-KV producer contract.")
            producers[index] = producer
            if sources.get(layer["layer_type"]) != producer:
                raise ValueError(
                    "Gemma4 shared-KV mapping must use the current storage producer."
                )
        elif producer is not None:
            raise ValueError("Gemma4 KV producer cannot itself reference a producer.")
        elif layer["store_full_length_kv"]:
            sources[layer["layer_type"]] = index
        if index >= boundary:
            if (
                not layer["is_shared"]
                or layer["finalize_artifact"] != layer["decode_artifact"]
            ):
                raise ValueError(
                    "Gemma4 finalize must alias a shared-consumer decode artifact."
                )
        elif layer["finalize_artifact"] is not None:
            raise ValueError("Gemma4 prefix cannot declare a finalize artifact.")
        for decode, destination in ((False, prefills), (True, decodes)):
            qlen = 1 if decode else capacity
            inputs = gemma4_text_stage_inputs(
                layer,
                query_length=qlen,
                capacity=decode_capacity if decode else capacity,
                hidden_size=hidden,
                ple_dim=ple_dim,
                decode=decode,
            )
            outputs = [[1, qlen, hidden]]
            if not layer["is_shared"]:
                cache_shape = [1, layer["num_kv_heads"], qlen, layer["head_dim"]]
                outputs += [cache_shape, cache_shape]
            destination.append(
                stage(
                    layer["decode_artifact" if decode else "prefill_artifact"],
                    inputs,
                    outputs,
                )
            )
        # Shapes above prove that C=1 has empty past inputs. No persistent
        # samples cross steps in that case, so their unused FLOAT32 inputs may
        # have no grid even when the current storage delta is affine-quantized.
        if not layer["is_shared"] and decode_capacity > 1:
            input_domains = dict(
                zip(decodes[-1].input_spec.names, decodes[-1].runtime.input_tensors)
            )
            for position, role in enumerate(
                ("past_key_value_0", "past_key_value_1"), start=1
            ):
                output = decodes[-1].runtime.output_tensors[position]
                input_tensor = input_domains[role]
                if (output.tensor_type, output.quantization) != (
                    input_tensor.tensor_type,
                    input_tensor.quantization,
                ):
                    raise ValueError(
                        "Gemma4 persistent cache input/output qparams do not match."
                    )
        # Shape specialization must keep every interface on the same frozen
        # grid. Comparing storage type/qparams also rejects mixed calibration
        # exports before a sliced prompt can enter the wrong domain.
        prefill, decode_stage = prefills[-1], decodes[-1]
        prefill_domains = {
            name: (tensor.tensor_type, tensor.quantization)
            for name, tensor in zip(
                prefill.input_spec.names, prefill.runtime.input_tensors
            )
        }
        for name, tensor in zip(
            decode_stage.input_spec.names, decode_stage.runtime.input_tensors
        ):
            if name in prefill_domains and prefill_domains[name] != (
                tensor.tensor_type,
                tensor.quantization,
            ):
                raise ValueError("Gemma4 prefill/decode boundary qparams do not match.")
        for left, right in zip(
            prefill.runtime.output_tensors, decode_stage.runtime.output_tensors
        ):
            if (left.tensor_type, left.quantization) != (
                right.tensor_type,
                right.quantization,
            ):
                raise ValueError("Gemma4 prefill/decode output qparams do not match.")
    for consumer, producer in producers.items():
        for loaded_stage in (prefills[consumer], decodes[consumer]):
            input_domains = dict(
                zip(loaded_stage.input_spec.names, loaded_stage.runtime.input_tensors)
            )
            for position, role in enumerate(
                ("shared_key_value_0", "shared_key_value_1"), start=1
            ):
                output = prefills[producer].runtime.output_tensors[position]
                input_tensor = input_domains[role]
                if (output.tensor_type, output.quantization) != (
                    input_tensor.tensor_type,
                    input_tensor.quantization,
                ):
                    raise ValueError(
                        "Gemma4 shared-KV producer/consumer qparams do not match."
                    )
    lm_head = stage(
        manifest["lm_head_artifact"],
        {"hidden_states": [1, 1, hidden]},
        [[1, 1, manifest["vocab_size"]]],
    )
    return Gemma4TextPipeline(
        tuple(prefills),
        tuple(decodes),
        tuple(decodes[boundary:]),
        lm_head,
        boundary,
        producers,
        capacity,
        tuple(layers),
        manifest["finalize_supported"],
        manifest.get("finalize_unsupported_reason"),
        hidden,
        ple_dim,
        manifest["rope_convention"],
        policy,
        tuple(decode_capacities),
    )

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

"""Reproduce the PR #1083 real-model comparison without downloading any data.

Run ``init --fixture DIR`` once, then ``export --fixture DIR --repo WORKTREE
--output DIR`` separately for main and candidate, using the same Python executable.
Finally run ``compare --main DIR --candidate DIR --output comparison.json``.
Export writes all binaries, arrays, graph details and metrics outside the repository.
The only patch supplies CompileConfigV1 to the real converter; no exporter is mocked.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def config_dict() -> dict[str, Any]:
    return dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        vocab_size=32,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


def initialize(directory: Path) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1083)
    model = LlamaForCausalLM(LlamaConfig(**config_dict())).eval()
    torch.save(model.state_dict(), directory / "float_state.pt")
    write_json(directory / "config.json", config_dict())


def qcontract(tensor: Any) -> dict[str, Any]:
    quant = tensor.quantization
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "signature": tensor.shape_signature,
        "scale": list(quant.scale) if quant else [],
        "zero_point": list(quant.zero_point) if quant else [],
        "axis": quant.quantized_dimension if quant else 0,
    }


def grid(contract: dict[str, Any]) -> tuple[Any, ...]:
    return contract["dtype"], contract["scale"], contract["zero_point"]


def options_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [options_json(v) for v in value]
    return {
        key: options_json(item)
        for key, item in sorted(vars(value).items())
        if not key.startswith("_")
    }


def normalized_graph(program: Any) -> dict[str, Any]:
    """Canonical references use graph roles, producer positions, and content hashes."""
    references = {-1: "optional"}
    for position, index in enumerate(program.inputs):
        references[index] = f"input:{position}"
    for tensor in program.tensors:
        if tensor.constant is not None:
            digest = hashlib.sha256(tensor.constant.tobytes()).hexdigest()
            references[tensor.index] = f"constant:{digest}"
    operators = []
    anomalies = []
    redundant = []
    for position, operator in enumerate(program.operators):
        inputs = [
            qcontract(program.tensor(i)) if i >= 0 else None for i in operator.inputs
        ]
        outputs = [qcontract(program.tensor(i)) for i in operator.outputs]
        for output, index in enumerate(operator.outputs):
            references[index] = f"op:{position}:{output}"
        record = {
            "type": operator.name,
            "version": operator.version,
            "inputs": [references[i] for i in operator.inputs],
            "outputs": [references[i] for i in operator.outputs],
            "input_contracts": inputs,
            "output_contracts": outputs,
            "options": options_json(operator.options),
        }
        operators.append(record)
        if operator.name in {
            "RESHAPE",
            "TRANSPOSE",
            "SQUEEZE",
            "EXPAND_DIMS",
            "STRIDED_SLICE",
            "SLICE",
        }:
            if grid(inputs[0]) != grid(outputs[0]):
                anomalies.append({"position": position, **record})
        if operator.name == "CONCATENATION":
            if any(grid(item) != grid(outputs[0]) for item in inputs):
                anomalies.append({"position": position, **record})
        if operator.name == "QUANTIZE" and grid(inputs[0]) == grid(outputs[0]):
            redundant.append(position)
    return {
        "inputs": [qcontract(program.tensor(i)) for i in program.inputs],
        "outputs": [qcontract(program.tensor(i)) for i in program.outputs],
        "operators": operators,
        "counts": dict(Counter(op.name for op in program.operators)),
        "value_preserving_anomalies": anomalies,
        "redundant_quantize": redundant,
    }


def domain_paths(program: Any) -> dict[str, Any]:
    """Trace serialized KV interfaces back through conversions and view operations.

    Arithmetic anchors use type/occurrence, independent of inserted Quantize nodes
    and temporary tensor names. These expose shared P ancestry of the C/S branches.
    """
    producers = {i: op for op in program.operators for i in op.outputs}
    counters = Counter()
    anchors = {}
    for operator in program.operators:
        anchors[operator.index] = f"{operator.name}:{counters[operator.name]}"
        counters[operator.name] += 1
    inputs = {index: position for position, index in enumerate(program.inputs)}
    views = {"RESHAPE", "TRANSPOSE", "SQUEEZE", "EXPAND_DIMS", "STRIDED_SLICE", "SLICE"}

    def path(index: int, depth: int = 0) -> dict[str, Any]:
        tensor = program.tensor(index)
        result: dict[str, Any] = {"contract": qcontract(tensor)}
        if index in inputs:
            return {**result, "source": f"input:{inputs[index]}"}
        if tensor.is_constant:
            return {**result, "source": "constant"}
        operator = producers[index]
        result.update(source=anchors[operator.index], type=operator.name)
        if depth < 20 and operator.name in views | {"QUANTIZE", "DEQUANTIZE"}:
            result["data_input"] = path(operator.inputs[0], depth + 1)
        return result

    concats = []
    for operator in program.operators:
        if operator.name != "CONCATENATION" or len(operator.inputs) != 2:
            continue
        shapes = [program.tensor(i).shape for i in operator.inputs]
        if len(shapes[0]) == 3 and {shapes[0][1], shapes[1][1]} == {1, 7}:
            concats.append(
                {
                    "anchor": anchors[operator.index],
                    "inputs": [path(i) for i in operator.inputs],
                    "output": qcontract(program.tensor(operator.outputs[0])),
                }
            )
    return {
        "compute_concats": concats,
        "cache_delta_outputs": [path(i) for i in program.outputs[1:]],
    }


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "exact": bool(np.array_equal(actual, expected)),
        "max_abs": float(delta.max(initial=0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "differing": int(np.count_nonzero(actual != expected)),
        "elements": int(delta.size),
    }


def encode(value: torch.Tensor, tensor: Any) -> torch.Tensor:
    """Encode/decode the serialized affine ABI before giving inputs to eager."""
    if not tensor.is_quantized_integer:
        return value
    quant = tensor.quantization
    assert quant is not None and len(quant.scale) == 1
    limits = np.iinfo(tensor.dtype)
    return torch.fake_quantize_per_tensor_affine(
        value, quant.scale[0], quant.zero_point[0], limits.min, limits.max
    )


def bind_inputs(runtime: Any, values: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    result = []
    for index in runtime.program.inputs:
        tensor = runtime.program.tensor(index)
        name = tensor.name
        if "hidden" in name:
            key = "hidden"
        elif "attention_mask" in name:
            key = "mask"
        elif "past_key_value" in name:
            key = "past_v" if name.endswith("_1") else "past_k"
        elif "position_embeddings" in name:
            key = "sin" if name.endswith("_1") else "cos"
        else:
            raise ValueError(f"Unrecognized serialized input {name!r}")
        # Cache values already came from serialized storage outputs. Copy/append
        # them directly; ABI mismatches are reported rather than hidden by a host
        # conversion. Other external inputs are encoded to their serialized grid.
        if key not in {"past_k", "past_v"}:
            values[key] = encode(values[key], tensor)
        result.append(values[key])
    return result


def runtime_results(model: Any, folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
    from tico.quantization.wrapq.wrappers.llama.static_inputs import (
        prepare_llama_static_inputs,
    )

    summaries = {}
    arrays = {}
    layers = model.wrapped.model.wrapped.layers
    for layer_index, wrapper in enumerate(layers):
        layer = wrapper.wrapped
        static = prepare_llama_static_inputs(layer, 8)
        cache = None
        for step in range(3):
            mode = "prefill" if step == 0 else "decode"
            key = f"layer{layer_index}_{mode}{step}"
            path = folder / f"decoder_layer_{mode}_{layer_index}.q.circle"
            runtime = CircleReferenceRuntime(path.read_bytes())
            length = 8 if step == 0 else 1
            values = {
                "hidden": torch.linspace(-0.08, 0.09, length * 16).reshape(
                    1, length, 16
                ),
                "mask": (
                    static.mask(8, 8, 0, device="cpu")
                    if step == 0
                    else torch.full((1, 1, 8), -120.0)
                ),
            }
            position = 0 if step == 0 else step + 2
            values["cos"], values["sin"] = static.rope(
                position, length, device="cpu", dtype=torch.float32
            )
            if cache is not None:
                values["past_k"], values["past_v"] = cache
                values["mask"][:, :, : step + 2] = 0.0
                values["mask"][:, :, 7] = 0.0
            inputs = bind_inputs(runtime, values)
            kwargs: dict[str, Any] = dict(
                attention_mask=values["mask"],
                position_embeddings=(values["cos"], values["sin"]),
            )
            if cache is not None:
                kwargs["past_key_value"] = (values["past_k"], values["past_v"])
            with torch.no_grad():
                eager = layer.as_export_module(mode)(values["hidden"], **kwargs)
            actual = runtime.run(
                [v.numpy() for v in inputs], mode=ExecutionMode.FAKE_QUANTIZE
            ).outputs
            summaries[key] = {}
            for index, (circle, reference) in enumerate(zip(actual, eager)):
                output_name = ("hidden", "k", "v")[index]
                name = f"{key}_{output_name}"
                summaries[key][output_name] = {
                    **metrics(circle, reference.numpy()),
                    "contract": qcontract(
                        runtime.program.tensor(runtime.program.outputs[index])
                    ),
                }
                arrays[name] = circle
                arrays[name + "_eager"] = reference.numpy()
            # Persist decoded storage-domain values; append/copy only, no host requantization.
            if cache is None:
                cache = tuple(
                    torch.cat(
                        (torch.from_numpy(v[:, :, :3].copy()), torch.zeros(1, 1, 4, 8)),
                        dim=2,
                    )
                    for v in actual[1:]
                )
            else:
                for stored, delta in zip(cache, actual[1:]):
                    stored[:, :, step + 2 : step + 3] = torch.from_numpy(delta)
    np.savez(folder / "runtime_outputs.npz", **arrays)
    return summaries, arrays


@contextmanager
def compile_options(optimize: bool):
    import tico
    from tico.config import CompileConfigV1

    original = tico.convert

    def convert_with_config(*args, **kwargs):
        return original(
            *args, **kwargs, config=CompileConfigV1(circle_optimize=optimize)
        )

    with patch.object(tico, "convert", convert_with_config):
        yield


def export(args: argparse.Namespace) -> None:
    import transformers

    from tico.circle import CircleDocument
    from tico.circle.runtime import CircleProgram
    from tico.quantization import convert, prepare
    from tico.quantization.config.ptq import PTQConfig
    from tico.quantization.recipes.export.llama import export_llama_per_layer
    from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
    from tico.quantization.wrapq.wrappers.llama.quant_attention import (
        QuantLlamaAttention,
    )
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    state = torch.load(args.fixture / "float_state.pt", weights_only=True)
    config = LlamaConfig(**json.loads((args.fixture / "config.json").read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "sha": subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version,
        "config": config_dict(),
        "fixture_sha256": hashlib.sha256(
            (args.fixture / "float_state.pt").read_bytes()
        ).hexdigest(),
        "scenarios": {},
    }
    for scenario in ("natural", "distinct"):
        torch.manual_seed(1083)
        model = LlamaForCausalLM(config).eval()
        model.load_state_dict(state)
        prepared = prepare(model, PTQConfig(model_args={"profile": "npu_export"}))
        with torch.no_grad():
            prepared(torch.tensor([[1, 5, 3, 7, 9, 4, 6, 8]]), use_cache=True)
            output = prepared(
                torch.tensor([[1, 5, 3]]), use_cache=True, return_dict=True
            )
            for token in (7, 9):
                output = prepared(
                    torch.tensor([[token]]),
                    past_key_values=output.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        if scenario == "distinct":
            # Valid per-tensor UINT8 domains, deliberately using non-integral scale ratios.
            for attention in (
                m for m in prepared.modules() if isinstance(m, QuantLlamaAttention)
            ):
                for name, scale, zero in (
                    ("obs_new_k", 0.0017, 119),
                    ("obs_new_v", 0.0019, 123),
                    ("obs_present_key", 0.0011, 127),
                    ("obs_present_value", 0.0013, 129),
                    ("obs_k_rot", 0.0014, 121),
                ):
                    getattr(attention, name).load_qparams(
                        torch.tensor(scale), torch.tensor(zero)
                    )
                attention.v_proj.wrapped.obs_act_out.load_qparams(
                    torch.tensor(0.0015), torch.tensor(125)
                )
        quantized = convert(prepared).eval()
        observers = {
            name: {
                "scale": module.compute_qparams()[0].tolist(),
                "zero_point": module.compute_qparams()[1].tolist(),
                "dtype": str(module.dtype),
            }
            for name, module in quantized.named_modules()
            if isinstance(module, AffineObserverBase) and module.has_qparams
        }
        manifest["scenarios"][scenario] = {"observers": observers, "exports": {}}
        for optimize in (False, True):
            folder = args.output / scenario / ("o1" if optimize else "unoptimized")
            with compile_options(optimize):
                torch.manual_seed(817)
                export_llama_per_layer(
                    q_model=copy.deepcopy(quantized),
                    max_seq_len=8,
                    output_dir=folder,
                    prefill_decode=True,
                )
            graphs = {}
            for artifact in sorted(folder.glob("*.circle")):
                document = CircleDocument.from_bytes(artifact.read_bytes())
                document.verify(raise_on_error=True)
                program = CircleProgram(document)
                graphs[artifact.name] = normalized_graph(program)
                graphs[artifact.name]["kv_domain_paths"] = domain_paths(program)
            numerical, _ = runtime_results(quantized, folder)
            abi = {}
            for layer_index in range(2):
                prefill_name = f"decoder_layer_prefill_{layer_index}.q.circle"
                decode_name = f"decoder_layer_decode_{layer_index}.q.circle"
                decode_program = CircleProgram.from_bytes(
                    (folder / decode_name).read_bytes()
                )
                past = [
                    qcontract(decode_program.tensor(i))
                    for i in decode_program.inputs
                    if "past_key_value" in decode_program.tensor(i).name
                ]
                abi[str(layer_index)] = {
                    kind: {
                        "prefill_output": graphs[prefill_name]["outputs"][position + 1],
                        "decode_input": past[position],
                        "decode_output": graphs[decode_name]["outputs"][position + 1],
                        "equal": grid(graphs[prefill_name]["outputs"][position + 1])
                        == grid(past[position])
                        == grid(graphs[decode_name]["outputs"][position + 1]),
                    }
                    for position, kind in enumerate(("k", "v"))
                }
            summary = {"graphs": graphs, "numerical": numerical, "cache_abi": abi}
            write_json(folder / "summary.json", summary)
            manifest["scenarios"][scenario]["exports"][folder.name] = summary
    write_json(args.output / "manifest.json", manifest)


def semantic_topology(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize away Quantize positions to compare all other producer/consumer edges."""
    references = {}
    occurrences = Counter()
    result = []
    for operator in graph["operators"]:
        inputs = [references.get(value, value) for value in operator["inputs"]]
        if operator["type"] == "QUANTIZE":
            references.update((value, inputs[0]) for value in operator["outputs"])
            continue
        name = operator["type"]
        anchor = f"{name}:{occurrences[name]}"
        occurrences[name] += 1
        references.update(
            (value, f"{anchor}:{position}")
            for position, value in enumerate(operator["outputs"])
        )
        result.append(
            {
                "anchor": anchor,
                "inputs": inputs,
                "options": operator["options"],
                "input_shapes": [
                    c["shape"] if c else None for c in operator["input_contracts"]
                ],
                "output_shapes": [c["shape"] for c in operator["output_contracts"]],
            }
        )
    return result


def compare(args: argparse.Namespace) -> None:
    main = json.loads((args.main / "manifest.json").read_text())
    candidate = json.loads((args.candidate / "manifest.json").read_text())
    assert main["fixture_sha256"] == candidate["fixture_sha256"]
    assert all(
        main[key] == candidate[key]
        for key in ("torch", "transformers", "python", "config")
    )
    result: dict[str, Any] = {
        "main_sha": main["sha"],
        "candidate_sha": candidate["sha"],
        "scenarios": {},
    }
    for scenario in main["scenarios"]:
        old = main["scenarios"][scenario]
        new = candidate["scenarios"][scenario]
        common = old["observers"].keys() & new["observers"].keys()
        entry: dict[str, Any] = {
            "common_observer_count": len(common),
            "observer_changes": {
                name: [old["observers"][name], new["observers"][name]]
                for name in sorted(common)
                if old["observers"][name] != new["observers"][name]
            },
            "exports": {},
        }
        for option in old["exports"]:
            old_arrays = np.load(args.main / scenario / option / "runtime_outputs.npz")
            new_arrays = np.load(
                args.candidate / scenario / option / "runtime_outputs.npz"
            )
            exports = {
                "artifacts": {},
                "main_candidate_numerical": {
                    name: metrics(new_arrays[name], old_arrays[name])
                    for name in old_arrays.files
                    if not name.endswith("_eager")
                },
            }
            for artifact, old_graph in old["exports"][option]["graphs"].items():
                new_graph = new["exports"][option]["graphs"][artifact]
                old_ops = old_graph["operators"]
                new_ops = new_graph["operators"]
                matcher = difflib.SequenceMatcher(
                    a=[op["type"] for op in old_ops],
                    b=[op["type"] for op in new_ops],
                    autojunk=False,
                )
                edits = []
                changed_aligned = []
                for tag, a, b, c, d in matcher.get_opcodes():
                    if tag != "equal":
                        edits.append(
                            {
                                "tag": tag,
                                "main_positions": [a, b],
                                "candidate_positions": [c, d],
                                "main_types": [op["type"] for op in old_ops[a:b]],
                                "candidate_types": [op["type"] for op in new_ops[c:d]],
                            }
                        )
                    else:
                        for i, j in zip(range(a, b), range(c, d)):
                            fields = [
                                key
                                for key in (
                                    "input_contracts",
                                    "output_contracts",
                                    "options",
                                )
                                if old_ops[i][key] != new_ops[j][key]
                            ]
                            if fields:
                                changed_aligned.append(
                                    {
                                        "main_position": i,
                                        "candidate_position": j,
                                        "type": old_ops[i]["type"],
                                        "fields": fields,
                                        "main": old_ops[i],
                                        "candidate": new_ops[j],
                                    }
                                )
                exports["artifacts"][artifact] = {
                    "identical_normalized_graph": old_graph == new_graph,
                    "topology_ignoring_quantize_equal": semantic_topology(old_graph)
                    == semantic_topology(new_graph),
                    "main_kv_paths": old_graph["kv_domain_paths"],
                    "candidate_kv_paths": new_graph["kv_domain_paths"],
                    "main_count": len(old_ops),
                    "candidate_count": len(new_ops),
                    "main_types": old_graph["counts"],
                    "candidate_types": new_graph["counts"],
                    "opcode_edits": edits,
                    "aligned_changes": changed_aligned,
                    "interface_equal": old_graph["inputs"] == new_graph["inputs"]
                    and old_graph["outputs"] == new_graph["outputs"],
                    "main_anomalies": old_graph["value_preserving_anomalies"],
                    "candidate_anomalies": new_graph["value_preserving_anomalies"],
                    "candidate_redundant_quantize": new_graph["redundant_quantize"],
                }
            entry["exports"][option] = exports
        result["scenarios"][scenario] = entry
    write_json(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--fixture", type=Path, required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--fixture", type=Path, required=True)
    export_parser.add_argument("--repo", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--main", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.fixture)
    elif args.command == "export":
        sys.path.insert(0, str(args.repo.resolve()))
        export(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

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

"""Compare actual exports from two source trees and after candidate eager use.

Run this file directly so the coordinator does not import TICO from the wrong
checkout. Workers use the same interpreter and the same trusted frozen Torch
checkpoint. No calibration, model download, or NPU execution is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _write_json(path: Path, value: Any) -> None:
    """Write a strict, deterministic JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _file_sha256(path: Path) -> str:
    """Hash a file without loading a complete checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value: Any):
    """Detach a tensor for numerical reporting without changing its dtype."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _array_record(value: Any) -> dict:
    """Record shape, dtype, and exact bytes without embedding large arrays."""
    import numpy as np

    array = _array(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays do not have a stable byte representation.")
    contiguous = np.ascontiguousarray(array).reshape(-1).view(np.uint8)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(memoryview(contiguous)).hexdigest(),
    }


def _json_value(value: Any) -> Any:
    """Normalize generated schema objects while preserving semantic fields."""
    import numpy as np

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return {"nonfinite": repr(value)}
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "values": _json_value(value.tolist())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if hasattr(value, "detach"):
        import torch
        raw = value.detach().contiguous().cpu()
        return {
            "tensor_shape": list(raw.shape), "tensor_dtype": str(raw.dtype),
            "bytes": _array_record(raw.reshape(-1).view(torch.uint8)),
        }
    if hasattr(value, "__dict__"):
        return {
            "class": type(value).__name__,
            "fields": {
                key: _json_value(item)
                for key, item in sorted(vars(value).items())
                if not key.startswith("_")
            },
        }
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def circle_contract(document) -> dict:
    """Normalize executable Circle structure, excluding storage offsets/names.

    Tensor ordering and graph topology remain significant. Constants are hashed
    by payload, not by buffer-table index. I/O names remain significant because
    the runtime binder uses them. This is a conservative structural comparison,
    not a proof of general graph equivalence.
    """
    model = document.model
    opcodes = model.operatorCodes or []
    buffers = model.buffers or []
    payloads = {}
    subgraphs = []
    for graph in model.subgraphs or []:
        input_indices = [] if graph.inputs is None else list(map(int, graph.inputs))
        output_indices = [] if graph.outputs is None else list(map(int, graph.outputs))
        boundary = set(input_indices + output_indices)
        tensors = []
        for index, tensor in enumerate(graph.tensors or []):
            fields = {
                key: _json_value(value)
                for key, value in vars(tensor).items()
                if key not in ("name", "buffer")
            }
            if index in boundary:
                fields["name"] = _json_value(tensor.name)
            buffer_index = int(tensor.buffer)
            if buffer_index not in payloads:
                data = buffers[buffer_index].data
                payloads[buffer_index] = (
                    None if data is None or len(data) == 0 else _array_record(data)
                )
            fields["constant"] = payloads[buffer_index]
            tensors.append(fields)
        operators = []
        for operator in graph.operators or []:
            fields = {
                key: _json_value(value)
                for key, value in vars(operator).items()
                if key not in ("opcodeIndex",)
            }
            fields["opcode"] = _json_value(opcodes[int(operator.opcodeIndex)])
            operators.append(fields)
        subgraphs.append({
            "inputs": input_indices, "outputs": output_indices,
            "tensors": tensors, "operators": operators,
        })
    return {
        "version": int(model.version),
        "subgraphs": subgraphs,
        "signatureDefs": _json_value(getattr(model, "signatureDefs", None)),
    }


def _tree_arrays(value: Any, prefix: str = "") -> dict:
    """Flatten output tensors without silently dropping unsupported leaves."""
    import numpy as np
    import torch

    if isinstance(value, (torch.Tensor, np.ndarray)):
        return {prefix or "0": _array(value).copy()}
    if value is None:
        return {}
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, (list, tuple)):
        items = list(enumerate(value))
    else:
        # TICO registers HF caches as PyTrees. Use that registered ordering rather
        # than guessing the order of keys/values for a transformers version.
        from torch.utils._pytree import tree_flatten
        leaves, _ = tree_flatten(value)
        if len(leaves) == 1 and leaves[0] is value:
            raise TypeError(f"Unsupported output artifact: {type(value).__name__}")
        items = list(enumerate(leaves))
    flattened = {}
    for key, item in items:
        flattened.update(_tree_arrays(item, f"{prefix}/{key}"))
    return flattened


def compare_arrays(left, right, *, atol: float, rtol: float) -> dict:
    """Compare actual values, rejecting NaN/Inf even when both sides contain them."""
    import numpy as np

    left, right = _array(left), _array(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"ok": False, "reason": "shape_or_dtype", "left": _array_record(left), "right": _array_record(right)}
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"ok": False, "reason": "nonfinite"}
    if left.dtype.kind in "biu":
        ok = np.array_equal(left, right)
    else:
        ok = np.allclose(left, right, atol=atol, rtol=rtol, equal_nan=False)
    maximum = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
    return {"ok": bool(ok), "max_abs_error": maximum}


def _first_differences(left: Any, right: Any, path: str = "$", limit: int = 20) -> list:
    """Locate structural differences without printing whole weight-scale arrays."""
    if left == right:
        return []
    if type(left) is not type(right):
        return [path + ": type differs"]
    result = []
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            child = path + "." + str(key)
            if key not in left or key not in right:
                result.append(child + ": missing key")
            else:
                result.extend(_first_differences(left[key], right[key], child, limit))
            if len(result) >= limit:
                return result[:limit]
    elif isinstance(left, list):
        if len(left) != len(right):
            result.append(path + ": list length differs")
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(_first_differences(a, b, f"{path}[{index}]", limit))
            if len(result) >= limit:
                return result[:limit]
    else:
        result.append(path + ": value differs")
    return result[:limit]


def _state_contract(model) -> dict:
    """Fingerprint weights, all observer buffers/qparams, and quantization modes."""
    import torch
    from tico.quantization.wrapq.mode import Mode
    from tico.quantization.wrapq.observers.base import ObserverBase
    from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

    def tensor_record(tensor):
        raw = tensor.detach().contiguous().cpu()
        # Byte view also supports BF16, which NumPy may not expose as a dtype.
        return {"shape": list(raw.shape), "dtype": str(raw.dtype), "bytes": _array_record(raw.reshape(-1).view(torch.uint8))}

    weights = {name: tensor_record(tensor) for name, tensor in model.named_parameters()}
    observers, modes = {}, {}
    for name, module in model.named_modules():
        if isinstance(module, QuantModuleBase):
            if module._mode is Mode.CALIB:
                raise ValueError(f"Calibration is enabled at {name}; provide a frozen checkpoint.")
            modes[name] = module._mode.name
        if isinstance(module, ObserverBase):
            fields = {
                key: repr(getattr(module, key, None))
                for key in ("enabled", "fake_quant_enabled", "dtype", "qscheme", "channel_axis", "_qparams_locked")
            }
            fields["buffers"] = {
                key: tensor_record(tensor) for key, tensor in module.named_buffers()
            }
            # Some observers keep qparams outside registered buffers.
            for key, value in vars(module).items():
                if isinstance(value, torch.Tensor):
                    fields[key] = tensor_record(value)
                elif key == "_qparams" and value is not None:
                    if isinstance(value, (tuple, list)):
                        fields[key] = [tensor_record(item) if isinstance(item, torch.Tensor) else _json_value(item) for item in value]
                    elif isinstance(value, dict):
                        fields[key] = {str(k): tensor_record(v) if isinstance(v, torch.Tensor) else _json_value(v) for k, v in value.items()}
                    else:
                        fields[key] = _json_value(value)
            observers[name] = fields
    if not observers or not any(mode == "QUANT" for mode in modes.values()):
        raise ValueError("This regression runner requires a frozen quantized checkpoint.")
    return {"weights": weights, "observers": observers, "modes": modes}


def _seed(seed: int) -> None:
    """Reset all random sources before every export, not only before model load."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clone_inputs(value):
    """Protect tensor/list inputs from in-place mutation by a model forward."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(item) for key, item in value.items()}
    return value


def _probes(args, kwargs):
    """Include fixed-shape decode empty/full and padded-prefill mask probes."""
    import torch

    probes = [("export_example", _clone_inputs(args), _clone_inputs(kwargs))]
    mask = kwargs.get("attention_mask")
    if not isinstance(mask, torch.Tensor) or not mask.is_floating_point() or mask.ndim < 3:
        return probes
    if mask.shape[-2] == 1:
        for name, empty in (("decode_empty_past", True), ("decode_full_past", False)):
            new_args, new_kwargs = _clone_inputs(args), _clone_inputs(kwargs)
            new_mask = new_kwargs["attention_mask"]
            new_mask.fill_(-120.0 if empty else 0.0)
            new_mask[..., -1] = 0.0
            if empty:
                for key in ("past_key_value", "past_key_values"):
                    past = new_kwargs.get(key)
                    if isinstance(past, (tuple, list)) and all(isinstance(t, torch.Tensor) for t in past):
                        new_kwargs[key] = tuple(torch.zeros_like(t) for t in past)
            probes.append((name, new_args, new_kwargs))
    elif mask.shape[-1] == mask.shape[-2] and mask.shape[-1] > 1:
        new_args, new_kwargs = _clone_inputs(args), _clone_inputs(kwargs)
        new_kwargs["attention_mask"][..., -1] = -120.0
        probes.append(("prefill_last_key_padded", new_args, new_kwargs))
    return probes


class ExportRecorder:
    """Intercept real conversions/saves without replacing compiler or runtime."""

    def __init__(self, root: Path, *, atol: float, rtol: float):
        self.root = root
        self.records = root / "records"
        self.records.mkdir(parents=True)
        self.atol, self.rtol = atol, rtol
        self.stages = {}
        self.pending = 0

    def __enter__(self):
        import tico
        from tico.utils.model import CircleModel

        self.convert_module = importlib.import_module("tico.utils.convert")
        self.old_convert = tico.convert
        self.old_save = CircleModel.save
        self.old_export = self.convert_module.export
        self.fx = None

        def capture_export(*args, **kwargs):
            program = self.old_export(*args, **kwargs)
            self.fx = [
                {"op": node.op, "target": str(node.target)}
                for node in program.graph.nodes
                if node.op == "call_function"
            ]
            return program

        def capture_convert(module, args, *positional, **options):
            import numpy as np
            import torch
            from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
            from tico.utils.signature import ModelInputSpec

            if positional:
                raise TypeError("Regression capture requires named conversion options.")
            kwargs = options.get("kwargs") or {}
            probes = _probes(args, kwargs)
            expected = []
            with torch.no_grad():
                for name, probe_args, probe_kwargs in probes:
                    tensors = list(_tree_arrays(module(*_clone_inputs(probe_args), **_clone_inputs(probe_kwargs))).values())
                    expected.append(tensors)
                self.fx = None
                converted = self.old_convert(module, args, **options)
            if self.fx is None:
                raise RuntimeError("No Torch Export graph was captured; conversion path changed.")
            runtime = CircleReferenceRuntime(converted.circle_binary)
            mode = ExecutionMode.FAKE_QUANTIZE if runtime.has_quantized_activations() else ExecutionMode.NATIVE
            specification = ModelInputSpec(converted.circle_binary)
            arrays, numerical = {}, []
            for probe_index, (name, probe_args, probe_kwargs) in enumerate(probes):
                # Binding is name/order aware. Runtime validates effective dtypes.
                bound = specification.bind(probe_args, probe_kwargs, check=False)
                actual = runtime.run(bound, mode=mode).outputs
                for index, value in enumerate(bound):
                    arrays[f"p{probe_index}_input_{index}"] = _array(value).copy()
                for index, value in enumerate(actual):
                    arrays[f"p{probe_index}_circle_{index}"] = _array(value).copy()
                golden = expected[probe_index]
                comparisons = []
                if len(golden) != len(actual):
                    comparisons.append({"ok": False, "reason": "output_count"})
                else:
                    for index, (before, after) in enumerate(zip(golden, actual)):
                        arrays[f"p{probe_index}_torch_{index}"] = before
                        comparisons.append(compare_arrays(before, after, atol=self.atol, rtol=self.rtol))
                numerical.append({"probe": name, "torch_parity": comparisons})
            converted._export_regression = {"fx": self.fx, "arrays": arrays, "probes": numerical, "mode": mode.value}
            self.pending += 1
            return converted

        def capture_save(converted, path):
            from tico.circle import CircleDocument

            self.old_save(converted, path)
            payload = getattr(converted, "_export_regression", None)
            if payload is None:
                raise RuntimeError(f"Saved Circle was not captured: {path}")
            relative = Path(path).resolve().relative_to((self.root / "artifacts").resolve()).as_posix()
            if relative in self.stages:
                raise RuntimeError(f"Duplicate exported stage: {relative}")
            import numpy as np
            snapshot_name = relative.replace("/", "__")
            np.savez(self.records / (snapshot_name + ".npz"), **payload.pop("arrays"))
            with CircleDocument.load_mapped(path) as document:
                document.verify(raise_on_error=True)
                contract = circle_contract(document)
            _write_json(self.records / (snapshot_name + ".json"), {"contract": contract, **payload})
            self.stages[relative] = snapshot_name
            delattr(converted, "_export_regression")
            self.pending -= 1

        tico.convert = capture_convert
        CircleModel.save = capture_save
        self.convert_module.export = capture_export
        return self

    def __exit__(self, exc_type, exc_value, trace):
        import tico
        from tico.utils.model import CircleModel

        tico.convert = self.old_convert
        CircleModel.save = self.old_save
        self.convert_module.export = self.old_export
        if exc_type is None and self.pending:
            raise RuntimeError("Some converted stages were not saved and verified.")


def _host_contract(root: Path) -> dict:
    """Fingerprint host arrays and JSON contracts, not serialization timestamps."""
    import numpy as np
    import torch

    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".circle":
            continue
        name = path.relative_to(root).as_posix()
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as values:
                result[name] = {key: _array_record(values[key]) for key in sorted(values.files)}
        elif path.suffix == ".npy":
            result[name] = _array_record(np.load(path, allow_pickle=False, mmap_mode="r"))
        elif path.suffix in (".pt", ".pth"):
            values = torch.load(path, map_location="cpu", weights_only=True)
            result[name] = _json_value(values)
        elif path.suffix == ".json":
            result[name] = json.loads(path.read_text())
        else:
            # Text/vocabulary artifacts are intentionally compared byte-for-byte.
            result[name] = {"sha256": _file_sha256(path)}
    return result


def _export_full_text_probe(ctx, path: Path, options: argparse.Namespace) -> None:
    """Exercise the VLM full-text path in addition to deployment layer stages.

    This text-only probe is not a multimodal product artifact. It is useful for
    catching mask/RoPE generation accidentally reintroduced inside the trace.
    """
    from contextlib import nullcontext
    import torch
    import tico

    owners = [
        module for module in ctx.model.modules()
        if type(module).__name__ in ("QuantQwen3VLTextModel", "QuantGemma4TextModel")
    ]
    if len(owners) != 1:
        raise ValueError(
            "--include-full-text requires exactly one Qwen3-VL or Gemma4 text wrapper. "
            "For LLaMA, select --artifacts circle_bundle,circle_full instead."
        )
    owner = owners[0]

    class TextProbe(torch.nn.Module):
        """Make cache policy and output flattening static without cloning weights."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(
                input_ids=input_ids, use_cache=False, return_dict=True
            ).last_hidden_state

    adapter = TextProbe(owner).eval()
    inputs = torch.randint(0, int(owner.config.vocab_size), (1, options.seq_len))
    builder = getattr(owner, "_build_static_export_inputs", None)
    if callable(builder):
        from tico.quantization.wrapq.utils.static_export import prepared_static_export
        context = prepared_static_export(adapter, options.seq_len)
    else:
        context = nullcontext()
    previous_force_export = owner.force_export
    try:
        owner.force_export = True
        with context, torch.no_grad():
            result = tico.convert(adapter, (inputs,), strict=False)
        result.save(path / "full_text_regression.q.circle")
    finally:
        owner.force_export = previous_force_export


def _export_phase(ctx, phase: Path, options: argparse.Namespace, initial_state: dict) -> dict:
    """Export one phase and check that it preserved the frozen quantization state."""
    phase.mkdir(parents=True, exist_ok=False)
    ctx.cfg["export"]["output_dir"] = str(phase / "artifacts")
    _seed(options.seed)
    with ExportRecorder(phase, atol=options.atol, rtol=options.rtol) as recorder:
        ctx.adapter.export(ctx)
        if options.include_full_text:
            _export_full_text_probe(ctx, phase / "artifacts", options)
    if not recorder.stages:
        raise RuntimeError("No Circle stages were exported; check export.artifacts.")
    actual_stages = {
        path.relative_to(phase / "artifacts").as_posix()
        for path in (phase / "artifacts").rglob("*.circle")
    }
    if actual_stages != recorder.stages.keys():
        raise RuntimeError("Some Circle artifacts bypassed conversion capture.")
    state = _state_contract(ctx.model)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stages": recorder.stages,
        "state_preserved": state == initial_state,
        "state_differences": _first_differences(initial_state, state),
        "host": _host_contract(phase / "artifacts"),
    }
    _write_json(phase / "snapshot.json", result)
    return result


def _exercise_eager(ctx, options: argparse.Namespace) -> dict:
    """Run teacher-forced eager calls without changing the model's config."""
    import torch

    _seed(options.seed + 1)
    if options.eager_inputs:
        # User-supplied model kwargs are trusted for the same reason as checkpoint.
        loaded = torch.load(options.eager_inputs, map_location="cpu", weights_only=True)
        calls = loaded if isinstance(loaded, list) else [loaded]
        if not calls or not all(isinstance(item, dict) for item in calls):
            raise ValueError("--eager-inputs must contain a dict or a list of kwargs dicts.")
        with torch.no_grad():
            for call in calls:
                ctx.model(**_clone_inputs(call))
        return {"kind": "user_kwargs", "calls": len(calls), "file_sha256": _file_sha256(Path(options.eager_inputs))}
    config = getattr(ctx.model, "config", None)
    if config is None and hasattr(ctx.model, "wrapped"):
        config = ctx.model.wrapped.config
    text_config = getattr(config, "text_config", config)
    capacity = int(text_config.max_position_embeddings)
    length = options.eager_seq_len
    if length + options.decode_steps > capacity:
        raise ValueError(
            f"Eager stress requires {length + options.decode_steps} tokens but this "
            f"checkpoint records capacity={capacity}. Do not silently enlarge its "
            "RoPE config; select a valid stress length or a native-context checkpoint."
        )
    inputs = torch.randint(0, int(text_config.vocab_size), (1, length))
    with torch.no_grad():
        output = ctx.model(input_ids=inputs, use_cache=True, logits_to_keep=1, return_dict=True)
        cache = output.past_key_values
        if cache is None and options.decode_steps:
            raise RuntimeError("Eager prefill did not produce a KV cache.")
        for _ in range(options.decode_steps):
            token = torch.randint(0, int(text_config.vocab_size), (1, 1))
            output = ctx.model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            cache = output.past_key_values
    return {"kind": "text_only", "prefill_tokens": length, "decode_steps": options.decode_steps, "capacity": capacity}


def _worker(options: argparse.Namespace) -> None:
    """Load one checkout/checkpoint in an isolated process and collect snapshots."""
    repo = Path(options.repo).resolve()
    sys.path.insert(0, str(repo))
    import torch
    import tico
    from tico.quantization.recipes.adapters import resolve_adapter
    from tico.quantization.recipes.config import load_recipe_config
    from tico.quantization.recipes.context import RecipeContext
    from tico.quantization.recipes.extensions import load_recipe_extensions

    if not Path(tico.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("TICO was imported from a different source tree.")
    torch.set_num_threads(options.threads)
    root = Path(options.output)
    root.mkdir(parents=True, exist_ok=False)
    overrides = list(options.set)
    overrides += [
        "runtime.device=cpu", "runtime.dtype=float32", "evaluation.enabled=false",
        "export.enabled=true", "export.prefill_decode=true",
        "export.artifacts=[" + options.artifacts + "]",
        f"export.max_seq_len={options.seq_len}",
        # The pre-PR adapter takes its export length from calibration.seq_len.
        # No calibration runs: both fields merely select the same artifact shape.
        f"calibration.seq_len={options.seq_len}",
    ]
    cfg = load_recipe_config(options.config, overrides=overrides)
    if options.artifacts == "auto":
        family = str(cfg.get("model", {}).get("family", ""))
        cfg["export"]["artifacts"] = [
            "circle_bundle" if family == "llama" else "circle_per_layer"
        ]
    supported = {"circle_per_layer", "circle_bundle", "circle_full"}
    if not set(cfg["export"]["artifacts"]).issubset(supported):
        raise ValueError("Only Circle artifacts may be requested by this runner.")
    load_recipe_extensions(cfg)
    adapter = resolve_adapter(cfg)
    _seed(options.seed)
    model = torch.load(options.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The checkpoint must be a frozen TICO model, not a state_dict.")
    non_fp32 = [
        name for name, value in model.named_parameters()
        if value.is_floating_point() and value.dtype != torch.float32
    ]
    if non_fp32:
        raise ValueError(
            "The CPU export regression requires a float32 frozen checkpoint; "
            "implicit casting would change the comparison. Non-float32 parameters: "
            + ", ".join(non_fp32[:5])
        )
    model.eval()
    ctx = RecipeContext(cfg=cfg, adapter=adapter, model=model, device=torch.device("cpu"), dtype=torch.float32)
    initial = _state_contract(model)
    rope_migration = None
    if options.migrate_llama_rope:
        if str(cfg.get("model", {}).get("family", "")) != "llama":
            raise ValueError("--migrate-llama-rope is only valid for the LLaMA family.")
        from tico.quantization.wrapq.wrappers.llama.static_inputs import (
            migrate_llama_rope_state,
        )
        rope_migration = migrate_llama_rope_state(model)
        if _state_contract(model) != initial:
            raise RuntimeError("RoPE migration changed frozen quantization state.")
    _write_json(root / "quantization_state.json", initial)
    versions = {}
    for package in ("torch", "transformers", "circle-schema", "numpy"):
        versions[package] = importlib.metadata.version(package)
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True)
    _write_json(root / "environment.json", {
        "repo": str(repo), "commit": git.stdout.strip() if git.returncode == 0 else None,
        "working_tree": dirty.stdout if dirty.returncode == 0 else None,
        "versions": versions, "checkpoint_sha256": _file_sha256(Path(options.checkpoint)),
        "effective_config": cfg,
        "llama_rope_migration": rope_migration,
    })
    _export_phase(ctx, root / "before", options, initial)
    if options.stress:
        exercise = _exercise_eager(ctx, options)
        after_eager = _state_contract(model)
        exercise["state_preserved"] = after_eager == initial
        exercise["state_differences"] = _first_differences(initial, after_eager)
        _write_json(root / "eager.json", exercise)
        _export_phase(ctx, root / "after_eager", options, initial)


def compare_snapshots(left_root: Path, right_root: Path, *, atol: float, rtol: float, require_torch_parity: bool = False) -> dict:
    """Compare structure, actual inputs/outputs, host data, and state invariants."""
    import numpy as np

    left = json.loads((left_root / "snapshot.json").read_text())
    right = json.loads((right_root / "snapshot.json").read_text())
    failures, stages = [], {}
    for side, snapshot in (("left", left), ("right", right)):
        if snapshot["schema_version"] != SCHEMA_VERSION:
            failures.append(side + ": incompatible snapshot schema")
        if not snapshot["state_preserved"]:
            failures.append(side + ": weights/observers/qparams/modes changed")
    if left["stages"].keys() != right["stages"].keys():
        failures.append("Exported stage sets differ")
    if left["host"] != right["host"]:
        failures.extend("Host " + item for item in _first_differences(left["host"], right["host"]))
    for name in sorted(left["stages"].keys() & right["stages"].keys()):
        a_path = left_root / "records" / left["stages"][name]
        b_path = right_root / "records" / right["stages"][name]
        a, b = json.loads(Path(str(a_path) + ".json").read_text()), json.loads(Path(str(b_path) + ".json").read_text())
        record = {
            "circle_structure_equal": a["contract"] == b["contract"],
            "circle_structure_differences": _first_differences(a["contract"], b["contract"]),
            "torch_export_ops_equal": a["fx"] == b["fx"],
            "probe_names_equal": [p["probe"] for p in a["probes"]] == [p["probe"] for p in b["probes"]],
            "execution_mode_equal": a["mode"] == b["mode"],
            "torch_parity_left": a["probes"], "torch_parity_right": b["probes"],
            "arrays": {},
        }
        with np.load(str(a_path) + ".npz", allow_pickle=False) as av, np.load(str(b_path) + ".npz", allow_pickle=False) as bv:
            record["array_keys_equal"] = set(av.files) == set(bv.files)
            record["numerical_coverage_complete"] = all(
                bool(metadata["probes"])
                and all(
                    any(key.startswith(f"p{index}_circle_") for key in values.files)
                    for index in range(len(metadata["probes"]))
                )
                for metadata, values in ((a, av), (b, bv))
            )
            for key in sorted(set(av.files) & set(bv.files)):
                # Inputs must match exactly; output tolerances are explicit.
                tol = (0.0, 0.0) if "_input_" in key else (atol, rtol)
                record["arrays"][key] = compare_arrays(av[key], bv[key], atol=tol[0], rtol=tol[1])
        stage_ok = all(record[key] for key in ("circle_structure_equal", "probe_names_equal", "execution_mode_equal", "array_keys_equal", "numerical_coverage_complete"))
        stage_ok &= all(value["ok"] for value in record["arrays"].values())
        if require_torch_parity:
            stage_ok &= all(item["ok"] for p in a["probes"] + b["probes"] for item in p["torch_parity"])
        record["ok"] = bool(stage_ok)
        if not stage_ok:
            failures.append(name + ": structural, input, output, or requested Torch parity mismatch")
        stages[name] = record
    if not stages:
        failures.append("No common exported stages were compared")
    return {"ok": not failures, "failures": failures, "stages": stages}


def _run(options: argparse.Namespace) -> int:
    """Run two workers, then compare baseline/candidate and before/after eager."""
    output = Path(options.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    script = str(Path(__file__).resolve())
    common = [
        "--config", str(Path(options.config).resolve()),
        "--checkpoint", str(Path(options.checkpoint).resolve()),
        "--seq-len", str(options.seq_len), "--eager-seq-len", str(options.eager_seq_len),
        "--decode-steps", str(options.decode_steps), "--artifacts", options.artifacts,
        "--seed", str(options.seed), "--threads", str(options.threads),
        "--atol", str(options.atol), "--rtol", str(options.rtol),
    ]
    if options.eager_inputs:
        common += ["--eager-inputs", str(Path(options.eager_inputs).resolve())]
    if options.include_full_text:
        common.append("--include-full-text")
    for override in options.set:
        common += ["--set", override]
    workers = [("candidate", options.candidate_repo, True)]
    if options.command == "run":
        workers.insert(0, ("baseline", options.baseline_repo, False))
    for role, repo, stress in workers:
        command = [sys.executable, script, "_worker", "--repo", str(Path(repo).resolve()), "--output", str(output / role), *common]
        if stress:
            command.append("--stress")
        if role == "candidate" and options.migrate_llama_rope:
            command.append("--migrate-llama-rope")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        with (output / (role + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=Path(repo).resolve(), env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            _write_json(output / "summary.json", {"status": "ERROR", "worker": role, "returncode": result.returncode, "log": role + ".log", "comparisons_complete": False})
            print(f"ERROR: {role} worker failed; inspect {output / (role + '.log')}")
            return 2
    identities_match, initial_match = True, True
    old_new = {"status": "NOT_RUN", "reason": "Isolation-only run; no baseline checkout."}
    if options.command == "run":
        baseline = json.loads((output / "baseline/environment.json").read_text())
        candidate = json.loads((output / "candidate/environment.json").read_text())
        identities_match = baseline["versions"] == candidate["versions"] and baseline["checkpoint_sha256"] == candidate["checkpoint_sha256"]
        initial_match = json.loads((output / "baseline/quantization_state.json").read_text()) == json.loads((output / "candidate/quantization_state.json").read_text())
        old_new = compare_snapshots(output / "baseline/before", output / "candidate/before", atol=options.atol, rtol=options.rtol, require_torch_parity=options.require_torch_parity)
    before_after = compare_snapshots(output / "candidate/before", output / "candidate/after_eager", atol=options.atol, rtol=options.rtol, require_torch_parity=options.require_torch_parity)
    eager = json.loads((output / "candidate/eager.json").read_text())
    ok = identities_match and initial_match and old_new.get("ok", options.command == "isolation") and before_after["ok"] and eager["state_preserved"]
    summary = {
        "status": "PASS" if ok else "FAIL", "comparisons_complete": True,
        "requested_scope": options.command,
        "baseline_comparison_performed": options.command == "run",
        "same_checkpoint_and_dependencies": identities_match,
        "same_initial_quantization_state": initial_match,
        "baseline_vs_candidate": old_new, "candidate_before_vs_after_eager": before_after,
        "eager": eager, "atol": options.atol, "rtol": options.rtol,
        "torch_parity_required": options.require_torch_parity,
        "llama_rope_migration_requested": options.migrate_llama_rope,
        "intentional_change_note": (
            "The candidate uses unified LLaMA RoPE. Differences from an old "
            "unscaled decoder are reported as FAIL, never waived automatically."
            if options.migrate_llama_rope else None
        ),
        "scope": "Conservative Circle structure and stage-level reference numerics; not NPU or end-to-end generation validation.",
    }
    _write_json(output / "summary.json", summary)
    print(f"{summary['status']}: {output / 'summary.json'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    """Provide isolated collection, complete orchestration, and offline comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Export both revisions and candidate after eager use.")
    run.add_argument("--baseline-repo", required=True)
    run.add_argument("--candidate-repo", default=".")
    isolation = sub.add_parser("isolation", help="Compare only candidate export before/after eager use.")
    isolation.add_argument("--candidate-repo", default=".")
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--repo", required=True)
    worker.add_argument("--stress", action="store_true")
    for command in (run, isolation, worker):
        command.add_argument("--config", required=True)
        command.add_argument("--checkpoint", required=True, help="Trusted frozen model checkpoint; torch.load uses weights_only=False.")
        command.add_argument("--output", required=True, help="A new output directory; existing results are never overwritten.")
        command.add_argument("--seq-len", type=int, required=True)
        command.add_argument("--eager-seq-len", type=int, required=True)
        command.add_argument("--eager-inputs", help="Optional torch-saved kwargs dict/list for real multimodal eager stress.")
        command.add_argument("--include-full-text", action="store_true", help="Also compare a text-only full-model probe for Qwen3-VL/Gemma4.")
        command.add_argument(
            "--migrate-llama-rope", action="store_true",
            help="Explicitly rebuild candidate LLaMA RoPE from its unified snapshot or recorded config. The baseline and frozen qparams are unchanged.",
        )
        command.add_argument("--decode-steps", type=int, default=2)
        command.add_argument("--artifacts", default="auto", help="Comma-separated Circle artifacts; auto selects LLaMA bundle or VLM per-layer export.")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--set", action="append", default=[])
    compare = sub.add_parser("compare", help="Compare previously collected phase directories.")
    compare.add_argument("left")
    compare.add_argument("right")
    compare.add_argument("--report", required=True)
    for command in (run, isolation, worker, compare):
        command.add_argument("--atol", type=float, default=1e-5)
        command.add_argument("--rtol", type=float, default=1e-4)
        command.add_argument("--require-torch-parity", action="store_true", help="Also require PyTorch-vs-Circle parity, not just before-vs-after parity.")
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error("Tolerances must be finite and nonnegative.")
    if args.command != "compare":
        if min(args.seq_len, args.eager_seq_len, args.threads) < 1 or args.decode_steps < 0:
            parser.error("Lengths/threads must be positive and decode steps nonnegative.")
        if not Path(args.checkpoint).is_file() or not Path(args.config).is_file():
            parser.error("Checkpoint and config must be existing files.")
        if Path(args.output).exists():
            parser.error("Output already exists; choose a new directory.")
    try:
        if args.command == "_worker":
            _worker(args)
            return 0
        if args.command in ("run", "isolation"):
            return _run(args)
        result = compare_snapshots(Path(args.left), Path(args.right), atol=args.atol, rtol=args.rtol, require_torch_parity=args.require_torch_parity)
        _write_json(Path(args.report), result)
        print("PASS" if result["ok"] else "FAIL")
        return 0 if result["ok"] else 1
    except Exception as exc:
        traceback.print_exc()
        output = getattr(args, "output", None)
        if output and Path(output).is_dir():
            _write_json(Path(output) / "error.json", {
                "status": "ERROR", "error_type": type(exc).__name__,
                "message": str(exc), "comparisons_complete": False,
            })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

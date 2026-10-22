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

"""Connected image-conditioned execution of the saved Qwen static stage ABI."""

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.recipes.debug.cpu_fixtures import qwen_batch
from tico.quantization.recipes.debug.cpu_followup import _write
from tico.quantization.recipes.debug.export_regression import (
    _array,
    _file_sha256,
    _state_contract,
    _tree_arrays,
    compare_arrays,
)
from tico.quantization.recipes.export.qwen3_vl import _unwrap_qwen3_vl_components
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLDeepstackFusionExportAdapter,
    Qwen3VLLMHeadExportAdapter,
    Qwen3VLMultimodalEmbeddingExportAdapter,
    Qwen3VLTextEmbeddingExportAdapter,
)
from tico.utils.signature import ModelInputSpec


def decode_mask(*, capacity: int, past_len: int, fill: float) -> torch.Tensor:
    """Validate logical capacity before constructing the fixed-slot host mask."""
    if type(capacity) is not int or capacity < 2:
        raise ValueError("Static decode capacity must be an integer of at least two.")
    if type(past_len) is not int or not 0 <= past_len < capacity:
        raise ValueError("Static cache has no room for the requested current token.")
    if not math.isfinite(fill):
        raise ValueError("The static attention mask fill must be finite.")
    mask = torch.full((1, 1, 1, capacity), fill)
    mask[..., :past_len] = 0.0
    mask[..., -1] = 0.0
    return mask


class _Stages:
    """Execute saved bytes, retaining every input and Torch/Circle output."""

    def __init__(self, artifacts: Path, root: Path):
        self.artifacts = artifacts
        self.root = root
        self.records: list[dict[str, Any]] = []
        self.runtime: dict[str, tuple[CircleReferenceRuntime, ModelInputSpec]] = {}

    def run(
        self, name: str, module: torch.nn.Module, args: tuple, **kwargs
    ) -> list[torch.Tensor]:
        """Use the real runtime and same host inputs for local stage parity."""
        if name not in self.runtime:
            binary = (self.artifacts / name).read_bytes()
            self.runtime[name] = (
                CircleReferenceRuntime(binary),
                ModelInputSpec(binary),
            )
        runtime, binder = self.runtime[name]
        bound = binder.bind(args, kwargs, check=False)
        index = len(self.records)
        torch.save(
            {"args": args, "kwargs": kwargs}, self.root / f"call-{index:03d}.inputs.pt"
        )
        golden = list(_tree_arrays(module(*args, **kwargs)).values())
        mode = (
            ExecutionMode.FAKE_QUANTIZE
            if runtime.has_quantized_activations()
            else ExecutionMode.NATIVE
        )
        actual = runtime.run(bound, mode=mode).outputs
        if len(golden) != len(actual) or not actual:
            raise RuntimeError(f"{name}: incomplete output evidence.")
        values = {f"input_{index}": _array(value) for index, value in enumerate(bound)}
        for kind, tensors in (("torch", golden), ("circle", actual)):
            values.update(
                {
                    f"{kind}_{index}": _array(value)
                    for index, value in enumerate(tensors)
                }
            )
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"{name}: nonfinite input/output evidence.")
        comparisons = [
            compare_arrays(left, right, atol=1e-5, rtol=1e-4)
            for left, right in zip(golden, actual)
        ]
        np.savez(self.root / f"call-{index:03d}.npz", **values)
        self.records.append(
            {
                "stage": name,
                "evidence": f"call-{index:03d}.npz",
                "structured_inputs": f"call-{index:03d}.inputs.pt",
                "input_count": len(bound),
                "output_count": len(actual),
                "outputs": comparisons,
                "ok": all(result["ok"] for result in comparisons),
                "mode": mode.value,
            }
        )
        return [torch.from_numpy(np.array(value, copy=True)) for value in actual]


def connected_qwen(checkpoint: Path, artifacts: Path, root: Path, capacity: int) -> int:
    """Write actual prefill deltas to fixed buffers and consume them in decode.

    The current token occupies the final attention column. Valid past values
    occupy a left-aligned prefix of S-1 host slots; other slots stay masked.
    The last decode consumes S-1 valid past positions and fills capacity S.
    """
    if capacity not in (16, 32):
        raise ValueError("The CPU static integration profile must be 16 or 32.")
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=False)
    model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    initial = _state_contract(model)
    top, multimodal_model, vision, text = _unwrap_qwen3_vl_components(model)
    stages = _Stages(artifacts, root)
    prompt_len = capacity - 4
    batch = qwen_batch(image=True, length=prompt_len)
    ids = torch.zeros((1, capacity), dtype=torch.long)
    ids[:, :prompt_len] = batch["input_ids"]
    embedding = Qwen3VLTextEmbeddingExportAdapter(top).eval()
    multimodal = Qwen3VLMultimodalEmbeddingExportAdapter(top, visual_start_idx=2).eval()
    lm_head = Qwen3VLLMHeadExportAdapter(top).eval()
    vision_adapter = vision.as_export_module("prefill", grid_thw=(1, 2, 2)).eval()
    buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        vision_outputs = stages.run(
            "vision_prefill_t1_h2_w2.q.circle", vision_adapter, (batch["pixel_values"],)
        )
        hidden = stages.run(
            "multimodal_embedding_prefill_t1_h2_w2.q.circle",
            multimodal,
            (ids, vision_outputs[0]),
        )[0]
        prompt_positions, rope_delta = multimodal_model.module.get_rope_index(
            input_ids=batch["input_ids"],
            image_grid_thw=batch["image_grid_thw"],
            attention_mask=batch["attention_mask"],
            mm_token_type_ids=batch["mm_token_type_ids"],
        )
        positions = torch.zeros((3, 1, capacity), dtype=torch.long)
        positions[..., :prompt_len] = prompt_positions
        positions[..., prompt_len:] = torch.arange(prompt_len, capacity).view(
            1, 1, -1
        ) + rope_delta.view(1, 1, 1)
        np.savez(
            root / "host-positions.npz",
            prompt_positions=prompt_positions.numpy(),
            rope_delta=rope_delta.numpy(),
            padded_positions=positions.numpy(),
        )
        cos, sin = text.rotary_emb(hidden, positions)
        cos, sin = text._fq(cos, text.obs_cos), text._fq(sin, text.obs_sin)
        columns = torch.arange(capacity)
        visible = (columns[None, :] <= columns[:, None]) & (
            columns[None, :] < prompt_len
        )
        fill = float(text.qcfg.attention_mask_fill_value)
        mask = torch.where(visible, 0.0, fill)[None, None]
        mask = text._fq(mask, text.obs_attention_mask)
        for index, layer in enumerate(text.layers):
            outputs = stages.run(
                f"decoder_layer_prefill_{index}.q.circle",
                cast(PTQWrapper, layer).as_export_module("prefill", return_kv=True),
                (hidden,),
                attention_mask=mask,
                position_embeddings=(cos, sin),
            )
            hidden, keys, values = outputs
            if keys.shape[2] != capacity or values.shape != keys.shape:
                raise RuntimeError(
                    "Prefill delta KV does not match the selected static capacity."
                )
            past_key = torch.zeros_like(keys[:, :, :-1])
            past_value = torch.zeros_like(past_key)
            past_key[:, :, :prompt_len] = keys[:, :, :prompt_len]
            past_value[:, :, :prompt_len] = values[:, :, :prompt_len]
            buffers.append((past_key, past_value))
            if index < len(vision.deepstack_merger_list):
                hidden = stages.run(
                    f"deepstack_fusion_{index}_t1_h2_w2.q.circle",
                    Qwen3VLDeepstackFusionExportAdapter(visual_start_idx=2),
                    (hidden, vision_outputs[index + 1]),
                )[0]
        logits = stages.run(
            "lm_head.q.circle", lm_head, (hidden[:, prompt_len - 1 : prompt_len],)
        )[0]
        lengths = [prompt_len]
        for step in range(4):
            past_len = prompt_len + step
            token = logits[:, -1].argmax(-1, keepdim=True)
            hidden = stages.run("token_embedding.q.circle", embedding, (token,))[0]
            mask = decode_mask(capacity=capacity, past_len=past_len, fill=fill)
            mask = text._fq(mask, text.obs_attention_mask)
            position_ids = torch.full(
                (3, 1, 1), past_len, dtype=torch.long
            ) + rope_delta.view(1, 1, 1)
            rope = text.rotary_emb(hidden, position_ids)
            rope = (text._fq(rope[0], text.obs_cos), text._fq(rope[1], text.obs_sin))
            for index, layer in enumerate(text.layers):
                outputs = stages.run(
                    f"decoder_layer_decode_{index}.q.circle",
                    cast(PTQWrapper, layer).as_export_module("decode", return_kv=True),
                    (hidden,),
                    attention_mask=mask,
                    position_embeddings=rope,
                    past_key_values=buffers[index],
                )
                hidden, new_key, new_value = outputs
                if (
                    tuple(new_key.shape[2:]) != (1, int(text.config.head_dim))
                    or new_value.shape != new_key.shape
                ):
                    raise RuntimeError(
                        "Decode must return exactly one new KV position."
                    )
                if past_len < capacity - 1:
                    buffers[index][0][:, :, past_len : past_len + 1] = new_key
                    buffers[index][1][:, :, past_len : past_len + 1] = new_value
            logits = stages.run("lm_head.q.circle", lm_head, (hidden,))[0]
            lengths.append(past_len + 1)
        cache_arrays: dict[str, Any] = {
            f"layer_{index}_{kind}": value.numpy()
            for index, pair in enumerate(buffers)
            for kind, value in zip(("key", "value"), pair)
        }
        np.savez(root / "final-host-cache.npz", **cache_arrays)
    capacity_error = ""
    try:
        decode_mask(capacity=capacity, past_len=capacity, fill=fill)
    except ValueError as error:
        capacity_error = str(error)
    else:
        raise AssertionError("A full static cache must reject another decode token.")
    preserved = initial == _state_contract(model)
    parity = all(record["ok"] for record in stages.records)
    _write(
        root / "summary.json",
        {
            "status": "PASS" if parity and preserved else "FAIL",
            "capacity": capacity,
            "checkpoint_sha256": _file_sha256(checkpoint),
            "full_cache_rejection": capacity_error,
            "prefill_valid_length": prompt_len,
            "connected_valid_lengths": lengths,
            "past_buffer_slots": capacity - 1,
            "cache_source": "actual saved Circle image-conditioned prefill and connected decode deltas",
            "local_torch_circle_parity": parity,
            "frozen_state_preserved": preserved,
            "stages": stages.records,
            "strict_tolerances": {"atol": 1e-5, "rtol": 1e-4},
        },
    )
    return 0 if parity and preserved else 1

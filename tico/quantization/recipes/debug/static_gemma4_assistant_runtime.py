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

"""Reference debug flow for the static Gemma4 assistant draft-one runtime.

The flow mirrors the intended NPU execution:

1. Run the FP target on a prompt (or build a synthetic target-side fixture)
   to obtain the last-token embedding, last hidden state, and shared KV.
2. Concatenate embedding and hidden state into the assistant input.
3. Run the FP assistant draft-one step (reference).
4. Prepare → calibrate → convert the assistant and run the quantized
   draft-one step (computed independently from the FP reference).
5. Run the static canonicalized core through the export adapter.
6. Compute the next draft token with the host ordered sparse head.
7. Print parity metrics.
"""

import copy
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_assistant_builders import (
    build_gemma4_assistant_ptq_config,
)
from tico.quantization.recipes.utils import set_seed, torch_dtype_from_name
from tico.quantization.wrapq.wrappers.gemma4_assistant.export_adapters import (
    Gemma4AssistantCoreExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.sparse_head import (
    select_sparse_candidates,
    sparse_top1_token,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
    canonicalize_gemma4_assistant_static_inputs,
    Gemma4AssistantStaticShapeConfig,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    assistant_layer_type_head_dim,
    assistant_shared_kv_num_heads,
    extract_assistant_text_config,
    validate_gemma4_assistant_architecture,
)


@dataclass
class StaticGemma4AssistantRuntimeConfig:
    """Configuration for the static Gemma4 assistant debug runtime."""

    model: str = "google/gemma-4-E2B-it-assistant"
    target_model: Optional[str] = None
    device: str = "cpu"
    dtype: str = "float32"
    prompt: str = "The capital of France is"
    full_kv_length: int = 64
    sliding_kv_length: int = 64
    calibration_samples: int = 4
    seed: int = 42
    synthetic: bool = False


def _make_tiny_synthetic_assistant() -> torch.nn.Module:
    """Build the tiny synthetic assistant used when no checkpoint is given."""
    from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
        make_tiny_gemma4_assistant_model,
    )

    return make_tiny_gemma4_assistant_model()


def _load_assistant(cfg: StaticGemma4AssistantRuntimeConfig) -> torch.nn.Module:
    """Load the assistant checkpoint or fall back to the tiny synthetic model."""
    if cfg.synthetic:
        return _make_tiny_synthetic_assistant()
    from transformers import AutoModelForCausalLM

    path = os.environ.get("GEMMA4_ASSISTANT_PATH", cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch_dtype_from_name(cfg.dtype)
    )
    return model.eval()


def _synthetic_target_fixture(
    assistant: torch.nn.Module, *, kv_len: int
) -> dict[str, Any]:
    """Build a target-side fixture with the correct shared-KV geometry."""
    text_config = extract_assistant_text_config(assistant.config)
    kv_heads = assistant_shared_kv_num_heads(text_config)
    backbone = int(assistant.config.backbone_hidden_size)
    shared_kv_states = {
        layer_type: (
            torch.randn(
                1,
                kv_heads,
                kv_len,
                assistant_layer_type_head_dim(text_config, layer_type),
            ),
            torch.randn(
                1,
                kv_heads,
                kv_len,
                assistant_layer_type_head_dim(text_config, layer_type),
            ),
        )
        for layer_type in ("full_attention", "sliding_attention")
    }
    return {
        "inputs_embeds": torch.cat(
            [torch.randn(1, 1, backbone), torch.randn(1, 1, backbone)], dim=-1
        ),
        "position_ids": torch.tensor([[kv_len - 1]]),
        "attention_mask": torch.ones(1, kv_len, dtype=torch.long),
        "shared_kv_states": shared_kv_states,
    }


def _target_fixture_from_prompt(
    cfg: StaticGemma4AssistantRuntimeConfig,
    assistant: torch.nn.Module,
) -> Optional[dict[str, Any]]:
    """Run the FP target on the prompt and collect the assistant inputs."""
    target_path = os.environ.get("GEMMA4_TARGET_PATH", cfg.target_model)
    if not target_path or cfg.synthetic:
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(cfg.device)
    dtype = torch_dtype_from_name(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(target_path)
    target = AutoModelForCausalLM.from_pretrained(target_path, dtype=dtype)
    target = target.to(device).eval()

    encoded = tokenizer(cfg.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = target(
            **encoded,
            output_hidden_states=True,
            return_shared_kv_states=True,
            use_cache=False,
        )
    input_ids = encoded["input_ids"]
    last_embedding = target.get_input_embeddings()(input_ids[:, -1:])
    last_hidden = outputs.hidden_states[-1][:, -1:, :]
    fixture = {
        "inputs_embeds": torch.cat([last_embedding, last_hidden], dim=-1)
        .to(torch.float32)
        .cpu(),
        "position_ids": torch.tensor([[input_ids.shape[1] - 1]]),
        "attention_mask": torch.ones_like(input_ids).cpu(),
        "shared_kv_states": {
            layer_type: (
                key.to(torch.float32).cpu(),
                value.to(torch.float32).cpu(),
            )
            for layer_type, (key, value) in outputs.shared_kv_states.items()
        },
    }
    del target
    return fixture


def _print_metric(label: str, value: float) -> None:
    print(f"{label:<29}: {value:.6e}")


def run_static_gemma4_assistant_runtime(
    cfg: StaticGemma4AssistantRuntimeConfig,
) -> dict[str, Any]:
    """Run the FP/PTQ/static parity reference flow and print metrics."""
    set_seed(cfg.seed)

    assistant = _load_assistant(cfg).to(torch.float32).eval()
    validate_gemma4_assistant_architecture(assistant)
    text_config = extract_assistant_text_config(assistant.config)

    fixture = _target_fixture_from_prompt(cfg, assistant)
    if fixture is None:
        print(
            "[Info] No target model available; using a synthetic target-side "
            "fixture with the correct shared-KV geometry."
        )
        fixture = _synthetic_target_fixture(
            assistant, kv_len=max(2, cfg.full_kv_length // 2)
        )

    forward_kwargs = dict(fixture, use_cache=False)

    # 1) FP reference draft-one step.
    with torch.no_grad():
        fp_out = assistant(**forward_kwargs)
    fp_token = int(fp_out.logits[:, -1].argmax(dim=-1).item())

    # 2) PTQ prepare → calibrate → convert on an independent copy.
    quant_source = copy.deepcopy(assistant)
    qcfg = build_gemma4_assistant_ptq_config(
        num_hidden_layers=int(text_config.num_hidden_layers),
        model_args={
            "assistant": {
                "full_kv_length": cfg.full_kv_length,
                "sliding_kv_length": cfg.sliding_kv_length,
            }
        },
    )
    prepared = prepare(quant_source, qcfg)
    kv_len = int(fixture["shared_kv_states"]["full_attention"][0].shape[2])
    with torch.no_grad():
        prepared(**forward_kwargs)
        for _ in range(max(0, cfg.calibration_samples - 1)):
            prepared(**dict(_synthetic_target_fixture(assistant, kv_len=kv_len)))
    quantized = convert(prepared)

    with torch.no_grad():
        ptq_out = quantized(**forward_kwargs)
    ptq_token = int(ptq_out.logits[:, -1].argmax(dim=-1).item())

    # 3) Static canonicalized execution through the export adapter.
    shape = Gemma4AssistantStaticShapeConfig(
        full_kv_length=cfg.full_kv_length,
        sliding_kv_length=cfg.sliding_kv_length,
    )
    static = canonicalize_gemma4_assistant_static_inputs(
        inputs_embeds=fixture["inputs_embeds"],
        position_ids=fixture["position_ids"],
        attention_mask=fixture["attention_mask"],
        shared_kv_states=fixture["shared_kv_states"],
        shape=shape,
        model_or_config=assistant.config,
        rotary_emb=assistant.model.rotary_emb,
        mask_fill_value=float(qcfg.attention_mask_fill_value),
    )
    adapter = Gemma4AssistantCoreExportAdapter(quantized)
    with torch.no_grad():
        projected, hidden, centroid_logits = adapter(*static.as_tuple())

    # 4) Host sparse head on the static core outputs.
    quant_assistant = quantized.wrapped
    lm_head_weight = quant_assistant._lm_head_weight()
    candidates = select_sparse_candidates(
        hidden,
        lm_head_weight,
        centroid_logits=centroid_logits,
        token_ordering=quant_assistant.masked_embedding.token_ordering,
        num_centroids=quant_assistant.masked_embedding.num_centroids,
        centroid_top_k=quant_assistant.masked_embedding.centroid_top_k,
    )
    static_token = int(sparse_top1_token(candidates).item())

    fp_projected = fp_out.last_hidden_state
    with torch.no_grad():
        projected_inputs = assistant.pre_projection(fixture["inputs_embeds"])
        fp_backbone_hidden = assistant.model(
            input_ids=None,
            inputs_embeds=projected_inputs,
            attention_mask=assistant.create_attention_masks(
                projected_inputs,
                fixture["attention_mask"],
                fixture["shared_kv_states"],
            ),
            position_ids=fixture["position_ids"],
            shared_kv_states=fixture["shared_kv_states"],
            use_cache=False,
        ).last_hidden_state
        fp_centroid = assistant.masked_embedding.centroids(fp_backbone_hidden)

    fp_selected = select_sparse_candidates(
        fp_backbone_hidden,
        assistant.lm_head.weight,
        centroid_logits=fp_centroid,
        token_ordering=assistant.masked_embedding.token_ordering,
        num_centroids=int(assistant.config.num_centroids),
        centroid_top_k=int(assistant.config.centroid_intermediate_top_k),
    )

    print("\n=== Assistant core parity ===")
    _print_metric(
        "projected_state mean_abs_diff",
        (fp_projected - projected).abs().mean().item(),
    )
    _print_metric(
        "projected_state max_abs_diff ",
        (fp_projected - projected).abs().max().item(),
    )
    _print_metric(
        "hidden mean_abs_diff         ",
        (fp_backbone_hidden - hidden).abs().mean().item(),
    )
    _print_metric(
        "centroid mean_abs_diff       ",
        (fp_centroid - centroid_logits).abs().mean().item(),
    )

    fp_candidate_ids = set(fp_selected.selected_token_ids.flatten().tolist())
    ptq_candidate_ids = set(candidates.selected_token_ids.flatten().tolist())
    candidate_recall = (
        len(fp_candidate_ids & ptq_candidate_ids) / len(fp_candidate_ids)
        if fp_candidate_ids
        else 0.0
    )
    if torch.equal(fp_selected.selected_token_ids, candidates.selected_token_ids):
        selected_mae = (
            (fp_selected.selected_logits - candidates.selected_logits)
            .abs()
            .mean()
            .item()
        )
    else:
        # Different centroid selections make positional logit comparison
        # meaningless; report NaN and rely on candidate recall instead.
        selected_mae = float("nan")

    print("\n=== Sparse head parity ===")
    print(f"fp token                     : {fp_token}")
    print(f"ptq token                    : {ptq_token}")
    print(f"static token                 : {static_token}")
    print(f"top1 agreement               : {int(fp_token == static_token)}")
    _print_metric("candidate recall             ", candidate_recall)
    _print_metric("selected logits MAE          ", selected_mae)

    return {
        "fp_token": fp_token,
        "ptq_token": ptq_token,
        "static_token": static_token,
        "projected_state_max_abs_diff": (fp_projected - projected).abs().max().item(),
    }

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

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization.config.builders import build_llm_ptq_config
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.config import get_by_path
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.llm import build_wikitext_calibration_inputs
from tico.quantization.recipes.evaluation.llm import (
    evaluate_lm_tasks,
    evaluate_perplexity,
)
from tico.quantization.recipes.export.checkpoint import save_checkpoint
from tico.quantization.recipes.export.circle import export_full_circle
from tico.quantization.recipes.export.llama import export_llama_per_layer
from tico.quantization.recipes.utils import (
    quant_spec_from_config,
    quant_specs_equivalent,
    torch_dtype_from_name,
)


def _weights_share_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return True if two weight tensors share the exact same storage slice."""
    if left is right:
        return True

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False

    if left.device != right.device:
        return False

    if left.device.type == "meta" or right.device.type == "meta":
        return False

    if left.numel() == 0 or right.numel() == 0:
        return False

    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
    )


def has_tied_input_output_embeddings(model: torch.nn.Module) -> bool:
    """Return True if the input embedding and LM head weights are tied."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)

    if not callable(get_input_embeddings) or not callable(get_output_embeddings):
        return False

    input_embeddings = get_input_embeddings()
    output_embeddings = get_output_embeddings()

    if input_embeddings is None or output_embeddings is None:
        return False

    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)

    if input_weight is None or output_weight is None:
        return False

    return _weights_share_storage(input_weight, output_weight)


def _resolve_lm_head_weight(stage_cfg: Mapping[str, Any]) -> Any:
    """Resolve the LM head weight spec using the embedding spec as its default."""
    return stage_cfg.get("lm_head_weight", stage_cfg.get("embedding_weight"))


def _resolve_embedding_weight(stage_cfg: Mapping[str, Any]) -> Any:
    """Resolve the optional input embedding weight spec."""
    return stage_cfg.get("embedding_weight")


def validate_tied_embedding_weight_specs(
    model: torch.nn.Module,
    embedding_weight: Any,
    lm_head_weight: Any,
) -> None:
    """
    Reject different embedding and LM head weight specs for tied weights.

    Args:
        model: Model whose input embedding and output projection are inspected.
        embedding_weight: QuantSpec-like value requested for input embeddings.
        lm_head_weight: QuantSpec-like value requested for lm_head weights.

    Raises:
        ValueError: If the model ties input embedding and LM head weights while
            their requested quantization specs differ.
    """
    if embedding_weight is None or lm_head_weight is None:
        return

    if quant_specs_equivalent(embedding_weight, lm_head_weight):
        return

    if not has_tied_input_output_embeddings(model):
        return

    raise ValueError(
        "Cannot use different quantization specs for tied input embedding and "
        "lm_head weights. Set both options to the same value or use a model with untied "
        "input/output embeddings."
    )


class LlamaAdapter(ModelAdapter):
    family = "llama"

    def _ptq_decode_calibration_steps(self, cfg: Mapping[str, Any]) -> int:
        calib = cfg.get("calibration", {})
        if not isinstance(calib, Mapping):
            calib = {}
        decode_steps = int(calib.get("decode_steps", 0))

        for stage in cfg.get("pipeline", []):
            if not isinstance(stage, Mapping):
                continue
            if stage.get("name") == "ptq" and stage.get("enabled", True):
                return int(stage.get("decode_calibration_steps", decode_steps))

        return decode_steps

    def load_model(self, ctx: RecipeContext) -> RecipeContext:
        cfg = ctx.cfg
        model_cfg = cfg.get("model", {})
        runtime_cfg = cfg.get("runtime", {})

        ctx.device = torch.device(
            runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        ctx.dtype = torch_dtype_from_name(runtime_cfg.get("dtype", "float32"))

        name = model_cfg["name_or_path"]
        trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
        hf_token = model_cfg.get("hf_token")
        cache_dir = model_cfg.get("cache_dir")
        device_map = runtime_cfg.get("device_map")
        if device_map is None:
            device_map = "balanced" if ctx.device.type != "cpu" else "cpu"

        ctx.tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            cache_dir=cache_dir,
            legacy=False,
        )
        if ctx.tokenizer.pad_token_id is None:
            ctx.tokenizer.pad_token = ctx.tokenizer.eos_token

        ctx.model = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=ctx.dtype,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            cache_dir=cache_dir,
            device_map=device_map,
        ).eval()

        calib_seq_len = get_by_path(cfg, "calibration.seq_len")
        if calib_seq_len is not None and hasattr(
            ctx.model.config, "max_position_embeddings"
        ):
            ctx.model.config.max_position_embeddings = min(
                int(ctx.model.config.max_position_embeddings),
                int(calib_seq_len),
            )
        return ctx

    def build_calibration_inputs(self, ctx: RecipeContext) -> list[torch.Tensor]:
        cfg = ctx.cfg
        calib = cfg.get("calibration", {})
        runtime = cfg.get("runtime", {})
        seq_len = int(calib.get("seq_len") or ctx.model.config.max_position_embeddings)
        decode_steps = self._ptq_decode_calibration_steps(cfg)
        seq_len = seq_len - decode_steps
        if seq_len <= 0:
            raise ValueError(
                "calibration.seq_len must be larger than decode calibration steps"
            )

        return build_wikitext_calibration_inputs(
            tokenizer=ctx.tokenizer,
            cache_dir=ctx.cfg.get("model", {}).get("cache_dir"),
            n_samples=int(calib.get("n_samples", 128)),
            seq_len=seq_len,
            seed=int(runtime.get("seed", 42)),
            device=ctx.device,
            dataset_name=calib.get("dataset", "Salesforce/wikitext"),
            dataset_config=calib.get("dataset_config", "wikitext-2-raw-v1"),
            split=calib.get("split", "train"),
        )

    def forward_calibration(
        self,
        ctx: RecipeContext,
        model: torch.nn.Module,
        calibration_inputs: Sequence[Any],
        *,
        desc: str,
    ) -> None:
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))
        iterator = tqdm.tqdm(calibration_inputs, desc=desc, disable=not show_progress)
        model.eval()
        with torch.no_grad():
            for inp in iterator:
                model(inp.to(ctx.device))

    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        decode_steps = int(
            stage_cfg.get(
                "decode_calibration_steps",
                ctx.cfg.get("calibration", {}).get("decode_steps", 0),
            )
        )
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))
        iterator = tqdm.tqdm(
            ctx.calibration_inputs, desc="PTQ calibration", disable=not show_progress
        )
        prepared_model.eval()

        with torch.no_grad():
            for inp in iterator:
                inp = inp.to(ctx.device)
                if decode_steps <= 0:
                    prepared_model(inp)
                    continue

                outputs = prepared_model(
                    input_ids=inp, use_cache=True, return_dict=True
                )
                past_key_values = outputs.past_key_values
                next_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

                for _ in range(decode_steps):
                    outputs = prepared_model(
                        input_ids=next_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_input_ids = outputs.logits[:, -1, :].argmax(
                        dim=-1, keepdim=True
                    )

    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        num_hidden_layers = len(ctx.model.model.layers)
        profile = stage_cfg.get(
            "profile", ctx.cfg.get("model_args", {}).get("profile", "npu_export")
        )
        spin_rotation_weight = (
            stage_cfg.get("spin_rotation_weight")
            if _is_stage_enabled(ctx.cfg, "spinquant")
            else None
        )
        embedding_weight = _resolve_embedding_weight(stage_cfg)
        lm_head_weight = _resolve_lm_head_weight(stage_cfg)
        validate_tied_embedding_weight_specs(
            ctx.model,
            embedding_weight,
            lm_head_weight,
        )

        return build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=num_hidden_layers,
            activation=quant_spec_from_config(stage_cfg.get("activation", "int16")),
            linear_weight=quant_spec_from_config(stage_cfg.get("linear_weight")),
            embedding_weight=quant_spec_from_config(embedding_weight),
            lm_head_weight=quant_spec_from_config(lm_head_weight),
            spin_rotation_weight=quant_spec_from_config(spin_rotation_weight),
            norm=quant_spec_from_config(stage_cfg.get("norm")),
            norm_weight=quant_spec_from_config(stage_cfg.get("norm_weight")),
            strict_wrap=bool(stage_cfg.get("strict_wrap", True)),
            profile=profile,
        )

    def evaluate(self, ctx: RecipeContext) -> None:
        eval_cfg = ctx.cfg.get("evaluation", {})
        if not eval_cfg.get("enabled", False):
            return

        max_seq_len = int(
            eval_cfg.get("max_seq_len")
            or ctx.cfg.get("calibration", {}).get("seq_len")
            or ctx.model.config.max_position_embeddings
        )

        ppl_cfg = eval_cfg.get("perplexity")
        if ppl_cfg:
            ppl = evaluate_perplexity(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                device=str(ctx.device),
                cache_dir=ctx.cfg.get("model", {}).get("cache_dir"),
                max_seq_len=max_seq_len,
                dataset_name=ppl_cfg.get("dataset", "Salesforce/wikitext"),
                dataset_config=ppl_cfg.get("dataset_config", "wikitext-2-raw-v1"),
                split=ppl_cfg.get("split", "test"),
            )
            print("\n┌── Perplexity ─────────────────────────────")
            print(f"│ {ppl:8.2f}")
            print("└───────────────────────────────────────────")

        tasks = eval_cfg.get("lm_eval_tasks")
        if tasks:
            print("\n=== lm-eval ===")
            evaluate_lm_tasks(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                tasks=tasks,
                max_seq_len=max_seq_len,
            )

    def export(self, ctx: RecipeContext) -> None:
        export_cfg = ctx.cfg.get("export", {})
        if not export_cfg.get("enabled", False):
            return

        output_dir = Path(export_cfg.get("output_dir", "./out/llama"))
        artifacts = set(export_cfg.get("artifacts", []))

        if "ptq_checkpoint" in artifacts or "checkpoint" in artifacts:
            save_checkpoint(ctx.model, output_dir)

        if not artifacts.intersection({"circle_full", "circle_per_layer"}):
            return

        profile = _find_ptq_profile(ctx.cfg)
        if profile != "npu_export":
            raise ValueError(
                "Circle export for LLaMA is restricted to profile='npu_export'. "
                f"Current profile={profile!r}."
            )

        max_seq_len = int(
            export_cfg.get("max_seq_len")
            or ctx.cfg.get("evaluation", {}).get("max_seq_len")
            or ctx.cfg.get("calibration", {}).get("seq_len", 2048)
        )

        if "circle_full" in artifacts:
            if not ctx.calibration_inputs:
                raise RuntimeError(
                    "Circle export requires at least one calibration input."
                )
            export_full_circle(
                model=ctx.model,
                example_input=ctx.calibration_inputs[0],
                output_dir=output_dir,
                name="model.q.circle",
                strict=bool(export_cfg.get("strict", False)),
            )

        if "circle_per_layer" in artifacts:
            export_llama_per_layer(
                q_model=ctx.model,
                max_seq_len=max_seq_len,
                output_dir=output_dir,
                prefill_decode=bool(export_cfg.get("prefill_decode", False)),
            )


def _find_ptq_profile(cfg: dict[str, Any]) -> str:
    for stage in cfg.get("pipeline", []):
        if stage.get("name") == "ptq":
            return str(
                stage.get(
                    "profile", cfg.get("model_args", {}).get("profile", "npu_export")
                )
            )
    return str(cfg.get("model_args", {}).get("profile", "npu_export"))


def _is_stage_enabled(cfg: dict[str, Any], stage_name: str) -> bool:
    for stage in cfg.get("pipeline", []):
        if stage.get("name") == stage_name:
            return bool(stage.get("enabled", True))
    return False

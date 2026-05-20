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
    qscheme_from_name,
    torch_dtype_from_name,
    wrapq_dtype_from_name,
)


class LlamaAdapter(ModelAdapter):
    family = "llama"

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
        decode_steps = int(calib.get("decode_steps", 0))
        seq_len = seq_len - decode_steps
        if seq_len <= 0:
            raise ValueError(
                "calibration.seq_len must be larger than calibration.decode_steps"
            )

        return build_wikitext_calibration_inputs(
            tokenizer=ctx.tokenizer,
            cache_dir=ctx.cfg.get("model", {}).get("cache_dir"),
            n_samples=int(calib.get("n_samples", 128)),
            seq_len=seq_len,
            seed=int(runtime.get("seed", 42)),
            device=ctx.device,
            dataset_name=calib.get("dataset", "wikitext"),
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
        activation_dtype = wrapq_dtype_from_name(
            stage_cfg.get("activation_dtype", "int16")
        )
        default_qscheme = qscheme_from_name(
            stage_cfg.get("default_qscheme", "per_tensor_symm")
        )
        profile = stage_cfg.get(
            "profile", ctx.cfg.get("model_args", {}).get("profile", "npu_export")
        )
        spin_rotation_weight_bits = (
            stage_cfg.get("spin_rotation_weight_bits")
            if _is_stage_enabled(ctx.cfg, "spinquant")
            else None
        )

        return build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=num_hidden_layers,
            activation_dtype=activation_dtype,
            default_qscheme=default_qscheme,
            linear_weight_bits=stage_cfg.get("linear_weight_bits"),
            embedding_weight_bits=stage_cfg.get("embedding_weight_bits"),
            lm_head_weight_bits=stage_cfg.get("lm_head_weight_bits"),
            spin_rotation_weight_bits=spin_rotation_weight_bits,
            norm_dtype=wrapq_dtype_from_name(stage_cfg["norm_dtype"])
            if stage_cfg.get("norm_dtype")
            else None,
            norm_weight_dtype=wrapq_dtype_from_name(stage_cfg["norm_weight_dtype"])
            if stage_cfg.get("norm_weight_dtype")
            else None,
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
                dataset_name=ppl_cfg.get("dataset", "wikitext"),
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

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

"""Recipe adapter for the Gemma4 assistant (MTP draft) family."""

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import tqdm
from transformers import AutoTokenizer

from tico.quantization.config.gemma4_assistant_builders import (
    build_gemma4_assistant_ptq_config,
)
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.llm import build_wikitext_calibration_inputs
from tico.quantization.recipes.export.checkpoint import save_checkpoint
from tico.quantization.recipes.export.gemma4_assistant import (
    export_gemma4_assistant_core_circle,
    export_gemma4_assistant_sparse_head,
    resolve_assistant_quant_wrapper,
    write_gemma4_assistant_manifest,
)
from tico.quantization.recipes.utils import (
    quant_spec_from_config,
    torch_dtype_from_name,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
    Gemma4AssistantStaticShapeConfig,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    extract_assistant_text_config,
    Gemma4AssistantGenerationAdapter,
    validate_gemma4_assistant_architecture,
)


TARGET_MODEL_ENV_VAR = "GEMMA4_TARGET_PATH"
ASSISTANT_MODEL_ENV_VAR = "GEMMA4_ASSISTANT_PATH"


def _load_causal_lm(name: str, **kwargs: Any) -> torch.nn.Module:
    """Load a HF causal LM, falling back to the image-text auto class."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except (ValueError, OSError):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(name, **kwargs)


class _AssistantCallRecorder:
    """Count assistant draft calls and record drafted top-1 tokens."""

    def __init__(self, assistant_module: torch.nn.Module):
        self.calls = 0
        self.drafted_tokens: list[int] = []
        self._handle = assistant_module.register_forward_hook(self._on_forward)

    def _on_forward(self, module, args, output):
        self.calls += 1
        logits = getattr(output, "logits", None)
        if isinstance(logits, torch.Tensor):
            self.drafted_tokens.append(int(logits[:, -1].argmax(dim=-1).item()))

    def remove(self) -> None:
        self._handle.remove()


class _ForwardCounter:
    """Count forward invocations of one module."""

    def __init__(self, module: torch.nn.Module):
        self.calls = 0
        self._handle = module.register_forward_pre_hook(self._on_forward)

    def _on_forward(self, module, args):
        self.calls += 1

    def remove(self) -> None:
        self._handle.remove()


class Gemma4AssistantAdapter(ModelAdapter):
    """Adapter that quantizes the Gemma4 assistant with real assisted decoding.

    Calibration runs genuine Hugging Face assisted generation: the FP target
    drafts with the *prepared* assistant, so the assistant observers see the
    exact ``inputs_embeds`` / ``shared_kv_states`` / mask / position
    distribution of the MTP runtime, streamed sample by sample.
    """

    family = "gemma4_assistant"
    evaluation_targets = frozenset({"assisted"})

    # --- Model loading -------------------------------------------------------

    @staticmethod
    def _resolve_assistant_path(cfg: Mapping[str, Any]) -> str:
        env_path = os.environ.get(ASSISTANT_MODEL_ENV_VAR)
        if env_path:
            return env_path
        return cfg["model"]["name_or_path"]

    @staticmethod
    def _resolve_target_path(cfg: Mapping[str, Any]) -> str:
        env_path = os.environ.get(TARGET_MODEL_ENV_VAR)
        if env_path:
            return env_path
        target_cfg = cfg.get("target_model", {})
        if isinstance(target_cfg, Mapping) and target_cfg.get("name_or_path"):
            return str(target_cfg["name_or_path"])
        raise ValueError(
            "Gemma4 assistant calibration/evaluation requires the target "
            "model. Set target_model.name_or_path in the recipe config or "
            f"the {TARGET_MODEL_ENV_VAR} environment variable."
        )

    def load_model(self, ctx: RecipeContext) -> RecipeContext:
        """Load the assistant model and the target tokenizer."""
        cfg = ctx.cfg
        model_cfg = cfg.get("model", {})
        runtime_cfg = cfg.get("runtime", {})

        ctx.device = torch.device(
            runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        ctx.dtype = torch_dtype_from_name(runtime_cfg.get("dtype", "float32"))

        name = self._resolve_assistant_path(cfg)
        trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        hf_token = model_cfg.get("hf_token")
        cache_dir = model_cfg.get("cache_dir")

        ctx.model = _load_causal_lm(
            name,
            dtype=ctx.dtype,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            cache_dir=cache_dir,
        ).to(ctx.device)
        ctx.model.eval()
        validate_gemma4_assistant_architecture(ctx.model)

        try:
            tokenizer_path = self._resolve_target_path(cfg)
        except ValueError:
            tokenizer_path = name
        ctx.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            cache_dir=cache_dir,
        )
        return ctx

    def _load_target_model(self, ctx: RecipeContext) -> torch.nn.Module:
        """Load (and cache) the FP target model used for assisted decoding."""
        cached = ctx.artifacts.get("gemma4_assistant_target_model")
        if cached is not None:
            return cached

        cfg = ctx.cfg
        target_cfg = cfg.get("target_model", {})
        if not isinstance(target_cfg, Mapping):
            raise TypeError("target_model must be a mapping.")
        model_cfg = cfg.get("model", {})

        target = _load_causal_lm(
            self._resolve_target_path(cfg),
            dtype=torch_dtype_from_name(
                target_cfg.get("dtype", cfg.get("runtime", {}).get("dtype"))
            ),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            token=target_cfg.get("hf_token", model_cfg.get("hf_token")),
            cache_dir=target_cfg.get("cache_dir", model_cfg.get("cache_dir")),
        ).to(ctx.device)
        target.eval()

        # Validate target/assistant compatibility. The assistant consumes the
        # target hidden states through ``backbone_hidden_size`` (top-level
        # assistant config); its own text-config ``hidden_size`` is the much
        # smaller draft width and must not be compared against the target.
        assistant = ctx.require_model()
        assistant_cfg = extract_assistant_text_config(assistant.config)
        target_cfg = extract_assistant_text_config(target.config)
        backbone_hidden_size = int(assistant.config.backbone_hidden_size)

        if backbone_hidden_size != int(target_cfg.hidden_size):
            raise ValueError(
                f"Assistant backbone_hidden_size {backbone_hidden_size} does not match "
                f"target hidden_size {target_cfg.hidden_size}."
            )
        if assistant_cfg.vocab_size != target_cfg.vocab_size:
            raise ValueError(
                f"Assistant vocab_size {assistant_cfg.vocab_size} does not match "
                f"target vocab_size {target_cfg.vocab_size}."
            )

        ctx.artifacts["gemma4_assistant_target_model"] = target
        return target

    def _release_target_model(self, ctx: RecipeContext) -> None:
        """Free the cached target model after calibration/evaluation."""
        target = ctx.artifacts.pop("gemma4_assistant_target_model", None)
        if target is not None:
            del target
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Calibration ---------------------------------------------------------

    def build_calibration_inputs(
        self, ctx: RecipeContext
    ) -> list[dict[str, torch.Tensor]]:
        """Build prompts with varied padding lengths for assisted decoding.

        Returns prompt buckets with different lengths to cover:
        - short (32-128), medium (256-512), long (1024-1536), near-capacity (1800-2048)
        - left-padded variants to test attention mask handling
        """
        calib = ctx.cfg.get("calibration", {})
        runtime = ctx.cfg.get("runtime", {})
        base_prompts = build_wikitext_calibration_inputs(
            tokenizer=ctx.tokenizer,
            cache_dir=ctx.cfg.get("model", {}).get("cache_dir"),
            n_samples=int(calib.get("n_prompts", calib.get("n_samples", 64))),
            seq_len=int(calib.get("max_prompt_length", 512)),
            seed=int(runtime.get("seed", 42)),
            device="cpu",
            dataset_name=calib.get("dataset", "wikitext2"),
            dataset_config=calib.get("dataset_config", "wikitext-2-raw-v1"),
            split=calib.get("split", "train"),
            allow_benchmark_overlap=bool(calib.get("allow_benchmark_overlap", False)),
            allow_unregistered_dataset=bool(
                calib.get("allow_unregistered_dataset", False)
            ),
        )

        # Distribute prompts across length buckets for padding diversity.
        prompt_buckets = {
            "short": [p for p in base_prompts[: len(base_prompts) // 4]],
            "medium": [
                p for p in base_prompts[len(base_prompts) // 4 : len(base_prompts) // 2]
            ],
            "long": [
                p
                for p in base_prompts[
                    len(base_prompts) // 2 : 3 * len(base_prompts) // 4
                ]
            ],
            "near_capacity": [p for p in base_prompts[3 * len(base_prompts) // 4 :]],
        }

        result = []
        for bucket_name, prompts in prompt_buckets.items():
            for prompt in prompts:
                valid_len = int(prompt.shape[1])
                result.append(
                    {
                        "input_ids": prompt,
                        "attention_mask": torch.ones_like(prompt),
                    }
                )
                # Add left-padded variant for short/medium buckets.
                if bucket_name in ("short", "medium") and valid_len < 256:
                    pad_len = min(256, valid_len + 64)
                    padded_input = torch.cat(
                        [
                            torch.full(
                                (1, pad_len - valid_len),
                                ctx.tokenizer.pad_token_id or 0,
                                dtype=torch.long,
                            ),
                            prompt,
                        ],
                        dim=1,
                    )
                    mask = torch.cat(
                        [
                            torch.zeros(1, pad_len - valid_len, dtype=torch.long),
                            torch.ones(1, valid_len, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    result.append({"input_ids": padded_input, "attention_mask": mask})

        return result

    def _run_assisted_generation(
        self,
        ctx: RecipeContext,
        *,
        target: torch.nn.Module,
        assistant: torch.nn.Module,
        prompts: Sequence[dict[str, torch.Tensor]],
        max_new_tokens: int,
        num_assistant_tokens: int,
        max_assistant_calls: int | None,
        desc: str,
    ) -> _AssistantCallRecorder:
        """Stream prompts through real HF assisted generation."""
        generation_adapter = Gemma4AssistantGenerationAdapter(assistant)
        generation_config = generation_adapter.generation_config
        generation_config.num_assistant_tokens = int(num_assistant_tokens)
        generation_config.num_assistant_tokens_schedule = "constant"

        recorder = _AssistantCallRecorder(generation_adapter.assistant)
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))
        pad_token_id = ctx.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = ctx.tokenizer.eos_token_id

        try:
            iterator = tqdm.tqdm(prompts, desc=desc, disable=not show_progress)
            with torch.no_grad():
                for prompt_dict in iterator:
                    input_ids = prompt_dict["input_ids"].to(ctx.device)
                    attention_mask = prompt_dict["attention_mask"].to(ctx.device)
                    target.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        assistant_model=generation_adapter,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                    if (
                        max_assistant_calls is not None
                        and recorder.calls >= max_assistant_calls
                    ):
                        break
        finally:
            recorder.remove()
        return recorder

    def forward_calibration(
        self,
        ctx: RecipeContext,
        model: torch.nn.Module,
        calibration_inputs: Sequence[Any],
        *,
        desc: str,
    ) -> None:
        """Calibrate the prepared assistant inside real assisted generation."""
        calib = ctx.cfg.get("calibration", {})
        target = self._load_target_model(ctx)
        recorder = self._run_assisted_generation(
            ctx,
            target=target,
            assistant=model,
            prompts=calibration_inputs,
            max_new_tokens=int(calib.get("max_new_tokens", 16)),
            num_assistant_tokens=int(calib.get("num_assistant_tokens", 6)),
            max_assistant_calls=int(calib.get("max_assistant_calls", 384)),
            desc=desc,
        )
        if recorder.calls == 0:
            raise RuntimeError(
                "Assisted-generation calibration produced no assistant draft "
                "calls. Check target_model configuration, prompt lengths, and "
                "max_new_tokens."
            )
        print(f"[Info] Calibrated on {recorder.calls} assistant draft call(s).")

    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        """Calibrate a prepared PTQ assistant with real assisted decoding."""
        try:
            self.forward_calibration(
                ctx,
                prepared_model,
                ctx.calibration_inputs,
                desc="Gemma4 assistant PTQ calibration",
            )
        finally:
            if not ctx.cfg.get("evaluation", {}).get("enabled", False):
                self._release_target_model(ctx)

    # --- PTQ config ----------------------------------------------------------

    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        """Build the assistant PTQConfig from recipe stage settings."""
        text_config = ctx.model.config.get_text_config()
        model_args = dict(ctx.cfg.get("model_args", {}))

        return build_gemma4_assistant_ptq_config(
            num_hidden_layers=int(text_config.num_hidden_layers),
            model_args=model_args,
            activation=quant_spec_from_config(stage_cfg.get("activation", "int16")),
            weight=quant_spec_from_config(stage_cfg.get("weight")),
            linear_weight=quant_spec_from_config(stage_cfg.get("linear_weight")),
            projection_weight=quant_spec_from_config(
                stage_cfg.get("projection_weight")
            ),
            centroid_weight=quant_spec_from_config(stage_cfg.get("centroid_weight")),
            lm_head_weight=quant_spec_from_config(stage_cfg.get("lm_head_weight")),
            norm_weight=quant_spec_from_config(stage_cfg.get("norm_weight")),
            strict_wrap=bool(stage_cfg.get("strict_wrap", True)),
        )

    # --- Evaluation ----------------------------------------------------------

    def evaluate(self, ctx: RecipeContext) -> None:
        """Compare target-only and assisted decoding on a small prompt set.

        Assistant quality is measured by candidate acceptance, not perplexity:
        greedy assisted decoding must reproduce the target-only sequence, and
        fewer target verification forwards mean better drafts.
        """
        eval_cfg = ctx.cfg.get("evaluation", {})
        if not eval_cfg.get("enabled", False):
            return
        self.validate_evaluation_config(ctx.cfg)

        runtime = ctx.cfg.get("runtime", {})
        n_prompts = int(eval_cfg.get("n_prompts", 4))
        max_new_tokens = int(eval_cfg.get("max_new_tokens", 32))
        num_assistant_tokens = int(eval_cfg.get("num_assistant_tokens", 6))

        prompts = build_wikitext_calibration_inputs(
            tokenizer=ctx.tokenizer,
            cache_dir=ctx.cfg.get("model", {}).get("cache_dir"),
            n_samples=n_prompts,
            seq_len=int(eval_cfg.get("max_prompt_length", 256)),
            seed=int(runtime.get("seed", 42)) + 1,
            device="cpu",
            dataset_name=eval_cfg.get("dataset", "wikitext2"),
            dataset_config=eval_cfg.get("dataset_config", "wikitext-2-raw-v1"),
            split=eval_cfg.get("split", "train"),
            allow_benchmark_overlap=True,
        )

        target = self._load_target_model(ctx)
        pad_token_id = ctx.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = ctx.tokenizer.eos_token_id
        generation_adapter = Gemma4AssistantGenerationAdapter(ctx.require_model())
        generation_config = generation_adapter.generation_config
        generation_config.num_assistant_tokens = num_assistant_tokens
        generation_config.num_assistant_tokens_schedule = "constant"

        greedy_matches = 0
        total_new_tokens = 0
        total_target_forwards = 0
        total_assistant_calls = 0

        with torch.no_grad():
            for prompt in prompts:
                input_ids = prompt.to(ctx.device)
                attention_mask = torch.ones_like(input_ids)
                baseline = target.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
                target_counter = _ForwardCounter(target)
                recorder = _AssistantCallRecorder(generation_adapter.assistant)
                try:
                    assisted = target.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        assistant_model=generation_adapter,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                    )
                finally:
                    target_counter.remove()
                    recorder.remove()

                greedy_matches += int(torch.equal(baseline, assisted))
                total_new_tokens += int(assisted.shape[1] - input_ids.shape[1])
                total_target_forwards += target_counter.calls
                total_assistant_calls += recorder.calls

        print("\n=== Gemma4 assistant assisted-decoding evaluation ===")
        print(f"prompts                        : {len(prompts)}")
        print(f"greedy sequence match          : {greedy_matches}/{len(prompts)}")
        print(f"generated tokens               : {total_new_tokens}")
        print(f"target verification forwards   : {total_target_forwards}")
        print(f"assistant draft forwards       : {total_assistant_calls}")
        if total_target_forwards:
            print(
                "generated tokens / target forward : "
                f"{total_new_tokens / total_target_forwards:.3f}"
            )
        self._release_target_model(ctx)

    # --- Export ---------------------------------------------------------------

    @staticmethod
    def _quantization_profile(cfg: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize the effective PTQ dtypes for the manifest."""
        for stage_cfg in cfg.get("pipeline", []):
            if isinstance(stage_cfg, Mapping) and stage_cfg.get("name") == "ptq":
                keys = (
                    "activation",
                    "weight",
                    "linear_weight",
                    "projection_weight",
                    "centroid_weight",
                    "lm_head_weight",
                    "norm_weight",
                )
                return {
                    key: stage_cfg.get(key)
                    for key in keys
                    if stage_cfg.get(key) is not None
                }
        # Export-only: infer profile from effective config or defaults.
        # For now, return the documented safe_w8a16 default.
        return {
            "activation": "int16",
            "weight": "uint8",
            "linear_weight": "uint8",
            "projection_weight": "uint8",
            "centroid_weight": "uint8",
            "lm_head_weight": "uint8",
            "norm_weight": "int16",
        }

    def export(self, ctx: RecipeContext) -> None:
        """Export the configured Gemma4 assistant artifacts."""
        export_cfg = ctx.cfg.get("export", {})
        if not export_cfg.get("enabled", False):
            return

        output_dir = Path(export_cfg.get("output_dir", "./out/gemma4_assistant"))
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = set(export_cfg.get("artifacts", []))

        if "ptq_checkpoint" in artifacts or "checkpoint" in artifacts:
            save_checkpoint(ctx.require_model(), output_dir)

        core_artifacts = {
            "assistant_core_circle",
            "assistant_sparse_head",
            "assistant_manifest",
        }
        if not artifacts & core_artifacts:
            return

        model = ctx.require_model().eval().cpu()
        assistant = resolve_assistant_quant_wrapper(model)
        shape = Gemma4AssistantStaticShapeConfig.from_model_args(
            ctx.cfg.get("model_args", {})
        )
        shape.validate(assistant.text_config)

        circle_path = None
        if "assistant_core_circle" in artifacts:
            circle_path = export_gemma4_assistant_core_circle(
                assistant,
                shape,
                output_dir,
                strict=bool(export_cfg.get("strict", False)),
            )
        if "assistant_sparse_head" in artifacts:
            export_gemma4_assistant_sparse_head(assistant, output_dir)
        if "assistant_manifest" in artifacts:
            if circle_path is None:
                raise RuntimeError(
                    "assistant_manifest requires assistant_core_circle to be exported first "
                    "(for actual I/O type extraction from the Circle graph)."
                )
            write_gemma4_assistant_manifest(
                assistant,
                shape,
                output_dir,
                source_model=self._resolve_assistant_path(ctx.cfg),
                quantization_profile=self._quantization_profile(ctx.cfg),
                circle_path=circle_path,
            )

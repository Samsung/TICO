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

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import tqdm
from transformers import AutoProcessor

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.config import get_by_path
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.vlm import build_vlm_calibration_inputs
from tico.quantization.recipes.evaluation.hellaswag import evaluate_and_print_hellaswag
from tico.quantization.recipes.evaluation.llava_bench_judge import (
    evaluate_and_print_llava_bench_judge,
)
from tico.quantization.recipes.evaluation.mmlu import evaluate_and_print_mmlu
from tico.quantization.recipes.evaluation.mmmu import evaluate_and_print_mmmu
from tico.quantization.recipes.evaluation.selection import (
    get_mapping_evaluation_config,
    should_run_evaluation,
    should_run_mapping_evaluation,
)
from tico.quantization.recipes.evaluation.video_mme import evaluate_and_print_video_mme
from tico.quantization.recipes.evaluation.vlm import (
    evaluate_coco,
    evaluate_llava_bench,
    evaluate_vlm_text_ppl,
    evaluate_vlm_text_ppl_chat_prefix,
    evaluate_vqa_tasks,
    print_coco_score_results,
    print_vqa_results,
)
from tico.quantization.recipes.export.checkpoint import save_checkpoint
from tico.quantization.recipes.export.gemma4 import export_gemma4_per_layer
from tico.quantization.recipes.utils import (
    move_to_device,
    quant_spec_from_config,
    torch_dtype_from_name,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    build_gemma4_static_vision_profile,
    canonicalize_gemma4_static_vision_model_args,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe


class Gemma4Adapter(ModelAdapter):
    """Model adapter for Gemma4 E2B PTQ and static runtime experiments."""

    family = "gemma4"
    evaluation_targets = frozenset(
        {
            "vqa",
            "coco",
            "llava_bench",
            "videomme",
            "mmlu",
            "hellaswag",
            "mmmu",
            "ppl",
        }
    )
    evaluation_target_requirements = {"vqa": "vlm_tasks"}

    def load_model(self, ctx: RecipeContext) -> RecipeContext:
        """Load the Gemma4 E2B model and processor."""
        cfg = ctx.cfg
        model_cfg = cfg.get("model", {})
        runtime_cfg = cfg.get("runtime", {})

        ctx.device = torch.device(
            runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        ctx.dtype = torch_dtype_from_name(runtime_cfg.get("dtype", "float32"))

        name = model_cfg["name_or_path"]
        trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        hf_token = model_cfg.get("hf_token")
        cache_dir = model_cfg.get("cache_dir")
        device_map = runtime_cfg.get("device_map")
        if device_map is None:
            if ctx.device.type == "cpu":
                device_map = "cpu"
            else:
                # Use specific device index instead of "auto" to avoid
                # multi-GPU model split during PTQ calibration.
                device_map = str(ctx.device)

        ctx.processor = AutoProcessor.from_pretrained(
            name,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            cache_dir=cache_dir,
        )

        try:
            from transformers import AutoModelForImageTextToText

            ctx.model = AutoModelForImageTextToText.from_pretrained(
                name,
                dtype=ctx.dtype,
                trust_remote_code=trust_remote_code,
                token=hf_token,
                cache_dir=cache_dir,
                device_map=device_map,
            )
        except Exception:
            from transformers import AutoModelForVision2Seq

            ctx.model = AutoModelForVision2Seq.from_pretrained(
                name,
                dtype=ctx.dtype,
                trust_remote_code=trust_remote_code,
                token=hf_token,
                cache_dir=cache_dir,
                device_map=device_map,
            )

        ctx.model.eval()
        self._disable_cache(ctx.model)
        assert_gemma4_e2b_no_moe(ctx.model)

        calib_seq_len = get_by_path(cfg, "calibration.seq_len")
        if calib_seq_len is not None and hasattr(ctx.model.config, "text_config"):
            ctx.model.config.text_config.max_position_embeddings = min(
                int(ctx.model.config.text_config.max_position_embeddings),
                int(calib_seq_len),
            )

        model_args = ctx.cfg.get("model_args", {})
        if isinstance(model_args, Mapping):
            vision_args = model_args.get("vision", {})
            if (
                isinstance(vision_args, Mapping)
                and vision_args.get("profile") is not None
            ):
                normalized_model_args = canonicalize_gemma4_static_vision_model_args(
                    model_args
                )
                max_seq_len = int(
                    calib_seq_len
                    or ctx.model.config.get_text_config().max_position_embeddings
                )
                profile = build_gemma4_static_vision_profile(
                    normalized_model_args,
                    vision_config=ctx.model.config.vision_config,
                    max_seq_len=max_seq_len,
                )
                profile.validate_processor(ctx.processor)
                ctx.cfg["model_args"] = normalized_model_args
                ctx.artifacts["gemma4_static_vision_profile"] = profile
        return ctx

    @staticmethod
    def _disable_cache(model: Any) -> None:
        """Disable HF dynamic cache paths during PTQ calibration."""
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False

    @staticmethod
    def _enable_cache(model: Any) -> None:
        """Re-enable HF dynamic cache for autoregressive generation."""
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = True

    @staticmethod
    def _static_calibration_image_size(
        cfg: Mapping[str, Any],
    ) -> tuple[int, int] | None:
        """Return the configured static image size for calibration samples."""
        model_args = cfg.get("model_args", {})
        if not isinstance(model_args, Mapping):
            return None

        vision_cfg = model_args.get("vision", {})
        if not isinstance(vision_cfg, Mapping):
            return None

        height = vision_cfg.get("image_height")
        width = vision_cfg.get("image_width")
        if height is None and width is None:
            return None
        if height is None or width is None:
            raise ValueError(
                "Both model_args.vision.image_height and "
                "model_args.vision.image_width must be set for static Gemma4 "
                "calibration resizing."
            )

        parsed_height = int(height)
        parsed_width = int(width)
        if parsed_height <= 0 or parsed_width <= 0:
            raise ValueError(
                "Gemma4 static calibration image dimensions must be positive: "
                f"image_height={parsed_height}, image_width={parsed_width}."
            )
        return parsed_height, parsed_width

    @staticmethod
    @contextmanager
    def _fixed_image_processor_size(
        processor: Any,
        image_size: tuple[int, int] | None,
    ) -> Iterator[None]:
        """Temporarily force processor image resizing for static calibration."""
        if image_size is None:
            yield
            return

        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            yield
            return

        height, width = image_size
        updates: dict[str, Any] = {
            "do_resize": True,
            "size": {"height": height, "width": width},
        }
        if hasattr(image_processor, "crop_size"):
            updates["crop_size"] = {"height": height, "width": width}

        originals: dict[str, tuple[bool, Any]] = {}
        for attr, value in updates.items():
            originals[attr] = (
                hasattr(image_processor, attr),
                getattr(image_processor, attr, None),
            )
            setattr(image_processor, attr, value)

        try:
            yield
        finally:
            for attr, (had_attr, original) in originals.items():
                if had_attr:
                    setattr(image_processor, attr, original)
                elif hasattr(image_processor, attr):
                    delattr(image_processor, attr)

    def build_calibration_inputs(self, ctx: RecipeContext) -> list[dict]:
        """Build VLM calibration inputs for fixed image-text PTQ."""
        calib = ctx.cfg.get("calibration", {})
        runtime = ctx.cfg.get("runtime", {})
        image_size = self._static_calibration_image_size(ctx.cfg)
        with self._fixed_image_processor_size(ctx.processor, image_size):
            return build_vlm_calibration_inputs(
                processor=ctx.processor,
                dataset=calib.get("dataset", "vqav2"),
                datasets=calib.get("datasets"),
                n_samples=int(calib.get("n_samples", 128)),
                split=calib.get("split"),
                max_seq_len=calib.get("seq_len"),
                seed=int(runtime.get("seed", 42)),
                allow_benchmark_overlap=bool(
                    calib.get("allow_benchmark_overlap", False)
                ),
                allow_unregistered_dataset=bool(
                    calib.get("allow_unregistered_dataset", False)
                ),
            )

    def forward_calibration(
        self,
        ctx: RecipeContext,
        model: torch.nn.Module,
        calibration_inputs: Sequence[Any],
        *,
        desc: str,
    ) -> None:
        """Run calibration samples through the prepared Gemma4 model."""
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))
        iterator = tqdm.tqdm(calibration_inputs, desc=desc, disable=not show_progress)
        model.eval()
        with torch.no_grad():
            for batch in iterator:
                model(**move_to_device(batch, ctx.device))

    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        """Calibrate a prepared PTQ model."""
        self.forward_calibration(
            ctx, prepared_model, ctx.calibration_inputs, desc="Gemma4 PTQ calibration"
        )

    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        """Build the Gemma4 E2B PTQConfig from recipe stage settings."""
        text_config = ctx.model.config.get_text_config()
        vision_config = ctx.model.config.vision_config
        model_args = dict(ctx.cfg.get("model_args", {}))

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(text_config.num_hidden_layers),
            num_vision_layers=int(vision_config.num_hidden_layers),
            model_args=model_args,
            activation=quant_spec_from_config(stage_cfg.get("activation", "int16")),
            weight=quant_spec_from_config(stage_cfg.get("weight")),
            linear_weight=quant_spec_from_config(stage_cfg.get("linear_weight")),
            embedding_weight=quant_spec_from_config(stage_cfg.get("embedding_weight")),
            lm_head_weight=quant_spec_from_config(stage_cfg.get("lm_head_weight")),
            vision_patch_embed_weight=quant_spec_from_config(
                stage_cfg.get("vision_patch_embed_weight")
            ),
            norm_weight=quant_spec_from_config(stage_cfg.get("norm_weight")),
            strict_wrap=bool(stage_cfg.get("strict_wrap", True)),
        )

    def apply_ptq(
        self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Prepare, calibrate, and convert Gemma4 E2B with PTQ."""
        qcfg = self.build_ptq_config(ctx, stage_cfg)
        prepared = prepare(
            ctx.require_model(), qcfg, inplace=bool(stage_cfg.get("inplace", True))
        )
        self.calibrate_prepared_model(ctx, prepared, stage_cfg)
        return convert(prepared, inplace=True)

    def evaluate(self, ctx: RecipeContext) -> None:
        """Evaluate Gemma4 E2B with the configured top-level targets."""
        eval_cfg = ctx.cfg.get("evaluation", {})
        if not eval_cfg.get("enabled", False):
            return

        self.validate_evaluation_config(ctx.cfg)

        # Re-enable KV cache for autoregressive generation during evaluation.
        # ``_disable_cache`` in ``load_model`` turns it off for calibration,
        # but ``generate()`` requires cache for acceptable speed.
        self._enable_cache(ctx.model)

        max_seq_len = eval_cfg.get("max_seq_len")
        n_samples = int(eval_cfg.get("n_samples", 50))
        raw_vqa_tasks = eval_cfg.get("vlm_tasks") or []
        verbose = bool(eval_cfg.get("verbose", False))
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))

        if should_run_evaluation(
            eval_cfg,
            "vqa",
            default_enabled=bool(raw_vqa_tasks),
        ):
            if isinstance(raw_vqa_tasks, str):
                vqa_tasks = [
                    task.strip() for task in raw_vqa_tasks.split(",") if task.strip()
                ]
            elif isinstance(raw_vqa_tasks, Sequence):
                vqa_tasks = []
                for task in raw_vqa_tasks:
                    if not isinstance(task, str):
                        raise TypeError(
                            "evaluation.vlm_tasks must contain only strings."
                        )
                    normalized_task = task.strip()
                    if normalized_task:
                        vqa_tasks.append(normalized_task)
            else:
                raise TypeError(
                    "evaluation.vlm_tasks must be a sequence or "
                    "comma-separated string."
                )

            if not vqa_tasks:
                raise ValueError(
                    "evaluation.vlm_tasks must be non-empty when vqa runs."
                )
            vqa_results = evaluate_vqa_tasks(
                model=ctx.model,
                processor=ctx.processor,
                tasks=vqa_tasks,
                device=str(ctx.device),
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                verbose=verbose,
                show_progress=show_progress,
            )
            print_vqa_results("VQA evaluation", vqa_results)

        if should_run_evaluation(
            eval_cfg,
            "coco",
            default_enabled=bool(eval_cfg.get("coco", False)),
        ):
            coco_results = evaluate_coco(
                model=ctx.model,
                processor=ctx.processor,
                device=str(ctx.device),
                dataset_name="coco",
                n_samples=n_samples,
                max_seq_len=max_seq_len,
            )
            print_coco_score_results("\n=== COCO Evaluation ===", coco_results)

        raw_llava_cfg = eval_cfg.get("llava_bench")
        llava_default_enabled = (
            bool(raw_llava_cfg.get("enabled", False))
            if isinstance(raw_llava_cfg, Mapping)
            else bool(raw_llava_cfg)
        )
        if should_run_evaluation(
            eval_cfg,
            "llava_bench",
            default_enabled=llava_default_enabled,
        ):
            if isinstance(raw_llava_cfg, Mapping):
                llava_cfg = raw_llava_cfg
                mode = str(llava_cfg.get("mode", "judge")).lower()
                if mode in {"judge", "llm_judge"}:
                    evaluate_and_print_llava_bench_judge(
                        model=ctx.model,
                        processor=ctx.processor,
                        device=str(ctx.device),
                        llava_cfg=llava_cfg,
                        model_cfg=ctx.cfg.get("model", {}),
                        runtime_cfg=ctx.cfg.get("runtime", {}),
                        default_n_samples=n_samples,
                        default_max_seq_len=max_seq_len,
                    )
                elif mode in {"legacy", "coco", "caption"}:
                    llava_results = evaluate_llava_bench(
                        model=ctx.model,
                        processor=ctx.processor,
                        device=str(ctx.device),
                        n_samples=int(llava_cfg.get("n_samples", n_samples)),
                        max_seq_len=llava_cfg.get(
                            "max_seq_len",
                            max_seq_len,
                        ),
                    )
                    print_coco_score_results(
                        "\n=== LLaVA Bench Legacy COCO-style Evaluation ===",
                        llava_results,
                    )
                else:
                    raise ValueError(
                        "evaluation.llava_bench.mode must be one of "
                        "{'judge', 'llm_judge', 'legacy', 'coco', 'caption'}, "
                        f"got {mode!r}."
                    )
            elif raw_llava_cfg is None or raw_llava_cfg is False:
                evaluate_and_print_llava_bench_judge(
                    model=ctx.model,
                    processor=ctx.processor,
                    device=str(ctx.device),
                    llava_cfg={},
                    model_cfg=ctx.cfg.get("model", {}),
                    runtime_cfg=ctx.cfg.get("runtime", {}),
                    default_n_samples=n_samples,
                    default_max_seq_len=max_seq_len,
                )
            elif raw_llava_cfg is True:
                print(
                    "[WARNING] evaluation.llava_bench=true uses the legacy "
                    "COCO-style CIDEr/BLEU path. Prefer the nested judge config: "
                    "evaluation.llava_bench.enabled=true, mode=judge."
                )
                llava_results = evaluate_llava_bench(
                    model=ctx.model,
                    processor=ctx.processor,
                    device=str(ctx.device),
                    n_samples=n_samples,
                    max_seq_len=max_seq_len,
                )
                print_coco_score_results(
                    "\n=== Llava Bench Evaluation ===",
                    llava_results,
                )
            else:
                raise TypeError(
                    "evaluation.llava_bench must be a mapping, boolean, or null."
                )

        if should_run_mapping_evaluation(eval_cfg, "videomme"):
            videomme = get_mapping_evaluation_config(eval_cfg, "videomme")
            video_n_samples = int(videomme.get("n_samples", -1))
            # Gemma4 uses at least 70 soft tokens per frame at this profile.
            max_num_frames = int(videomme.get("max_num_frames", 21))
            if max_num_frames <= 0:
                raise ValueError(
                    "evaluation.videomme.max_num_frames must be a positive integer."
                )

            evaluate_and_print_video_mme(
                model=ctx.model,
                processor=ctx.processor,
                device=str(ctx.device),
                batch_size=int(videomme.get("batch_size", 1)),
                max_new_tokens=int(videomme.get("max_new_tokens", 30)),
                n_samples=video_n_samples if video_n_samples > 0 else None,
                max_num_frames=max_num_frames,
                use_cache=videomme.get("use_cache", None),
                verbose=bool(videomme.get("verbose", verbose)),
            )

        if should_run_mapping_evaluation(eval_cfg, "mmlu"):
            mmlu = get_mapping_evaluation_config(eval_cfg, "mmlu")
            evaluate_and_print_mmlu(
                model=ctx.model,
                tokenizer=ctx.processor.tokenizer,
                subjects=mmlu.get("subjects") or ["mmlu"],
                device=str(ctx.device),
                n_shots=int(mmlu.get("n_shots", 5)),
                n_samples=int(mmlu.get("n_samples", -1)),
                batch_size=int(mmlu.get("batch_size", 1)),
                max_seq_len=int(
                    max_seq_len or ctx.cfg.get("calibration", {}).get("seq_len", 2048)
                ),
            )

        if should_run_mapping_evaluation(eval_cfg, "hellaswag"):
            hellaswag = get_mapping_evaluation_config(eval_cfg, "hellaswag")
            evaluate_and_print_hellaswag(
                model=ctx.model,
                tokenizer=ctx.processor.tokenizer,
                device=str(ctx.device),
                n_shots=int(hellaswag.get("n_shots", 10)),
                n_samples=int(hellaswag.get("n_samples", -1)),
                batch_size=int(hellaswag.get("batch_size", 1)),
                max_seq_len=int(
                    max_seq_len or ctx.cfg.get("calibration", {}).get("seq_len", 2048)
                ),
            )

        if should_run_mapping_evaluation(eval_cfg, "mmmu"):
            mmmu = get_mapping_evaluation_config(eval_cfg, "mmmu")
            subjects = mmmu.get("subjects")
            if subjects == ["mmmu"] or subjects == "mmmu":
                subjects = None
            evaluate_and_print_mmmu(
                model=ctx.model,
                processor=ctx.processor,
                dataset=mmmu.get("dataset") or "MMMU/MMMU",
                subjects=subjects,
                device=str(ctx.device),
                n_shots=int(mmmu.get("n_shots", 5)),
                n_samples=int(mmmu.get("n_samples", -1)),
                max_new_tokens=int(mmmu.get("max_new_tokens", 16)),
                max_seq_len=max_seq_len,
                temperature=float(mmmu.get("temperature", 0.0)),
                verbose=bool(mmmu.get("verbose", verbose)),
            )

        if should_run_mapping_evaluation(eval_cfg, "ppl"):
            ppl_cfg = get_mapping_evaluation_config(eval_cfg, "ppl")
            ppl_mode = str(ppl_cfg.get("mode", "raw")).lower()

            valid_ppl_modes = {"raw", "chat-prefix"}
            if ppl_mode not in valid_ppl_modes:
                raise ValueError(
                    f"Unsupported ppl mode: {ppl_mode!r}. "
                    f"Must be one of {sorted(valid_ppl_modes)}."
                )

            eval_fn = (
                evaluate_vlm_text_ppl_chat_prefix
                if ppl_mode == "chat-prefix"
                else evaluate_vlm_text_ppl
            )
            ppl_value = eval_fn(
                model=ctx.model,
                processor=ctx.processor,
                dataset_name=ppl_cfg.get("dataset", "wikitext2"),
                split=ppl_cfg.get("split", "test"),
                device=str(ctx.device),
                stride=int(ppl_cfg.get("stride", 512)),
                max_seq_len=int(
                    max_seq_len or ctx.cfg.get("calibration", {}).get("seq_len", 2048)
                ),
                show_progress=show_progress,
            )
            print(
                f"\nPPL[{ppl_mode}]"
                f"({ppl_cfg.get('dataset', 'wikitext2')}): {ppl_value:.2f}"
            )

    def export(self, ctx: RecipeContext) -> None:
        """Export configured Gemma4 E2B artifacts."""
        export_cfg = ctx.cfg.get("export", {})
        if not export_cfg.get("enabled", False):
            return

        output_dir = Path(export_cfg.get("output_dir", "./out/gemma4"))
        artifacts = set(export_cfg.get("artifacts", []))
        if "ptq_checkpoint" in artifacts or "checkpoint" in artifacts:
            save_checkpoint(ctx.require_model(), output_dir)

        if "circle_per_layer" in artifacts:
            calibration_cfg = ctx.cfg.get("calibration", {})
            max_seq_len = int(
                export_cfg.get(
                    "max_seq_len",
                    calibration_cfg.get("seq_len", 2048),
                )
            )
            model_args = ctx.cfg.get("model_args", {})
            if not isinstance(model_args, Mapping):
                raise TypeError("model_args must be a mapping for Gemma4 export.")
            export_gemma4_per_layer(
                q_model=ctx.require_model(),
                max_seq_len=max_seq_len,
                output_dir=output_dir,
                model_args=model_args,
                prefill_decode=bool(export_cfg.get("prefill_decode", True)),
                strict=bool(export_cfg.get("strict", False)),
            )

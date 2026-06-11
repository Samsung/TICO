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
from transformers import AutoProcessor

from tico.quantization import convert, prepare
from tico.quantization.config.builders import build_qwen3_vl_ptq_config
from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig
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
from tico.quantization.recipes.evaluation.video_mme import evaluate_and_print_video_mme
from tico.quantization.recipes.evaluation.vlm import (
    evaluate_coco,
    evaluate_llava_bench,
    evaluate_vlm_text_ppl,
    evaluate_vqa_tasks,
    print_coco_score_results,
    print_vqa_results,
)
from tico.quantization.recipes.export.checkpoint import save_checkpoint
from tico.quantization.recipes.utils import (
    move_to_device,
    quant_spec_from_config,
    quant_specs_equivalent,
    torch_dtype_from_name,
)


class Qwen3VLAdapter(ModelAdapter):
    family = "qwen3_vl"

    def load_model(self, ctx: RecipeContext) -> RecipeContext:
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
            device_map = "auto" if ctx.device.type != "cpu" else "cpu"

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

        calib_seq_len = get_by_path(cfg, "calibration.seq_len")
        if calib_seq_len is not None and hasattr(ctx.model.config, "text_config"):
            ctx.model.config.text_config.max_position_embeddings = min(
                int(ctx.model.config.text_config.max_position_embeddings),
                int(calib_seq_len),
            )
        return ctx

    @staticmethod
    def _disable_cache(model: Any) -> None:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "config") and hasattr(model.config, "text_config"):
            if hasattr(model.config.text_config, "use_cache"):
                model.config.text_config.use_cache = False

    def build_calibration_inputs(self, ctx: RecipeContext) -> list[dict]:
        calib = ctx.cfg.get("calibration", {})
        runtime = ctx.cfg.get("runtime", {})
        return build_vlm_calibration_inputs(
            processor=ctx.processor,
            dataset=calib.get("dataset", "vqav2"),
            datasets=calib.get("datasets"),
            n_samples=int(calib.get("n_samples", 128)),
            split=calib.get("split", "testdev"),
            max_seq_len=calib.get("seq_len"),
            seed=int(runtime.get("seed", 42)),
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
            for batch in iterator:
                model(**move_to_device(batch, ctx.device))

    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        self.forward_calibration(
            ctx,
            prepared_model,
            ctx.calibration_inputs,
            desc="PTQ calibration",
        )

    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        num_vision_blocks = self.get_num_vision_blocks(ctx.model)
        num_text_layers = self.get_num_text_layers(ctx.model)
        num_deepstack_mergers = self.get_num_deepstack_mergers(ctx.model)

        model_args = dict(ctx.cfg.get("model_args", {}))
        if "vision" not in model_args:
            model_args["vision"] = {}
        vision_args = model_args["vision"]
        if "grid_thw" in vision_args:
            vision_args["grid_thw"] = tuple(vision_args["grid_thw"])

        return build_qwen3_vl_ptq_config(
            num_vision_blocks=num_vision_blocks,
            num_text_layers=num_text_layers,
            num_deepstack_mergers=num_deepstack_mergers,
            model_args=model_args,
            activation=quant_spec_from_config(stage_cfg.get("activation", "int16")),
            linear_weight=quant_spec_from_config(stage_cfg.get("linear_weight")),
            vision_patch_embed_weight=quant_spec_from_config(
                stage_cfg.get("vision_patch_embed_weight")
            ),
            embedding_weight=quant_spec_from_config(stage_cfg.get("embedding_weight")),
            lm_head_weight=quant_spec_from_config(stage_cfg.get("lm_head_weight")),
            spin_rotation_weight=quant_spec_from_config(
                stage_cfg.get("spin_rotation_weight")
            ),
            norm=quant_spec_from_config(stage_cfg.get("norm")),
            norm_weight=quant_spec_from_config(stage_cfg.get("norm_weight")),
            strict_wrap=bool(stage_cfg.get("strict_wrap", True)),
        )

    @staticmethod
    def get_num_vision_blocks(model: Any) -> int:
        vision_config = getattr(model.config, "vision_config", None)
        for attr in ("num_hidden_layers", "num_layers", "depth"):
            if vision_config is not None and hasattr(vision_config, attr):
                return int(getattr(vision_config, attr))
        raise ValueError(
            "Cannot determine Qwen3-VL vision block count from model.config.vision_config."
        )

    @staticmethod
    def get_num_text_layers(model: Any) -> int:
        if hasattr(model.config, "text_config"):
            return int(model.config.text_config.num_hidden_layers)
        if hasattr(model.config, "num_hidden_layers"):
            return int(model.config.num_hidden_layers)
        raise ValueError("Cannot determine Qwen3-VL text layer count.")

    @staticmethod
    def get_num_deepstack_mergers(model: Any) -> int:
        vision_config = getattr(model.config, "vision_config", None)
        indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        return 0 if indexes is None else len(indexes)

    def apply_spinquant(
        self,
        ctx: RecipeContext,
        stage_cfg: Mapping[str, Any],
    ) -> torch.nn.Module:
        self._ensure_spinquant_compatible(ctx, stage_cfg)

        r1 = _load_torch_object(stage_cfg.get("r1_path"))
        r2_map = _load_torch_object(stage_cfg.get("r2_map_path"))
        vision_r1 = _load_torch_object(stage_cfg.get("vision_r1_path"))
        vision_r2_map = _load_torch_object(stage_cfg.get("vision_r2_map_path"))

        enable_r1 = bool(stage_cfg.get("enable_r1", True))
        enable_r2 = bool(stage_cfg.get("enable_r2", True))
        enable_vision_r1 = bool(stage_cfg.get("enable_vision_r1", False))
        enable_vision_r2 = bool(stage_cfg.get("enable_vision_r2", False))
        fuse_vision_layer_norms = bool(
            stage_cfg.get("fuse_vision_layer_norms", enable_vision_r1)
        )

        spinquant_config = Qwen3VLSpinQuantConfig(
            init_method=stage_cfg.get("init_method", "random"),
            r1=r1,
            r2_map=r2_map,
            enable_r1=enable_r1,
            enable_r2=enable_r2,
            fuse_deepstack_visual_outputs=bool(
                stage_cfg.get("fuse_deepstack_visual_outputs", True)
            ),
            show_progress=bool(
                stage_cfg.get(
                    "show_progress",
                    ctx.cfg.get("runtime", {}).get("show_progress", True),
                )
            ),
            fuse_vision_layer_norms=fuse_vision_layer_norms,
            enable_vision_r1=enable_vision_r1,
            enable_vision_r2=enable_vision_r2,
            vision_init_method=stage_cfg.get("vision_init_method"),
            vision_r1=vision_r1,
            vision_r2_map=vision_r2_map,
            require_vision_r1_layernorm_compatible=bool(
                stage_cfg.get("require_vision_r1_layernorm_compatible", True)
            ),
            vision_rotation_tolerance=float(
                stage_cfg.get("vision_rotation_tolerance", 1e-4)
            ),
        )

        print("Applying Qwen3-VL SpinQuant …")
        q_model = prepare(ctx.require_model(), spinquant_config, inplace=True)
        q_model = convert(q_model, inplace=True)
        q_model.eval()
        print("Qwen3-VL SpinQuant complete.")
        return q_model

    def _ensure_spinquant_compatible(
        self,
        ctx: RecipeContext,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        smooth_stage = _find_stage(ctx.cfg, "smoothquant")
        if (
            smooth_stage is not None
            and smooth_stage.get("enabled", True)
            and smooth_stage.get("components") in {"text", "both"}
        ):
            raise ValueError(
                "Qwen3-VL SpinQuant Phase 1 does not support text SmoothQuant "
                "together. Use smoothquant.components=vision or disable SmoothQuant."
            )

        ptq_stage = _find_stage(ctx.cfg, "ptq")
        if ptq_stage is None:
            return

        embedding_weight = ptq_stage.get("embedding_weight")
        lm_head_weight = ptq_stage.get("lm_head_weight")
        if embedding_weight is not None and lm_head_weight is not None:
            if not quant_specs_equivalent(embedding_weight, lm_head_weight):
                raise ValueError(
                    "Qwen3-VL SpinQuant assumes tied word embeddings, so "
                    "ptq.embedding_weight and ptq.lm_head_weight must match."
                )

    def apply_smoothquant(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        from tico.quantization.algorithm.smoothquant.smooth_quant import apply_smoothing

        alpha = float(stage_cfg.get("alpha", 0.5))
        components = stage_cfg.get("components")
        if components not in {"vision", "text", "both"}:
            raise ValueError(
                "SmoothQuant for Qwen3-VL requires components to be one of "
                "{'vision', 'text', 'both'}."
            )

        exclude_appliers: list[str] = []
        if components == "text":
            exclude_appliers.extend(
                [
                    "_apply_if_qwen3vl_vision_block",
                    "_apply_if_qwen3vl_vision_patch_merger",
                ]
            )
        elif components == "vision":
            exclude_appliers.append("_apply_if_qwen3vl_text_decoder")

        print(
            f"Applying SmoothQuant smoothing … components={components}, alpha={alpha}"
        )

        activation_max: dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, inputs, output):
                x = inputs[0] if isinstance(inputs, tuple) else inputs
                if isinstance(x, torch.Tensor) and x.dim() >= 2:
                    x_flat = x.reshape(-1, x.shape[-1])
                    amax = x_flat.abs().max(dim=0)[0].detach()
                    activation_max[name] = (
                        amax
                        if name not in activation_max
                        else torch.maximum(activation_max[name], amax)
                    )

            return hook

        for name, module in ctx.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            self.forward_calibration(
                ctx, ctx.model, ctx.calibration_inputs, desc="SmoothQuant stats"
            )
        finally:
            for hook in hooks:
                hook.remove()

        apply_smoothing(
            ctx.model,
            activation_max,
            alpha=alpha,
            custom_alpha_map=stage_cfg.get("custom_alpha_map"),
            exclude_appliers=exclude_appliers,
        )
        return ctx.model

    def evaluate(self, ctx: RecipeContext) -> None:
        eval_cfg = ctx.cfg.get("evaluation", {})
        if not eval_cfg.get("enabled", False):
            return

        max_seq_len = eval_cfg.get("max_seq_len")
        n_samples = int(eval_cfg.get("n_samples", 50))
        tasks = eval_cfg.get("vlm_tasks") or []
        verbose = bool(eval_cfg.get("verbose", False))
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]

        if tasks:
            vqa_results = evaluate_vqa_tasks(
                model=ctx.model,
                processor=ctx.processor,
                tasks=tasks,
                device=str(ctx.device),
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                verbose=verbose,
                show_progress=show_progress,
            )
            print_vqa_results("VQA evaluation", vqa_results)

        if eval_cfg.get("coco", False):
            coco_results = evaluate_coco(
                model=ctx.model,
                processor=ctx.processor,
                device=str(ctx.device),
                dataset_name="coco",
                n_samples=n_samples,
                max_seq_len=max_seq_len,
            )
            print_coco_score_results("\n=== COCO Evaluation ===", coco_results)

        llava_bench_cfg = eval_cfg.get("llava_bench", False)
        if isinstance(llava_bench_cfg, Mapping):
            if llava_bench_cfg.get("enabled", False):
                mode = str(llava_bench_cfg.get("mode", "judge")).lower()
                if mode in {"judge", "llm_judge"}:
                    evaluate_and_print_llava_bench_judge(
                        model=ctx.model,
                        processor=ctx.processor,
                        device=str(ctx.device),
                        llava_cfg=llava_bench_cfg,
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
                        n_samples=int(llava_bench_cfg.get("n_samples", n_samples)),
                        max_seq_len=llava_bench_cfg.get("max_seq_len", max_seq_len),
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
        elif llava_bench_cfg:
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
            print_coco_score_results("\n=== Llava Bench Evaluation ===", llava_results)

        videomme = eval_cfg.get("videomme", {})
        if videomme.get("enabled", False):
            n_samples = int(videomme.get("n_samples", -1))
            max_num_frames = int(videomme.get("max_num_frames", 32))
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
                n_samples=n_samples if n_samples > 0 else None,
                max_num_frames=max_num_frames,
                use_cache=videomme.get("use_cache", None),
                verbose=bool(videomme.get("verbose", eval_cfg.get("verbose", False))),
            )

        mmlu = eval_cfg.get("mmlu", {})
        if mmlu.get("enabled", False):
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

        hellaswag = eval_cfg.get("hellaswag", {})
        if hellaswag.get("enabled", False):
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

        mmmu = eval_cfg.get("mmmu", {})
        if mmmu.get("enabled", False):
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
                verbose=bool(mmmu.get("verbose", eval_cfg.get("verbose", False))),
            )

        ppl = eval_cfg.get("ppl", {})
        if ppl.get("enabled", False):
            ppl_value = evaluate_vlm_text_ppl(
                model=ctx.model,
                processor=ctx.processor,
                dataset_name=ppl.get("dataset", "wikitext2"),
                split=ppl.get("split", "test"),
                device=str(ctx.device),
                stride=int(ppl.get("stride", 512)),
                max_seq_len=int(
                    max_seq_len or ctx.cfg.get("calibration", {}).get("seq_len", 2048)
                ),
                show_progress=bool(
                    ctx.cfg.get("runtime", {}).get("show_progress", True)
                ),
            )
            print(f"\nPPL({ppl.get('dataset', 'wikitext2')}): {ppl_value:.2f}")

    def export(self, ctx: RecipeContext) -> None:
        export_cfg = ctx.cfg.get("export", {})
        if not export_cfg.get("enabled", False):
            return

        output_dir = Path(export_cfg.get("output_dir", "./out/qwen3_vl"))
        artifacts = set(export_cfg.get("artifacts", []))
        if "ptq_checkpoint" in artifacts or "checkpoint" in artifacts:
            save_checkpoint(ctx.model, output_dir)


def _load_torch_object(path: str | None) -> Any:
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


def _find_stage(cfg: dict[str, Any], stage_name: str) -> Mapping[str, Any] | None:
    for stage in cfg.get("pipeline", []):
        if isinstance(stage, Mapping) and stage.get("name") == stage_name:
            return stage
    return None

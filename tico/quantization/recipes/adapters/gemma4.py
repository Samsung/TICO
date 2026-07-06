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

from typing import Any, Mapping, Sequence

import torch
import tqdm
from transformers import AutoProcessor

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.vlm import build_vlm_calibration_inputs
from tico.quantization.recipes.evaluation.llava_bench_judge import (
    evaluate_and_print_llava_bench_judge,
)
from tico.quantization.recipes.evaluation.vlm import (
    evaluate_llava_bench,
    print_coco_score_results,
)
from tico.quantization.recipes.utils import (
    move_to_device,
    quant_spec_from_config,
    torch_dtype_from_name,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe


class Gemma4Adapter(ModelAdapter):
    """Model adapter for Gemma4 E2B PTQ and static runtime experiments."""

    family = "gemma4"

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
        return ctx

    @staticmethod
    def _disable_cache(model: Any) -> None:
        """Disable HF dynamic cache paths during PTQ calibration."""
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        text_config = getattr(getattr(model, "config", None), "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False

    def build_calibration_inputs(self, ctx: RecipeContext) -> list[dict]:
        """Build VLM calibration inputs for fixed image-text PTQ."""
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
        """Evaluate Gemma4 E2B via LLaVA-Bench judge and other VLM tasks."""
        eval_cfg = ctx.cfg.get("evaluation", {})
        if not eval_cfg.get("enabled", False):
            return

        max_seq_len = eval_cfg.get("max_seq_len")
        n_samples = int(eval_cfg.get("n_samples", 50))

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

    def export(self, ctx: RecipeContext) -> None:
        """Export Gemma4 E2B artifacts.

        TODO: Implement Circle export via the static runtime submodule adapters.
        """
        export_cfg = ctx.cfg.get("export", {})
        if not export_cfg.get("enabled", False):
            return
        raise NotImplementedError("Gemma4 E2B export adapter is not wired yet.")

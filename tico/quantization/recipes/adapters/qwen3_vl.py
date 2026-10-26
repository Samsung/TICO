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
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.data.vlm import build_vlm_calibration_inputs
from tico.quantization.recipes.evaluation.eager import prepare_eager_evaluation
from tico.quantization.recipes.evaluation.hellaswag import evaluate_and_print_hellaswag
from tico.quantization.recipes.evaluation.llava_bench import run_llava_bench_evaluation
from tico.quantization.recipes.evaluation.mmlu import evaluate_and_print_mmlu
from tico.quantization.recipes.evaluation.mmmu import (
    run_mmmu_evaluation,
    validate_mmmu_recipe_config,
)
from tico.quantization.recipes.evaluation.selection import (
    get_mapping_evaluation_config,
    should_run_evaluation,
    should_run_mapping_evaluation,
)
from tico.quantization.recipes.evaluation.video_mme import evaluate_and_print_video_mme
from tico.quantization.recipes.evaluation.vlm import (
    evaluate_coco,
    evaluate_vlm_text_ppl,
    print_coco_score_results,
    run_vqa_evaluation,
)
from tico.quantization.recipes.export.checkpoint import save_checkpoint
from tico.quantization.recipes.export.qwen3_vl import export_qwen3_vl_per_layer
from tico.quantization.recipes.override_policies import (
    COMMON_LINEAR_MODULES,
    ComponentQuantTargetInfo,
    QuantTargetResolverContext,
)
from tico.quantization.recipes.sequence_length import (
    model_context_length,
    resolve_export_max_seq_len,
    resolve_ppl_max_seq_len,
)
from tico.quantization.recipes.utils import (
    get_attr_path,
    move_to_device,
    quant_spec_from_config,
    quant_specs_equivalent,
    torch_dtype_from_name,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

_VISION_LINEAR_MODULES = (
    "attn.qkv",
    "attn.proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)


class Qwen3VLAdapter(ModelAdapter):
    family = "qwen3_vl"
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

    def validate_evaluation_config(self, cfg: Mapping[str, Any]) -> None:
        super().validate_evaluation_config(cfg)
        validate_mmmu_recipe_config(cfg)

    def get_gptq_config_class(self) -> type[Qwen3VLGPTQConfig]:
        """Select the stagewise vision + text GPTQ config for Qwen3-VL."""
        return Qwen3VLGPTQConfig

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

        # Every Transformers release that ships Qwen3-VL also exports this auto
        # class, so load with it directly and let any failure propagate as-is.
        from transformers import AutoModelForImageTextToText

        ctx.model = AutoModelForImageTextToText.from_pretrained(
            name,
            dtype=ctx.dtype,
            trust_remote_code=trust_remote_code,
            token=hf_token,
            cache_dir=cache_dir,
            device_map=device_map,
        )

        ctx.model.eval()
        self._disable_cache(ctx.model)

        ctx.artifacts["pretrained_context_length"] = model_context_length(ctx.model)
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
            split=calib.get("split"),
            max_seq_len=calib.get("seq_len"),
            seed=int(runtime.get("seed", 42)),
            allow_benchmark_overlap=bool(calib.get("allow_benchmark_overlap", False)),
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
        """Reserve trailing dataset tokens for connected cache calibration.

        The default one-token decode covers the static past-K/V boundary. Zero
        explicitly requests prompt-only calibration. Short samples clamp the
        decode count to leave at least one prefill token; a one-token sample
        cannot calibrate a nonempty past boundary. No extra tokens are generated.
        """
        decode_steps = int(
            stage_cfg.get(
                "decode_calibration_steps",
                ctx.cfg.get("calibration", {}).get("decode_steps", 1),
            )
        )
        if decode_steps < 0:
            raise ValueError("decode_calibration_steps must be non-negative.")
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))
        iterator = tqdm.tqdm(
            ctx.calibration_inputs, desc="PTQ calibration", disable=not show_progress
        )
        prepared_model.eval()
        owner = (
            prepared_model.wrapped
            if isinstance(prepared_model, PTQWrapper)
            else prepared_model
        )
        model_config = owner.config
        # Qwen owns request-local mRoPE offsets. Calibration must not leave its
        # final image/text request installed for the next evaluation request.
        rope_states = [
            (module, module.rope_deltas)
            for module in prepared_model.modules()
            if hasattr(module, "rope_deltas")
        ]
        try:
            with torch.no_grad():
                for batch in iterator:
                    batch = move_to_device(batch, ctx.device)
                    ids = batch.get("input_ids")
                    if (
                        not isinstance(ids, torch.Tensor)
                        or ids.ndim != 2
                        or ids.size(1) == 0
                    ):
                        raise ValueError(
                            "Qwen PTQ calibration requires nonempty rank-two input_ids."
                        )
                    budget = ctx.cfg.get("calibration", {}).get("seq_len")
                    if budget is not None and ids.size(1) > int(budget):
                        raise ValueError(
                            "Qwen calibration input exceeds calibration.seq_len token budget."
                        )
                    if batch.get("past_key_values") is not None:
                        raise ValueError(
                            "Qwen calibration inputs must start without a cache."
                        )
                    steps = min(decode_steps, ids.size(1) - 1)
                    prefix = ids.size(1) - steps
                    if steps:
                        special_ids = [
                            getattr(model_config, name, None)
                            for name in (
                                "image_token_id",
                                "video_token_id",
                                "vision_start_token_id",
                                "vision_end_token_id",
                            )
                        ]
                        if any(
                            value is not None and bool((ids[:, prefix:] == value).any())
                            for value in special_ids
                        ):
                            raise ValueError(
                                "Qwen cached calibration requires trailing text tokens after the complete image/video span."
                            )
                    mask = batch.get("attention_mask")
                    if mask is not None and (mask.ndim != 2 or mask.shape != ids.shape):
                        raise ValueError(
                            "Qwen PTQ calibration requires a rank-two attention_mask matching input_ids."
                        )
                    prefill = dict(batch)
                    prefill.pop("past_key_values", None)
                    for name in (
                        "input_ids",
                        "attention_mask",
                        "position_ids",
                        "token_type_ids",
                        "mm_token_type_ids",
                    ):
                        value = prefill.get(name)
                        if isinstance(value, torch.Tensor):
                            prefill[name] = value[..., :prefix]
                    prefill["use_cache"] = bool(steps)
                    prefill["return_dict"] = True
                    outputs = prepared_model(**prefill)
                    for position in range(prefix, ids.size(1)):
                        decode = {
                            "input_ids": ids[:, position : position + 1],
                            "past_key_values": outputs.past_key_values,
                            "use_cache": True,
                            "return_dict": True,
                        }
                        if outputs.past_key_values is None:
                            raise RuntimeError(
                                "Qwen calibration prefill did not return a usable cache."
                            )
                        if mask is not None:
                            decode["attention_mask"] = mask[:, : position + 1]
                        for name in (
                            "position_ids",
                            "token_type_ids",
                            "mm_token_type_ids",
                        ):
                            value = batch.get(name)
                            if isinstance(value, torch.Tensor):
                                decode[name] = value[..., position : position + 1]
                        # Image/video pixels belong only to prefill. The text
                        # suffix follows exactly the supplied dataset IDs.
                        outputs = prepared_model(**decode)
        finally:
            for module, value in rope_states:
                module.rope_deltas = value

    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        num_vision_blocks = self.get_num_vision_blocks(ctx.model)
        num_text_layers = self.get_num_text_layers(ctx.model)
        num_deepstack_mergers = self.get_num_deepstack_mergers(ctx.model)

        # Copy the mappings that are written below so grid_thw normalization
        # lands only in the builder arguments, never in the recipe cfg.
        model_args = dict(ctx.cfg.get("model_args", {}))
        vision_args = dict(model_args.get("vision", {}))
        if "grid_thw" in vision_args:
            vision_args["grid_thw"] = tuple(vision_args["grid_thw"])
        model_args["vision"] = vision_args

        return build_qwen3_vl_ptq_config(
            num_vision_blocks=num_vision_blocks,
            num_text_layers=num_text_layers,
            num_deepstack_mergers=num_deepstack_mergers,
            model_args=model_args,
            activation=quant_spec_from_config(stage_cfg.get("activation", "int16")),
            linear_weight=quant_spec_from_config(stage_cfg.get("linear_weight")),
            text_linear_weight=quant_spec_from_config(
                stage_cfg.get("text_linear_weight")
            ),
            vision_linear_weight=quant_spec_from_config(
                stage_cfg.get("vision_linear_weight")
            ),
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

    def build_quant_target_context(self, model: Any) -> QuantTargetResolverContext:
        """Declare the text decoder and vision block components for selectors.

        Layer counts are read with a wider fallback chain than
        ``get_num_text_layers``/``get_num_vision_blocks`` (config fields first,
        then the actual module lists), so selector resolution keeps accepting
        the model shapes it accepted before.
        """
        return QuantTargetResolverContext(
            components={
                "text": ComponentQuantTargetInfo(
                    name="text",
                    layer_path_prefixes=(("model", "language_model", "layers"),),
                    num_layers=_quant_target_text_layers(model),
                    op_aliases={"linear": COMMON_LINEAR_MODULES},
                ),
                "vision": ComponentQuantTargetInfo(
                    name="vision",
                    layer_path_prefixes=(("model", "visual", "blocks"),),
                    num_layers=_quant_target_vision_blocks(model),
                    op_aliases={"linear": _VISION_LINEAR_MODULES},
                ),
            }
        )

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

        self.validate_evaluation_config(ctx.cfg)
        max_seq_len = prepare_eager_evaluation(ctx)
        n_samples = int(eval_cfg.get("n_samples", 50))
        verbose = bool(eval_cfg.get("verbose", False))
        show_progress = bool(ctx.cfg.get("runtime", {}).get("show_progress", True))

        run_vqa_evaluation(
            eval_cfg=eval_cfg,
            model=ctx.model,
            processor=ctx.processor,
            device=str(ctx.device),
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            verbose=verbose,
            show_progress=show_progress,
        )

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

        run_llava_bench_evaluation(
            eval_cfg=eval_cfg,
            model=ctx.model,
            processor=ctx.processor,
            device=str(ctx.device),
            model_cfg=ctx.cfg.get("model", {}),
            runtime_cfg=ctx.cfg.get("runtime", {}),
            default_n_samples=n_samples,
            default_max_seq_len=max_seq_len,
        )

        if should_run_mapping_evaluation(eval_cfg, "videomme"):
            videomme = get_mapping_evaluation_config(eval_cfg, "videomme")
            video_n_samples = int(videomme.get("n_samples", -1))
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
                n_samples=video_n_samples if video_n_samples > 0 else None,
                max_num_frames=max_num_frames,
                max_seq_len=max_seq_len,
                kv_cache=eval_cfg.get("use_cache", True),
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
                max_seq_len=max_seq_len,
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
                max_seq_len=max_seq_len,
            )

        run_mmmu_evaluation(
            eval_cfg=eval_cfg,
            model=ctx.model,
            processor=ctx.processor,
            device=str(ctx.device),
            max_seq_len=max_seq_len,
            verbose=verbose,
        )

        if should_run_mapping_evaluation(eval_cfg, "ppl"):
            ppl_cfg = get_mapping_evaluation_config(eval_cfg, "ppl")
            ppl_value = evaluate_vlm_text_ppl(
                model=ctx.model,
                processor=ctx.processor,
                dataset_name=ppl_cfg.get("dataset", "wikitext2"),
                split=ppl_cfg.get("split", "test"),
                device=str(ctx.device),
                stride=int(ppl_cfg.get("stride", 512)),
                max_seq_len=resolve_ppl_max_seq_len(
                    ppl_cfg, evaluation_max_seq_len=max_seq_len
                ),
                show_progress=show_progress,
            )
            print(f"\nPPL({ppl_cfg.get('dataset', 'wikitext2')}): {ppl_value:.2f}")

    def export(self, ctx: RecipeContext) -> None:
        export_cfg = ctx.cfg.get("export", {})
        if not export_cfg.get("enabled", False):
            return

        output_dir = Path(export_cfg.get("output_dir", "./out/qwen3_vl"))
        artifacts = set(export_cfg.get("artifacts", []))
        if "ptq_checkpoint" in artifacts or "checkpoint" in artifacts:
            save_checkpoint(ctx.model, output_dir)

        if "circle_per_layer" in artifacts:
            max_seq_len = resolve_export_max_seq_len(ctx)
            model_args = ctx.cfg.get("model_args", {})
            if not isinstance(model_args, Mapping):
                raise TypeError(
                    "Qwen3-VL Circle export requires model_args to be a mapping."
                )

            export_qwen3_vl_per_layer(
                q_model=ctx.model,
                max_seq_len=max_seq_len,
                output_dir=output_dir,
                model_args=model_args,
                prefill_decode=bool(export_cfg.get("prefill_decode", True)),
                strict=bool(export_cfg.get("strict", False)),
            )


def _quant_target_text_layers(model: Any) -> int:
    """Text layer count for selector overrides: text_config, config, then modules."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    if config is not None and hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers)
    layers = get_attr_path(
        model,
        ("model", "language_model", "layers"),
        context="Qwen3-VL text layers",
    )
    return len(layers)


def _quant_target_vision_blocks(model: Any) -> int:
    """Vision block count for selector overrides: config fields, then modules."""
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    for attr in ("num_hidden_layers", "num_layers", "depth"):
        if vision_config is not None and hasattr(vision_config, attr):
            return int(getattr(vision_config, attr))
    blocks = get_attr_path(
        model,
        ("model", "visual", "blocks"),
        context="Qwen3-VL vision blocks",
    )
    return len(blocks)


def _load_torch_object(path: str | None) -> Any:
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


def _find_stage(cfg: dict[str, Any], stage_name: str) -> Mapping[str, Any] | None:
    for stage in cfg.get("pipeline", []):
        if isinstance(stage, Mapping) and stage.get("name") == stage_name:
            return stage
    return None

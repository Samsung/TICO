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

import argparse
import contextlib
import io
from typing import Any, Dict, List, Optional

import torch
import tqdm
from transformers import AutoProcessor

from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.algorithm.smoothquant.smooth_quant import apply_smoothing
from tico.quantization.config.builders import build_qwen3_vl_ptq_config
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig
from tico.quantization.config.specs import affine
from tico.quantization.evaluation.hellaswag_eval_utils import (
    evaluate_hellaswag,
    get_hellaswag_accuracy,
    print_hellaswag_results,
)
from tico.quantization.evaluation.lmms_eval_utils import (
    evaluate_vlm_on_tasks,
    print_lmms_eval_results,
)
from tico.quantization.evaluation.mmlu_eval_utils import (
    evaluate_mmlu,
    print_mmlu_results,
)
from tico.quantization.evaluation.mmmu_eval_utils import (
    evaluate_mmmu,
    MMMU_DATASETS,
    print_mmmu_results,
)
from tico.quantization.evaluation.vlm_eval_utils import (
    evaluate_ppl,
    get_accuracy_on_dataset,
    get_coco_scores_on_dataset,
    get_dataset,
    get_mixed_calib_inputs,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO: Support more dtypes if needed.
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL modular GPTQ plus PTQ pipeline"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF repo name or local path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda|cpu|mps).",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="float32",
        help="Model dtype for load.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )
    parser.add_argument(
        "--no_GPTQ",
        action="store_true",
        default=False,
        help="Skip GPTQ. PTQ still runs unless --no_PTQ is also set.",
    )
    parser.add_argument(
        "--gptq_vision",
        dest="gptq_vision",
        action="store_true",
        default=True,
        help="Apply GPTQ to the vision tower. Enabled by default.",
    )
    parser.add_argument(
        "--no_gptq_vision",
        dest="gptq_vision",
        action="store_false",
        help="Disable GPTQ for the vision tower.",
    )
    parser.add_argument(
        "--gptq_text",
        dest="gptq_text",
        action="store_true",
        default=True,
        help="Apply GPTQ to the text decoder. Enabled by default.",
    )
    parser.add_argument(
        "--no_gptq_text",
        dest="gptq_text",
        action="store_false",
        help="Disable GPTQ for the text decoder.",
    )
    parser.add_argument(
        "--gptq_lm_head",
        action="store_true",
        default=False,
        help=(
            "Apply GPTQ to lm_head. Disabled by default because "
            "lm_head.weight can be tied with the input embedding table."
        ),
    )
    parser.add_argument(
        "--no_PTQ",
        action="store_true",
        default=False,
        help="Skip PTQ activation quantization.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for model/dataset loading.",
    )
    parser.add_argument(
        "--nsamples_for_qcalibration",
        type=str,
        default="vqav2:testdev:128,wikitext2:train:128",
        help=(
            "Just a number -> uses default dataset `vqav2` with `testdev` split and N samples. "
            "Comma-separated list: "
            "a) dataset:n_samples -> uses default split. "
            "b) dataset:split:n_samples -> uses specified split. "
            "c) Multiple datasets example: vqav2:testdev:128,wikitext2:train:128,coco:32"
        ),
    )
    parser.add_argument(
        "--nsamples_for_evaluation",
        type=int,
        default=50,
        help="Number of samples for evaluation. -1 means full dataset.",
    )
    parser.add_argument(
        "--calib_seq_len",
        type=int,
        default=2048,
        help=(
            "Maximum text sequence length for calibration inputs. "
            "If not set, processor default behavior is used."
        ),
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help=(
            "Maximum text sequence length for evaluation and export. "
            "If not set, processor default behavior is used."
        ),
    )
    parser.add_argument(
        "--linear_weight_bits",
        type=int,
        default=4,
        help="Weight bit-width for linear weight quantization.",
    )
    parser.add_argument(
        "--vision_patch_embed_weight_bits",
        type=int,
        default=8,
        help="Number of bits for vision patch embedding (Conv3d) quantization.",
    )
    parser.add_argument(
        "--embedding_weight_bits",
        type=int,
        default=8,
        help="Number of bits for input Embedding quantization.",
    )
    parser.add_argument(
        "--lm_head_weight_bits",
        type=int,
        default=4,
        help=(
            "Number of bits for lm_head quantization. Qwen3-VL SpinQuant assumes "
            "tied word embeddings, so this must match --embedding_weight_bits."
        ),
    )
    parser.add_argument(
        "--gptq_mse",
        type=str,
        default=None,
        choices=["mse", "smse"],
        help="Whether and how to use mse in GPTQ.",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help="Tasks to evaluate, e.g. `vqav2,textvqa`.",
    )
    parser.add_argument(
        "--sensitivity_path",
        type=str,
        default=None,
        help="Optional path to precomputed sensitivity tensors.",
    )
    parser.add_argument(
        "--move_cache_to_cpu",
        action="store_true",
        default=False,
        help="Move cached stage inputs to CPU between stages to save device memory.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="GPTQ group size. -1 disables grouping.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="GPTQ percdamp value.",
    )
    parser.add_argument(
        "--actorder",
        action="store_true",
        default=True,
        help="Enable activation-order column permutation.",
    )
    parser.add_argument(
        "--static_groups",
        action="store_true",
        default=False,
        help="Enable static group quantizers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose GPTQ logging.",
    )
    parser.add_argument(
        "--hide_progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--grid_thw",
        type=int,
        nargs=3,
        default=[8, 24, 24],
        help="Grid temporal-height-width for vision model (e.g. 8 24 24).",
    )
    parser.add_argument(
        "--visual_start_idx",
        type=int,
        default=0,
        help="Starting index for visual tokens in the input sequence.",
    )
    parser.add_argument(
        "--spatial_merge_size",
        type=int,
        default=2,
        help="Spatial merge size for vision tokens.",
    )

    # SpinQuant arguments
    parser.add_argument(
        "--spinquant",
        action="store_true",
        default=False,
        help="Apply tied-embedding-safe Qwen3-VL SpinQuant before SmoothQuant/GPTQ/PTQ.",
    )
    parser.add_argument(
        "--spinquant_init_method",
        choices=["random", "hadamard", "external"],
        default="random",
        help="Rotation initialization method for Qwen3-VL SpinQuant.",
    )
    parser.add_argument(
        "--spinquant_disable_r1",
        action="store_true",
        default=False,
        help="Disable global hidden-dimension R1 rotation.",
    )
    parser.add_argument(
        "--spinquant_disable_r2",
        action="store_true",
        default=False,
        help="Disable OV-side head-dimension R2 rotation.",
    )
    parser.add_argument(
        "--spinquant_disable_deepstack_fusion",
        action="store_true",
        default=False,
        help="Disable R1 fusion for Qwen3-VL DeepStack visual output projections.",
    )
    parser.add_argument(
        "--spinquant_r1_path",
        type=str,
        default=None,
        help="Optional path to an external R1 tensor.",
    )
    parser.add_argument(
        "--spinquant_r2_map_path",
        type=str,
        default=None,
        help="Optional path to an external R2 tensor map.",
    )

    # SmoothQuant arguments (for LayerNorm-based vision components and RMSNorm-based text components)
    parser.add_argument(
        "--smoothquant",
        action="store_true",
        default=False,
        help="Apply SmoothQuant smoothing for vision and text layers.",
    )
    parser.add_argument(
        "--smoothquant_alpha",
        type=float,
        default=0.5,
        help="SmoothQuant alpha for vision components (Qwen3VLVisionBlock, Qwen3VLVisionPatchMerger). "
        "Range: 0.0-1.0. Higher = more weight smoothing.",
    )
    parser.add_argument(
        "--smoothquant_components",
        choices=["vision", "text", "both"],
        default=None,
        help="Target components for SmoothQuant.",
    )
    parser.add_argument(
        "--print_quantized_model",
        action="store_true",
        default=False,
        help="Print model after quantization",
    )

    # MMLU evaluation arguments
    parser.add_argument(
        "--mmlu_subjects",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Space-separated list of MMLU subjects to evaluate. Use 'mmlu' for all subjects."
            "Use 'stem', 'humanities', 'social_sciences', or 'other' for the broader domains of knowledge."
            "Or use narrower subjects like 'college_physics', 'abstract_algebra', etc."
            "See https://huggingface.co/datasets/cais/mmlu for the full list."
        ),
    )
    parser.add_argument(
        "--mmlu_n_shots",
        type=int,
        default=5,
        help="Number of few-shot examples for MMLU evaluation.",
    )
    parser.add_argument(
        "--mmlu_n_samples",
        type=int,
        default=-1,
        help="Number of samples per MMLU subject. Use -1 for full test set.",
    )
    parser.add_argument(
        "--mmlu_batch_size",
        type=int,
        default=1,
        help="Number of samples in a batch for MMLU evaluation.",
    )

    # MMMU evaluation arguments
    parser.add_argument(
        "--mmmu_dataset",
        type=str,
        choices=MMMU_DATASETS,
        default=None,
        help="MMMU dataset name.",
    )

    parser.add_argument(
        "--mmmu_subjects",
        type=str,
        default=None,
        nargs="+",
        help=(
            "Space-separated list of MMMU subjects to evaluate. "
            "Use 'Accounting', 'Agriculture', 'Art', etc. for specific subjects. "
            "See https://huggingface.co/datasets/MMMU/MMMU for the full list. "
            "If not specified, all subjects will be evaluated."
        ),
    )
    parser.add_argument(
        "--mmmu_n_shots",
        type=int,
        default=5,
        help="Number of few-shot examples for MMMU evaluation.",
    )
    parser.add_argument(
        "--mmmu_n_samples",
        type=int,
        default=-1,
        help="Number of samples per MMMU subject. Use -1 for full test set.",
    )

    # HellaSwag evaluation arguments
    parser.add_argument(
        "--hellaswag",
        action="store_true",
        default=False,
        help="Evaluate on HellaSwag benchmark.",
    )
    parser.add_argument(
        "--hellaswag_n_shots",
        type=int,
        default=10,
        help="Number of few-shot examples for HellaSwag evaluation.",
    )
    parser.add_argument(
        "--hellaswag_n_samples",
        type=int,
        default=-1,
        help="Number of samples for HellaSwag evaluation. Use -1 for full test set.",
    )
    parser.add_argument(
        "--hellaswag_batch_size",
        type=int,
        default=1,
        help="Batch size for HellaSwag evaluation.",
    )

    # PPL evaluation arguments
    parser.add_argument(
        "--ppl_dataset",
        type=str,
        default=None,
        choices=["wikitext2"],
        help="Text dataset for PPL (perplexity) evaluation.",
    )
    parser.add_argument(
        "--ppl_stride",
        type=int,
        default=512,
        help="Sliding window stride for perplexity calculation.",
    )

    parser.add_argument(
        "--ppl_split",
        type=str,
        default="test",
        help="Split for PPL evaluation",
    )

    # Video-MME evaluation arguments
    parser.add_argument(
        "--video_mme",
        action="store_true",
        default=False,
        help="Evaluate on Video-MME benchmark via lmms-eval.",
    )
    parser.add_argument(
        "--video_mme_max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate per Video-MME sample.",
    )
    parser.add_argument(
        "--video_mme_limit",
        type=int,
        default=None,
        help="Limit the number of Video-MME samples to evaluate (and download). "
        "E.g. --video_mme_limit 10 evaluates only the first 10 samples.",
    )
    parser.add_argument(
        "--video_mme_max_num_frames",
        type=int,
        default=5,
        help="Maximum number of frames to sample from each video for Video-MME evaluation. "
        "Default is 5 for fitting within max_position_embeddings (2048).",
    )

    return parser.parse_args()


def build_processor_inputs(
    processor: Any,
    image: Any,
    question: str,
    seq_len: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """
    Build one multimodal processor input with optional text truncation.

    Args:
        processor: Hugging Face processor for the target model.
        image: Input image.
        question: User question text.
        seq_len: Optional maximum text sequence length. If None, processor
            default behavior is used.

    Returns:
        Processor output mapping.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        "Return ONLY the final answer with no extra words."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    processor_kwargs = {"text": prompt, "images": image, "return_tensors": "pt"}
    if seq_len is not None and seq_len > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = seq_len
    return dict(processor(**processor_kwargs))


def evaluate_model(
    model,
    processor,
    tasks: str,
    device: str,
    nsamples: int = 50,
    max_seq_len: Optional[int] = None,
) -> dict[str, tuple[int, int]]:
    """
    Evaluate a VLM on one or more mini VQA tasks.

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor.
        tasks: Comma-separated task names.
        device: Target device string.
        nsamples: Number of evaluation samples per task. -1 means full dataset.
        max_seq_len: Optional maximum text sequence length for evaluation inputs.

    Returns:
        Mapping from task name to (exact_match_count, total_count).
    """
    tasks_list = tasks.split(",")
    results: dict[str, tuple[int, int]] = {}

    for task in tasks_list:
        if "vqa" not in task:
            continue
        with (
            io.StringIO() as buffer,
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
        ):
            ds, adapter = get_dataset(task, n=nsamples)
            em_cnt, total = get_accuracy_on_dataset(
                model,
                processor,
                ds,
                adapter,
                device,
                max_seq_len=max_seq_len,
            )
            results[task] = (em_cnt, total)

    return results


def evaluate_model_with_coco_score(
    model,
    processor,
    device: str,
    dataset_name: str,
    nsamples: int = 50,
    max_seq_len: Optional[int] = None,
):
    """
    Evaluate a model on the provided dataset with COCO score

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor.
        device: Target device string.
        dataset_name: Dataset name for evaluation
        nsamples: Number of evaluation samples. -1 means full dataset.
        max_seq_len: Optional maximum text sequence length.

    Returns:
        COCO metric dictionary.
    """

    ds, _ = get_dataset(dataset_name, n=nsamples)
    result = get_coco_scores_on_dataset(
        model=model,
        processor=processor,
        dataset_name=dataset_name,
        ds=ds,
        device=device,
        max_seq_len=max_seq_len,
    )
    return result


def move_batch_to_device(
    batch: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    """
    Move one processor batch to the target device.

    Args:
        batch: Processor output mapping.
        device: Target device.

    Returns:
        Device-moved batch.
    """
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def print_eval_results(
    title: str,
    results: dict[str, tuple[int, int]],
) -> None:
    """
    Print evaluation results in a simple readable format.

    Args:
        title: Section title.
        results: Task result mapping.
    """
    print(title)
    for key, (correct, total) in results.items():
        print(f"{key}: EM={correct / total:.4f}  (n={total})")


def print_markdown_comparison(
    original_results: dict[str, tuple[int, int]],
    quantized_results: dict[str, tuple[int, int]],
) -> None:
    """
    Print a markdown table comparing original and quantized metrics.

    Args:
        original_results: Baseline results.
        quantized_results: Quantized results.
    """
    tasks = list(quantized_results.keys())

    header = "|model|" + "|".join(tasks) + "|"
    sep = "|--|" + "|".join(["--"] * len(tasks)) + "|"

    original_row = "|original|"
    for task in tasks:
        correct, total = original_results[task]
        original_row += f"{correct / total:.4f}|"

    quantized_row = "|quantized|"
    for task in tasks:
        correct, total = quantized_results[task]
        quantized_row += f"{correct / total:.4f}|"

    print(header)
    print(sep)
    print(original_row)
    print(quantized_row)


def load_torch_object(path: Optional[str]) -> Any:
    """
    Load a torch object from disk if a path is provided.

    Args:
        path: Optional file path.

    Returns:
        Loaded object, or None if the path is None.
    """
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


def ensure_spinquant_compatible_args(args: argparse.Namespace) -> None:
    """
    Validate command-line options that interact with Qwen3-VL SpinQuant.

    Args:
        args: Parsed command-line arguments.

    Raises:
        ValueError: If incompatible options are enabled together.
    """
    if not args.spinquant:
        return

    if args.smoothquant and args.smoothquant_components in {"text", "both"}:
        raise ValueError(
            "Qwen3-VL SpinQuant Phase 1 does not support text SmoothQuant "
            "together. Use --smoothquant_components vision or disable SmoothQuant."
        )

    if args.embedding_weight_bits != args.lm_head_weight_bits:
        raise ValueError(
            "Qwen3-VL SpinQuant assumes tied word embeddings, so "
            "--embedding_weight_bits and --lm_head_weight_bits must match."
        )


def apply_spinquant_if_enabled(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.nn.Module:
    """
    Apply Qwen3-VL SpinQuant before SmoothQuant/GPTQ/PTQ if requested.

    Args:
        model: Floating-point Qwen3-VL model.
        args: Parsed command-line arguments.

    Returns:
        The original model if SpinQuant is disabled, otherwise a SpinQwen3VL
        model with fused rotations.
    """
    if not args.spinquant:
        return model

    r1 = load_torch_object(args.spinquant_r1_path)
    r2_map = load_torch_object(args.spinquant_r2_map_path)

    spinquant_config = Qwen3VLSpinQuantConfig(
        init_method=args.spinquant_init_method,
        r1=r1,
        r2_map=r2_map,
        enable_r1=not args.spinquant_disable_r1,
        enable_r2=not args.spinquant_disable_r2,
        fuse_deepstack_visual_outputs=not args.spinquant_disable_deepstack_fusion,
        show_progress=not args.hide_progress,
    )

    print("Applying Qwen3-VL SpinQuant …")
    model = prepare(model, spinquant_config, inplace=True)
    model = convert(model, inplace=True)
    model.eval()
    print("Qwen3-VL SpinQuant complete.")
    return model


def has_gptq_targets(args) -> bool:
    """
    Return whether at least one module is selected for GPTQ.

    Args:
        args: Command-line arguments.

    Returns:
        True when GPTQ is enabled for vision, text, or lm_head.
    """
    return args.gptq_vision or args.gptq_text or args.gptq_lm_head


def build_qwen3_vl_gptq_config(
    args,
    sensitivity: dict[str, torch.Tensor] | None = None,
) -> Qwen3VLGPTQConfig:
    """
    Build a Qwen3-VL GPTQ configuration from command-line arguments.

    GPTQ can be configured independently for the vision tower, text decoder,
    and lm_head. Vision and text GPTQ are enabled by default; lm_head GPTQ is
    disabled by default.
    """
    weight_bits_overrides: dict[str, int] = {}

    if args.gptq_lm_head:
        weight_bits_overrides["lm_head"] = args.lm_head_weight_bits

    return Qwen3VLGPTQConfig(
        verbose=args.verbose,
        show_progress=not args.hide_progress,
        weight_bits=args.linear_weight_bits,
        weight_bits_overrides=weight_bits_overrides,
        mse=args.gptq_mse,
        sensitivity=sensitivity,
        percdamp=args.percdamp,
        groupsize=args.groupsize,
        actorder=args.actorder,
        static_groups=args.static_groups,
        quantize_vision=args.gptq_vision,
        quantize_text=args.gptq_text,
        quantize_lm_head=args.gptq_lm_head,
        quantize_vision_patch_embed=args.gptq_vision,
        quantize_vision_blocks=args.gptq_vision,
        quantize_vision_merger=args.gptq_vision,
        quantize_vision_deepstack_mergers=args.gptq_vision,
        quantize_text_layers=args.gptq_text,
        move_cache_to_cpu=args.move_cache_to_cpu,
    )


# -------------------------------------------------------------------------
# Helper — copy GPTQ (scale, zp) into PTQ observers
# -------------------------------------------------------------------------
def inject_gptq_qparams(
    root: torch.nn.Module,
    gptq_quantizers: dict[str, Any],  # {fp_name: quantizer}
    weight_obs_name: str = "weight",
):
    """
    For every `QuantModuleBase` whose `fp_name` matches a GPTQ key,
    locate the observer called `weight_obs_name` and overwrite its
    (scale, zero-point), then lock them against further updates.
    """
    for m in root.modules():
        if not isinstance(m, QuantModuleBase):
            continue
        if m.fp_name is None:
            continue

        quantizer = gptq_quantizers.get(m.fp_name)
        if quantizer is None:
            continue
        obs = m.get_observer(weight_obs_name)
        if obs is None:
            continue
        assert isinstance(obs, AffineObserverBase)
        # GPTQ quantizer attributes
        obs.load_qparams(quantizer.scale, quantizer.zero, lock=True)


def quantize_using_PTQ(
    q_m,
    calib_inputs,
    args,
    grid_thw,
    num_vision_blocks: int,
    num_text_layers: int,
    num_deepstack_mergers: int,
):
    """
    Wrap the full model with PTQWrapper and calibrate activation observers.

    Args:
        q_m: Model after optional GPTQ quantization.
        calib_inputs: Calibration inputs for PTQ calibration.
        args: Command-line arguments.
        grid_thw: Vision grid temporal-height-width tuple.
        num_vision_blocks: Number of vision transformer blocks.
        num_text_layers: Number of text decoder layers.
        num_deepstack_mergers: Number of deepstack merger modules.

    Returns:
        PTQ-quantized model.
    """
    print("Wrapping model with PTQWrapper …")
    qcfg = build_qwen3_vl_ptq_config(
        num_vision_blocks=num_vision_blocks,
        num_text_layers=num_text_layers,
        num_deepstack_mergers=num_deepstack_mergers,
        activation=affine(DType.int(16)),
        linear_weight=affine(DType.uint(args.linear_weight_bits)),
        vision_patch_embed_weight=affine(
            DType.uint(args.vision_patch_embed_weight_bits)
        ),
        embedding_weight=affine(DType.uint(args.embedding_weight_bits)),
        lm_head_weight=affine(DType.uint(args.lm_head_weight_bits)),
        norm=affine(DType.int(16)),
        norm_weight=affine(DType.int(16)),
        strict_wrap=True,
        model_args={
            "vision": {
                "grid_thw": grid_thw,
                "visual_start_idx": args.visual_start_idx,
                "spatial_merge_size": args.spatial_merge_size,
            }
        },
    )

    q_m = prepare(q_m, qcfg)

    # -------------------------------------------------------------------------
    # Single-pass activation calibration
    # -------------------------------------------------------------------------
    print("Calibrating PTQ observers…")

    # Overwrite weight observers with GPTQ statistics when GPTQ was applied.
    if not args.no_GPTQ and has_gptq_targets(args):
        if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
            inject_gptq_qparams(q_m, q_m.quantizers)
        elif (
            hasattr(q_m, "wrapped")
            and hasattr(q_m.wrapped, "module")
            and hasattr(q_m.wrapped.module, "quantizers")
            and isinstance(q_m.wrapped.module.quantizers, dict)
        ):
            inject_gptq_qparams(q_m.wrapped, q_m.wrapped.module.quantizers)
        else:
            print(
                "[Warn] q_m.quantizers not found or not a dict; skipping GPTQ qparam injection."
            )

    with torch.no_grad():
        for inp in tqdm.tqdm(calib_inputs):
            dev_inp = move_batch_to_device(inp, args.device)
            q_m(**dev_inp)

    # Freeze all Q-params (scale, zero-point)
    q_m = convert(q_m)

    return q_m


def get_num_vision_blocks(q_m) -> int:
    """
    Get the number of vision transformer blocks from model config.

    Args:
        q_m: Model with config attribute containing vision_config.

    Returns:
        Number of vision blocks.

    Raises:
        ValueError: If vision config or layer count cannot be determined.
    """
    if hasattr(q_m, "config") and hasattr(q_m.config, "vision_config"):
        vision_config = q_m.config.vision_config
        if hasattr(vision_config, "num_hidden_layers"):
            return vision_config.num_hidden_layers
        elif hasattr(vision_config, "num_layers"):
            return vision_config.num_layers
        elif hasattr(vision_config, "depth"):
            return vision_config.depth
        else:
            raise ValueError(
                "Cannot determine num_vision_blocks from vision_config. "
                "Expected vision_config.num_hidden_layers, num_layers, or depth."
            )
    else:
        raise ValueError(
            "Cannot determine num_vision_blocks from model config. "
            "Please ensure the model has config.vision_config."
        )


def get_num_text_layers(q_m) -> int:
    """
    Get the number of text decoder layers from model config.

    Args:
        q_m: Model with config attribute containing text_config.

    Returns:
        Number of text layers.

    Raises:
        ValueError: If text config or layer count cannot be determined.
    """
    if hasattr(q_m, "config") and hasattr(q_m.config, "text_config"):
        return q_m.config.text_config.num_hidden_layers
    elif hasattr(q_m, "config") and hasattr(q_m.config, "num_hidden_layers"):
        return q_m.config.num_hidden_layers
    else:
        raise ValueError(
            "Cannot determine num_text_layers from model config. "
            "Please ensure the model has config.text_config.num_hidden_layers "
            "or config.num_hidden_layers."
        )


def get_num_deepstack_mergers(q_m) -> int:
    """
    Get the number of deepstack merger modules from model config.

    Args:
        q_m: Model with config attribute containing vision_config.

    Returns:
        Number of deepstack mergers (0 if not present).
    """
    num_deepstack_mergers = 0
    if hasattr(q_m, "config") and hasattr(q_m.config, "vision_config"):
        vision_config = q_m.config.vision_config
        if hasattr(vision_config, "deepstack_visual_indexes"):
            num_deepstack_mergers = len(vision_config.deepstack_visual_indexes)
    return num_deepstack_mergers


def apply_smoothquant_if_requested(
    model: torch.nn.Module,
    calib_inputs: list[dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> None:
    """
    Apply SmoothQuant smoothing when requested.

    Calibration maxima are collected from Linear module inputs, then passed to
    SmoothQuant appliers for the selected Qwen3-VL components.

    Args:
        model: Target model.
        calib_inputs: Calibration inputs.
        args: Parsed command-line arguments.

    Raises:
        ValueError: If SmoothQuant is enabled without target components.
    """
    if not args.smoothquant:
        return

    if args.smoothquant_components is None:
        raise ValueError(
            "--smoothquant_components must be specified when --smoothquant is enabled."
        )

    exclude_appliers = []
    if args.smoothquant_components == "text":
        exclude_appliers.extend(
            [
                "_apply_if_qwen3vl_vision_block",
                "_apply_if_qwen3vl_vision_patch_merger",
            ]
        )
    if args.smoothquant_components == "vision":
        exclude_appliers.append("_apply_if_qwen3vl_text_decoder")

    print(
        f"Applying SmoothQuant smoothing for {args.smoothquant_components} components"
    )
    print("Computing activation maximum values for SmoothQuant …")
    activation_max: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                x = input[0]
            else:
                x = input
            if not isinstance(x, torch.Tensor):
                return
            if x.dim() < 2:
                return
            x_flat = x.reshape(-1, x.shape[-1])
            amax = x_flat.abs().max(dim=0)[0].detach()
            if name not in activation_max:
                activation_max[name] = amax
            else:
                activation_max[name] = torch.maximum(activation_max[name], amax)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            iterator = tqdm.tqdm(
                calib_inputs,
                desc="SmoothQuant calibration",
                disable=args.hide_progress,
            )
            for inp in iterator:
                dev_inp = move_batch_to_device(inp, args.device)
                model(**dev_inp)
    finally:
        for hook in hooks:
            hook.remove()

    apply_smoothing(
        model,
        activation_max,
        alpha=args.smoothquant_alpha,
        exclude_appliers=exclude_appliers,
    )


def load_or_compute_sensitivity(model, calib_inputs, args):
    """
    Load or compute sensitivity tensors used by sensitivity-aware GPTQ.

    Sensitivity information is only needed when `--gptq_mse smse` is used.
    """
    if args.gptq_mse != "smse":
        return None

    if args.sensitivity_path is not None:
        print(f"Loading sensitivity tensors from {args.sensitivity_path}")
        return torch.load(args.sensitivity_path, map_location="cpu")

    print("Computing sensitivity tensors for GPTQ …")
    calibrator = SensitivityCalibrator(
        model,
        calib_inputs,
        show_progress=not args.hide_progress,
    )
    return calibrator.compute_sensitivity_info()


def quantize_using_GPTQ(model, calib_inputs, args):
    """
    Apply Qwen3-VL GPTQ to the selected model modules.

    GPTQ for vision, text, and lm_head is controlled independently. Vision
    and text GPTQ are enabled by default and can be disabled with
    `--no_gptq_vision` and `--no_gptq_text`. lm_head GPTQ remains opt-in via
    `--gptq_lm_head`.
    """
    if args.no_GPTQ:
        print("Skipping GPTQ because --no_GPTQ was set.")
        return model

    if not has_gptq_targets(args):
        print("Skipping GPTQ because no GPTQ targets were selected.")
        return model

    print("Applying GPTQ …")
    sensitivity = load_or_compute_sensitivity(model, calib_inputs, args)
    qcfg = build_qwen3_vl_gptq_config(args, sensitivity=sensitivity)

    q_m = prepare(model, qcfg)

    with torch.no_grad():
        iterator = tqdm.tqdm(
            calib_inputs,
            desc="GPTQ calibration",
            disable=args.hide_progress,
        )
        for inp in iterator:
            dev_inp = move_batch_to_device(inp, args.device)
            q_m(**dev_inp)

    q_m = convert(q_m, inplace=True)
    return q_m


def evaluate_original_model(model, processor, args):
    """
    Run requested evaluations on the original floating-point model.
    """
    original_results = None

    if args.eval_tasks is not None:
        if "vqa" in args.eval_tasks:
            original_results = evaluate_model(
                model,
                processor,
                args.eval_tasks,
                args.device,
                args.nsamples_for_evaluation,
                max_seq_len=args.max_seq_len,
            )
            print_eval_results("Evaluating original model", original_results)

        if "coco" in args.eval_tasks:
            print("\n=== COCO Evaluation (Original Model) ===")
            results = evaluate_model_with_coco_score(
                model=model,
                processor=processor,
                device=args.device,
                dataset_name="coco",
                nsamples=args.nsamples_for_evaluation,
                max_seq_len=args.max_seq_len,
            )
            for metric, value in results.items():
                print(f"{metric:<10} {value:.3f}")

        if "llava_bench" in args.eval_tasks:
            print("\n=== Llava Bench Evaluation (Original Model) ===")
            results = evaluate_model_with_coco_score(
                model=model,
                processor=processor,
                device=args.device,
                dataset_name="llava_bench",
                nsamples=args.nsamples_for_evaluation,
                max_seq_len=args.max_seq_len,
            )
            for metric, value in results.items():
                print(f"{metric:<10} {value:.3f}")

    if args.mmlu_subjects is not None:
        print("\n=== MMLU Evaluation (Original Model) ===")
        original_mmlu_results = evaluate_mmlu(
            model=model,
            tokenizer=processor.tokenizer,
            subjects=args.mmlu_subjects,
            device=args.device,
            n_shots=args.mmlu_n_shots,
            n_samples=args.mmlu_n_samples,
            batch_size=args.mmlu_batch_size,
            max_seq_len=args.max_seq_len,
        )
        print_mmlu_results(original_mmlu_results)

    if args.hellaswag:
        print("\n=== HellaSwag Evaluation (Original Model) ===")
        original_hellaswag_results = evaluate_hellaswag(
            model=model,
            tokenizer=processor.tokenizer,
            device=args.device,
            n_shots=args.hellaswag_n_shots,
            n_samples=args.hellaswag_n_samples,
            batch_size=args.hellaswag_batch_size,
            max_seq_len=args.max_seq_len,
        )
        print_hellaswag_results(original_hellaswag_results)
        acc = get_hellaswag_accuracy(original_hellaswag_results)
        print(f"Accuracy: {acc['acc']:.4f}, Accuracy (norm): {acc['acc_norm']:.4f}")

    if args.mmmu_dataset is not None:
        print("\n=== MMMU Evaluation (Original Model) ===")
        original_mmmu_results = evaluate_mmmu(
            model=model,
            processor=processor,
            dataset=args.mmmu_dataset,
            subjects=args.mmmu_subjects,
            device=args.device,
            n_shots=args.mmmu_n_shots,
            n_samples=args.mmmu_n_samples,
            max_seq_len=args.max_seq_len,
            verbose=args.verbose,
        )
        print_mmmu_results(original_mmmu_results)

    if args.video_mme:
        print(
            f"\n=== Video-MME Evaluation (Original Model) (limit={args.video_mme_limit}) ==="
        )
        original_video_mme_results = evaluate_vlm_on_tasks(
            model=model,
            processor=processor,
            tasks=["videomme"],
            device=args.device,
            max_new_tokens=args.video_mme_max_new_tokens,
            limit=args.video_mme_limit,
            verbose=args.verbose,
            max_num_frames=args.video_mme_max_num_frames,
        )
        print_lmms_eval_results(original_video_mme_results)

    if args.ppl_dataset:
        print("\n=== PPL Evaluation (Original Model) ===")
        ds_ppl, _ = get_dataset(args.ppl_dataset, split=args.ppl_split, n=-1)
        original_ppl = evaluate_ppl(
            model=model,
            tokenizer=processor.tokenizer,
            ds=ds_ppl,
            device=args.device,
            stride=args.ppl_stride,
            max_seq_len=args.max_seq_len,
            show_progress=not args.hide_progress,
        )
        print(f"Original PPL: {original_ppl:.2f}")

    return original_results


def evaluate_quantized_model(model, processor, args, original_results=None) -> None:
    """
    Run requested evaluations on the quantized model.
    """
    quantized_results = None

    if args.eval_tasks is not None:
        if "vqa" in args.eval_tasks:
            quantized_results = evaluate_model(
                model,
                processor,
                args.eval_tasks,
                args.device,
                args.nsamples_for_evaluation,
                max_seq_len=args.max_seq_len,
            )
            print_eval_results("Evaluating quantized model", quantized_results)

            if original_results is not None:
                print_markdown_comparison(original_results, quantized_results)

        if "coco" in args.eval_tasks:
            print("\n=== COCO Evaluation (Quantized Model) ===")
            results = evaluate_model_with_coco_score(
                model=model,
                processor=processor,
                device=args.device,
                dataset_name="coco",
                nsamples=args.nsamples_for_evaluation,
                max_seq_len=args.max_seq_len,
            )
            for metric, value in results.items():
                print(f"{metric:<10} {value:.3f}")

        if "llava_bench" in args.eval_tasks:
            print("\n=== Llava Bench Evaluation (Quantized Model) ===")
            results = evaluate_model_with_coco_score(
                model=model,
                processor=processor,
                device=args.device,
                dataset_name="llava_bench",
                nsamples=args.nsamples_for_evaluation,
                max_seq_len=args.max_seq_len,
            )
            for metric, value in results.items():
                print(f"{metric:<10} {value:.3f}")

    if args.mmlu_subjects is not None:
        print("\n=== MMLU Evaluation (Quantized Model) ===")
        quantized_mmlu_results = evaluate_mmlu(
            model=model,
            tokenizer=processor.tokenizer,
            subjects=args.mmlu_subjects,
            device=args.device,
            n_shots=args.mmlu_n_shots,
            n_samples=args.mmlu_n_samples,
            batch_size=args.mmlu_batch_size,
            max_seq_len=args.max_seq_len,
        )
        print_mmlu_results(quantized_mmlu_results)

    if args.hellaswag:
        print("\n=== HellaSwag Evaluation (Quantized Model) ===")
        quantized_hellaswag_results = evaluate_hellaswag(
            model=model,
            tokenizer=processor.tokenizer,
            device=args.device,
            n_shots=args.hellaswag_n_shots,
            n_samples=args.hellaswag_n_samples,
            batch_size=args.hellaswag_batch_size,
            max_seq_len=args.max_seq_len,
        )
        print_hellaswag_results(quantized_hellaswag_results)
        acc = get_hellaswag_accuracy(quantized_hellaswag_results)
        print(f"Accuracy: {acc['acc']:.4f}, Accuracy (norm): {acc['acc_norm']:.4f}")

    if args.mmmu_dataset is not None:
        print("\n=== MMMU Evaluation (Quantized Model) ===")
        quantized_mmmu_results = evaluate_mmmu(
            model=model,
            processor=processor,
            dataset=args.mmmu_dataset,
            subjects=args.mmmu_subjects,
            device=args.device,
            n_shots=args.mmmu_n_shots,
            n_samples=args.mmmu_n_samples,
            max_seq_len=args.max_seq_len,
            verbose=args.verbose,
        )
        print_mmmu_results(quantized_mmmu_results)

    if args.video_mme:
        print(
            f"\n=== Video-MME Evaluation (Quantized Model) (limit={args.video_mme_limit}) ==="
        )
        quantized_video_mme_results = evaluate_vlm_on_tasks(
            model=model,
            processor=processor,
            tasks=["videomme"],
            device=args.device,
            max_new_tokens=args.video_mme_max_new_tokens,
            limit=args.video_mme_limit,
            verbose=args.verbose,
            max_num_frames=args.video_mme_max_num_frames,
        )
        print_lmms_eval_results(quantized_video_mme_results)

    if args.ppl_dataset:
        print("\n=== PPL Evaluation (Quantized Model) ===")
        ds_ppl, _ = get_dataset(args.ppl_dataset, split=args.ppl_split, n=-1)
        quantized_ppl = evaluate_ppl(
            model=model,
            tokenizer=processor.tokenizer,
            ds=ds_ppl,
            device=args.device,
            stride=args.ppl_stride,
            max_seq_len=args.max_seq_len,
            show_progress=not args.hide_progress,
        )
        print(f"Quantized PPL: {quantized_ppl:.2f}")


def load_calib_inputs(processor, args) -> List[Dict[str, Any]]:
    """
    Load calibration inputs
    Parse the nsamples_for_qcalibration argument

    Formats supported:
    - Just a number -> uses default dataset "vqav2" with N samples
    - "dataset:n_samples" -> uses default split
    - "dataset:split:n_samples" -> uses specified split
    - Multiple datasets: "vqav2:32,coco:val:32,wikitext2:test:32"
    """
    dataset_config: Dict[str, Dict[str, Any]] = {}
    raw_input = args.nsamples_for_qcalibration.strip()

    # Check if the input is just a number
    # In this case, use the default dataset "vqav2"
    if raw_input.isdigit():
        n_samples = int(raw_input)
        dataset_config["vqav2"] = {"n_samples": n_samples, "split": "testdev"}
        print(f"[info] Using default dataset 'vqav2' with {n_samples} samples")
    else:
        # Parse the full format
        for item in raw_input.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) == 2:
                # Format: "dataset:n_samples" (use default split)
                name, n_samples = parts
                dataset_config[name.strip()] = {"n_samples": int(n_samples)}
            elif len(parts) == 3:
                # Format: "dataset:split:n_samples"
                name, split, n_samples = parts
                dataset_config[name.strip()] = {
                    "n_samples": int(n_samples),
                    "split": split.strip(),
                }
            else:
                print(
                    f"[warn] Invalid format '{item}', expected 'dataset:n_samples' or 'dataset:split:n_samples'"
                )

    return get_mixed_calib_inputs(
        processor=processor,
        dataset_config=dataset_config,
        max_seq_len=args.calib_seq_len,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    ensure_spinquant_compatible_args(args)
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    use_gptq = not args.no_GPTQ and has_gptq_targets(args)
    gptq_vision = use_gptq and args.gptq_vision
    gptq_text = use_gptq and args.gptq_text
    gptq_lm_head = use_gptq and args.gptq_lm_head

    grid_thw = tuple(args.grid_thw)

    print("=== Config ===")
    print(f"Model               : {args.model}")
    print(f"Device              : {device.type}")
    print(f"DType               : {args.dtype}")
    print(f"Calib seq len       : {args.calib_seq_len}")
    print(f"Max seq len         : {args.max_seq_len}")
    print(f"Use SpinQuant       : {args.spinquant}")
    if args.spinquant:
        print(f"SpinQuant init      : {args.spinquant_init_method}")
        print(f"SpinQuant R1        : {not args.spinquant_disable_r1}")
        print(f"SpinQuant R2        : {not args.spinquant_disable_r2}")
    print(f"Use GPTQ            : {use_gptq}")
    print(f"GPTQ vision         : {gptq_vision}")
    print(f"GPTQ text           : {gptq_text}")
    print(f"GPTQ lm_head        : {gptq_lm_head}")
    print(f"Use PTQ             : {not args.no_PTQ}")
    print(f"Use SmoothQuant     : {args.smoothquant}")
    if args.smoothquant:
        print(f"SmoothQuant alpha   : {args.smoothquant_alpha}")
    print(f"grid_thw            : {grid_thw}")
    print(f"spatial_merge_size  : {args.spatial_merge_size}")
    print(f"visual_start_idx    : {args.visual_start_idx}")
    print()

    print("Loading FP model …")

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    dev_map = "auto" if args.device != "cpu" else "cpu"

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            device_map=dev_map,
        )
    except Exception:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            device_map=dev_map,
        )

    model.eval()

    if args.calib_seq_len is not None:
        model.config.text_config.max_position_embeddings = min(
            model.config.text_config.max_position_embeddings, args.calib_seq_len
        )

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        if hasattr(model.config.text_config, "use_cache"):
            model.config.text_config.use_cache = False

    original_results = evaluate_original_model(model, processor, args)

    calib_inputs = load_calib_inputs(processor, args)

    model = apply_spinquant_if_enabled(model, args)
    apply_smoothquant_if_requested(model, calib_inputs, args)

    q_m = model
    q_m = quantize_using_GPTQ(q_m, calib_inputs, args)

    if not args.no_PTQ:
        num_vision_blocks = get_num_vision_blocks(q_m)
        num_text_layers = get_num_text_layers(q_m)
        num_deepstack_mergers = get_num_deepstack_mergers(q_m)
        q_m = quantize_using_PTQ(
            q_m,
            calib_inputs,
            args,
            grid_thw=grid_thw,
            num_vision_blocks=num_vision_blocks,
            num_text_layers=num_text_layers,
            num_deepstack_mergers=num_deepstack_mergers,
        )

    if args.print_quantized_model:
        print(q_m)

    evaluate_quantized_model(q_m, processor, args, original_results=original_results)


if __name__ == "__main__":
    main()

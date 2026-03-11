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

# =============================================================================
# PTQ + GPTQ HYBRID QUANTIZATION PIPELINE
# -----------------------------------------------------------------------------
# This script shows how to:
#   1. Load a pretrained FP Llama-3 model.
#   2. Run GPTQ to quantize weights only (optional).
#   3. Wrap every Transformer layer with a PTQWrapper to quantize activations.
#   4. Calibrate activations observers in a single pass over a text corpus.
#   5. Inject GPTQ’s per-tensor weight scales / zero-points into the PTQ graph.
#   6. Freeze all Q-params and compute Wikitext-2 perplexity.
#   7. Save model/layers (optional)
# =============================================================================

import argparse

import copy
import pathlib
import random

import types

from typing import Any, List, Optional, Tuple, Union

import torch
import tqdm
from datasets import load_dataset
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer

import tico

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.metrics import perplexity
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from tico.utils.utils import SuppressWarning

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}

# Hardcoded dataset settings
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"


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


# -------------------------------------------------------------------------
# Save model/layers in circle format
# -------------------------------------------------------------------------
def save_circles_to(q_m, calib_inputs, save_circle_to_folder):
    q_m.eval()
    q_m.cpu()

    save_path = pathlib.Path(save_circle_to_folder, "model.q.circle")
    print(f"saving the whole model to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(q_m, (calib_inputs[0],), strict=False)

            cm.save(save_path)


def quantize_using_PTQ(q_m, calib_inputs, args):
    print("Wrapping layers with PTQWrapper …")

    w_cfg = {
        "mlp": {
            "gate_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "up_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "down_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
        },
        "self_attn": {
            "q_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "k_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "v_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
            "o_proj": {
                "weight": {
                    "dtype": DType.uint(args.linear_weight_bits),
                },
            },
        },
        "input_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16)},
        },
        "post_attention_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16)},
        },
    }

    cfg = PTQConfig(
        default_dtype=DType.int(16),
        default_qscheme=QScheme.PER_TENSOR_SYMM,
        wrapper_variant="prefill",
        overrides={
            "model": {
                "embed_tokens": {
                    "weight": {
                        "dtype": (
                            DType.uint(args.embedding_weight_bits)
                            if args.embedding_weight_bits < 16
                            else DType.int(args.embedding_weight_bits)
                        ),
                    },
                },
                "layers": {},
                "norm": {
                    "weight": {"dtype": DType.int(16)},
                },
            },
            "lm_head": {
                "weight": {
                    "dtype": (
                        DType.uint(args.lm_head_weight_bits)
                        if args.lm_head_weight_bits < 16
                        else DType.int(args.lm_head_weight_bits)
                    ),
                },
            },
        },
    )
    for i in range(len(q_m.model.layers)):
        child_scope = f"{i}"
        cfg.overrides["model"]["layers"][child_scope] = w_cfg  # type: ignore[index]

    qcfg = cfg
    q_m = prepare(q_m, qcfg)

    # -------------------------------------------------------------------------
    # Single-pass activation calibration
    # -------------------------------------------------------------------------
    print("Calibrating PTQ obeservers…")

    # Overwrite weight observers with GPTQ statistics
    if hasattr(q_m, "quantizers") and isinstance(q_m.quantizers, dict):
        inject_gptq_qparams(q_m, q_m.quantizers)
    elif (
        hasattr(q_m, "wrapped")
        and hasattr(q_m.wrapped, "quantizers")
        and isinstance(q_m.wrapped.quantizers, dict)
    ):
        inject_gptq_qparams(q_m.wrapped, q_m.wrapped.quantizers)
    else:
        print(
            "[Warn] q_m.quantizers not found or not a dict; skipping GPTQ qparam injection."
        )

    device = torch.device(args.device)
    with torch.no_grad():
        for inp in tqdm.tqdm(calib_inputs):
            q_m(inp.to(device))

    # Freeze all Q-params (scale, zero-point)
    q_m = convert(q_m)

    return q_m


def evaluate(q_m, tokenizer, dataset_test, args):
    # -------------------------------------------------------------------------
    # Evaluate perplexity on Wikitext-2
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_uint8 = perplexity(
        q_m, enc, args.device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ int16 : {ppl_uint8:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            q_m, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        print("Quantized RESULTS ARE:")
        print(make_table(results))


def get_dataset_for_calibration(model, dataset):
    class DataSetWithLabels(torch.utils.data.Dataset):
        def __init__(self, inputs, targets, transform=None):
            self.n_inputs = len(inputs)
            self.inputs = inputs
            self.labels = targets
            self.transform = transform

        def __len__(self):
            return self.n_inputs

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            sample = self.inputs[idx]
            if self.transform:
                sample = self.transform(sample)

            return (sample, self.labels[idx])

    targets = []

    with torch.no_grad():
        print("Computing calibration set")
        for prompt in tqdm.tqdm(dataset):
            results = model(prompt.to(model.device)).logits.detach()
            results = torch.argmax(results.detach(), dim=-1).cpu()

            targets.append(results)

    labeled_data = DataSetWithLabels(dataset, targets)
    dataloader = torch.utils.data.DataLoader(labeled_data, batch_size=1, shuffle=False)
    return dataloader


class SensitivityCalibrator:
    """
    Sensitivity calibrator - compute sensitivies using empirical Fisher information
    """

    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def compute_sensitivity_info(self):

        data_loader = get_dataset_for_calibration(self.model, self.dataset)

        dtype = self.model.dtype
        model = self.model.float()

        sensitivity = {}
        modules_to_process = {}
        name_of_module: dict[torch.nn.Linear, str] = {}

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules_to_process[name] = module
                name_of_module[module] = name
                sensitivity[name] = torch.zeros_like(module.weight).cpu()

        print("Calibrating sensitivity")
        num_of_backwards = 0
        for inputs, targets in tqdm.tqdm(data_loader):
            model.zero_grad()
            inp_ids = inputs.view(-1, inputs.shape[-1])
            logits = model(inp_ids.to(model.device)).logits

            outputs = logits.squeeze()
            targets = targets.squeeze()

            b_indices = [outputs.shape[0] - 1]  # priority to the last token
            for token_index, b_index in enumerate(b_indices):
                outputs_el = outputs[b_index : b_index + 1, :]  # noqa E203
                targets_el = targets[b_index : b_index + 1]  # noqa E203

                model.zero_grad()
                loss = torch.nn.CrossEntropyLoss()(
                    outputs_el, targets_el.to(model.device)
                )  # for Fisher this must be CrossEntropy

                # last retain_graph should be set to False to delete intermediate activations
                retain_graph = False if token_index == len(b_indices) - 1 else True

                loss.backward(retain_graph=retain_graph)

                # update second order information as current weights gradients are ready
                for name in modules_to_process:
                    cur_module = modules_to_process[name]
                    cur_grad = copy.deepcopy(cur_module.weight.grad.detach())  # type: ignore[union-attr]
                    if torch.isnan(cur_grad).any().item():
                        print("WARNING NaN detected")

                    sensitivity[name] += torch.mul(cur_grad, cur_grad).cpu()

                    cur_grad = None
                    del cur_grad

                    if model.device.type != "cpu":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                loss.detach()

                loss = None
                del loss

                num_of_backwards += 1

            del logits, outputs, targets

            if model.device.type != "cpu":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        for name in modules_to_process:
            sensitivity[name] /= num_of_backwards

        model = model.to(dtype)

        return sensitivity


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ+PTQ pipeline (weight-only + activation)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HF repo name or local path."
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        "--no-tqdm", action="store_true", help="Disable tqdm progress bars."
    )
    parser.add_argument(
        "--no_GPTQ",
        action="store_true",
        default=False,
        help="Don't use GPTQ",
    )
    parser.add_argument(
        "--no_PTQ",
        action="store_true",
        default=False,
        help="Leave model float",
    )
    parser.add_argument(
        "--save_circle_to_folder",
        type=str,
        default=None,
        help="Save embedding/lm_head/all_layers/model.model/the_whole_model to the folder specified",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache_dir for using model/datasets loading",
    )
    parser.add_argument(
        "--nsamples_for_qcalibration",
        type=int,
        default="128",  # almost standard
        help="number of samples to be used in GPTQ/PTQ calibration",
    )
    parser.add_argument(
        "--linear_weight_bits",
        type=int,
        default=4,
        help="Number of bits to be used in quantizer for matmul weight quantization",
    )
    parser.add_argument(
        "--gptq_mse",
        type=str,
        default=None,
        help="Whether and how to use mse in gptq (none/mse/smse/)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="seq_len to use in model evaluation and conversion to circle",
    )
    parser.add_argument(
        "--calibrate_seq_len",
        type=int,
        default=2048,
        help="seq_len to use in quantized model calibration. More the better",
    )
    parser.add_argument(
        "--embedding_weight_bits",
        type=int,
        default=8,
        help="Number of bits to be used to quantize input Embedding",
    )
    parser.add_argument(
        "--lm_head_weight_bits",
        type=int,
        default=4,
        help="Number of bits to be used to quantize lm_head",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help="tasks to be evaluated using lm_eval, e.g. `winogrande,arc_easy,arc_challenge,openbookqa,mmlu_pro,ifeval,bbh`",
    )
    parser.add_argument(
        "--sensitivity_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print(args)

    # Basic setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print()

    # -------------------------------------------------------------------------
    # 2. Load the FP backbone and tokenizer
    # -------------------------------------------------------------------------
    print("Loading FP model …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
        )
        .to(device)
        .eval()
    )

    model.config.use_cache = False  # TODO use args for it
    if args.calibrate_seq_len is not None:
        model.config.max_position_embeddings = min(
            model.config.max_position_embeddings, args.calibrate_seq_len
        )

    dataset_test = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT, cache_dir=args.cache_dir
    )

    print("\nCalculating original perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_fp32 = perplexity(
        model, enc, device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ FP32 : {ppl_fp32:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            model, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        print("Original RESULTS ARE:")
        print(make_table(results))

    # -------------------------------------------------------------------------
    # Prepare calibration dataset
    # -------------------------------------------------------------------------
    dataset_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
    calib_txt = " ".join(dataset_train["text"])
    train_ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(device)
    calib_inputs = []
    nsamples = args.nsamples_for_qcalibration
    seqlen = model.config.max_position_embeddings
    random.seed(args.seed)
    for _ in range(nsamples):
        i = random.randint(0, train_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = train_ids[:, i:j]
        calib_inputs.append(inp.cpu())

    # -------------------------------------------------------------------------
    # Run GPTQ (weight-only) pass
    # -------------------------------------------------------------------------
    if not args.no_GPTQ:
        if not args.no_GPTQ:
            print("Applying GPTQ …")

        sens = None
        if args.gptq_mse is not None and (
            args.gptq_mse == "smse" or args.gptq_mse == "smse_for_gptq"
        ):
            if args.sensitivity_path is not None:
                sens = torch.load(args.sensitivity_path)
            else:
                calibrator = SensitivityCalibrator(model, tokenizer, calib_inputs)
                sens = calibrator.compute_sensitivity_info()

        gptq_config = GPTQConfig(
            weight_bits=args.linear_weight_bits,
            perchannel=True,
            mse=args.gptq_mse,
            sensitivity=sens,
        )
        q_m = prepare(model, gptq_config, inplace=True)
        with torch.no_grad():
            for inp in calib_inputs:
                q_m(inp.to(args.device))

        q_m = convert(q_m, inplace=True)  # materialize INT-weight tensors
    else:
        q_m = model

    # -------------------------------------------------------------------------
    # Wrap every layer with PTQWrapper
    # -------------------------------------------------------------------------
    if not args.no_PTQ:
        q_m = quantize_using_PTQ(q_m, calib_inputs, args)

    # after PTQ quantizer only fixed-length input sequences are valid
    evaluate(q_m, tokenizer, dataset_test, args)

    if args.save_circle_to_folder is not None:
        calib_inputs = list(torch.stack(calib_inputs).reshape(-1, 1, args.max_seq_len))
        save_circles_to(q_m, calib_inputs, args.save_circle_to_folder)


if __name__ == "__main__":
    main()

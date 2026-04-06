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
import pathlib
import random
from typing import Any

import torch
import tqdm
from datasets import load_dataset
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer

import tico
from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.builders import build_llm_ptq_config
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.spinquant import SpinQuantConfig
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
# Save model in circle format
# -------------------------------------------------------------------------
def save_model_to(q_m, calib_inputs, save_circle_to_folder):
    q_m.eval()
    q_m.cpu()

    save_path = pathlib.Path(save_circle_to_folder, "model.q.circle")
    print(f"saving the whole model to {save_path.resolve()}")
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(q_m, (calib_inputs[0],), strict=False)

            cm.save(save_path)


def save_layers_to(q_m, max_seq_len, save_layers_to_folder):
    q_m.eval()
    q_m.cpu()

    if not hasattr(q_m, "wrapped"):
        print("Saving layers currently is supported only for PTQ quantized model")
        return

    layers = q_m.wrapped.model.wrapped.layers
    config = q_m.wrapped.config
    for i, qlayer in enumerate(layers):
        save_path = pathlib.Path(save_layers_to_folder, f"decoder_layer_{i}.q.circle")
        B, S, D = 1, max_seq_len, config.hidden_size
        example_hidden = torch.randn(B, S, D)

        attention_mask = qlayer.wrapped._slice_causal(S, "cpu").squeeze(0)
        dtype = example_hidden.dtype
        pos_embeds = qlayer.wrapped._slice_rope(S, "cpu", dtype)

        print(f"Saving model layer_{i} to {save_path.resolve()}")
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                cm = tico.convert(
                    qlayer,
                    (example_hidden,),
                    kwargs={
                        "attention_mask": attention_mask,
                        "position_embeddings": pos_embeds,
                    },
                )
        cm.save(save_path)


def quantize_using_PTQ(q_m, calib_inputs, args):
    print("Wrapping layers with PTQWrapper …")

    qcfg = build_llm_ptq_config(
        model_type="llama",
        num_hidden_layers=len(q_m.model.layers),
        wrapper_variant="prefill",
        activation_dtype=DType.int(16),
        default_qscheme=QScheme.PER_TENSOR_SYMM,
        linear_weight_bits=args.linear_weight_bits,
        embedding_weight_bits=args.embedding_weight_bits,
        lm_head_weight_bits=args.lm_head_weight_bits,
        norm_weight_dtype=DType.int(16),
        strict_wrap=True,
    )
    q_m = prepare(q_m, qcfg)

    # -------------------------------------------------------------------------
    # Single-pass activation calibration
    # -------------------------------------------------------------------------
    print("Calibrating PTQ observers…")

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


def evaluate(q_m, tokenizer, dataset_test, args, quantized=False):
    # -------------------------------------------------------------------------
    # Evaluate perplexity on Wikitext-2
    # -------------------------------------------------------------------------
    print("\nCalculating perplexities …")
    enc = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    ppl_uint8 = perplexity(
        q_m, enc, args.device, max_length=args.max_seq_len, stride=args.max_seq_len
    )

    ppl_info_str = "int16" if quantized is True else "FP32"
    print("\n┌── Wikitext-2 test perplexity ─────────────")
    print(f"│ {ppl_info_str} : {ppl_uint8:8.2f}")
    print("└───────────────────────────────────────────")

    if args.eval_tasks is not None:
        results = evaluate_llm_on_tasks(
            q_m, tokenizer, args.eval_tasks, max_length=args.max_seq_len
        )
        acc_info_str = "Quantized" if quantized is True else "Original"
        print(f"{acc_info_str} RESULTS ARE:")
        print(make_table(results))


class QModelProcessor:
    """Processor for quantization model handling GPTQ and PTQ steps.

    Attributes:
        model: The underlying FP model.
        tokenizer: Tokenizer associated with the model.
        device: Torch device derived from args.
        args: Parsed command-line arguments.
    """

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(args.device)
        self.args = args

    def get_tokenized_inputs(self, dataset, shuffle=True):
        """Generate tokenized inputs for calibration.

        Concatenates all text from the provided dataset, tokenizes it using the stored
        tokenizer, and extracts ``nsamples`` slices of length ``max_position_embeddings``.
        If ``shuffle`` is True, slices are selected randomly; otherwise they are taken
        sequentially with a fixed stride.

        Args:
            dataset: The dataset object containing a ``text`` field.
            shuffle (bool): Whether to randomly sample slices.

        Returns:
            List[torch.Tensor]: A list of tokenized input tensors.
        """
        text = " ".join(dataset["text"])
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        tokenized_inputs = []
        nsamples = self.args.nsamples_for_qcalibration
        seqlen = self.model.config.max_position_embeddings
        if shuffle is True:
            random.seed(self.args.seed)
        else:
            stride = (ids.shape[1] - seqlen - 1) // seqlen
        for index in range(nsamples):
            if shuffle is True:
                i = random.randint(0, ids.shape[1] - seqlen - 1)
            else:
                i = index * stride
            j = i + seqlen
            inp = ids[:, i:j]
            tokenized_inputs.append(inp.cpu())
        return tokenized_inputs

    def run_gptq(self, calib_inputs):
        """Run GPTQ weight‑only quantization on the model.

        Args:
            calib_inputs (List[torch.Tensor]): Calibration inputs used for
                GPTQ weight quantization.

        Returns:
            torch.nn.Module: The quantized model with INT‑weight tensors.
        """
        print("Applying GPTQ …")

        sens = None
        if self.args.gptq_mse is not None and self.args.gptq_mse == "smse":
            if self.args.sensitivity_path is not None:
                sens = torch.load(self.args.sensitivity_path)
            else:
                calibrator = SensitivityCalibrator(self.model, calib_inputs)
                sens = calibrator.compute_sensitivity_info()

        gptq_config = GPTQConfig(
            weight_bits=self.args.linear_weight_bits,
            perchannel=True,
            mse=self.args.gptq_mse,
            sensitivity=sens,
        )
        q_m = prepare(self.model, gptq_config, inplace=True)
        with torch.no_grad():
            for inp in calib_inputs:
                q_m(inp.to(self.device))

        q_m = convert(q_m, inplace=True)  # materialize INT-weight tensors
        return q_m

    def evaluate_original(self, dataset_test):
        """Evaluate the original (FP) model on the test dataset.

        Args:
            dataset_test: The test split of the dataset.

        Returns:
            Any: The result of the ``evaluate`` helper (typically prints metrics).
        """
        return evaluate(
            self.model,
            self.tokenizer,
            dataset_test,
            self.args,
            quantized=False,
        )

    def evaluate_quantized(self, dataset_test):
        """Placeholder for evaluating the quantized model.

        This method should be overridden in subclasses to implement evaluation of the
        quantized model.

        Args:
            dataset_test: The test dataset.

        Raises:
            NotImplementedError: Indicates the method is not implemented in the base class.
        """
        raise NotImplementedError

    def save_quantized(self, model, calib_inputs):
        """Placeholder for saving the quantized model.

        Subclasses should implement saving logic for the quantized model or its layers.

        Args:
            model: The quantized model instance.
            calib_inputs: Calibration inputs used for possible model saving.

        Raises:
            NotImplementedError: Indicates the method is not implemented in the base class.
        """
        raise NotImplementedError


class PrefillQModelProcessor(QModelProcessor):
    """
    PrefillQModelProcessor extends QModelProcessor for models that operate in
    a simple prefill mode without KV cache.

    It provides implementations for PTQ quantization, evaluation of the
    quantized model, and optional saving of layers or the whole model.
    """

    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)

    def run_ptq(self, q_m, calib_inputs):
        """Run PTQ activation quantization on the model.

        Args:
            q_m: The model (potentially already weight‑quantized) to wrap with PTQ.
            calib_inputs: Calibration inputs for activation observers.

        Returns:
            torch.nn.Module: The PTQ‑wrapped and calibrated model.
        """
        return quantize_using_PTQ(q_m, calib_inputs, self.args)

    def evaluate_quantized(self, model, dataset_test):
        """Evaluate the quantized model on the test dataset.

        Args:
            model: The quantized model instance.
            dataset_test: The test split of the dataset.

        Returns:
            None. The function prints evaluation metrics.
        """
        evaluate(model, self.tokenizer, dataset_test, self.args, quantized=True)

    def save_quantized(self, model, calib_inputs):
        """Save the quantized model and its components.

        This method respects the command‑line arguments ``save_layers_to_folder``
        and ``save_circle_to_folder``. If a folder is provided, the corresponding
        helper is invoked to persist either individual layers and/or the whole model.

        Args:
            model: The quantized model instance.
            calib_inputs: Calibration inputs required when saving the full model
                (used to construct a dummy batch for conversion to Circle format).
        """
        if self.args.save_layers_to_folder is not None:
            save_layers_to(
                model, self.args.max_seq_len, self.args.save_layers_to_folder
            )

        if self.args.save_circle_to_folder is not None:
            calib_inputs = list(
                torch.stack(calib_inputs).reshape(-1, 1, self.args.max_seq_len)
            )
            save_model_to(model, calib_inputs, self.args.save_circle_to_folder)


def get_qmodel_processor(model, tokenizer, args):
    # TODO add more processors

    return PrefillQModelProcessor(model, tokenizer, args)


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
        "--no_spinquant",
        action="store_true",
        default=False,
        help="Disable SpinQuant preprocessing.",
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
        help="Save the whole model to the folder specified",
    )
    parser.add_argument(
        "--save_layers_to_folder",
        type=str,
        default=None,
        help="Save all layers to the folder specified",
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
        choices=["mse", "smse"],
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
    dev_map = "balanced" if args.device != "cpu" else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        legacy=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        device_map=dev_map,
    ).eval()

    if not args.no_spinquant:
        print("Applying SpinQuant preprocessing …")
        model = prepare(model, SpinQuantConfig())
        model = convert(model)
    else:
        print("Skipping SpinQuant preprocessing …")

    model.config.use_cache = False  # TODO use args for it
    if args.calibrate_seq_len is not None:
        model.config.max_position_embeddings = min(
            model.config.max_position_embeddings, args.calibrate_seq_len
        )

    dataset_test = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=TEST_SPLIT, cache_dir=args.cache_dir
    )

    # -------------------------------------------------------------------------
    # Create a processor for the model
    # -------------------------------------------------------------------------
    qmodel_processor = get_qmodel_processor(model, tokenizer, args)

    # -------------------------------------------------------------------------
    # Compute original metrics to estimate metrics degradation
    # -------------------------------------------------------------------------
    qmodel_processor.evaluate_original(dataset_test)

    # -------------------------------------------------------------------------
    # Prepare calibration dataset
    # -------------------------------------------------------------------------
    dataset_train = load_dataset(DATASET_NAME, DATASET_CONFIG, split=TRAIN_SPLIT)
    calib_inputs = qmodel_processor.get_tokenized_inputs(dataset_train, shuffle=True)

    # -------------------------------------------------------------------------
    # Run GPTQ (weight-only) pass
    # -------------------------------------------------------------------------
    if not args.no_GPTQ:
        q_m = qmodel_processor.run_gptq(calib_inputs)
    else:
        q_m = model

    # -------------------------------------------------------------------------
    # Wrap every layer with PTQWrapper
    # -------------------------------------------------------------------------
    if not args.no_PTQ:
        q_m = qmodel_processor.run_ptq(q_m, calib_inputs)

    # -------------------------------------------------------------------------
    # Compute quantized model metrics to estimate metrics degradation
    # -------------------------------------------------------------------------
    qmodel_processor.evaluate_quantized(q_m, dataset_test)

    # -------------------------------------------------------------------------
    # Save layers and model
    # -------------------------------------------------------------------------
    qmodel_processor.save_quantized(q_m, calib_inputs)


if __name__ == "__main__":
    main()

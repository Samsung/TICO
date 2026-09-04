# Copyright (c) 2024 Intel Corporation
# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import copy
import functools
import multiprocessing
import types
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm.auto import tqdm

from tico.quantization.algorithm.gptq.gptq import GPTQ
from tico.quantization.algorithm.gptq.utils import (
    find_layers,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
)
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer
from tico.utils.utils import move_to_device


class FPInputsCache:
    """
    Class for saving full-precision output in each layer (GPTQv2).
    """

    def __init__(self, sequential):
        self.fp_cache = {}
        self.names = tuple(name for names in sequential for name in names)
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        self.fp_cache[name] += [inp.cpu()]

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []


def move_to_cpu(obj):
    return move_to_device(obj, "cpu")


def _format_elapsed_dhms(seconds: float) -> str:
    """Format elapsed seconds as DD:HH:MM:SS, dropping leading zero components.

    Always shows at least MM:SS.

    Examples:
        59    -> "00:59"
        297   -> "04:57"
        3601  -> "01:00:01"
        90061 -> "01:01:01:01"
    """
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = [days, hours, minutes, seconds]
    formatted = [f"{p:02d}" for p in parts]
    # Find first non-zero component (default to 2 so we always show MM:SS)
    first_nonzero = 2
    for i, p in enumerate(parts):
        if p > 0:
            first_nonzero = i
            break
    # Always keep at least MM:SS (indices 2 and 3)
    start = min(first_nonzero, 2)
    return ":".join(formatted[start:])



class StopForward(Exception):
    """Custom exception used to stop the forward pass after the first layer."""

    pass


def _gptq_layer_worker_gpu(worker_args: dict) -> dict:
    """Module-level worker function for parallel GPTQ layer quantization on GPU.

    This function runs in a separate process (spawned by multiprocessing.Pool).
    It reconstructs a decoder layer from its state_dict, collects Hessian
    statistics using pre-computed layer inputs, and applies GPTQ quantization.

    Args:
        worker_args: Dictionary containing:
            - l_idx: Layer index
            - layer_state_dict: Layer state_dict (CPU tensors)
            - layer_class_name: Full class path for layer reconstruction
            - layer_inputs: List of pre-computed layer input args per batch (CPU)
            - layer_kwargs: List of pre-computed layer kwargs per batch (CPU)
            - gptq_conf_dict: GPTQ config as plain dict
            - module_names: List of submodule full names
            - sample_weights: Optional sample weights for Hessian
            - sensitivity_data: Optional sensitivity dict (CPU tensors)
            - fp_inputs: Optional FP inputs per submodule (GPTQv2, CPU tensors)

    Returns:
        Dictionary with:
            - l_idx: Layer index
            - quantized_weights: {submodule_name: CPU tensor}
            - quantizer_params: {submodule_name: (scale_cpu, zero_cpu)}
    """
    import torch
    from tico.quantization.algorithm.gptq.gptq import GPTQ
    from tico.quantization.algorithm.gptq.quant import Quantizer

    l_idx = worker_args["l_idx"]
    layer = worker_args["target_layer"]
    layer_inputs = worker_args["layer_inputs"]
    layer_kwargs = worker_args["layer_kwargs"]
    gptq_conf_dict = worker_args["gptq_conf_dict"]
    module_names = worker_args["module_names"]
    sample_weights = worker_args.get("sample_weights", None)
    sensitivity_data = worker_args.get("sensitivity_data", None)
    fp_inputs = worker_args.get("fp_inputs", None)
    device = worker_args.get("device", "cuda")
    # Set the CUDA device for this worker process so that .to(device)
    # uses the correct GPU when multiple GPUs are available.
    if torch.cuda.is_available() and "cuda" in str(device):
        gpu_id = int(str(device).split(":")[1]) if ":" in str(device) else 0
        torch.cuda.set_device(gpu_id)

    # Move the received layer to the target device
    layer = layer.to(device)
    layer.eval()

    # Find quantizable submodules
    full = find_layers(
        layer,
        layers=[
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.Conv1d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
        ],
    )



    # Use sequential_groups if provided, otherwise process all at once
    sequential_groups = worker_args.get("sequential_groups", None)
    if sequential_groups is not None:
        # Filter groups to only include names that exist in the reconstructed layer
        existing_names = set(full.keys())
        sequential = []
        for names in sequential_groups:
            cur_seq = [name for name in names if name in existing_names]
            if cur_seq:
                sequential.append(cur_seq)
    else:
        sequential = [list(full.keys())]

    # Set up GPTQ objects and gather stats
    quantized_weights = {}
    quantizer_params = {}

    for names in sequential:
        subset = {n: full[n] for n in names}

        gptq: Dict[str, GPTQ] = {}
        for name in subset:
            full_module_name = module_names.get(name, name)
            gptq[name] = GPTQ(
                subset[name],
                double_precision=gptq_conf_dict.get("double_precision", False),
                layer_name=full_module_name,
            )
            gptq[name].saturation_threshold = gptq_conf_dict.get("saturation_threshold", None)
            gptq[name].saturation_min_batches = gptq_conf_dict.get("saturation_min_batches", 4)
            # Resolve weight bits
            weight_bits = gptq_conf_dict.get("weight_bits", 8)
            weight_bits_overrides = gptq_conf_dict.get("weight_bits_overrides", {})
            if full_module_name in weight_bits_overrides:
                weight_bits = weight_bits_overrides[full_module_name]
            elif name in weight_bits_overrides:
                weight_bits = weight_bits_overrides[name]
            else:
                suffix_matches = [
                    bits
                    for pattern, bits in weight_bits_overrides.items()
                    if full_module_name.endswith(f".{pattern}")
                ]
                if suffix_matches:
                    weight_bits = suffix_matches[-1]

            # Resolve sensitivity
            cur_sensitivity = None
            if sensitivity_data is not None and full_module_name in sensitivity_data:
                cur_sensitivity = sensitivity_data[full_module_name]

            gptq[name].quantizer.configure(
                bits=weight_bits,
                perchannel=gptq_conf_dict.get("perchannel", True),
                sym=gptq_conf_dict.get("symmetric", False),
                mse=gptq_conf_dict.get("mse", None),
                sensitivity=cur_sensitivity,
                mse_tolerance=gptq_conf_dict.get("mse_tolerance", 1e-2),
                chunk_size=gptq_conf_dict.get("chunk_size", 64),
                use_batched_gptq=gptq_conf_dict.get("use_batched_gptq", True),
            )

            # GPTQv2: Assign native_inp
            if fp_inputs is not None and name in fp_inputs:
                gptq[name].native_inp = fp_inputs[name]

        # Set weights and reset batch_id
        for name in subset:
            gptq[name].weights = sample_weights
            gptq[name].batch_id = 0

        # Register hooks
        def add_batch(name):
            def _hook(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return _hook

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Run layer forward over all batches to collect Hessian
        batch_num = len(layer_inputs)
        for batch_idx in range(batch_num):
            cache_args_batch = layer_inputs[batch_idx]
            cache_args_batch = move_to_device(cache_args_batch, device)
            cache_kwargs_batch = layer_kwargs[batch_idx]
            cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)

            if gptq_conf_dict.get("double_precision", False):
                # Cast to double for batch-size-independence
                for pname, param in layer.named_parameters():
                    param.data = param.data.double()
                args_d = GPTQQuantizer._cast_to_double(cache_args_batch)
                kwargs_d = GPTQQuantizer._cast_to_double(cache_kwargs_batch)
                layer(*args_d, **kwargs_d)
                for pname, param in layer.named_parameters():
                    param.data = param.data.float()
            else:
                layer(*cache_args_batch, **cache_kwargs_batch)

        # Remove hooks
        for h in handles:
            h.remove()

        # Quantize each submodule
        for name in subset:
            full_module_name = module_names.get(name, name)

            if gptq_conf_dict.get("verbose", False):
                print(f"[Layer {l_idx}] {name} -> Quantizing (parallel) ...")

            gptq[name].fasterquant(
                percdamp=gptq_conf_dict.get("percdamp", 0.01),
                groupsize=gptq_conf_dict.get("groupsize", -1),
                actorder=gptq_conf_dict.get("actorder", True),
                static_groups=gptq_conf_dict.get("static_groups", False),
                verbose=gptq_conf_dict.get("verbose", False),
                adaptive_percdamp=gptq_conf_dict.get("adaptive_percdamp", False),
                cond_threshold_good=gptq_conf_dict.get("cond_threshold_good", 100000.0),
                use_iterate=gptq_conf_dict.get("use_iterate", False),
                actorder_precision=gptq_conf_dict.get("actorder_precision", 1e-2),
            )

            # Collect quantized weights and quantizer params
            quantized_weights[name] = subset[name].weight.data.cpu().clone()
            quantizer_params[name] = (
                gptq[name].quantizer.scale.cpu().clone(),
                gptq[name].quantizer.zero.cpu().clone(),
            )
            gptq[name].free()

        # If processing subgroups sequentially, re-run the layer forward
        # (without hooks) so that the next subgroup's Hessian sees the
        # effect of this subgroup's quantized weights.
        if len(sequential) > 1:
            for batch_idx in range(batch_num):
                cache_args_batch = layer_inputs[batch_idx]
                cache_args_batch = move_to_device(cache_args_batch, device)
                cache_kwargs_batch = layer_kwargs[batch_idx]
                cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)

                if gptq_conf_dict.get("double_precision", False):
                    for pname, param in layer.named_parameters():
                        param.data = param.data.double()
                    args_d = GPTQQuantizer._cast_to_double(cache_args_batch)
                    kwargs_d = GPTQQuantizer._cast_to_double(cache_kwargs_batch)
                    layer(*args_d, **kwargs_d)
                    for pname, param in layer.named_parameters():
                        param.data = param.data.float()
                else:
                    layer(*cache_args_batch, **cache_kwargs_batch)


    # Clean up GPU memory
    layer.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "l_idx": l_idx,
        "quantized_weights": quantized_weights,
        "quantizer_params": quantizer_params,
    }



@register_quantizer(GPTQConfig)
class GPTQQuantizer(BaseQuantizer):
    """
    Quantizer for applying the GPTQ algorithm (typically for weight quantization).
    This implementation expects:
        1) prepare(model, ...) to only attach hooks/Catchers and NOT run the model internally.
        2) The user runs the model with arbitrary number of batches to collect calibration data.
        3) convert(model) to consume the collected data and apply GPTQ.
    """

    def __init__(self, config: GPTQConfig):
        super().__init__(config)

        # cache_args[i] -> list of the i-th positional argument for each batch
        self.cache_args: List[List[Any]] = []
        # cache_kwargs[k] -> list of the value for keyword k for each batch
        self.cache_kwargs: Dict[str, List[Any]] = {}
        self.num_batches: int = 0
        # sample_weights for weighted Hessian accumulation (default: uniform)
        self.sample_weights: Optional[List[float]] = None

        # References to original forwards for restoration
        self._orig_model_forward: Optional[Callable[..., Any]] = None
        self._orig_layer_forward: Optional[Callable[..., Any]] = None
        self._first_layer_ref: Optional[torch.nn.Module] = None

        # Reference to original model for use_orig_model_inference and GPTQv2
        self.orig_model: Optional[torch.nn.Module] = None

    @staticmethod
    def _cast_to_double(obj):
        """Recursively cast all tensors in a list/dict/tensor to float64."""
        if isinstance(obj, torch.Tensor):
            return obj.double()
        if isinstance(obj, (list, tuple)):
            return type(obj)(GPTQQuantizer._cast_to_double(o) for o in obj)
        if isinstance(obj, dict):
            return {k: GPTQQuantizer._cast_to_double(v) for k, v in obj.items()}
        return obj

    def _run_layer_forward_double_precision(
        self,
        layer: torch.nn.Module,
        args: List[Any],
        kwargs: Dict[str, Any],
        double_precision: bool,
    ):
        """Run a layer forward, optionally in float64 for batch-size-independence.

        When double_precision=True, temporarily casts the layer and its inputs to float64
        so that GPU matmul tiling differences don't produce different results
        for different batch sizes. Outputs are kept in float64 to avoid losing
        precision when they become the next layer's input.
        """
        if not double_precision:
            return layer(*args, **kwargs)

        # Save original dtypes and cast layer to double
        orig_dtypes: Dict[str, torch.dtype] = {}
        for name, param in layer.named_parameters():
            orig_dtypes[name] = param.dtype
            param.data = param.data.double()
        for name, buf in layer.named_buffers():
            orig_dtypes[f"buf:{name}"] = buf.dtype
            buf.data = buf.data.double()

        # Cast inputs to double
        args_d = self._cast_to_double(args)
        kwargs_d = self._cast_to_double(kwargs)

        outs = layer(*args_d, **kwargs_d)

        # Keep outputs in float64 — casting to float32 would introduce
        # batch-size-dependent rounding when these outputs become the
        # next layer's input, defeating the purpose of double precision.
        if isinstance(outs, tuple):
            outs = outs[0]

        # Restore original dtypes
        for name, param in layer.named_parameters():
            param.data = param.data.to(orig_dtypes[name])
        for name, buf in layer.named_buffers():
            buf.data = buf.data.to(orig_dtypes[f"buf:{name}"])

        return outs

    def _resolve_weight_bits(
        self,
        gptq_conf: GPTQConfig,
        *,
        full_module_name: str,
        local_module_name: str,
    ) -> int:
        """Resolve the effective bit-width for a quantized submodule."""
        if full_module_name in gptq_conf.weight_bits_overrides:
            return gptq_conf.weight_bits_overrides[full_module_name]

        if local_module_name in gptq_conf.weight_bits_overrides:
            return gptq_conf.weight_bits_overrides[local_module_name]

        suffix_matches = [
            bits
            for pattern, bits in gptq_conf.weight_bits_overrides.items()
            if full_module_name.endswith(f".{pattern}")
        ]

        if suffix_matches:
            return suffix_matches[-1]

        return gptq_conf.weight_bits

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Overrides the forward method of the first LLaMA layer (layer 0) to capture the
         input required for calibration.

        When the user calls `model(...)`, we intercept (and store) the inputs to that
         layer, then raise an exception to stop the forward pass immediately. These
        captured inputs are then utilized to calibrate the quantization parameters
         for the GPTQ.

        Parameters:
            model (torch.nn.Module): The target PyTorch model
            args (Any, optional): Unused (kept for API compatibility)
            kwargs (Dict[str, Any], optional): Unused (kept for API compatibility)

        Returns:
            torch.nn.Module: The model with the catcher attached
        """
        # Define the catcher to store inputs/kwargs and stop the execution
        def forward(layer, *args, **kwargs):
            """
            Stores this batch's inputs and kwargs, then raises StopForward to stop computation.
            """
            # Store positional args
            for idx, item in enumerate(args):
                if (idx + 1) > len(self.cache_args):
                    self.cache_args.append([])
                self.cache_args[idx].append(move_to_cpu(item))
            # Store keyword args
            for k, v in kwargs.items():
                if self.cache_kwargs.get(k, None) is None:
                    self.cache_kwargs[k] = []
                self.cache_kwargs[k].append(move_to_cpu(v))

            self.num_batches += 1
            raise StopForward  # stop after the first layer

        gptq_conf = self.config
        assert isinstance(gptq_conf, GPTQConfig)
        if gptq_conf.use_orig_model_inference is True or gptq_conf.gptq_v2:
            device = next(model.parameters()).device
            model = model.cpu()
            self.orig_model = copy.deepcopy(model)
            model = model.to(device)
        else:
            self.orig_model = None
        # Replace the first layer with defined function to capture calibration data.
        if hasattr(model, "model"):
            if hasattr(model.model, "layers") and isinstance(
                model.model.layers, torch.nn.ModuleList
            ):
                self._first_layer_ref = model.model.layers[0]
            else:
                self._first_layer_ref = (
                    model  # let's treat it as a single layer (fallback)
                )
        else:
            # fallback if the model is not LLaMA-like; treat whole model as single layer
            self._first_layer_ref = model

        assert hasattr(self._first_layer_ref, "forward")
        # Backup the original forward of the first layer
        assert isinstance(self._first_layer_ref, torch.nn.Module)
        self._orig_layer_forward = self._first_layer_ref.forward
        self._first_layer_ref.forward = types.MethodType(forward, self._first_layer_ref)

        def model_forward_wrapper(_model, *m_args, **m_kwargs):
            """
            Wrapper to ignore StopForward exceptions so the user's training loop doesn't crash.
            """
            try:
                assert self._orig_model_forward is not None
                return self._orig_model_forward(*m_args, **m_kwargs)
            except StopForward:
                # We stopped after the first layer; return None or dummy output if needed.
                return None

        # Backup model.forward so we can suppress StopForward
        self._orig_model_forward = model.forward
        model.forward = types.MethodType(model_forward_wrapper, model)

        # Disable use_cache during calibration
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            self.orig_use_cache = model.config.use_cache
            model.config.use_cache = False
        else:
            self.orig_use_cache = None

        return model

    @torch.no_grad()
    def convert(self, model):
        """
        Perform GPTQ quantization using cached first-layer inputs.

        Steps:
          1) Restore original forwards (no more catching).
          2) Iterate through each Transformer layer sequentially:
             a) For each layer, register forward hooks to collect (inp, out) stats for GPTQ.
             b) Run the layer on cached inputs for all batches.
             c) Apply GPTQ and update the weights.
             d) Re-run the layer to produce outputs for the next layer; update cached inputs.
          3) Optionally apply GPTQ to lm_head when configured.
          4) Restore model.config.use_cache if needed and clear internal caches.

        Parameters:
            model (torch.nn.Module): The prepared model.

        Returns:
            torch.nn.Module: Quantized model.
        """
        # Restore original forwards (we no longer want to stop after first layer)
        assert self._orig_model_forward is not None
        model.forward = self._orig_model_forward
        assert (
            self._first_layer_ref is not None and self._orig_layer_forward is not None
        )
        self._first_layer_ref.forward = self._orig_layer_forward

        gptq_conf = self.config
        assert isinstance(gptq_conf, GPTQConfig)
        gptq_conf.validate()
        
        # Set sample_weights from config for weighted Hessian accumulation
        self.sample_weights = getattr(gptq_conf, 'sample_weights', None)

        # Identify layers
        orig_layers = None
        if hasattr(model, "model"):
            if hasattr(model.model, "layers"):
                target_layers = model.model.layers
                if self.orig_model is not None:
                    orig_layers = self.orig_model.model.layers
            else:
                target_layers = [model]
        else:
            target_layers = [model]

        module_name = {}
        for name, module in model.named_modules():
            module_name[module] = name

        quantizers: Dict[str, Any] = {}
        
        # GPTQv2: Collect FP inputs from original model before quantization
        need_float_inference = gptq_conf.gptq_v2
        fp_inps = None
        if need_float_inference and orig_layers is not None:
            fp_inps = copy.deepcopy(self.cache_args)

        # ---- Parallel path: when parallel_workers > 1 ----
        if gptq_conf.parallel_workers > 1:
            assert orig_layers is not None, (
                "parallel_workers > 0 requires use_orig_model_inference=True "
                "which creates orig_model"
            )
            model = self._convert_parallel(
                model, target_layers, orig_layers, module_name, quantizers,
                gptq_conf, fp_inps
            )
            return self.finalize(model, quantizers)

        need_next_orig_layer_inference = gptq_conf.use_orig_model_inference
        
        # ---- Sequential path (original, runs when parallel_workers <= 1) ----
        for l_idx, layer in enumerate(
            tqdm(
                target_layers,
                desc="Quantizing layers",
                unit="layer",
                disable=not gptq_conf.show_progress,
            )
        ):
            # 1) Identify quantizable submodules within the layer
            full = find_layers(
                layer,
                layers=[
                    torch.nn.Linear,
                    torch.nn.Conv2d,
                    torch.nn.Conv1d,
                    torch.nn.Conv3d,
                    torch.nn.ConvTranspose2d,
                ],
            )
            sequential = [list(full.keys())]

            # GPTQv2: Set up FPInputsCache for collecting FP inputs per submodule
            fp_inputs_cache = None
            if need_float_inference and orig_layers is not None:
                fp_inputs_cache = FPInputsCache(sequential)
                orig_full = find_layers(
                    orig_layers[l_idx],
                    layers=[
                        torch.nn.Linear,
                        torch.nn.Conv2d,
                        torch.nn.Conv1d,
                        torch.nn.Conv3d,
                        torch.nn.ConvTranspose2d,
                    ],
                )
                fp_inputs_cache.add_hook(orig_full)
                device = next(model.parameters()).device
                batch_num = self.num_batches
                for batch_idx in range(batch_num):
                    cache_args_batch = gather_single_batch_from_list(fp_inps, batch_idx)
                    cache_args_batch = move_to_device(cache_args_batch, device)
                    cache_kwargs_batch = gather_single_batch_from_dict(
                        self.cache_kwargs, batch_idx
                    )
                    cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)
                    
                    orig_layer = orig_layers[l_idx].to(device)
                    if gptq_conf.double_precision:
                        self._run_layer_forward_double_precision(
                            orig_layer, cache_args_batch, cache_kwargs_batch, True
                        )
                    else:
                        orig_layer(*cache_args_batch, **cache_kwargs_batch)
                    orig_layer.cpu()
                    
                fp_inputs_cache.clear_hook()

            # 2) Set up GPTQ objects and gather stats
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq: Dict[str, GPTQ] = {}
                for name in subset:
                    full_module_name = module_name[subset[name]]
                    gptq[name] = GPTQ(subset[name], double_precision=gptq_conf.double_precision, layer_name=full_module_name)
                    gptq[name].saturation_threshold = gptq_conf.saturation_threshold
                    gptq[name].saturation_min_batches = gptq_conf.saturation_min_batches
                    weight_bits = self._resolve_weight_bits(
                        gptq_conf,
                        full_module_name=full_module_name,
                        local_module_name=name,
                    )
                    if (
                        gptq_conf.sensitivity is not None
                        and isinstance(gptq_conf.sensitivity, dict)
                        and full_module_name in gptq_conf.sensitivity
                    ):
                        cur_sensitivity = gptq_conf.sensitivity[full_module_name]
                    else:
                        cur_sensitivity = None
                    gptq[name].quantizer.configure(
                        bits=weight_bits,
                        perchannel=gptq_conf.perchannel,
                        sym=gptq_conf.symmetric,
                        mse=gptq_conf.mse,
                        sensitivity=cur_sensitivity,
                        mse_tolerance=gptq_conf.mse_tolerance,
                        chunk_size=gptq_conf.chunk_size,
                        use_batched_gptq=gptq_conf.use_batched_gptq,
                    )

                    # GPTQv2: Assign native_inp from FPInputsCache
                    if fp_inputs_cache is not None and name in fp_inputs_cache.fp_cache:
                        gptq[name].native_inp = fp_inputs_cache.fp_cache[name]

                # Hook to collect (inp, out) for GPTQ with optional weights
                # Set weights on each GPTQ instance and reset batch_id
                batch_weights = self.sample_weights
                for name in subset:
                    gptq[name].weights = batch_weights
                    gptq[name].batch_id = 0  # Reset batch_id before collecting
                
                def add_batch(name):
                    def _hook(_, inp, out):
                        # GPTQ instance internally tracks batch_id and uses its weights
                        gptq[name].add_batch(inp[0].data, out.data)

                    return _hook

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

                # Run layer forward over all cached batches to build Hessian/statistics
                batch_num = self.num_batches
                device = next(model.parameters()).device
                for batch_idx in tqdm(
                    range(batch_num),
                    desc=f"[L{l_idx}] collecting",
                    leave=False,
                    unit="batch",
                    disable=not gptq_conf.show_progress,
                ):
                    cache_args_batch = gather_single_batch_from_list(
                        self.cache_args, batch_idx
                    )
                    cache_args_batch = move_to_device(cache_args_batch, device)

                    cache_kwargs_batch = gather_single_batch_from_dict(
                        self.cache_kwargs, batch_idx
                    )
                    cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)

                    if gptq_conf.double_precision:
                        self._run_layer_forward_double_precision(
                            layer, cache_args_batch, cache_kwargs_batch, True
                        )
                    else:
                        layer(*cache_args_batch, **cache_kwargs_batch)

                # Remove handles
                for h in handles:
                    h.remove()

                # 3) Quantize each submodule
                for name in subset:
                    full_module_name = module_name[subset[name]]

                    if gptq_conf.verbose:
                        print(f"[Layer {l_idx}] {name} -> Quantizing ...")

                    gptq[name].fasterquant(
                        percdamp=gptq_conf.percdamp,
                        groupsize=gptq_conf.groupsize,
                        actorder=gptq_conf.actorder,
                        static_groups=gptq_conf.static_groups,
                        verbose=gptq_conf.verbose,
                        adaptive_percdamp=gptq_conf.adaptive_percdamp,
                        cond_threshold_good=gptq_conf.cond_threshold_good,
                        use_iterate=gptq_conf.use_iterate,
                        actorder_precision=gptq_conf.actorder_precision,
                    )
                    quantizers[full_module_name] = gptq[name].quantizer
                    gptq[name].free()

            # 4) After quantization, re-run the layer to produce outputs for the next layer
            device = next(model.parameters()).device
            batch_num = self.num_batches
            for batch_idx in tqdm(
                range(batch_num),
                desc=f"[L{l_idx}] re-forward",
                leave=False,
                unit="batch",
                disable=not gptq_conf.show_progress,
            ):
                cache_args_batch = gather_single_batch_from_list(
                    self.cache_args, batch_idx
                )
                cache_args_batch = move_to_device(cache_args_batch, device)

                cache_kwargs_batch = gather_single_batch_from_dict(
                    self.cache_kwargs, batch_idx
                )
                cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)

                if fp_inps is not None:
                    fp_cache_args_batch = gather_single_batch_from_list(fp_inps, batch_idx)
                    fp_cache_args_batch = move_to_device(fp_cache_args_batch, device)
                    orig_layer = orig_layers[l_idx].to(device)
                    if gptq_conf.double_precision:
                        fp_outs = self._run_layer_forward_double_precision(
                            orig_layer, fp_cache_args_batch, cache_kwargs_batch, True
                        )
                    else:
                        fp_outs = orig_layer(*fp_cache_args_batch, **cache_kwargs_batch)
                        fp_outs = fp_outs[0] if isinstance(fp_outs, tuple) else fp_outs
                    orig_layer.cpu()
                    # Update inputs for next iteration.
                    if len(fp_inps) > 0:
                        if hasattr(fp_outs, "to") and hasattr(
                            fp_inps[0][batch_idx], "device"
                        ):
                            fp_inps[0][batch_idx] = fp_outs.to(
                                fp_inps[0][batch_idx].device
                            )
                        else:
                            fp_inps[0][batch_idx] = fp_outs
                
                if gptq_conf.double_precision:
                    if not need_next_orig_layer_inference :#orig_layers is None or gptq_conf.gptq_v2 is True:
                        outs = self._run_layer_forward_double_precision(
                            layer, cache_args_batch, cache_kwargs_batch, True
                        )
                    else:
                        assert orig_layers is not None
                        orig_layer = orig_layers[l_idx].to(device)
                        outs = self._run_layer_forward_double_precision(
                            orig_layer, cache_args_batch, cache_kwargs_batch, True
                        )
                        orig_layer.cpu()
                else:
                    if not need_next_orig_layer_inference:#orig_layers is None or gptq_conf.gptq_v2 is True:
                        outs = layer(*cache_args_batch, **cache_kwargs_batch)
                    else:
                        assert orig_layers is not None
                        orig_layer = orig_layers[l_idx].to(device)
                        outs = orig_layer(*cache_args_batch, **cache_kwargs_batch)
                        orig_layer.cpu()
                    # LLaMA's decoder layer return type differs across Transformers versions:
                    # some return a tuple (hidden_states, ...), others return just a tensor.
                    # This line ensures we always take the first element when it's a tuple.
                    outs = outs[0] if isinstance(outs, tuple) else outs
                # Update inputs for next iteration.
                if len(self.cache_args) > 0:
                    if hasattr(outs, "to") and hasattr(
                        self.cache_args[0][batch_idx], "device"
                    ):
                        self.cache_args[0][batch_idx] = outs.to(
                            self.cache_args[0][batch_idx].device
                        )
                    else:
                        self.cache_args[0][batch_idx] = outs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.finalize(model, quantizers)

    def finalize(self, model, quantizers):
        
        gptq_conf = self.config
        assert isinstance(gptq_conf, GPTQConfig)
                
        if (
            gptq_conf.quantize_lm_head
            and hasattr(model, "model")
            and hasattr(model.model, "norm")
            and hasattr(model, "lm_head")
        ):
            self._quantize_lm_head(model, quantizers)

        # Restore the original cache configuration.
        if self.orig_use_cache is not None:
            model.config.use_cache = self.orig_use_cache

        # Clear caches to free memory
        self.cache_args.clear()
        self.cache_kwargs.clear()
        self.num_batches = 0

        model.quantizers = quantizers

        # Finalization: cast entire model from float64 to float32 when
        # double_precision was used. This converts all weights, buffers,
        # and observer qparams (scale, zero_point) to float32 so that
        # subsequent evaluation and export use standard float32 inference.
        if gptq_conf.double_precision:
            print("Casting model from float64 to float32 for evaluation ...")
            model.float()

        return model
            
    def _convert_parallel(
        self,
        model: torch.nn.Module,
        target_layers,
        orig_layers,
        module_name: dict,
        quantizers: Dict[str, Any],
        gptq_conf: GPTQConfig,
        fp_inps,
    ) -> torch.nn.Module:
        """Parallel GPTQ quantization using multiprocessing.

        When ``use_orig_model_inference=True``, each layer's Hessian is
        computed from the original (unquantized) model, so layers are
        independent and can be quantized in parallel.

        This method:
          1. Pre-computes layer inputs for all layers using the original model.
          2. Dispatches each layer to a worker process via multiprocessing.Pool.
          3. Collects quantized weights and quantizer params from workers.
          4. Applies quantized weights back to the model.

        Args:
            model: The model to quantize (already prepared).
            target_layers: ModuleList of decoder layers.
            orig_layers: ModuleList of original (unquantized) decoder layers.
            module_name: Dict mapping module -> full module name.
            quantizers: Dict to populate with quantizer objects.
            gptq_conf: GPTQ configuration.
            fp_inps: Optional FP inputs for GPTQv2.

        Returns:
            The model with quantized weights applied.
        """
        import dataclasses
        import time

        device = next(model.parameters()).device
        num_layers = len(target_layers)
        num_batches = self.num_batches
        num_workers = gptq_conf.parallel_workers

        parallel_start = time.time()


        # Process layers in groups to reduce peak RAM usage.
        # Each group: compute inputs → dispatch workers → collect results.
        group_size = num_workers  # one layer per worker per group

        # Prepare static config data (shared across all groups)
        gptq_conf_dict = {}
        for f in dataclasses.fields(gptq_conf):
            val = getattr(gptq_conf, f.name)
            if f.name == "sensitivity":
                gptq_conf_dict[f.name] = None
            else:
                gptq_conf_dict[f.name] = val

        sensitivity_data = None
        if gptq_conf.sensitivity is not None and isinstance(gptq_conf.sensitivity, dict):
            sensitivity_data = {}
            for k, v in gptq_conf.sensitivity.items():
                if isinstance(v, torch.Tensor):
                    sensitivity_data[k] = v.cpu()
                else:
                    sensitivity_data[k] = v

        # Get model config for layer reconstruction
        layer_config = getattr(model, "config", None)

        # Detect available GPUs for multi-GPU distribution
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus > 1:
            gpu_devices = [f"cuda:{i}" for i in range(num_gpus)]
            print(f"[Parallel] Detected {num_gpus} GPUs, distributing workers across: {gpu_devices}")
        else:
            gpu_devices = [str(device)]

        # Use spawn context for CUDA compatibility
        ctx = multiprocessing.get_context("spawn")

        # Track current layer inputs (start from cached first-layer inputs)
        cur_args_per_batch: List[List[Any]] = self.cache_args
        cur_kwargs_per_batch: Dict[str, List[Any]] = self.cache_kwargs

        # Create pool once and reuse across groups
        with ctx.Pool(processes=num_workers) as pool:
            pbar = tqdm(
                range(num_layers),
                desc="[Parallel] Quantizing layers",
                unit="layer",
                disable=not gptq_conf.show_progress,
            )
            for group_start in range(0, num_layers, group_size):
                group_end = min(group_start + group_size, num_layers)
                group_indices = list(range(group_start, group_end))

                # ---- Step 1: Compute inputs for this group only ----
                # Note: we move individual orig_layers[l_idx] to GPU one at a time,
                # so there's no need to move the entire orig_model to GPU.

                group_inputs: List[List[Any]] = []
                group_kwargs: List[List[Dict[str, Any]]] = []
                group_fp_inputs: List[Optional[Dict[str, List[Any]]]] = []

                for l_idx in group_indices:
                    pbar.update(1)

                    layer_args: List[Any] = []
                    layer_kwargs: List[Dict[str, Any]] = []
                    next_args: List[Any] = []

                    # GPTQv2: Collect FP inputs from original model
                    fp_inputs_cache = None
                    if fp_inps is not None:
                        orig_full = find_layers(
                            orig_layers[l_idx],
                            layers=[
                                torch.nn.Linear,
                                torch.nn.Conv2d,
                                torch.nn.Conv1d,
                                torch.nn.Conv3d,
                                torch.nn.ConvTranspose2d,
                            ],
                        )
                        sequential = [list(orig_full.keys())]
                        fp_inputs_cache = FPInputsCache(sequential)
                        fp_inputs_cache.add_hook(orig_full)

                    for batch_idx in range(num_batches):
                        cache_args_batch = gather_single_batch_from_list(
                            cur_args_per_batch, batch_idx
                        )
                        cache_args_batch = move_to_device(cache_args_batch, device)
                        cache_kwargs_batch = gather_single_batch_from_dict(
                            cur_kwargs_per_batch, batch_idx
                        )
                        cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)

                        layer_args.append(move_to_cpu(cache_args_batch))
                        layer_kwargs.append(move_to_cpu(cache_kwargs_batch))

                        orig_layer = orig_layers[l_idx].to(device)
                        if gptq_conf.double_precision:
                            outs = self._run_layer_forward_double_precision(
                                orig_layer, cache_args_batch, cache_kwargs_batch, True
                            )
                        else:
                            outs = orig_layer(*cache_args_batch, **cache_kwargs_batch)
                            outs = outs[0] if isinstance(outs, tuple) else outs
                        orig_layer.cpu()

                        next_args.append(move_to_cpu(outs))

                    # GPTQv2: Collect FP inputs and clean up hooks
                    if fp_inputs_cache is not None:
                        fp_inputs_cache.clear_hook()
                        group_fp_inputs.append(fp_inputs_cache.fp_cache)
                    else:
                        group_fp_inputs.append(None)

                    group_inputs.append(layer_args)
                    group_kwargs.append(layer_kwargs)
                    cur_args_per_batch = [next_args]


                if self.orig_model is not None:
                    self.orig_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ---- Step 2: Build worker args for this group ----
                worker_args_list = []
                for i, l_idx in enumerate(group_indices):
                    layer = target_layers[l_idx]

                    local_module_names = {}
                    for name, module in layer.named_modules():
                        full_name = module_name.get(module, name)
                        local_module_names[name] = full_name

                    # Move layer to CPU for pickling to worker process
                    layer_cpu = move_to_cpu(layer)

                    worker_args = {
                        "l_idx": l_idx,
                        "target_layer": layer_cpu,
                        "layer_inputs": group_inputs[i],
                        "layer_kwargs": group_kwargs[i],
                        "gptq_conf_dict": gptq_conf_dict,
                        "module_names": local_module_names,
                        "sample_weights": self.sample_weights,
                        "sensitivity_data": sensitivity_data,
                        "fp_inputs": group_fp_inputs[i],
                        "ptq_wrapped": False,
                        "device": gpu_devices[i % len(gpu_devices)],
                    }
                    worker_args_list.append(worker_args)

                # ---- Step 3: Move model to CPU to free GPU for workers ----
                model = model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # ---- Step 4: Dispatch workers for this group ----
                group_results = {}
                for result in pool.imap_unordered(_gptq_layer_worker_gpu, worker_args_list):
                    l_idx = result["l_idx"]
                    group_results[l_idx] = result

                # ---- Step 5: Move model back to GPU and apply quantized weights ----
                model = model.to(device)

                for l_idx in group_indices:
                    layer = target_layers[l_idx]
                    result = group_results[l_idx]
                    quantized_weights = result["quantized_weights"]
                    quantizer_params = result["quantizer_params"]

                    full = find_layers(
                        layer,
                        layers=[
                            torch.nn.Linear,
                            torch.nn.Conv2d,
                            torch.nn.Conv1d,
                            torch.nn.Conv3d,
                            torch.nn.ConvTranspose2d,
                        ],
                    )

                    for name, module in full.items():
                        if name in quantized_weights:
                            module.weight.data = quantized_weights[name].to(device)
                            full_module_name = module_name.get(module, name)
                            if name in quantizer_params:
                                scale_cpu, zero_cpu = quantizer_params[name]
                                from tico.quantization.algorithm.gptq.quant import Quantizer
                                q = Quantizer()
                                q.configure(
                                    bits=gptq_conf.weight_bits,
                                    perchannel=gptq_conf.perchannel,
                                    sym=gptq_conf.symmetric,
                                    mse=gptq_conf.mse,
                                    sensitivity=(
                                        gptq_conf.sensitivity.get(full_module_name)
                                        if gptq_conf.sensitivity and isinstance(gptq_conf.sensitivity, dict)
                                        else None
                                    ),
                                    mse_tolerance=gptq_conf.mse_tolerance,
                                    chunk_size=gptq_conf.chunk_size,
                                    use_batched_gptq=gptq_conf.use_batched_gptq,
                                )
                                q.scale = scale_cpu.to(device)
                                q.zero = zero_cpu.to(device)
                                q.maxq = torch.tensor((1 << gptq_conf.weight_bits) - 1)
                                quantizers[full_module_name] = q

                # Free this group's inputs
                group_inputs.clear()
                group_kwargs.clear()
                group_results.clear()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.close()

        # Update cache_args with last layer outputs for lm_head quantization
        self.cache_args = cur_args_per_batch

        # Move model back to GPU for post-quantization steps
        model = model.to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parallel_elapsed = time.time() - parallel_start
        elapsed_dhms = _format_elapsed_dhms(parallel_elapsed)
        print(f"[Parallel] Total quantization time: {parallel_elapsed:.1f}s "
              f"[{elapsed_dhms}] "
              f"({num_layers} layers, {num_workers} workers)")

        return model


    def _quantize_lm_head(self, model, quantizers):
        """
        Apply GPTQ to the language-model output head.

        This method consumes cached decoder outputs, applies the final model
        normalization, collects GPTQ statistics for `lm_head`, and then
        quantizes the output head weights. It should only be called when
        `GPTQConfig.quantize_lm_head` is enabled.
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, GPTQConfig)
        # TODO reduce code duplication with layer-wise quantization

        # prepare data for lm_head
        batch_num = self.num_batches
        device = next(model.parameters()).device
        for batch_idx in tqdm(
            range(batch_num),
            desc=f"[model.norm] re-forward",
            leave=False,
            unit="batch",
            disable=not gptq_conf.show_progress,
        ):
            hidden_states = gather_single_batch_from_list(self.cache_args, batch_idx)[0]
            hidden_states = move_to_device(hidden_states, device)
            if self.orig_model is None:
                hidden_states = model.model.norm(hidden_states)
            else:
                norm = self.orig_model.model.norm.to(device)
                hidden_states = norm(hidden_states)
                norm = norm.cpu()
            if len(self.cache_args) > 0:
                self.cache_args[0][batch_idx] = move_to_cpu(hidden_states)

        layer = model.lm_head
        full_module_name = "lm_head"
        gptq = GPTQ(layer, double_precision=gptq_conf.double_precision, layer_name=full_module_name)
        weight_bits = self._resolve_weight_bits(
            gptq_conf,
            full_module_name=full_module_name,
            local_module_name="lm_head",
        )
        if (
            gptq_conf.sensitivity is not None
            and isinstance(gptq_conf.sensitivity, dict)
            and full_module_name in gptq_conf.sensitivity
        ):
            cur_sensitivity = gptq_conf.sensitivity[full_module_name]
        else:
            cur_sensitivity = None
        gptq.quantizer.configure(
            bits=weight_bits,
            perchannel=gptq_conf.perchannel,
            sym=gptq_conf.symmetric,
            mse=gptq_conf.mse,
            sensitivity=cur_sensitivity,
            mse_tolerance=gptq_conf.mse_tolerance,
            chunk_size=gptq_conf.chunk_size,
            use_batched_gptq=gptq_conf.use_batched_gptq,
        )

        # Hook to collect (inp, out) for GPTQ with optional weights
        gptq.weights = self.sample_weights
        gptq.batch_id = 0
        
        def add_batch():
            def _hook(_, inp, out):
                gptq.add_batch(inp[0].data, out.data)

            return _hook

        handles = [layer.register_forward_hook(add_batch())]

        # Run layer forward over all cached batches to build Hessian/statistics
        old_device = device
        model = model.to("cpu")
        model.lm_head = model.lm_head.to(old_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = next(layer.parameters()).device  # in case lm_head is located on cpu
        for batch_idx in tqdm(
            range(batch_num),
            desc=f"[lm_head] collecting",
            leave=False,
            unit="batch",
            disable=not gptq_conf.show_progress,
        ):
            hidden_states = gather_single_batch_from_list(self.cache_args, batch_idx)[0]
            hidden_states = move_to_device(hidden_states, device)

            layer(hidden_states)

        # Remove handles
        for h in handles:
            h.remove()

        # Quantize
        if gptq_conf.verbose:
            print(f"[lm_head] -> Quantizing ...")
        gptq.fasterquant(
            percdamp=gptq_conf.percdamp,
            groupsize=gptq_conf.groupsize,
            actorder=gptq_conf.actorder,
            static_groups=gptq_conf.static_groups,
            verbose=gptq_conf.verbose,
            adaptive_percdamp=gptq_conf.adaptive_percdamp,
            cond_threshold_good=gptq_conf.cond_threshold_good,
            use_iterate=gptq_conf.use_iterate,
            actorder_precision=gptq_conf.actorder_precision,
        )
        quantizers[f"lm_head"] = gptq.quantizer
        gptq.free()
        model = model.to(old_device)

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


def move_to_cpu(obj):
    return move_to_device(obj, "cpu")


class StopForward(Exception):
    """Custom exception used to stop the forward pass after the first layer."""

    pass


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

        # References to original forwards for restoration
        self._orig_model_forward: Optional[Callable[..., Any]] = None
        self._orig_layer_forward: Optional[Callable[..., Any]] = None
        self._first_layer_ref: Optional[torch.nn.Module] = None

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
          3) Restore model.config.use_cache if needed and clear internal caches.

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
        # Disable use_cache during calibration
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            orig_use_cache = model.config.use_cache
            model.config.use_cache = False
        else:
            orig_use_cache = None

        # Identify layers
        if hasattr(model, "model"):
            if hasattr(model.model, "layers"):
                target_layers = model.model.layers
            else:
                target_layers = [model]
        else:
            target_layers = [model]

        module_name = {}
        for name, module in model.named_modules():
            module_name[module] = name

        quantizers: Dict[str, Any] = {}
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

            # 2) Set up GPTQ objects and gather stats
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq: Dict[str, GPTQ] = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    full_module_name = module_name[subset[name]]
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
                    )

                # Hook to collect (inp, out) for GPTQ
                def add_batch(name):
                    def _hook(_, inp, out):
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

                    layer(*cache_args_batch, **cache_kwargs_batch)

                # Remove handles
                for h in handles:
                    h.remove()

                # 3) Quantize each submodule
                for name in subset:
                    if gptq_conf.verbose:
                        print(f"[Layer {l_idx}] {name} -> Quantizing ...")
                    gptq[name].fasterquant(
                        percdamp=gptq_conf.percdamp,
                        groupsize=gptq_conf.groupsize,
                        actorder=gptq_conf.actorder,
                        static_groups=gptq_conf.static_groups,
                        verbose=gptq_conf.verbose,
                    )
                    quantizers[f"{l_idx}.{name}"] = gptq[name].quantizer
                    gptq[name].free()

            # 4) After quantization, re-run the layer to produce outputs for the next layer
            device = next(model.parameters()).device
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

                outs = layer(*cache_args_batch, **cache_kwargs_batch)
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

        if (
            hasattr(model, "model")
            and hasattr(model.model, "norm")
            and hasattr(model, "lm_head")
        ):  # quantize lm_head
            self._quantize_lm_head(model, quantizers)

        # Restore the original cache configuration.
        if orig_use_cache is not None:
            model.config.use_cache = orig_use_cache

        # Clear caches to free memory
        self.cache_args.clear()
        self.cache_kwargs.clear()
        self.num_batches = 0

        model.quantizers = quantizers

        return model

    def _quantize_lm_head(self, model, quantizers):
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

            hidden_states = model.model.norm(hidden_states)
            if len(self.cache_args) > 0:
                self.cache_args[0][batch_idx] = move_to_cpu(hidden_states)

        layer = model.lm_head
        gptq = GPTQ(layer)
        full_module_name = "lm_head"
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
        )

        # Hook to collect (inp, out) for GPTQ
        def add_batch():
            def _hook(_, inp, out):
                gptq.add_batch(inp[0].data, out.data)

            return _hook

        handles = [layer.register_forward_hook(add_batch())]

        # Run layer forward over all cached batches to build Hessian/statistics
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
        )
        quantizers[f"lm_head"] = gptq.quantizer
        gptq.free()

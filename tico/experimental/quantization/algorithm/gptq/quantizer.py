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

from tico.experimental.quantization.algorithm.gptq.gptq import GPTQ
from tico.experimental.quantization.algorithm.gptq.utils import (
    find_layers,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
)
from tico.experimental.quantization.config import BaseConfig, GPTQConfig
from tico.experimental.quantization.quantizer import BaseQuantizer


class StopForward(Exception):
    """Custom exception used to stop the forward pass after the first layer."""

    pass


class GPTQQuantizer(BaseQuantizer):
    """
    Quantizer for applying the GPTQ algorithm (typically for weight quantization).
    This implementation expects:
        1) prepare(model, ...) to only attach hooks/Catchers and NOT run the model internally.
        2) The user runs the model with arbitrary number of batches to collect calibration data.
        3) convert(model) to consume the collected data and apply GPTQ.
    """

    def __init__(self, config: BaseConfig):
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
                self.cache_args[idx].append(item)
            # Store keyword args
            for k, v in kwargs.items():
                if self.cache_kwargs.get(k, None) is None:
                    self.cache_kwargs[k] = []
                self.cache_kwargs[k].append(v)

            self.num_batches += 1
            raise StopForward  # stop after the first layer

        # Replace the first layer with defined function to capture calibration data.
        if hasattr(model, "model"):
            if hasattr(model.model, "layers") and isinstance(
                model.model.layers, torch.nn.ModuleList
            ):
                self._first_layer_ref = model.model.layers[0]
            else:
                raise RuntimeError(
                    "GPTQ Quantizer assumes the model has a nested structure like `model.model.layers`, commonly found in LLaMA and other Hugging Face transformer models."
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
        # Disable use_cache during calibration
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            orig_use_cache = model.config.use_cache
            model.config.use_cache = False
        else:
            orig_use_cache = None

        # Identify layers
        if hasattr(model, "model"):
            target_layers = model.model.layers
        else:
            target_layers = [model]

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
            full = find_layers(layer)
            sequential = [list(full.keys())]

            # 2) Set up GPTQ objects and gather stats
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq: Dict[str, GPTQ] = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        bits=8, perchannel=True, sym=False, mse=False
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
                    cache_kwargs_batch = gather_single_batch_from_dict(
                        self.cache_kwargs, batch_idx
                    )
                    layer(*cache_args_batch, **cache_kwargs_batch)

                # Remove handles
                for h in handles:
                    h.remove()

                # 3) Quantize each submodule
                for name in subset:
                    if gptq_conf.verbose:
                        print(f"[Layer {l_idx}] {name} -> Quantizing ...")
                    gptq[name].fasterquant(
                        percdamp=0.01,
                        groupsize=-1,
                        actorder=True,
                        static_groups=False,
                        verbose=gptq_conf.verbose,
                    )
                    quantizers[f"model.layers.{l_idx}.{name}"] = gptq[name].quantizer
                    gptq[name].free()

            # 4) After quantization, re-run the layer to produce outputs for the next layer
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
                cache_kwargs_batch = gather_single_batch_from_dict(
                    self.cache_kwargs, batch_idx
                )
                outs = layer(*cache_args_batch, **cache_kwargs_batch)
                # LLaMA's decoder layer return type differs across Transformers versions:
                # some return a tuple (hidden_states, ...), others return just a tensor.
                # This line ensures we always take the first element when it's a tuple.
                outs = outs[0] if isinstance(outs, tuple) else outs
                # Update inputs for next iteration.
                self.cache_args[0][batch_idx] = outs

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore the original cache configuration.
        if orig_use_cache is not None:
            model.config.use_cache = orig_use_cache

        # Clear caches to free memory
        self.cache_args.clear()
        self.cache_kwargs.clear()
        self.num_batches = 0

        model.quantizers = quantizers

        return model

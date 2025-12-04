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

from typing import Any, Dict

import torch
from tqdm.auto import tqdm

from tico.quantization.algorithm.fpi_gptq.fpi_gptq import FPI_GPTQ
from tico.quantization.algorithm.gptq.quantizer import GPTQQuantizer
from tico.quantization.algorithm.gptq.utils import (
    find_layers,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
)
from tico.quantization.config.fpi_gptq import FPIGPTQConfig
from tico.quantization.quantizer_registry import register_quantizer


@register_quantizer(FPIGPTQConfig)
class FPIGPTQQuantizer(GPTQQuantizer):
    """
    Quantizer for applying the Fixed Point Iteration GPTQ algorithm (FPIGPTQ)
    This implementation expects the same steps as GPTQQuantizer.
    It should produce results very close to reference GPTQ but much faster when running on cuda.
    """

    def __init__(self, config: FPIGPTQConfig):
        super().__init__(config)

    @torch.no_grad()
    def convert(self, model):

        # Restore original forwards (we no longer want to stop after first layer)
        assert self._orig_model_forward is not None
        model.forward = self._orig_model_forward
        assert (
            self._first_layer_ref is not None and self._orig_layer_forward is not None
        )
        self._first_layer_ref.forward = self._orig_layer_forward

        gptq_conf = self.config
        assert isinstance(gptq_conf, FPIGPTQConfig)
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
            full = find_layers(
                layer,
                layers=[
                    torch.nn.Linear,
                    torch.nn.Conv2d,
                    torch.nn.Conv1d,
                    torch.nn.ConvTranspose2d,
                ],
            )
            sequential = [list(full.keys())]

            # 2) Set up (as in GPTQ)
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq: Dict[str, FPI_GPTQ] = {}
                for name in subset:
                    gptq[name] = FPI_GPTQ(subset[name])
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

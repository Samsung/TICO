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
import types
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm.auto import tqdm

from tico.quantization.algorithm.gptq.gptq import GPTQ
from tico.quantization.algorithm.gptq.utils import (
    find_layers,
    find_layers_deep,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
)
from tico.quantization.config.llama_gptq import LlamaGPTQConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.nn.quant_embedding import QuantEmbedding
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.utils.utils import move_to_device
from transformers import Conv1D


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

       # if isinstance(m, (torch.nn.Linear, Conv1D)):
       #     if len(inp.shape) == 3:
       #         inp = inp.reshape((-1, inp.shape[-1]))
       #     inp = inp.t()
       # elif isinstance(m, torch.nn.Conv2d):
       #     unfold = torch.nn.Unfold(
       #         m.kernel_size,
       #         dilation=m.dilation,
       #         padding=m.padding,
       #         stride=m.stride,
       #     )
       #     inp = unfold(inp)
       #     inp = inp.permute([1, 0, 2])
       #     inp = inp.flatten(1)

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

def print_minmax_values(model: torch.nn.Module) -> None:
    """
    Print min/max values from all PTQ observers in the quantized model.

    This function traverses the model hierarchy and prints the min/max statistics
    collected by each AffineObserverBase instance. Useful for debugging and
    inspecting quantization ranges after calibration.

    For per-tensor observers, prints scalar min/max values.
    For per-channel observers, prints the global min/max range and channel shape.

    Args:
        model: A PTQ-quantized model with observers containing min/max statistics.

    Example usage:
        # After calibration and before/after conversion:
        print_minmax_values(q_m)
    """
    from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
    from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

    print("\n" + "=" * 80)
    print("PTQ Model Min/Max Values")
    print("=" * 80)
    print(f"{'Module Name':<50} | {'Observer':<25} | Min/Max Values")
    print("-" * 80)

    count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, QuantModuleBase):
            continue

        for obs_name, obs in module.named_observers(recurse=True):
            if not isinstance(obs, AffineObserverBase):
                continue

            if not hasattr(obs, "min_val") or not hasattr(obs, "max_val"):
                continue

            min_val = obs.min_val
            max_val = obs.max_val

            # Format output based on per-tensor vs per-channel
            if min_val.numel() == 1:
                # Per-tensor: scalar values
                values_str = f"min={min_val.item():.6f}, max={max_val.item():.6f}"
            else:
                # Per-channel: show shape and range
                values_str = (
                    f"min={min_val.min().item():.6f}..{max_val.max().item():.6f} "
                    f"(shape={tuple(min_val.shape)})"
                )

            print(f"{module_name:<50} | {obs_name:<25} | {values_str}")
            count += 1

    print("-" * 80)
    print(f"Total observers: {count}")
    print("=" * 80 + "\n")

class StopForward(Exception):
    """Custom exception used to stop the forward pass after the first layer."""

    pass


@register_quantizer(LlamaGPTQConfig)
class LlamaGPTQQuantizer(BaseQuantizer):
    """
    Llama-specific quantizer for applying the GPTQ algorithm (typically for weight quantization).

    This quantizer is designed specifically for Llama-family models, including:
        - LlamaForCausalLM (standard Hugging Face Llama models)
        - SpinLlamaForCausalLM (Llama models with SpinQuant rotation layers)

    This implementation expects:
        1) prepare(model, ...) to only attach hooks/Catchers and NOT run the model internally.
        2) The user runs the model with arbitrary number of batches to collect calibration data.
        3) convert(model) to consume the collected data and apply GPTQ.

    Unlike the generic GPTQQuantizer, this implementation:
        - Properly handles Llama-specific architecture (model.layers, lm_head)
        - Supports SpinLlamaForCausalLM with rotate_lm_head layer
        - Provides Llama-specific configuration options
    """

    def __init__(self, config: LlamaGPTQConfig):
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

        # Reference to original model for use_orig_model_inference
        self.orig_model: Optional[torch.nn.Module] = None

    def _resolve_weight_bits(
        self,
        gptq_conf: LlamaGPTQConfig,
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

    def _is_spinllama_model(self, model: torch.nn.Module) -> bool:
        """Check if the model is a SpinLlamaForCausalLM (has rotate_lm_head)."""
        return hasattr(model, "rotate_lm_head") and model.rotate_lm_head is not None

    @staticmethod
    def _is_ptq_wrapped(model: torch.nn.Module) -> bool:
        """Check if the model has been wrapped with PTQ prepare().

        After PTQ prepare(), the top-level model becomes a
        ``QuantLlamaForCausalLM`` whose ``.model`` attribute is a
        ``PTQWrapper`` (instead of a plain ``LlamaModel``).
        """
        return isinstance(model, PTQWrapper)

    def _get_decoder_layers(self, model: torch.nn.Module):
        """Get the decoder layers from a Llama model.

        Handles both raw models and PTQ-wrapped models.

        After PTQ prepare() the top-level model is ``QuantLlamaForCausalLM``
        which stores the Llama body in ``self.model`` (a ``PTQWrapper``).
        The actual ``LlamaModel`` is at ``model.model.wrapped``.

        If the model is already a ``QuantLlamaModel`` (e.g. the body
        without the CausalLM wrapper), its layers are directly at
        ``model.layers``.
        """
        # Case 1: model has a .model child (LlamaForCausalLM / QuantLlamaForCausalLM)
        if hasattr(model, "model"):
            model_attr = model.model
            if isinstance(model_attr, QuantModuleBase):
                # PTQ-wrapped: .model is PTQWrapper → .wrapped is QuantLlamaModel
                return model_attr.wrapped.layers
            return model_attr.layers

        # Case 2: model IS the LlamaModel / QuantLlamaModel directly
        if isinstance(model, QuantModuleBase):
            return model.wrapped.model.wrapped.layers

        # Case 3: plain LlamaModel
        return model.layers

    def _get_orig_decoder_layers(self, model: torch.nn.Module):
        if self.orig_model is not None:
            if hasattr(self.orig_model, "model"):
                return self.orig_model.model.layers
            elif hasattr(self.orig_model, "wrapped"):
                return self.orig_model.wrapped.model.wrapped.layers
            return self.orig_model.layers
        return None

    @staticmethod
    def _find_ptq_layers(layer: torch.nn.Module, layers=None, name=""):
        """Find quantizable submodules inside a PTQ-wrapped decoder layer.

        Navigates the ``PTQWrapper(.wrapped)`` hierarchy transparently so
        that the returned names match the **original** model structure
        (e.g. ``"self_attn.q_proj"`` instead of
        ``"wrapped.self_attn.wrapped.q_proj.wrapped.module"``).

        Returns a dict mapping *local* name → raw ``nn.Module``
        (e.g. the ``nn.Linear`` inside ``QuantLinear.module``).
        """
        if layers is None:
            layers = [torch.nn.Linear]

        # Direct match
        if type(layer) in layers:
            return {name: layer}

        # Unwrap QuantModuleBase that stores the original layer in .module
        if hasattr(layer, "module") and isinstance(getattr(layer, "module"), torch.nn.Module):
            inner = layer.module
            if type(inner) in layers:
                return {name: inner}

        res: Dict[str, torch.nn.Module] = {}
        for child_name, child in layer.named_children():
            # Skip the "wrapped" level of PTQWrapper to keep names clean
            from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
            if child_name == "wrapped" and isinstance(layer, PTQWrapper):
                new_name = name  # don't append "wrapped"
            else:
                new_name = name + "." + child_name if name != "" else child_name

            res.update(
                LlamaGPTQQuantizer._find_ptq_layers(
                    child, layers=layers, name=new_name
                )
            )
        return res
    
    @staticmethod
    def reset_layer_observers(layer: torch.nn.Module, save_obs) -> None:
        """
        Reset all observers (weight and activation) in a layer.

        This clears the min/max statistics collected by observers, allowing
        them to collect fresh calibration data.

        Args:
            layer: A QuantModuleBase (e.g., PTQWrapper) containing observers
        """
        for m in layer.modules():
            if isinstance(m, QuantModuleBase):
                for name, obs in m.named_observers(recurse=False):
                    if obs in save_obs:
                        continue
                    obs.reset()

    @staticmethod
    def remove_wrapped_substrings(s: str) -> str:
        
        s = s.replace(".wrapped", "")
        s = s.replace("wrapped.", "")
        s = s.replace("wrapped", "")
        return s

    @staticmethod
    def _inject_gptq_qparams_into_layer(
        layer: torch.nn.Module,
        gptq_quantizers: Dict[str, Any],
        *,
        verbose: bool = False,
    ):
        """Inject GPTQ (scale, zero-point) into the PTQ weight observers
        of *all* ``QuantModuleBase`` descendants inside *layer*, then call
        ``freeze_qparams()`` to lock every observer (weight + activation).

        This is used when GPTQ runs on a PTQ-prepared model: GPTQ quantizes
        the weights, and we push the resulting qparams into the PTQ weight
        observers so the PTQ graph uses the same quantization parameters.
        """
        seen = set()
        missed_modules = []
        saved_obs = set()
        for m in layer.modules():
            if not isinstance(m, QuantModuleBase):
                continue
            if m.fp_name is None:
                continue

            quantizer = gptq_quantizers.get(m.fp_name)
            obs = m.get_observer("weight")

            # Only care about modules that should have weight observers
            if obs is None:
                continue

            if quantizer is None:
                missed_modules.append(m.fp_name)
                #saved_obs.add(obs) #not-gptq weight
                #obs.enabled = False
                continue

            assert isinstance(obs, AffineObserverBase)
            obs.load_qparams(quantizer.scale, quantizer.zero, lock=True)
            seen.add(m.fp_name)
            saved_obs.add(obs)
            
            #m.freeze_qparams()

        unused = set(gptq_quantizers.keys()) - seen
        LlamaGPTQQuantizer.reset_layer_observers(layer, saved_obs)
        
        if verbose:
            print(f"\n  [GPTQ → PTQ injection] matched={len(seen)}, "
                  f"missed={len(missed_modules)}, unused={len(unused)}")
            if missed_modules:
                print(f"    missed: {missed_modules[:5]}")
          # if unused:
          #     print(f"    unused: {list(unused)[:5]}")

        # Freeze all observers (weight + activation) for this layer.
        # The layer is a PTQWrapper (QuantModuleBase), and freeze_qparams()
        # propagates to all child QuantModuleBase descendants.
        # This transitions the layer from CALIB → QUANT mode.
       # if isinstance(layer, QuantModuleBase):
       #     layer.freeze_qparams()

    def _get_config(self, m):
        """Get config from model, handling PTQ wrappers."""
        if hasattr(m, 'config'):
            return m.config
        if hasattr(m, 'wrapped'):
            return self._get_config(m.wrapped)
        return None
    
    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Overrides the forward method of the first Llama layer (layer 0) to capture the
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
        assert isinstance(gptq_conf, LlamaGPTQConfig)
        if gptq_conf.use_orig_model_inference is True or gptq_conf.gptq_v2:
            device = next(model.parameters()).device
            model = model.cpu()
            self.orig_model = copy.deepcopy(model)
            model = model.to(device)
        else:
            self.orig_model = None
        
        # Replace the first layer with defined function to capture calibration data.
        layers = self._get_decoder_layers(model)
        self._first_layer_ref = layers[0]

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
        # Handle PTQ-wrapped models by unwrapping to get to the config
        config = self._get_config(model)
        if config is not None and hasattr(config, "use_cache"):
            self.orig_use_cache = config.use_cache
            config.use_cache = False
        else:
            self.orig_use_cache = None

        return model
    def _get_embed_tokens_ptq_wrapper(self, model: torch.nn.Module) -> Optional[QuantModuleBase]:
        """
        Get the PTQ wrapper for embed_tokens.
        
        This only handles PTQ-wrapped embed_tokens (QuantEmbedding).
        Returns None if not found or not PTQ-wrapped.
        """
        for m in model.modules():
            if isinstance(m, QuantEmbedding):
                fp_name = getattr(m, 'fp_name', None)
                if fp_name is not None and 'embed_tokens' in fp_name:
                    return m
        return None

    def _get_model_norm_ptq_wrapper(self, model: torch.nn.Module) -> Optional[QuantModuleBase]:
        """
        Get the PTQ wrapper for model.norm.
        
        This only handles PTQ-wrapped model.norm (QuantRMSNorm).
        Returns None if not found or not PTQ-wrapped.
        """
        from tico.quantization.wrapq.wrappers.ops.quant_rmsnorm import QuantRMSNorm
        from tico.quantization.wrapq.wrappers.nn.quant_layernorm import QuantLayerNorm
        
        for m in model.modules():
            if isinstance(m, (QuantRMSNorm, QuantLayerNorm)):
                fp_name = getattr(m, 'fp_name', None)
                if fp_name is not None and fp_name.endswith('.norm'):
                    return m
        return None

    def _get_lm_head_ptq_wrapper(self, model: torch.nn.Module) -> Optional[QuantModuleBase]:
        """
        Get the PTQ wrapper for lm_head.
        
        This only handles PTQ-wrapped lm_head (QuantLinear).
        Returns None if not found or not PTQ-wrapped.
        """
        for m in model.modules():
            if isinstance(m, QuantLinear):
                fp_name = getattr(m, 'fp_name', None)
                if fp_name is not None and 'lm_head' in fp_name:
                    return m
        return None
    
    def _calibrate_embed_tokens_ptq(self, model: torch.nn.Module) -> None:
        """
        Calibrate PTQ observers for embed_tokens (PTQ-only, no GPTQ).
        
        Calibrates weight, input activation, and output activation observers.
        """
        embed_tokens = self._get_embed_tokens_ptq_wrapper(model)
        if embed_tokens is None:
            return

        # Calibrate weight observer immediately (fixed)
        obs_weight = embed_tokens.get_observer("weight")
        if obs_weight is not None:
            obs_weight.collect(embed_tokens.module.weight)

        embed_tokens.freeze_qparams()

    def _calibrate_norm_lm_head_ptq(self, model: torch.nn.Module) -> None:
        """
        Calibrate PTQ observers for  norm and lm_head (PTQ-only, no GPTQ).
        
        Calibrates weights, input activations, and output activations observers.
        """
        lm_head = self._get_lm_head_ptq_wrapper(model)
        if lm_head is None:
            return

        # Calibrate weight observer immediately (fixed)
        obs_weight = lm_head.get_observer("weight")
        if obs_weight is not None:
            obs_weight.collect(lm_head.module.weight)
            obs_weight.enabled = False
            obs_weight.compute_qparams()

        # Calibrate input and output activation observers
        device = next(model.parameters()).device
        batch_num = self.num_batches
        model_norm = self._get_model_norm_ptq_wrapper(model)
        
        for batch_idx in range(batch_num):
            hidden_states = gather_single_batch_from_list(self.cache_args, batch_idx)[0]
            hidden_states = move_to_device(hidden_states, device)
            hidden_states = model_norm(hidden_states)
            lm_head(hidden_states)

        # Freeze activation observers
        model_norm.freeze_qparams()
        lm_head.freeze_qparams()

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
          3) Optionally apply GPTQ to lm_head and rotate_lm_head when configured.
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
        assert isinstance(gptq_conf, LlamaGPTQConfig)
        gptq_conf.validate()

        ptq_wrapped = self._is_ptq_wrapped(model)

        # Identify layers
        target_layers = self._get_decoder_layers(model)
        orig_layers = self._get_orig_decoder_layers(model)
        
        module_name: Dict[torch.nn.Module, str] = {}
        for name, module in model.named_modules():
            module_name[module] = name

        self._calibrate_embed_tokens_ptq(model)
        
        # Choose the right layer-finder depending on whether the model is
        # PTQ-wrapped.  When it is, nn.Linear modules are hidden inside
        # QuantLinear.module and PTQWrapper.wrapped layers, so we need
        # _find_ptq_layers which transparently skips the "wrapped" level.
        _find = self._find_ptq_layers if ptq_wrapped else find_layers
        
        quantizers: Dict[str, Any] = {}
        batch_num = self.num_batches
        
        # GPTQv2: Collect FP inputs from original model before quantization
        need_float_inference = gptq_conf.gptq_v2 
        fp_inps = None
        if need_float_inference and orig_layers is not None:
            fp_inps = copy.deepcopy(self.cache_args)
        for l_idx, layer in enumerate(
            tqdm(
                target_layers,
                desc="Quantizing layers",
                unit="layer",
                disable=not gptq_conf.show_progress,
            )
        ):
            # 1) Identify quantizable submodules within the layer
            full = _find(
                layer,
                layers=[
                    torch.nn.Linear,
                    QuantLinear
                ],
            )

            sequential = False#True #False
            # Define groups for quantizing by internal structure (standard Llama modules)
            if sequential is True:
                #sequential processing
                all_names = [
                    # Wrapped paths (for PTQ-wrapped models) - must come first
                    ["wrapped.self_attn.wrapped.q_proj.wrapped", "wrapped.self_attn.wrapped.k_proj.wrapped", "wrapped.self_attn.wrapped.v_proj.wrapped"],
                    ["wrapped.self_attn.wrapped.o_proj.wrapped"],
                    ["wrapped.mlp.wrapped.gate_proj.wrapped", "wrapped.mlp.wrapped.up_proj.wrapped"],
                    ["wrapped.mlp.wrapped.down_proj.wrapped"],
                    # Standard unwrapped paths
                    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                    ["self_attn.o_proj"],
                    ["mlp.gate_proj", "mlp.up_proj"],
                    ["mlp.down_proj"],
                ]
            else:
                #process all internal linears at once
                all_names = [
                    # Wrapped paths (for PTQ-wrapped models) - must come first
                    ["wrapped.self_attn.wrapped.q_proj.wrapped", "wrapped.self_attn.wrapped.k_proj.wrapped", "wrapped.self_attn.wrapped.v_proj.wrapped",
                    "wrapped.self_attn.wrapped.o_proj.wrapped",
                    "wrapped.mlp.wrapped.gate_proj.wrapped", "wrapped.mlp.wrapped.up_proj.wrapped",
                    "wrapped.mlp.wrapped.down_proj.wrapped"],
                    # Standard unwrapped paths
                    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj",
                    "mlp.down_proj"],
                ]

            # Filter to only existing modules and group them
            existing_names = set(full.keys())
            sequential = []
            for names in all_names:
                cur_seq = [name for name in names if name in existing_names]
                if cur_seq:
                    sequential.append(cur_seq)

            # GPTQv2: Set up FPInputsCache for collecting FP inputs per submodule
            fp_inputs_cache = None
            if need_float_inference and orig_layers is not None:
                fp_inputs_cache = FPInputsCache(sequential)
                orig_full = _find(
                    orig_layers[l_idx],
                    layers=[
                        torch.nn.Linear,
                        QuantLinear
                    ],
                )
                fp_inputs_cache.add_hook(orig_full)
                device = next(model.parameters()).device
                for batch_idx in range(batch_num):
                    cache_args_batch = gather_single_batch_from_list(fp_inps, batch_idx)
                    cache_args_batch = move_to_device(cache_args_batch, device)
                    cache_kwargs_batch = gather_single_batch_from_dict(self.cache_kwargs, batch_idx)
                    cache_kwargs_batch = move_to_device(cache_kwargs_batch, device)
                    
                    orig_layer = orig_layers[l_idx].to(device)
                    orig_layer(*cache_args_batch, **cache_kwargs_batch)
                    orig_layer.cpu()
                    
                fp_inputs_cache.clear_hook()

            # 2) Set up GPTQ objects and gather stats
            for names in sequential:
                subset = {n: full[n] for n in names}

                gptq: Dict[str, GPTQ] = {}
                for name in subset:
                    sub_layer = subset[name]
                    nn_layer = sub_layer.module if hasattr(sub_layer, "module") else sub_layer
                    gptq[name] = GPTQ(nn_layer)
                    full_module_name = module_name[subset[name]]
                    weight_bits = 4
                    self._resolve_weight_bits(
                        gptq_conf,
                        full_module_name=self.remove_wrapped_substrings(full_module_name),
                        local_module_name=self.remove_wrapped_substrings(name),
                    )
                    if (
                        gptq_conf.sensitivity is not None
                        and isinstance(gptq_conf.sensitivity, dict)
                        and self.remove_wrapped_substrings(full_module_name) in gptq_conf.sensitivity
                    ):
                        cur_sensitivity = gptq_conf.sensitivity[self.remove_wrapped_substrings(full_module_name)]
                    else:
                        cur_sensitivity = None
                    gptq[name].quantizer.configure(
                        bits=weight_bits,
                        perchannel=gptq_conf.perchannel,
                        sym=gptq_conf.symmetric,
                        mse=gptq_conf.mse,
                        sensitivity=cur_sensitivity,
                    )

                    # GPTQv2: Assign native_inp from FPInputsCache
                    if fp_inputs_cache is not None and name in fp_inputs_cache.fp_cache:
                        gptq[name].native_inp = fp_inputs_cache.fp_cache[name]

                # Hook to collect (inp, out) for GPTQ
                def add_batch(name):
                    def _hook(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)

                    return _hook

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

                # Run layer forward over all cached batches to build Hessian/statistics
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
                    full_module_name = module_name[subset[name]]

                    if gptq_conf.verbose:
                        print(f"[Layer {l_idx}] {name} -> Quantizing ...")

                    gptq[name].fasterquant(
                        percdamp=gptq_conf.percdamp,
                        groupsize=gptq_conf.groupsize,
                        actorder=gptq_conf.actorder,
                        static_groups=gptq_conf.static_groups,
                        verbose=gptq_conf.verbose,
                    )
                    quantizers[self.remove_wrapped_substrings(full_module_name)] = gptq[name].quantizer
                    gptq[name].free()
                    
            # --- PTQ-wrapped: inject GPTQ qparams and freeze the layer ---
            if ptq_wrapped:
                self._inject_gptq_qparams_into_layer(
                    layer,
                    quantizers,
                    verbose=gptq_conf.verbose,
                )
                layer.enable_calibration()
                calibrated = False
                device = next(model.parameters()).device
                for batch_idx in tqdm(
                    range(batch_num),
                    desc=f"[L{l_idx}] activations calibration",
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
                    
                if ptq_wrapped:
                    layer.freeze_qparams()   
                    calibrated = True
                    
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
                if fp_inps is not None:
                    fp_cache_args_batch = gather_single_batch_from_list(fp_inps, batch_idx)
                    fp_cache_args_batch = move_to_device(fp_cache_args_batch, device)
                    orig_layer = orig_layers[l_idx].to(device)
                    fp_outs = orig_layer(*fp_cache_args_batch, **cache_kwargs_batch)
                    #fp_outs = layer(*fp_cache_args_batch, **cache_kwargs_batch)
                    orig_layer.cpu()
                    fp_outs = fp_outs[0] if isinstance(fp_outs, tuple) else fp_outs
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
                    
                if orig_layers is None or self.config.gptq_v2 is True:
                    outs = layer(*cache_args_batch, **cache_kwargs_batch)
                else:
                    orig_layer = orig_layers[l_idx].to(device)
                    outs = orig_layer(*cache_args_batch, **cache_kwargs_batch)
                    orig_layer.cpu()
                    if ptq_wrapped and not calibrated:
                        # nevertheless we should calibrate
                        layer(*cache_args_batch, **cache_kwargs_batch)
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
            
            if ptq_wrapped and not calibrated:
                layer.freeze_qparams()   

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
#        if ptq_wrapped:
#            self._calibrate_norm_lm_head_ptq(model)
#            model.freeze_qparams()
        
        # Restore the original cache configuration.
        config = self._get_config(model)
        if self.orig_use_cache is not None:
            config.use_cache = self.orig_use_cache

        # Clear caches to free memory
        self.cache_args.clear()
        self.cache_kwargs.clear()
        self.num_batches = 0

       # model.quantizers = quantizers

        return model

    def _quantize_lm_head(
        self,
        model: torch.nn.Module,
        quantizers: Dict[str, Any],
        module_name: Dict[torch.nn.Module, str],
    ):
        """
        Apply GPTQ to the language-model output head.

        This method consumes cached decoder outputs, applies the final model
        normalization, collects GPTQ statistics for `lm_head`, and then
        quantizes the output head weights.

        When the model is PTQ-wrapped, the inner ``nn.Linear`` is used for
        GPTQ hooks and the outer PTQ wrapper is used for the forward pass
        (so activation observers collect data).  After GPTQ quantization,
        qparams are injected into PTQ weight observers and ``freeze_qparams()``
        is called.
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, LlamaGPTQConfig)

        ptq_wrapped = self._is_ptq_wrapped(model)

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
                # PTQ-wrapped model has .model.wrapped.norm; raw has .model.norm
                model_norm = model.model
                if ptq_wrapped:
                    model_norm = model_norm.wrapped
                hidden_states = model_norm.norm(hidden_states)
            else:
                norm = self.orig_model.model.norm.to(device)
                hidden_states = norm(hidden_states)
                norm = norm.cpu()
            if len(self.cache_args) > 0:
                self.cache_args[0][batch_idx] = move_to_cpu(hidden_states)

        # For PTQ-wrapped models, lm_head is a PTQWrapper → need inner nn.Linear
        lm_head_module = model.lm_head
        if ptq_wrapped:
            # model.lm_head is PTQWrapper → .wrapped is QuantLinear → .module is nn.Linear
            lm_head_inner = lm_head_module.wrapped.module
        else:
            lm_head_inner = lm_head_module

        gptq = GPTQ(lm_head_inner)
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

        handles = [lm_head_inner.register_forward_hook(add_batch())]

        # Run layer forward over all cached batches to build Hessian/statistics
        device = next(lm_head_inner.parameters()).device  # in case lm_head is located on cpu
        for batch_idx in tqdm(
            range(batch_num),
            desc=f"[lm_head] collecting",
            leave=False,
            unit="batch",
            disable=not gptq_conf.show_progress,
        ):
            hidden_states = gather_single_batch_from_list(self.cache_args, batch_idx)[0]
            hidden_states = move_to_device(hidden_states, device)

            # Forward through the PTQ-wrapped lm_head (activates observers)
            # or the raw lm_head.
            lm_head_module(hidden_states)

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

        # PTQ-wrapped: inject GPTQ qparams and freeze lm_head observers
        if ptq_wrapped:
            self._inject_gptq_qparams_into_layer(
                lm_head_module,
                quantizers,
                verbose=gptq_conf.verbose,
            )

    def _quantize_rotate_lm_head(
        self,
        model: torch.nn.Module,
        quantizers: Dict[str, Any],
        module_name: Dict[torch.nn.Module, str],
    ):
        """
        Apply GPTQ to the rotate_lm_head rotation layer (SpinLlamaForCausalLM only).

        This method quantizes the rotate_lm_head layer weights using GPTQ.
        It should only be called when `LlamaGPTQConfig.quantize_rotate_lm_head` is enabled
        and the model has a rotate_lm_head attribute (i.e., is a SpinLlamaForCausalLM).
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, LlamaGPTQConfig)

        if not self._is_spinllama_model(model):
            return

        # prepare data for rotate_lm_head
        batch_num = self.num_batches
        device = next(model.parameters()).device
        for batch_idx in tqdm(
            range(batch_num),
            desc=f"[rotate_lm_head] re-forward",
            leave=False,
            unit="batch",
            disable=not gptq_conf.show_progress,
        ):
            hidden_states = gather_single_batch_from_list(self.cache_args, batch_idx)[0]
            hidden_states = move_to_device(hidden_states, device)
            if len(self.cache_args) > 0:
                self.cache_args[0][batch_idx] = move_to_cpu(hidden_states)

        ptq_wrapped = self._is_ptq_wrapped(model)

        # For PTQ-wrapped models, rotate_lm_head is a PTQWrapper → need inner nn.Linear
        rotate_lm_head_module = model.rotate_lm_head
        if ptq_wrapped:
            rotate_lm_head_inner = rotate_lm_head_module.wrapped.module
        else:
            rotate_lm_head_inner = rotate_lm_head_module

        gptq = GPTQ(rotate_lm_head_inner)
        full_module_name = "rotate_lm_head"
        weight_bits = self._resolve_weight_bits(
            gptq_conf,
            full_module_name=full_module_name,
            local_module_name="rotate_lm_head",
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

        handles = [rotate_lm_head_inner.register_forward_hook(add_batch())]

        # Run layer forward over all cached batches to build Hessian/statistics
        device = next(rotate_lm_head_inner.parameters()).device
        for batch_idx in tqdm(
            range(batch_num),
            desc=f"[rotate_lm_head] collecting",
            leave=False,
            unit="batch",
            disable=not gptq_conf.show_progress,
        ):
            hidden_states = gather_single_batch_from_list(self.cache_args, batch_idx)[0]
            hidden_states = move_to_device(hidden_states, device)

            # Forward through the PTQ-wrapped rotate_lm_head (activates observers)
            # or the raw rotate_lm_head.
            rotate_lm_head_module(hidden_states)

        # Remove handles
        for h in handles:
            h.remove()

        # Quantize
        if gptq_conf.verbose:
            print(f"[rotate_lm_head] -> Quantizing ...")
        gptq.fasterquant(
            percdamp=gptq_conf.percdamp,
            groupsize=gptq_conf.groupsize,
            actorder=gptq_conf.actorder,
            static_groups=gptq_conf.static_groups,
            verbose=gptq_conf.verbose,
        )
        quantizers[f"rotate_lm_head"] = gptq.quantizer
        gptq.free()

        # PTQ-wrapped: inject GPTQ qparams and freeze rotate_lm_head observers
        if ptq_wrapped:
            self._inject_gptq_qparams_into_layer(
                rotate_lm_head_module,
                quantizers,
                verbose=gptq_conf.verbose,
            )

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

import copy
import functools
import types
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from tico.quantization.algorithm.qwen3_vl_gptq.gptq import GPTQ
from tico.quantization.algorithm.qwen3_vl_gptq.utils import (
    append_batch_to_cache,
    build_module_name_map,
    extract_primary_output,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
    get_deepstack_entry,
    get_quantizable_layers,
    iter_cached_batches,
    maybe_move_cache_to_cpu,
    move_tensor_tree,
    Qwen3VLComponents,
    resolve_qwen3_vl_components,
    should_quantize_text_stage,
    should_quantize_vision_stage,
)
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


class FPInputsCache:
    """
    Cache for saving full-precision inputs to each quantizable submodule (GPTQv2).

    Registers forward hooks on the specified modules and stores their FP inputs
    per batch. The cached inputs are later assigned to ``GPTQ.native_inp`` so that
    ``add_batch`` can compute the dXXT correction matrix.
    """

    def __init__(self, names: list[str]):
        self.names = tuple(names)
        self.fp_cache: dict[str, list] = {name: [] for name in self.names}
        self.handles: list = []

    def _cache_fp_input(self, m, inp, out, name):
        if not inp or not isinstance(inp[0], torch.Tensor):
            return
        self.fp_cache[name].append(inp[0].detach().cpu())

    def add_hook(self, full: dict[str, nn.Module]):
        for name in self.names:
            if name in full:
                self.handles.append(
                    full[name].register_forward_hook(
                        functools.partial(self._cache_fp_input, name=name)
                    )
                )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []


class StopReplay(Exception):
    """Internal exception used to stop model replay at a stage boundary."""


@register_quantizer(Qwen3VLGPTQConfig)
class Qwen3VLGPTQQuantizer(BaseQuantizer):
    """
    Qwen3-VL specific GPTQ quantizer.

    This quantizer stores raw calibration inputs during `prepare()` and performs
    stagewise GPTQ during `convert()`.

    High-level flow:
        1) prepare():
           - intercept model.forward
           - cache raw calibration batches only
           - do not run the real forward

        2) convert():
           - restore the original forward
           - resolve Qwen3-VL components
           - quantize vision stages
           - quantize text stages
           - optionally quantize lm_head
           - attach collected GPTQ quantizer objects to model.quantizers
    """

    def __init__(self, config: Qwen3VLGPTQConfig):
        super().__init__(config)

        self.cache_args: list[list[Any]] = []
        self.cache_kwargs: dict[str, list[Any]] = {}
        self.num_batches: int = 0

        self._orig_model_forward: Optional[Callable[..., Any]] = None
        self._quantizers: dict[str, Any] = {}

        # Separate caches for vision batches (batches with pixel_values)
        # This is needed because vision batches have different kwargs than text batches
        self._vision_cache_args: list[list[Any]] = []
        self._vision_cache_kwargs: dict[str, list[Any]] = {}
        self._num_vision_batches: int = 0

        # GPTQv2: reference to original FP model for collecting FP inputs
        self.orig_model: Optional[nn.Module] = None

    def _resolve_weight_bits(
        self,
        gptq_conf: Qwen3VLGPTQConfig,
        *,
        full_module_name: str,
        local_module_name: str,
    ) -> int:
        """
        Resolve the effective bit-width for a quantized submodule.

        Override keys are matched in the following order:
            1) Full module name.
            2) Stage-local module name.
            3) Full-name suffix.
        """
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
        model: nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Intercept model inputs and cache them without executing the real model.

        Parameters:
            model: Target Qwen3-VL model.
            args: Unused. Kept for API compatibility.
            kwargs: Unused. Kept for API compatibility.

        Returns:
            The model whose forward is temporarily replaced with an input-caching
            wrapper.
        """

        def model_forward_wrapper(_model, *m_args, **m_kwargs):
            assert isinstance(self.config, Qwen3VLGPTQConfig)
            cache_args = maybe_move_cache_to_cpu(
                m_args,
                enabled=self.config.move_cache_to_cpu,
                dtype=self.config.cache_dtype,
            )
            cache_kwargs = maybe_move_cache_to_cpu(
                m_kwargs,
                enabled=self.config.move_cache_to_cpu,
                dtype=self.config.cache_dtype,
            )

            append_batch_to_cache(
                self.cache_args,
                self.cache_kwargs,
                *cache_args,
                **cache_kwargs,
            )

            # Track whether this batch has vision inputs (pixel_values)
            # Vision inputs have 'pixel_values' or 'pixel_values_videos' in kwargs
            # Store vision batches separately for vision stage quantization
            has_vision_input = (
                "pixel_values" in m_kwargs and m_kwargs["pixel_values"] is not None
            ) or (
                "pixel_values_videos" in m_kwargs
                and m_kwargs["pixel_values_videos"] is not None
            )

            if has_vision_input:
                # Also store in separate vision cache
                append_batch_to_cache(
                    self._vision_cache_args,
                    self._vision_cache_kwargs,
                    *cache_args,
                    **cache_kwargs,
                )
                self._num_vision_batches += 1

            self.num_batches += 1
            return None

        self._orig_model_forward = model.forward
        model.forward = types.MethodType(model_forward_wrapper, model)
        return model

    @torch.no_grad()
    def convert(self, model: nn.Module) -> nn.Module:
        """
        Run stagewise GPTQ conversion for Qwen3-VL.

        Parameters:
            model: Prepared Qwen3-VL model.

        Returns:
            Quantized model.
        """
        assert self._orig_model_forward is not None, "prepare() must be called first."
        model.forward = self._orig_model_forward

        gptq_conf = self.config
        assert isinstance(gptq_conf, Qwen3VLGPTQConfig)
        gptq_conf.validate()

        orig_use_cache = self._disable_model_cache(model)
        components = resolve_qwen3_vl_components(model, gptq_conf)
        module_name = build_module_name_map(model)

        # GPTQv2: create a deep copy of the original (unquantized) model.
        # This pristine copy is used to collect true FP inputs for the dXXT
        # correction matrix.
        orig_model: Optional[nn.Module] = None
        orig_components: Optional[Qwen3VLComponents] = None
        orig_model_use_cache: dict[str, Any] = {}

        if gptq_conf.gptq_v2:
            orig_model = self._copy_original_model(model)
            self.orig_model = orig_model
            orig_model_use_cache = self._disable_model_cache(orig_model)
            orig_components = resolve_qwen3_vl_components(orig_model, gptq_conf)

        try:
            if should_quantize_vision_stage(gptq_conf, stage="patch_embed"):
                self._quantize_stage_from_raw_replay(
                    model=model,
                    stage_module=components.visual_patch_embed,
                    module_name=module_name,
                    stage_desc="vision.patch_embed",
                    vision_only=True,  # Only use vision inputs for vision stages
                    orig_model=orig_model,
                    orig_stage_module=(
                        orig_components.visual_patch_embed
                        if orig_components is not None
                        else None
                    ),
                )

            if should_quantize_vision_stage(gptq_conf, stage="blocks"):
                self._quantize_vision_blocks(
                    model=model,
                    components=components,
                    module_name=module_name,
                    orig_model=orig_model,
                    orig_components=orig_components,
                )

            if should_quantize_vision_stage(gptq_conf, stage="merger"):
                self._quantize_stage_from_raw_replay(
                    model=model,
                    stage_module=components.visual_merger,
                    module_name=module_name,
                    stage_desc="vision.merger",
                    vision_only=True,  # Only use vision inputs for vision stages
                    orig_model=orig_model,
                    orig_stage_module=(
                        orig_components.visual_merger
                        if orig_components is not None
                        else None
                    ),
                )

            if should_quantize_vision_stage(gptq_conf, stage="deepstack_mergers"):
                for idx, merger in enumerate(components.visual_deepstack_mergers):
                    self._quantize_stage_from_raw_replay(
                        model=model,
                        stage_module=merger,
                        module_name=module_name,
                        stage_desc=f"vision.deepstack_merger[{idx}]",
                        vision_only=True,  # Only use vision inputs for vision stages
                        orig_model=orig_model,
                        orig_stage_module=(
                            orig_components.visual_deepstack_mergers[idx]
                            if orig_components is not None
                            else None
                        ),
                    )

            if should_quantize_text_stage(gptq_conf, stage="layers"):
                self._quantize_text_layers(
                    model=model,
                    components=components,
                    module_name=module_name,
                    orig_model=orig_model,
                    orig_components=orig_components,
                )

            if should_quantize_text_stage(gptq_conf, stage="lm_head"):
                self._quantize_stage_from_raw_replay(
                    model=model,
                    stage_module=components.lm_head,
                    module_name=module_name,
                    stage_desc="lm_head",
                    orig_model=orig_model,
                    orig_stage_module=(
                        orig_components.lm_head if orig_components is not None else None
                    ),
                )

            model.quantizers = self._quantizers
            return model
        finally:
            self._restore_model_cache(model, orig_use_cache)
            if orig_model is not None:
                self._restore_model_cache(orig_model, orig_model_use_cache)
                orig_model.cpu()
            self.orig_model = None

            self.cache_args.clear()
            self.cache_kwargs.clear()
            self.num_batches = 0
            # Clear vision cache
            self._vision_cache_args.clear()
            self._vision_cache_kwargs.clear()
            self._num_vision_batches = 0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Vision path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quantize_vision_blocks(
        self,
        model: nn.Module,
        components: Qwen3VLComponents,
        module_name: dict[nn.Module, str],
        orig_model: Optional[nn.Module] = None,
        orig_components: Optional[Qwen3VLComponents] = None,
    ) -> None:
        """
        Quantize Qwen3-VL vision blocks in layerwise order using first-block entry
        caches and progressive re-forward.
        """
        # Only use vision inputs for vision block quantization
        block_args, block_kwargs, num_vision_batches = self._collect_stage_entry_inputs(
            model=model,
            target_module=components.visual_blocks[0],
            desc="vision block entry capture",
            vision_only=True,
        )

        if num_vision_batches == 0:
            print(
                "Warning: No vision inputs found in calibration data. "
                "Skipping vision block quantization."
            )
            return

        assert isinstance(self.config, Qwen3VLGPTQConfig)

        # GPTQv2: collect native (FP) entry inputs from the original model
        native_block_args: Optional[list[list[Any]]] = None
        native_block_kwargs: Optional[dict[str, list[Any]]] = None

        if self.config.gptq_v2:
            assert orig_model is not None and orig_components is not None
            self._move_module_to_device(orig_model, self._module_device(model))
            try:
                (
                    native_block_args,
                    native_block_kwargs,
                    native_num_batches,
                ) = self._collect_stage_entry_inputs(
                    model=orig_model,
                    target_module=orig_components.visual_blocks[0],
                    desc="native vision block entry capture",
                    vision_only=True,
                )
            finally:
                orig_model.cpu()
            if native_num_batches != num_vision_batches:
                raise RuntimeError(
                    "Native/current vision block cache sizes differ: "
                    f"{native_num_batches} vs {num_vision_batches}"
                )

        for block_idx, block in enumerate(
            tqdm(
                components.visual_blocks,
                desc="Quantizing vision blocks",
                unit="block",
                disable=not self.config.show_progress,
            )
        ):
            stage_name = module_name.get(block, f"visual.blocks.{block_idx}")
            orig_block = (
                orig_components.visual_blocks[block_idx]
                if orig_components is not None
                else None
            )

            self._quantize_stage_from_stage_cache(
                stage_module=block,
                module_name=module_name,
                cached_args=block_args,
                cached_kwargs=block_kwargs,
                stage_desc=stage_name,
                num_batches=num_vision_batches,
                orig_stage_module=orig_block,
                native_cached_args=native_block_args,
                native_cached_kwargs=native_block_kwargs,
            )

            for batch_idx in tqdm(
                range(num_vision_batches),
                desc=f"[vision block {block_idx}] re-forward",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(block_args, batch_idx)
                kwargs_batch = gather_single_batch_from_dict(block_kwargs, batch_idx)
                args_batch = self._move_batch_to_stage_device(block, args_batch)
                kwargs_batch = self._move_batch_to_stage_device(block, kwargs_batch)

                outs = block(*args_batch, **kwargs_batch)
                hidden_states = extract_primary_output(outs)

                block_args[0][batch_idx] = maybe_move_cache_to_cpu(
                    hidden_states.detach().clone(),
                    enabled=self.config.move_cache_to_cpu,
                    dtype=self.config.cache_dtype,
                )

            # GPTQv2: re-forward through the original (unquantized) block to
            # keep the native cache in sync with a pristine model.
            if native_block_args is not None and native_block_kwargs is not None:
                assert orig_block is not None
                self._move_module_to_device(orig_block, self._module_device(block))
                try:
                    for batch_idx in tqdm(
                        range(num_vision_batches),
                        desc=f"[native vision block {block_idx}] re-forward",
                        leave=False,
                        unit="batch",
                        disable=not self.config.show_progress,
                    ):
                        args_batch = gather_single_batch_from_list(
                            native_block_args, batch_idx
                        )
                        kwargs_batch = gather_single_batch_from_dict(
                            native_block_kwargs, batch_idx
                        )
                        args_batch = self._move_batch_to_stage_device(
                            orig_block, args_batch
                        )
                        kwargs_batch = self._move_batch_to_stage_device(
                            orig_block, kwargs_batch
                        )

                        outs = orig_block(*args_batch, **kwargs_batch)
                        hidden_states = extract_primary_output(outs)

                        native_block_args[0][batch_idx] = maybe_move_cache_to_cpu(
                            hidden_states.detach().clone(),
                            enabled=self.config.move_cache_to_cpu,
                            dtype=self.config.cache_dtype,
                        )
                finally:
                    orig_block.cpu()

    # ------------------------------------------------------------------
    # Text path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quantize_text_layers(
        self,
        model: nn.Module,
        components: Qwen3VLComponents,
        module_name: dict[nn.Module, str],
        orig_model: Optional[nn.Module] = None,
        orig_components: Optional[Qwen3VLComponents] = None,
    ) -> None:
        """
        Quantize text decoder layers in layerwise order using first-layer entry
        caches and progressive re-forward.
        """
        # Text layers process all batches (both vision and text-only)
        layer_args, layer_kwargs, num_batches = self._collect_stage_entry_inputs(
            model=model,
            target_module=components.text_layers[0],
            desc="text layer entry capture",
            vision_only=False,  # Text layers process all inputs
        )
        assert isinstance(self.config, Qwen3VLGPTQConfig)

        # GPTQv2: collect native (FP) entry inputs from the original model
        native_layer_args: Optional[list[list[Any]]] = None
        native_layer_kwargs: Optional[dict[str, list[Any]]] = None

        if self.config.gptq_v2:
            assert orig_model is not None and orig_components is not None
            self._move_module_to_device(orig_model, self._module_device(model))
            try:
                (
                    native_layer_args,
                    native_layer_kwargs,
                    native_num_batches,
                ) = self._collect_stage_entry_inputs(
                    model=orig_model,
                    target_module=orig_components.text_layers[0],
                    desc="native text layer entry capture",
                    vision_only=False,
                )
            finally:
                orig_model.cpu()
            if native_num_batches != num_batches:
                raise RuntimeError(
                    "Native/current text layer cache sizes differ: "
                    f"{native_num_batches} vs {num_batches}"
                )

        for layer_idx, layer in enumerate(
            tqdm(
                components.text_layers,
                desc="Quantizing text layers",
                unit="layer",
                disable=not self.config.show_progress,
            )
        ):
            stage_name = module_name.get(layer, f"text.layers.{layer_idx}")
            orig_layer = (
                orig_components.text_layers[layer_idx]
                if orig_components is not None
                else None
            )

            self._quantize_stage_from_stage_cache(
                stage_module=layer,
                module_name=module_name,
                cached_args=layer_args,
                cached_kwargs=layer_kwargs,
                stage_desc=stage_name,
                num_batches=num_batches,
                orig_stage_module=orig_layer,
                native_cached_args=native_layer_args,
                native_cached_kwargs=native_layer_kwargs,
            )

            for batch_idx in tqdm(
                range(num_batches),
                desc=f"[text layer {layer_idx}] re-forward",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(layer_args, batch_idx)
                kwargs_batch = gather_single_batch_from_dict(layer_kwargs, batch_idx)

                args_batch = self._move_batch_to_stage_device(layer, args_batch)
                kwargs_batch = self._move_batch_to_stage_device(layer, kwargs_batch)

                outs = layer(*args_batch, **kwargs_batch)
                hidden_states = extract_primary_output(outs)

                hidden_states = self._apply_text_post_layer_processing(
                    components=components,
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    kwargs_batch=kwargs_batch,
                )

                layer_args[0][batch_idx] = maybe_move_cache_to_cpu(
                    hidden_states.detach().clone(),
                    enabled=self.config.move_cache_to_cpu,
                    dtype=self.config.cache_dtype,
                )

            # GPTQv2: re-forward through the original (unquantized) layer to
            # keep the native cache in sync with a pristine model.
            if native_layer_args is not None and native_layer_kwargs is not None:
                assert orig_layer is not None and orig_components is not None
                self._move_module_to_device(orig_layer, self._module_device(layer))
                try:
                    for batch_idx in tqdm(
                        range(num_batches),
                        desc=f"[native text layer {layer_idx}] re-forward",
                        leave=False,
                        unit="batch",
                        disable=not self.config.show_progress,
                    ):
                        args_batch = gather_single_batch_from_list(
                            native_layer_args, batch_idx
                        )
                        kwargs_batch = gather_single_batch_from_dict(
                            native_layer_kwargs, batch_idx
                        )
                        args_batch = self._move_batch_to_stage_device(
                            orig_layer, args_batch
                        )
                        kwargs_batch = self._move_batch_to_stage_device(
                            orig_layer, kwargs_batch
                        )

                        outs = orig_layer(*args_batch, **kwargs_batch)
                        hidden_states = extract_primary_output(outs)
                        hidden_states = self._apply_text_post_layer_processing(
                            components=orig_components,
                            layer_idx=layer_idx,
                            hidden_states=hidden_states,
                            kwargs_batch=kwargs_batch,
                        )

                        native_layer_args[0][batch_idx] = maybe_move_cache_to_cpu(
                            hidden_states.detach().clone(),
                            enabled=self.config.move_cache_to_cpu,
                            dtype=self.config.cache_dtype,
                        )
                finally:
                    orig_layer.cpu()

    @torch.no_grad()
    def _apply_text_post_layer_processing(
        self,
        components: Qwen3VLComponents,
        layer_idx: int,
        hidden_states: torch.Tensor,
        kwargs_batch: dict[str, Any],
    ) -> torch.Tensor:
        """
        Apply Qwen3-VL deepstack post-processing after a text decoder layer.
        """
        deepstack_visual_embeds = kwargs_batch.get("deepstack_visual_embeds")
        visual_pos_masks = kwargs_batch.get("visual_pos_masks")
        cur_visual_embeds = get_deepstack_entry(deepstack_visual_embeds, layer_idx)

        if cur_visual_embeds is None:
            return hidden_states
        if visual_pos_masks is None:
            return hidden_states

        language_model = components.language_model
        if not hasattr(language_model, "_deepstack_process"):
            return hidden_states

        return language_model._deepstack_process(
            hidden_states=hidden_states,
            visual_pos_masks=visual_pos_masks,
            visual_embeds=cur_visual_embeds,
        )

    # ------------------------------------------------------------------
    # Generic stage quantization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_native_inputs_from_raw_replay(
        self,
        model: nn.Module,
        subset: dict[str, nn.Module],
        module_name: dict[nn.Module, str],
        cache_args: list[list[Any]],
        cache_kwargs: dict[str, list[Any]],
        num_batches: int,
        stage_desc: str,
    ) -> dict[str, list[torch.Tensor]]:
        """
        GPTQv2: Collect full-precision inputs from the original (unquantized) model
        by replaying raw model inputs.

        Returns:
            Dictionary mapping local_name -> list of FP input tensors (one per batch).
        """
        assert isinstance(self.config, Qwen3VLGPTQConfig)

        # Build full module name -> local name mapping
        full_to_local: dict[str, str] = {}
        for local_name, submodule in subset.items():
            full_name = module_name.get(submodule, local_name)
            full_to_local[full_name] = local_name

        fp_cache = FPInputsCache(list(full_to_local.keys()))

        # Build full name -> module mapping for hook registration
        full_modules: dict[str, nn.Module] = {}
        for local_name, submodule in subset.items():
            full_name = module_name.get(submodule, local_name)
            full_modules[full_name] = submodule

        fp_cache.add_hook(full_modules)

        try:
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"[{stage_desc}] FP input collection",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(cache_args, batch_idx)
                kwargs_batch = gather_single_batch_from_dict(cache_kwargs, batch_idx)
                args_batch = self._move_batch_to_model_device(model, args_batch)
                kwargs_batch = self._move_batch_to_model_device(model, kwargs_batch)
                model(*args_batch, **kwargs_batch)
        finally:
            fp_cache.clear_hook()

        # Remap from full_name -> local_name
        result: dict[str, list[torch.Tensor]] = {}
        for full_name, local_name in full_to_local.items():
            result[local_name] = fp_cache.fp_cache.get(full_name, [])

        return result

    @torch.no_grad()
    def _collect_native_inputs_from_stage_cache(
        self,
        stage_module: nn.Module,
        subset: dict[str, nn.Module],
        cached_args: list[list[Any]],
        cached_kwargs: dict[str, list[Any]],
        stage_desc: str,
        num_batches: int,
    ) -> dict[str, list[torch.Tensor]]:
        """
        GPTQv2: Collect full-precision inputs from the original (unquantized) stage
        module by replaying cached stage-entry inputs.
        """
        fp_cache = FPInputsCache(list(subset.keys()))
        full_modules: dict[str, nn.Module] = {}
        for local_name, submodule in subset.items():
            full_modules[local_name] = submodule
        fp_cache.add_hook(full_modules)

        try:
            for args_batch, kwargs_batch in tqdm(
                iter_cached_batches(cached_args, cached_kwargs, num_batches),
                desc=f"[{stage_desc}] FP input collection",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = self._move_batch_to_stage_device(stage_module, args_batch)
                kwargs_batch = self._move_batch_to_stage_device(
                    stage_module, kwargs_batch
                )
                stage_module(*args_batch, **kwargs_batch)
        finally:
            fp_cache.clear_hook()

        return fp_cache.fp_cache

    @torch.no_grad()
    def _quantize_stage_from_raw_replay(
        self,
        model: nn.Module,
        stage_module: nn.Module,
        module_name: dict[nn.Module, str],
        stage_desc: str,
        vision_only: bool = False,
        orig_model: Optional[nn.Module] = None,
        orig_stage_module: Optional[nn.Module] = None,
    ) -> None:
        """
        Quantize a stage by replaying raw model inputs and collecting statistics
        only for that stage's quantizable submodules.

        Args:
            model: The full model.
            stage_module: The specific module being quantized.
            module_name: Mapping from module to name.
            stage_desc: Description for logging.
            vision_only: If True, only replay batches that have vision inputs
                (pixel_values). This is needed for vision stages to avoid errors
                when text-only inputs lack image tokens.
            orig_model: GPTQv2 original (unquantized) model for FP input collection.
            orig_stage_module: GPTQv2 original stage module for FP input collection.
        """
        subset = get_quantizable_layers(stage_module)
        if not subset:
            return

        gptq_objs = self._build_gptq_objects(
            subset=subset,
            module_name=module_name,
        )

        assert isinstance(self.config, Qwen3VLGPTQConfig)

        # Use separate vision cache for vision-only quantization
        if vision_only:
            if self._num_vision_batches == 0:
                print(
                    f"[{stage_desc}] Warning: No vision inputs found in calibration data. "
                    f"Skipping vision stage quantization."
                )
                return
            cache_args = self._vision_cache_args
            cache_kwargs = self._vision_cache_kwargs
            num_batches = self._num_vision_batches
        else:
            cache_args = self.cache_args
            cache_kwargs = self.cache_kwargs
            num_batches = self.num_batches

        # GPTQv2: Collect FP inputs from the ORIGINAL (unquantized) model
        if self.config.gptq_v2:
            assert orig_model is not None and orig_stage_module is not None
            orig_subset = get_quantizable_layers(orig_stage_module)
            self._move_module_to_device(orig_model, self._module_device(model))
            try:
                native_inputs = self._collect_native_inputs_from_raw_replay(
                    model=orig_model,
                    subset=orig_subset,
                    module_name=module_name,
                    cache_args=cache_args,
                    cache_kwargs=cache_kwargs,
                    num_batches=num_batches,
                    stage_desc=stage_desc,
                )
            finally:
                orig_model.cpu()
            self._assign_native_inputs(gptq_objs, native_inputs)

        handles = []
        for local_name, submodule in subset.items():
            handles.append(
                submodule.register_forward_hook(
                    self._make_add_batch_hook(gptq_objs, local_name)
                )
            )

        try:
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"[{stage_desc}] collecting",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(cache_args, batch_idx)
                kwargs_batch = gather_single_batch_from_dict(cache_kwargs, batch_idx)
                args_batch = self._move_batch_to_model_device(model, args_batch)
                kwargs_batch = self._move_batch_to_model_device(model, kwargs_batch)
                model(*args_batch, **kwargs_batch)
        finally:
            for handle in handles:
                handle.remove()

        self._finalize_stage_quantization(
            gptq_objs=gptq_objs,
            subset=subset,
            module_name=module_name,
            stage_desc=stage_desc,
        )

    @torch.no_grad()
    def _quantize_stage_from_stage_cache(
        self,
        stage_module: nn.Module,
        module_name: dict[nn.Module, str],
        cached_args: list[list[Any]],
        cached_kwargs: dict[str, list[Any]],
        stage_desc: str,
        num_batches: Optional[int] = None,
        orig_stage_module: Optional[nn.Module] = None,
        native_cached_args: Optional[list[list[Any]]] = None,
        native_cached_kwargs: Optional[dict[str, list[Any]]] = None,
    ) -> None:
        """
        Quantize a stage by replaying cached stage-entry inputs.

        Args:
            stage_module: The module to quantize.
            module_name: Mapping from module to name.
            cached_args: Cached positional arguments.
            cached_kwargs: Cached keyword arguments.
            stage_desc: Description for logging.
            num_batches: Number of batches to use. If None, uses self.num_batches.
            orig_stage_module: GPTQv2 original (unquantized) stage module for FP
                input collection.
            native_cached_args: GPTQv2 native entry cache args.
            native_cached_kwargs: GPTQv2 native entry cache kwargs.
        """
        subset = get_quantizable_layers(stage_module)
        if not subset:
            return

        if num_batches is None:
            num_batches = self.num_batches

        gptq_objs = self._build_gptq_objects(
            subset=subset,
            module_name=module_name,
        )

        assert isinstance(self.config, Qwen3VLGPTQConfig)

        # GPTQv2: Collect FP inputs from the ORIGINAL (unquantized) stage module.
        # The native cache contains entry inputs from the pristine model, so
        # forwarding them through orig_stage_module gives true FP submodule inputs.
        if self.config.gptq_v2:
            assert (
                orig_stage_module is not None
                and native_cached_args is not None
                and native_cached_kwargs is not None
            )
            orig_subset = get_quantizable_layers(orig_stage_module)
            self._move_module_to_device(
                orig_stage_module, self._module_device(stage_module)
            )
            try:
                native_inputs = self._collect_native_inputs_from_stage_cache(
                    stage_module=orig_stage_module,
                    subset=orig_subset,
                    cached_args=native_cached_args,
                    cached_kwargs=native_cached_kwargs,
                    stage_desc=stage_desc,
                    num_batches=num_batches,
                )
            finally:
                orig_stage_module.cpu()
            self._assign_native_inputs(gptq_objs, native_inputs)

        handles = []
        for local_name, submodule in subset.items():
            handles.append(
                submodule.register_forward_hook(
                    self._make_add_batch_hook(gptq_objs, local_name)
                )
            )
        try:
            for args_batch, kwargs_batch in tqdm(
                iter_cached_batches(cached_args, cached_kwargs, num_batches),
                desc=f"[{stage_desc}] collecting",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = self._move_batch_to_stage_device(stage_module, args_batch)
                kwargs_batch = self._move_batch_to_stage_device(
                    stage_module, kwargs_batch
                )
                stage_module(*args_batch, **kwargs_batch)
        finally:
            for handle in handles:
                handle.remove()

        self._finalize_stage_quantization(
            gptq_objs=gptq_objs,
            subset=subset,
            module_name=module_name,
            stage_desc=stage_desc,
        )

    def _build_gptq_objects(
        self,
        subset: dict[str, nn.Module],
        module_name: dict[nn.Module, str],
    ) -> dict[str, GPTQ]:
        """
        Create GPTQ objects for a subset of quantizable submodules.
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, Qwen3VLGPTQConfig)

        gptq_objs: dict[str, GPTQ] = {}
        for local_name, submodule in subset.items():
            gptq_obj = GPTQ(submodule, normalize_H=gptq_conf.normalize_H)

            full_name = module_name.get(submodule, local_name)
            weight_bits = self._resolve_weight_bits(
                gptq_conf,
                full_module_name=full_name,
                local_module_name=local_name,
            )

            if (
                gptq_conf.sensitivity is not None
                and isinstance(gptq_conf.sensitivity, dict)
                and full_name in gptq_conf.sensitivity
            ):
                cur_sensitivity = gptq_conf.sensitivity[full_name]
            else:
                cur_sensitivity = None

            gptq_obj.quantizer.configure(
                bits=weight_bits,
                perchannel=gptq_conf.perchannel,
                sym=gptq_conf.symmetric,
                mse=gptq_conf.mse,
                sensitivity=cur_sensitivity,
            )

            # GPTQv2: initialize native_inp list if enabled
            if gptq_conf.gptq_v2:
                gptq_obj.native_inp = []

            gptq_objs[local_name] = gptq_obj

        return gptq_objs

    def _assign_native_inputs(
        self,
        gptq_objs: dict[str, Any],
        native_inputs: dict[str, list[torch.Tensor]],
    ) -> None:
        """
        Assign collected FP inputs to GPTQ objects' native_inp attribute.
        """
        for local_name, gptq_obj in gptq_objs.items():
            if local_name in native_inputs:
                gptq_obj.native_inp = list(native_inputs[local_name])

    def _make_add_batch_hook(
        self,
        gptq_objs: dict[str, GPTQ],
        name: str,
    ) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        """
        Create a forward hook that updates the GPTQ Hessian accumulator.
        """

        def _hook(_module: nn.Module, inp: tuple[Any, ...], out: Any) -> None:
            if not inp:
                return

            first_inp = inp[0]
            out_main = extract_primary_output(out)

            if not isinstance(first_inp, torch.Tensor):
                return
            if not isinstance(out_main, torch.Tensor):
                return

            gptq_objs[name].add_batch(first_inp.data, out_main.data)

        return _hook

    @torch.no_grad()
    def _finalize_stage_quantization(
        self,
        gptq_objs: dict[str, GPTQ],
        subset: dict[str, nn.Module],
        module_name: dict[nn.Module, str],
        stage_desc: str,
    ) -> None:
        """
        Run GPTQ.fasterquant() for all submodules in a stage and store resulting
        quantizer metadata.
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, Qwen3VLGPTQConfig)

        for local_name, submodule in subset.items():
            if gptq_conf.verbose:
                print(f"[{stage_desc}] {local_name} -> Quantizing ...")

            gptq_obj = gptq_objs[local_name]
            gptq_obj.fasterquant(
                percdamp=gptq_conf.percdamp,
                groupsize=gptq_conf.groupsize,
                actorder=gptq_conf.actorder,
                static_groups=gptq_conf.static_groups,
                verbose=gptq_conf.verbose,
                alpha=gptq_conf.gptq_v2_alpha,
            )

            full_name = module_name.get(submodule, local_name)
            self._quantizers[full_name] = gptq_obj.quantizer
            gptq_obj.free()

    # ------------------------------------------------------------------
    # Stage input capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_stage_entry_inputs(
        self,
        model: nn.Module,
        target_module: nn.Module,
        desc: str,
        vision_only: bool = False,
    ) -> tuple[list[list[Any]], dict[str, list[Any]], int]:
        """
        Capture the per-batch inputs fed into a specific stage module by replaying
        raw model inputs and stopping at the stage boundary.

        Args:
            model: The full model.
            target_module: The module whose inputs to capture.
            desc: Description for logging.
            vision_only: If True, only capture inputs from batches with vision data.

        Returns:
            Tuple of (stage_args, stage_kwargs, num_batches) where num_batches is the
            number of batches captured (may be less than self.num_batches if
            vision_only=True).
        """
        stage_args: list[list[Any]] = []
        stage_kwargs: dict[str, list[Any]] = {}
        orig_forward = target_module.forward

        def capture_forward(module, *args, **kwargs):
            append_batch_to_cache(stage_args, stage_kwargs, *args, **kwargs)

            assert isinstance(self.config, Qwen3VLGPTQConfig)
            cached_args = (
                gather_single_batch_from_list(stage_args, len(stage_args[0]) - 1)
                if stage_args
                else []
            )
            cached_kwargs = (
                gather_single_batch_from_dict(
                    stage_kwargs,
                    len(next(iter(stage_kwargs.values()))) - 1,
                )
                if stage_kwargs
                else {}
            )

            cached_args = maybe_move_cache_to_cpu(
                cached_args,
                enabled=self.config.move_cache_to_cpu,
                dtype=self.config.cache_dtype,
            )
            cached_kwargs = maybe_move_cache_to_cpu(
                cached_kwargs,
                enabled=self.config.move_cache_to_cpu,
                dtype=self.config.cache_dtype,
            )

            for idx, item in enumerate(cached_args):
                stage_args[idx][-1] = item
            for key, value in cached_kwargs.items():
                stage_kwargs[key][-1] = value

            raise StopReplay

        target_module.forward = types.MethodType(capture_forward, target_module)

        # Use separate vision cache for vision-only quantization
        if vision_only:
            cache_args = self._vision_cache_args
            cache_kwargs = self._vision_cache_kwargs
            num_batches = self._num_vision_batches
        else:
            cache_args = self.cache_args
            cache_kwargs = self.cache_kwargs
            num_batches = self.num_batches

        assert isinstance(self.config, Qwen3VLGPTQConfig)
        try:
            for batch_idx in tqdm(
                range(num_batches),
                desc=desc,
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(cache_args, batch_idx)
                kwargs_batch = gather_single_batch_from_dict(cache_kwargs, batch_idx)
                args_batch = self._move_batch_to_model_device(model, args_batch)
                kwargs_batch = self._move_batch_to_model_device(model, kwargs_batch)

                try:
                    model(*args_batch, **kwargs_batch)
                except StopReplay:
                    pass
        finally:
            target_module.forward = orig_forward

        return stage_args, stage_kwargs, num_batches

    # ------------------------------------------------------------------
    # Device / dtype helpers
    # ------------------------------------------------------------------

    def _move_batch_to_model_device(self, model: nn.Module, batch: Any) -> Any:
        """
        Move a cached batch to a model device for raw replay.
        """
        try:
            device = next(model.parameters()).device
        except StopIteration:
            return batch
        return move_tensor_tree(batch, device=device)

    def _move_batch_to_stage_device(self, stage_module: nn.Module, batch: Any) -> Any:
        """
        Move a cached stage batch to the stage module device.
        """
        try:
            device = next(stage_module.parameters()).device
        except StopIteration:
            return batch
        return move_tensor_tree(batch, device=device)

    def _module_device(self, module: nn.Module) -> torch.device:
        """
        Get the device of a module (from parameters, buffers, or CPU fallback).
        """
        try:
            return next(module.parameters()).device
        except StopIteration:
            try:
                return next(module.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    def _move_module_to_device(
        self,
        module: nn.Module,
        device: torch.device | str,
    ) -> None:
        """
        Move a module to the specified device.
        """
        module.to(device)

    def _copy_original_model(self, model: nn.Module) -> nn.Module:
        """
        Create a deep copy of the model on CPU for GPTQv2 FP input collection.

        The original model is moved to CPU before copying to avoid GPU memory
        duplication, then moved back to its original device.
        """
        device = self._module_device(model)
        model.cpu()
        orig_model = copy.deepcopy(model)
        model.to(device)
        orig_model.eval()
        return orig_model

    # ------------------------------------------------------------------
    # Cache control helpers
    # ------------------------------------------------------------------

    def _disable_model_cache(self, model: nn.Module) -> dict[str, Any]:
        """
        Disable cache-related flags commonly used by Qwen3-VL / HF models.
        """
        saved: dict[str, Any] = {}

        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            saved["model.config.use_cache"] = model.config.use_cache
            model.config.use_cache = False

        if hasattr(model, "config") and hasattr(model.config, "text_config"):
            text_config = model.config.text_config
            if hasattr(text_config, "use_cache"):
                saved["model.config.text_config.use_cache"] = text_config.use_cache
                text_config.use_cache = False

        return saved

    def _restore_model_cache(self, model: nn.Module, saved: dict[str, Any]) -> None:
        """
        Restore cache-related flags saved by `_disable_model_cache`.
        """
        if "model.config.use_cache" in saved:
            model.config.use_cache = saved["model.config.use_cache"]

        if "model.config.text_config.use_cache" in saved:
            model.config.text_config.use_cache = saved[
                "model.config.text_config.use_cache"
            ]

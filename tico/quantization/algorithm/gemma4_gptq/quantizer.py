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

import types
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from tico.quantization.algorithm.gemma4_gptq.gptq import GPTQ
from tico.quantization.algorithm.gemma4_gptq.utils import (
    append_batch_to_cache,
    build_module_name_map,
    extract_primary_output,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
    get_quantizable_layers,
    iter_cached_batches,
    maybe_move_cache_to_cpu,
    move_tensor_tree,
    Gemma4Components,
    resolve_gemma4_components,
    should_quantize_text_stage,
    should_quantize_vision_stage,
)
from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


class StopReplay(Exception):
    """Internal exception used to stop model replay at a stage boundary."""


@register_quantizer(Gemma4GPTQConfig)
class Gemma4GPTQQuantizer(BaseQuantizer):
    """
    Gemma4 specific GPTQ quantizer.

    This quantizer stores raw calibration inputs during ``prepare()`` and
    performs stagewise GPTQ during ``convert()``.

    High-level flow:
        1) prepare():
           - intercept model.forward
           - cache raw calibration batches only
           - do not run the real forward

        2) convert():
           - restore the original forward
           - resolve Gemma4 components
           - quantize vision stages
           - quantize text stages
           - optionally quantize lm_head
           - attach collected GPTQ quantizer objects to model.quantizers

    Differences from Qwen3VLGPTQQuantizer:

    - **No deepstack**: Qwen3-VL replays ``_deepstack_process`` after each text
      decoder layer.  Gemma4 has no deepstack; Per-Layer Embeddings (PLE) are
      handled internally by each text decoder layer, so no post-layer
      processing is needed.
    - **Vision stages**: Gemma4 has ``patch_embedder``, ``encoder.layers``,
      ``pooler``, and ``embed_vision`` (multimodal embedder).  Qwen3-VL has
      ``patch_embed``, ``blocks``, ``merger``, and ``deepstack_merger_list``.
    - **Vision detection**: Gemma4 checks for ``pixel_values`` only;
      Qwen3-VL also checks ``pixel_values_videos``.
    """

    def __init__(self, config: Gemma4GPTQConfig):
        super().__init__(config)

        self.cache_args: list[list[Any]] = []
        self.cache_kwargs: dict[str, list[Any]] = {}
        self.num_batches: int = 0

        self._orig_model_forward: Optional[Callable[..., Any]] = None
        self._quantizers: dict[str, Any] = {}

        # Separate caches for vision batches (batches with pixel_values)
        # This is needed because vision batches have different kwargs than
        # text-only batches.
        self._vision_cache_args: list[list[Any]] = []
        self._vision_cache_kwargs: dict[str, list[Any]] = {}
        self._num_vision_batches: int = 0

    def _resolve_weight_bits(
        self,
        gptq_conf: Gemma4GPTQConfig,
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
            model: Target Gemma4 model.
            args: Unused. Kept for API compatibility.
            kwargs: Unused. Kept for API compatibility.

        Returns:
            The model whose forward is temporarily replaced with an
            input-caching wrapper.
        """

        def model_forward_wrapper(_model, *m_args, **m_kwargs):
            assert isinstance(self.config, Gemma4GPTQConfig)
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
            # Gemma4 uses 'pixel_values' for image inputs.
            # Unlike Qwen3-VL, Gemma4 does not have 'pixel_values_videos'.
            has_vision_input = (
                "pixel_values" in m_kwargs and m_kwargs["pixel_values"] is not None
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
        Run stagewise GPTQ conversion for Gemma4.

        Parameters:
            model: Prepared Gemma4 model.

        Returns:
            Quantized model.
        """
        assert self._orig_model_forward is not None, "prepare() must be called first."
        model.forward = self._orig_model_forward

        gptq_conf = self.config
        assert isinstance(gptq_conf, Gemma4GPTQConfig)
        gptq_conf.validate()

        orig_use_cache = self._disable_model_cache(model)
        components = resolve_gemma4_components(model, gptq_conf)
        module_name = build_module_name_map(model)

        if should_quantize_vision_stage(gptq_conf, stage="patch_embed"):
            self._quantize_stage_from_raw_replay(
                model=model,
                stage_module=components.vision_patch_embed,
                module_name=module_name,
                stage_desc="vision.patch_embed",
                vision_only=True,  # Only use vision inputs for vision stages
            )

        if should_quantize_vision_stage(gptq_conf, stage="blocks"):
            self._quantize_vision_blocks(
                model=model,
                components=components,
                module_name=module_name,
            )

        if should_quantize_vision_stage(gptq_conf, stage="pooler"):
            self._quantize_stage_from_raw_replay(
                model=model,
                stage_module=components.vision_pooler,
                module_name=module_name,
                stage_desc="vision.pooler",
                vision_only=True,
            )

        if should_quantize_vision_stage(gptq_conf, stage="multimodal_embedder"):
            self._quantize_stage_from_raw_replay(
                model=model,
                stage_module=components.multimodal_embedder,
                module_name=module_name,
                stage_desc="vision.multimodal_embedder",
                vision_only=True,
            )

        if should_quantize_text_stage(gptq_conf, stage="layers"):
            self._quantize_text_layers(
                model=model,
                components=components,
                module_name=module_name,
            )

        if should_quantize_text_stage(gptq_conf, stage="lm_head"):
            self._quantize_stage_from_raw_replay(
                model=model,
                stage_module=components.lm_head,
                module_name=module_name,
                stage_desc="lm_head",
            )

        self._restore_model_cache(model, orig_use_cache)

        self.cache_args.clear()
        self.cache_kwargs.clear()
        self.num_batches = 0
        # Clear vision cache
        self._vision_cache_args.clear()
        self._vision_cache_kwargs.clear()
        self._num_vision_batches = 0
        model.quantizers = self._quantizers
        return model

    # ------------------------------------------------------------------
    # Vision path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quantize_vision_blocks(
        self,
        model: nn.Module,
        components: Gemma4Components,
        module_name: dict[nn.Module, str],
    ) -> None:
        """
        Quantize Gemma4 vision encoder blocks in layerwise order using
        first-block entry caches and progressive re-forward.
        """
        # Only use vision inputs for vision block quantization
        block_args, block_kwargs, num_vision_batches = self._collect_stage_entry_inputs(
            model=model,
            target_module=components.vision_encoder_layers[0],
            desc="vision block entry capture",
            vision_only=True,
        )

        if num_vision_batches == 0:
            print(
                "Warning: No vision inputs found in calibration data. "
                "Skipping vision block quantization."
            )
            return

        assert isinstance(self.config, Gemma4GPTQConfig)
        for block_idx, block in enumerate(
            tqdm(
                components.vision_encoder_layers,
                desc="Quantizing vision blocks",
                unit="block",
                disable=not self.config.show_progress,
            )
        ):
            stage_name = module_name.get(
                block, f"vision_tower.encoder.layers.{block_idx}"
            )

            self._quantize_stage_from_stage_cache(
                stage_module=block,
                module_name=module_name,
                cached_args=block_args,
                cached_kwargs=block_kwargs,
                stage_desc=stage_name,
                num_batches=num_vision_batches,
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

    # ------------------------------------------------------------------
    # Text path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quantize_text_layers(
        self,
        model: nn.Module,
        components: Gemma4Components,
        module_name: dict[nn.Module, str],
    ) -> None:
        """
        Quantize text decoder layers in layerwise order.

        Unlike Qwen3-VL, Gemma4 text decoder layers do not require deepstack
        post-processing.  Per-Layer Embeddings (PLE) are handled internally by
        each decoder layer, so the re-forward simply feeds the output hidden
        states as input to the next layer.

        **Key difference from Qwen3-VL**: Gemma4 uses mixed layer types
        (``sliding_attention`` with ``head_dim=256`` and ``full_attention``
        with ``global_head_dim=512``).  Each layer type receives different
        ``position_embeddings`` and ``attention_mask`` from the text model.
        Additionally, each layer receives a different ``per_layer_input`` slice.

        To handle this correctly, we intercept the text model's forward to
        capture the full per-layer data (per_layer_inputs tensor,
        position_embeddings dict, causal_mask_mapping dict, shared_kv_states,
        position_ids) for each batch.  Then for each layer, we select the
        correct slices before replaying.
        """
        assert isinstance(self.config, Gemma4GPTQConfig)

        language_model = components.language_model
        text_config = getattr(language_model, "config", None)
        layer_types = getattr(text_config, "layer_types", None)
        num_layers = len(components.text_layers)

        # ------------------------------------------------------------------
        # Step 1: Capture per-layer data by intercepting the text model forward
        # ------------------------------------------------------------------
        # For each batch, we capture:
        #   - hidden_states (input to layer 0)
        #   - per_layer_inputs (full tensor [batch, seq, num_layers, dim])
        #   - position_embeddings (dict: layer_type -> tensor)
        #   - causal_mask_mapping (dict: layer_type -> tensor)
        #   - shared_kv_states (dict)
        #   - position_ids
        #   - past_key_values
        #   - extra kwargs
        captured_batches: list[dict[str, Any]] = []

        orig_lm_forward = language_model.forward

        def capture_lm_forward(_lm, *args, **kwargs):
            """Capture all per-layer data from the text model forward."""
            # The full model passes inputs_embeds (not input_ids) to the
            # language model.  It also passes per_layer_inputs (already
            # computed) and attention_mask (already a dict mapping).
            input_ids = kwargs.get("input_ids")
            inputs_embeds = kwargs.get("inputs_embeds")
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = language_model.embed_tokens(input_ids)
            if inputs_embeds is None:
                raise StopReplay

            # position_ids may be passed or need to be computed
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ).unsqueeze(0)

            # per_layer_inputs: The text model computes PLE internally.
            # If per_layer_inputs is None, it computes token-identity via
            # get_per_layer_inputs(input_ids, inputs_embeds).
            # Then it ALWAYS projects via project_per_layer_inputs.
            # We must replicate this here because we bypass the text model's
            # forward and pass per_layer_input directly to each layer.
            per_layer_inputs = kwargs.get("per_layer_inputs")
            if hasattr(language_model, "hidden_size_per_layer_input") and language_model.hidden_size_per_layer_input:
                if per_layer_inputs is None:
                    per_layer_inputs = language_model.get_per_layer_inputs(
                        input_ids, inputs_embeds
                    )
                per_layer_inputs = language_model.project_per_layer_inputs(
                    inputs_embeds, per_layer_inputs
                )

            # Compute position_embeddings for each layer type.

            # This is the key difference from Qwen3-VL: Gemma4 has mixed
            # layer types with different head_dim, so each type needs its
            # own rotary embeddings.
            position_embeddings: dict[str, Any] = {}
            hidden_states = inputs_embeds
            for layer_type in language_model.unique_layer_types:
                pos_emb = language_model.rotary_emb(
                    hidden_states, position_ids, layer_type
                )
                # rotary_emb may return a tuple (cos, sin) or a single tensor
                if isinstance(pos_emb, tuple):
                    position_embeddings[layer_type] = tuple(
                        t.detach().clone() for t in pos_emb
                    )
                else:
                    position_embeddings[layer_type] = pos_emb.detach().clone()

            # attention_mask may already be a dict (causal_mask_mapping) if
            # passed by the full model, or a tensor if called directly.
            attention_mask = kwargs.get("attention_mask")
            if isinstance(attention_mask, dict):
                causal_mask_mapping = attention_mask
            else:
                # Compute causal_mask_mapping for each layer type
                from transformers.masking_utils import (
                    create_causal_mask,
                    create_sliding_window_causal_mask,
                )

                past_key_values = kwargs.get("past_key_values")
                mask_kwargs = {
                    "config": language_model.config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(
                        **mask_kwargs
                    ),
                }

            past_key_values = kwargs.get("past_key_values")

            captured_batches.append(
                {
                    "hidden_states": hidden_states.detach().clone(),
                    "per_layer_inputs": (
                        per_layer_inputs.detach().clone()
                        if per_layer_inputs is not None
                        else None
                    ),
                    "position_embeddings": position_embeddings,
                    "causal_mask_mapping": causal_mask_mapping,
                    # shared_kv_states is NOT captured here because it is a
                    # mutable dict that gets populated during the forward pass
                    # by layers with store_full_length_kv=True.  Instead, we
                    # create a fresh empty dict for each batch during replay
                    # and let the layers fill it naturally.
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                }
            )

            raise StopReplay

        language_model.forward = types.MethodType(capture_lm_forward, language_model)

        num_batches = self.num_batches

        try:
            for batch_idx in tqdm(
                range(num_batches),
                desc="text layer entry capture",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(self.cache_args, batch_idx)

                kwargs_batch = gather_single_batch_from_dict(
                    self.cache_kwargs, batch_idx
                )
                args_batch = self._move_batch_to_model_device(model, args_batch)
                kwargs_batch = self._move_batch_to_model_device(model, kwargs_batch)
                try:
                    model(*args_batch, **kwargs_batch)
                except StopReplay:
                    pass
        finally:
            language_model.forward = orig_lm_forward

        num_captured = len(captured_batches)

        # ------------------------------------------------------------------
        # Step 2: Quantize each layer using captured per-layer data
        # ------------------------------------------------------------------
        # shared_kv_states is a mutable dict that gets populated during the
        # forward pass by layers with store_full_length_kv=True and read by
        # layers with is_kv_shared_layer=True.  It must be created ONCE per
        # batch and shared across ALL layers, so that when we reach a shared
        # layer (e.g. layer 15), the dict already contains the KV states
        # stored by earlier layers.
        from collections import UserDict

        shared_kv_states_per_batch: list[UserDict] = [
            UserDict() for _ in range(num_captured)
        ]

        for layer_idx, layer in enumerate(
            tqdm(
                components.text_layers,
                desc="Quantizing text layers",
                unit="layer",
                disable=not self.config.show_progress,
            )
        ):
            stage_name = module_name.get(
                layer, f"language_model.layers.{layer_idx}"
            )

            layer_type = None
            if layer_types is not None and layer_idx < len(layer_types):
                layer_type = layer_types[layer_idx]

            # Build per-layer args/kwargs from captured data.
            # We pass ALL arguments as keyword arguments to avoid positional
            # argument conflicts with the layer's forward signature.
            # The cache structure must be: layer_args[arg_idx][batch_idx]

            # and layer_kwargs[key][batch_idx].
            layer_args: list[list[Any]] = []  # empty - all args passed as kwargs
            layer_kwargs: dict[str, list[Any]] = {}
            for batch_idx, batch_data in enumerate(captured_batches):
                pli = batch_data["per_layer_inputs"]
                per_layer_input = (
                    pli[:, :, layer_idx, :]
                    if pli is not None and layer_idx < pli.shape[2]
                    else None
                )

                pos_emb = batch_data["position_embeddings"].get(layer_type)
                attn_mask = batch_data["causal_mask_mapping"].get(layer_type)
                for key, value in [
                    ("hidden_states", batch_data["hidden_states"]),
                    ("per_layer_input", per_layer_input),
                    ("position_embeddings", pos_emb),
                    ("attention_mask", attn_mask),
                    ("position_ids", batch_data["position_ids"]),
                    # Use the shared_kv_states created once per batch.
                    # This dict is populated during the forward pass by layers
                    # with store_full_length_kv=True and read by layers with
                    # is_kv_shared_layer=True.  It must be the SAME dict across
                    # all layers for a given batch.
                    ("shared_kv_states", shared_kv_states_per_batch[batch_idx]),
                    ("past_key_values", batch_data["past_key_values"]),
                ]:
                    layer_kwargs.setdefault(key, []).append(value)

            self._quantize_stage_from_stage_cache(
                stage_module=layer,
                module_name=module_name,
                cached_args=layer_args,
                cached_kwargs=layer_kwargs,
                stage_desc=stage_name,
                num_batches=num_captured,
            )

            # Re-forward to get output hidden_states for next layer
            for batch_idx in tqdm(
                range(num_captured),
                desc=f"[text layer {layer_idx}] re-forward",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = gather_single_batch_from_list(layer_args, batch_idx)
                kwargs_batch = gather_single_batch_from_dict(layer_kwargs, batch_idx)

                args_batch = self._move_batch_to_stage_device(layer, args_batch)
                # Move all kwargs EXCEPT shared_kv_states to the stage device.
                # shared_kv_states is a mutable UserDict that must be shared
                # across all layers for a given batch.  move_tensor_tree would
                # create a COPY of it, breaking the sharing.  Since it starts
                # empty and gets populated by layers on the same device, we
                # pass it through unchanged.
                shared_kv = kwargs_batch.pop("shared_kv_states", None)
                kwargs_batch = self._move_batch_to_stage_device(layer, kwargs_batch)
                if shared_kv is not None:
                    kwargs_batch["shared_kv_states"] = shared_kv


                outs = layer(*args_batch, **kwargs_batch)
                hidden_states = extract_primary_output(outs)

                # No deepstack post-processing needed for Gemma4.
                # PLE is handled internally by each decoder layer.
                # Update hidden_states in kwargs for next layer.
                layer_kwargs["hidden_states"][batch_idx] = maybe_move_cache_to_cpu(
                    hidden_states.detach().clone(),
                    enabled=self.config.move_cache_to_cpu,
                    dtype=self.config.cache_dtype,
                )



    # ------------------------------------------------------------------
    # Generic stage quantization helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _quantize_stage_from_raw_replay(
        self,
        model: nn.Module,
        stage_module: nn.Module,
        module_name: dict[nn.Module, str],
        stage_desc: str,
        vision_only: bool = False,
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
        """
        subset = get_quantizable_layers(stage_module)
        if not subset:
            return

        gptq_objs = self._build_gptq_objects(
            subset=subset,
            module_name=module_name,
        )

        handles = []
        for local_name, submodule in subset.items():
            handles.append(
                submodule.register_forward_hook(
                    self._make_add_batch_hook(gptq_objs, local_name)
                )
            )
        assert isinstance(self.config, Gemma4GPTQConfig)

        # Use separate vision cache for vision-only quantization
        if vision_only:
            if self._num_vision_batches == 0:
                print(
                    f"[{stage_desc}] Warning: No vision inputs found in "
                    f"calibration data. Skipping vision stage quantization."
                )
                for handle in handles:
                    handle.remove()
                return
            cache_args = self._vision_cache_args
            cache_kwargs = self._vision_cache_kwargs
            num_batches = self._num_vision_batches
        else:
            cache_args = self.cache_args
            cache_kwargs = self.cache_kwargs
            num_batches = self.num_batches

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

        handles = []
        for local_name, submodule in subset.items():
            handles.append(
                submodule.register_forward_hook(
                    self._make_add_batch_hook(gptq_objs, local_name)
                )
            )
        assert isinstance(self.config, Gemma4GPTQConfig)
        try:
            for args_batch, kwargs_batch in tqdm(
                iter_cached_batches(cached_args, cached_kwargs, num_batches),
                desc=f"[{stage_desc}] collecting",
                leave=False,
                unit="batch",
                disable=not self.config.show_progress,
            ):
                args_batch = self._move_batch_to_stage_device(stage_module, args_batch)
                # shared_kv_states is a mutable UserDict that must NOT be
                # copied by move_tensor_tree.  Pop it out, move the rest,
                # then put it back.
                shared_kv = kwargs_batch.pop("shared_kv_states", None)
                kwargs_batch = self._move_batch_to_stage_device(
                    stage_module, kwargs_batch
                )
                if shared_kv is not None:
                    kwargs_batch["shared_kv_states"] = shared_kv
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
        assert isinstance(gptq_conf, Gemma4GPTQConfig)

        gptq_objs: dict[str, GPTQ] = {}
        for local_name, submodule in subset.items():
            gptq_obj = GPTQ(submodule)

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
            gptq_objs[local_name] = gptq_obj

        return gptq_objs

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
        Run GPTQ.fasterquant() for all submodules in a stage and store
        resulting quantizer metadata.
        """
        gptq_conf = self.config
        assert isinstance(gptq_conf, Gemma4GPTQConfig)

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
        Capture the per-batch inputs fed into a specific stage module by
        replaying raw model inputs and stopping at the stage boundary.

        Args:
            model: The full model.
            target_module: The module whose inputs to capture.
            desc: Description for logging.
            vision_only: If True, only capture inputs from batches with vision
                data.

        Returns:
            Tuple of (stage_args, stage_kwargs, num_batches) where num_batches
            is the number of batches captured (may be less than
            self.num_batches if vision_only=True).
        """
        stage_args: list[list[Any]] = []
        stage_kwargs: dict[str, list[Any]] = {}
        orig_forward = target_module.forward

        def capture_forward(module, *args, **kwargs):
            append_batch_to_cache(stage_args, stage_kwargs, *args, **kwargs)

            assert isinstance(self.config, Gemma4GPTQConfig)
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

        assert isinstance(self.config, Gemma4GPTQConfig)
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

    # ------------------------------------------------------------------
    # Cache control helpers
    # ------------------------------------------------------------------

    def _disable_model_cache(self, model: nn.Module) -> dict[str, Any]:
        """
        Disable cache-related flags commonly used by Gemma4 / HF models.
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
        Restore cache-related flags saved by ``_disable_model_cache``.
        """
        if "model.config.use_cache" in saved:
            model.config.use_cache = saved["model.config.use_cache"]

        if "model.config.text_config.use_cache" in saved:
            model.config.text_config.use_cache = saved[
                "model.config.text_config.use_cache"
            ]

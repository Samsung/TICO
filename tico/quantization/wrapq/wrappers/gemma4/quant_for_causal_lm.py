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

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM")
class QuantGemma4ForCausalLM(QuantModuleBase):
    """PTQ wrapper for Gemma4 text-only causal LM."""

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        assert_gemma4_e2b_no_moe(fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config
        self.model = PTQWrapper(
            fp_model.model,
            qcfg=qcfg.child("model") if qcfg else None,
            fp_name=join_name(fp_name, "model"),
        )
        self.lm_head = PTQWrapper(
            fp_model.lm_head,
            qcfg=qcfg.child("lm_head") if qcfg else None,
            fp_name=join_name(fp_name, "lm_head"),
        )

        # Observers for the logit softcapping path.
        self.obs_logit_softcapping_div = self._make_obs("logit_softcapping_div")
        self.obs_logit_softcapping_tanh = self._make_obs("logit_softcapping_tanh")
        self.obs_logits = self._make_obs("logits")

    def forward(self, *args, logits_to_keep: int | torch.Tensor = 0, **kwargs):
        """Run the wrapped causal LM model (calibration path).

        Mirrors ``Gemma4ForCausalLM.forward`` including logit softcapping.
        Fake-quantization observers are inserted after the ``tanh`` and on
        the final logits so that the export path carries correct qparam
        metadata.

        TODO: Return ``Gemma4CausalLMOutputWithPast`` for full HF compatibility.
        """
        outputs = self.model(*args, **kwargs)
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs
        )
        # Match the original's logits_to_keep handling: int → slice, tensor → index.
        if isinstance(logits_to_keep, int) and logits_to_keep:
            slice_indices = slice(-logits_to_keep, None)
        elif isinstance(logits_to_keep, torch.Tensor):
            slice_indices = logits_to_keep
        else:
            slice_indices = slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return self._apply_logit_softcapping(logits)

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logit softcapping with fake-quantization observers.

        Mirrors the original ``Gemma4ForCausalLM`` softcapping:
        ``logits = tanh(logits / softcap) * softcap``.

        Three observers are inserted so that every graph node in the
        softcapping chain carries quantization parameter metadata:
        - ``obs_logit_softcapping_div``  — after the division
        - ``obs_logit_softcapping_tanh`` — after the tanh
        - ``obs_logits``                 — on the final logits
        """
        final_logit_softcapping = self.config.final_logit_softcapping
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = self._fq(logits, self.obs_logit_softcapping_div)
            logits = torch.tanh(logits)
            logits = self._fq(logits, self.obs_logit_softcapping_tanh)
            logits = logits * final_logit_softcapping

        logits = self._fq(logits, self.obs_logits)
        return logits

    def forward_export(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
        attention_masks: Optional[dict] = None,
        position_embeddings: Optional[dict] = None,
        logits_to_keep: int = 0,
    ) -> torch.Tensor:
        """Run the export path for causal LM.

        Uses ``self.prefill_layers`` (created by ``as_export_module()``) to
        run the text decoder layers with static shapes, then applies the LM
        head with ``logits_to_keep`` slicing and logit softcapping.

        This method assumes the CPU runtime has already performed:
        - Token embedding
        - PLE computation (if enabled)
        - Mask and RoPE generation per layer type

        Args:
            inputs_embeds: Pre-fused text embeddings, shape ``(1, S, H)``.
            per_layer_inputs: PLE tensor, shape ``(1, S, L, P)`` or None.
            attention_masks: Dict mapping layer type to additive mask tensors.
            position_embeddings: Dict mapping layer type to ``(cos, sin)`` tuples.
            logits_to_keep: Number of trailing positions to compute logits for.

        Returns:
            Logits tensor with shape ``(1, S', vocab_size)`` where ``S'``
            depends on ``logits_to_keep``.
        """
        if not hasattr(self, "prefill_layers"):
            raise RuntimeError(
                "forward_export() requires as_export_module() to be called first."
            )

        text_model = self.model.wrapped  # QuantGemma4TextModel

        # mypy: attention_masks and position_embeddings are required when
        # running decoder layers; narrow away None before the loop.
        assert attention_masks is not None
        assert position_embeddings is not None

        hidden_states = inputs_embeds

        # Run text decoder layers with precomputed masks and RoPE.
        for i, decoder_layer in enumerate(self.prefill_layers):
            layer_type = self.config.layer_types[i]
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            output = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                attention_mask=attention_masks[layer_type],
                position_embeddings=position_embeddings[layer_type],
            )
            # Layer output is (hidden_states,) or (hidden_states, kv).
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

        # Final norm.
        hidden_states = text_model.norm(hidden_states)

        # Slice hidden states for logits_to_keep.
        slice_indices = slice(-logits_to_keep, None) if logits_to_keep else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return self._apply_logit_softcapping(logits)

    def as_export_module(self, mode: str = "prefill", **kwargs) -> nn.Module:
        """Prepare the model for torch.export and return an export adapter.

        This method:
        1. Asserts that the model is in QUANT mode
        2. Verifies all observers are calibrated
        3. Creates text decoder layer export adapters via ``as_export_module()``
        4. Returns a ``Gemma4ForCausalLMExportAdapter`` wrapping this module

        Args:
            mode: Export mode — ``"prefill"`` (logits_to_keep=0) or
                ``"decode"`` (logits_to_keep=1).
            **kwargs: Additional arguments (unused).

        Returns:
            ``Gemma4ForCausalLMExportAdapter`` wrapping this module.
        """
        assert self._mode is Mode.QUANT, "Must be in QUANT mode for export"

        if mode not in ("prefill", "decode"):
            raise ValueError(f"Unsupported export mode: {mode!r}")

        # Make sure that all observers are calibrated.
        for obs in self._all_observers():
            assert obs.has_qparams, f"Observer {obs.name} has not been calibrated"

        text_model = self.model.wrapped  # QuantGemma4TextModel

        # Create text decoder layer export adapters.
        if mode == "prefill":
            self.prefill_layers = nn.ModuleList(
                [
                    layer.wrapped.as_export_module(mode="prefill", return_kv=True)
                    for layer in text_model.layers
                ]
            )
        elif mode == "decode":
            self.prefill_layers = nn.ModuleList(
                [
                    layer.wrapped.as_export_module(mode="decode", return_kv=True)
                    for layer in text_model.layers
                ]
            )

        # Register fake-quant meta kernels for dynamic export.
        from tico.quantization.wrapq.wrappers.llama.export_adapters import (
            register_fake_quant_meta_kernels_for_dynamic_export,
        )

        register_fake_quant_meta_kernels_for_dynamic_export()

        logits_to_keep = 0 if mode == "prefill" else 1

        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4ForCausalLMExportAdapter,
        )

        return Gemma4ForCausalLMExportAdapter(
            wrapped_model=self, logits_to_keep=logits_to_keep
        )

    def generate(self, *args, **kwargs):
        """Delegate generation to the original module until static runtime is wired."""
        return self.module.generate(*args, **kwargs)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
            self.obs_logit_softcapping_div,
            self.obs_logit_softcapping_tanh,
            self.obs_logits,
        )

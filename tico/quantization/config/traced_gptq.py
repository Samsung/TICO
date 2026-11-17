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
"""Configuration for the structure-free, whole-model traced GPTQ quantizer."""

from dataclasses import dataclass

from tico.quantization.config.gptq import GPTQConfig

__all__ = ["TracedGPTQConfig"]


@dataclass
class TracedGPTQConfig(GPTQConfig):
    """Configuration for :class:`TracedGPTQQuantizer`.

    Inherits every GPTQ knob (``weight_bits``, ``percdamp``, ``groupsize``,
    ``actorder``, ``double_precision``, ...) from :class:`GPTQConfig`.

    Attributes
    ----------
    layer_group_regex : str | None
        Regex with one capture group that maps a quantized linear's weight
        FQN to its decoder-layer id, used by ``gptq_v2`` +
        ``use_orig_model_inference`` to locate decoder-layer boundaries in
        the traced graph (the boundary is the first quantized linear of each
        new layer group). The default matches Llama-family
        ``model.layers.N.*`` FQNs. With both flags and at least one boundary
        found, the Hessian activation chain resets to the float model's
        chain at every decoder-layer boundary (LlamaGPTQQuantizer
        semantics); within a layer the quantized chain carries the error.
        Without ``gptq_v2``, ``use_orig_model_inference`` alone keeps the
        WHOLE chain float regardless of the regex (float-weight
        substitution at every quantized linear); with ``gptq_v2`` alone the
        quantized chain is kept (a float chain would zero out the dXXT
        correction).
    The graph is evaluated segment-by-segment: per-sample activation envs
    live on pinned host memory and cross PCIe only at segment boundaries
    (the quantized linears), one large batched copy per boundary --
    LlamaGPTQQuantizer's ``move_to_device`` granularity. Inside a segment
    all data stays on the GPU.

    ``double_precision`` (inherited) follows LlamaGPTQQuantizer semantics:
    the whole activation chain -- quantized and float shadow -- is evaluated
    in float64 against per-segment fp64 copies of the consumed parameters,
    so Hessian accumulation and GPTQ math run in fp64 while the fake-quant
    write-back lands in the shared fp32 parameters.
    """

    layer_group_regex: str | None = r"\.layers\.(\d+)\."

    @property
    def name(self) -> str:
        return "traced_gptq"

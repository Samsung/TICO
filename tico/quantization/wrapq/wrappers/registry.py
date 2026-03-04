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

import importlib
from typing import Callable, Dict, Type

import torch.nn as nn

from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

# -----------------------------------------------------------------------------
# Wrapper registry
#
# Mapping:
#     fp_module_class
#         └── variant_name → quant_wrapper_class
#
# Example:
#     LlamaAttention →
#         {
#             "prefill": QuantLlamaAttentionPrefill,
#             "decode":  QuantLlamaAttentionDecode,
#         }
#
# This enables multiple execution-specialized wrappers (e.g. prefill vs decode)
# to coexist without overwriting each other.
# -----------------------------------------------------------------------------
_WRAPPERS: Dict[
    Type[nn.Module],
    Dict[str, Type[QuantModuleBase]],
] = {}
_IMPORT_ONCE = False
_CORE_MODULES = (
    "tico.quantization.wrapq.wrappers.quant_elementwise",
    ## nn ##
    "tico.quantization.wrapq.wrappers.nn.quant_embedding",
    "tico.quantization.wrapq.wrappers.nn.quant_layernorm",
    "tico.quantization.wrapq.wrappers.nn.quant_linear",
    "tico.quantization.wrapq.wrappers.nn.quant_conv3d",
    # This includes not only `nn.SiLU` but also `SiLUActivation` from transformers
    # as they are same operation.
    "tico.quantization.wrapq.wrappers.nn.quant_silu",
    ## ops ##
    "tico.quantization.wrapq.wrappers.ops.quant_rmsnorm",
    ## llama ##
    "tico.quantization.wrapq.wrappers.llama.quant_attn_prefill",
    "tico.quantization.wrapq.wrappers.llama.quant_attn_decode",
    "tico.quantization.wrapq.wrappers.llama.quant_decoder_layer_prefill",
    "tico.quantization.wrapq.wrappers.llama.quant_decoder_layer_decode",
    "tico.quantization.wrapq.wrappers.llama.quant_mlp",
    "tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm",
    "tico.quantization.wrapq.wrappers.llama.quant_model",
    ## fairseq ##
    "tico.quantization.wrapq.wrappers.fairseq.quant_decoder_layer",
    "tico.quantization.wrapq.wrappers.fairseq.quant_encoder",
    "tico.quantization.wrapq.wrappers.fairseq.quant_encoder_layer",
    "tico.quantization.wrapq.wrappers.fairseq.quant_mha",
    ## qwen_vl ##
    "tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attn",
    "tico.quantization.wrapq.wrappers.qwen_vl.quant_text_mlp",
    "tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attn",
    "tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_mlp",
    "tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_embed",
    "tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_merger",
    # add future core wrappers here
)


def _lazy_init():
    """
    Deferred one-shot import of "core wrapper modules".

    Why not import everything when the program first starts?
    --------------------------------------------------
    * **Avoid circular-import hell**
      Core wrappers often import `PTQWrapper`, which in turn calls
      `registry.lookup()`.  Importing those files eagerly here would create a
      cycle (`registry → wrapper → registry`).  Delaying the import until the
      *first* `lookup()` call lets Python finish constructing the registry
      module before any wrapper files are touched.

    * **Cold-start speed**
      Most user code never wraps layers explicitly; they only hit
      `PTQWrapper` if they are doing quantization.  Deferring half-a-dozen
      heavyweight `import torch …` files until they are really needed
      reduces library start-up latency in the common path.

    * **Optional dependencies**
      Core wrappers listed in `_CORE_MODULES` are chosen to be dependency-free
      (pure PyTorch).  Anything that needs `transformers`, `torchvision`,
      etc. uses the `@try_register()` decorator inside its own module.  Those
      optional modules are *not* imported here, so users without the extra
      packages still get a clean import.

    Implementation notes
    --------------------
    * `_IMPORT_ONCE` guard ensures we execute the import loop only once,
      even if `lookup()` is called from multiple threads.
    * Each path in `_CORE_MODULES` is a "fully-qualified module string"
      (e.g. "ptq.wrappers.linear_quant").  Importing the module runs all
      its `@register(nn.Layer)` decorators, populating `_WRAPPERS`.
    * After the first call the function becomes a cheap constant-time no-op.

    Variant Support
    ---------------
    Multiple wrappers may exist for the same floating-point module
    class but serve different execution purposes:

    * "common"  : variant-independent implementation
    * "prefill" : full-sequence inference
    * "decode"  : single-token static decoding
    * future backend-specific variants

    Each imported module registers its supported variants.
    """
    global _IMPORT_ONCE
    if _IMPORT_ONCE:
        return
    for mod in _CORE_MODULES:
        __import__(mod)  # triggers decorators
    _IMPORT_ONCE = True


# ───────────────────────────── decorator for always-present classes
def register(
    fp_cls: Type[nn.Module],
    *,
    variant: str = "common",
) -> Callable[[Type[QuantModuleBase]], Type[QuantModuleBase]]:
    """
    Register a quantization wrapper for a floating-point module class.

    Parameters
    ----------
    fp_cls:
        Floating-point module class to wrap.

    variant:
        Execution variant name.

        Typical values:
            - "common" (default)
            - "prefill"
            - "decode"

        Variants allow multiple wrappers to coexist for the same
        FP module without collision.

    Notes
    -----
    Registration is additive:
        multiple variants may be registered for the same fp_cls.
    """

    def _decorator(quant_cls: Type[QuantModuleBase]):
        _WRAPPERS.setdefault(fp_cls, {})[variant] = quant_cls
        return quant_cls

    return _decorator


# ───────────────────────────── conditional decorator
def try_register(
    *paths: str,
    variant: str = "common",
) -> Callable[[Type[QuantModuleBase]], Type[QuantModuleBase]]:
    """
    Conditionally register a wrapper if the target class exists.

    Example
    -------
    @try_register(
        "transformers.models.llama.modeling_llama.LlamaAttention",
        variant="decode",
    )

    Behavior
    --------
    • If import succeeds → register wrapper
    • If dependency missing → silently skip

    This allows optional integrations (e.g. transformers)
    without hard runtime dependencies.
    """

    def _decorator(quant_cls: Type[QuantModuleBase]):
        for path in paths:
            module_name, _, cls_name = path.rpartition(".")
            try:
                mod = importlib.import_module(module_name)
                fp_cls = getattr(mod, cls_name)
                _WRAPPERS.setdefault(fp_cls, {})[variant] = quant_cls
            except (ModuleNotFoundError, AttributeError):
                # optional dep missing or class renamed – skip silently
                pass
        return quant_cls

    return _decorator


# ───────────────────────────── lookup
def lookup(
    fp_cls: Type[nn.Module],
    *,
    variant: str = "common",
) -> Type[QuantModuleBase] | None:
    """
    Resolve the quantization wrapper class for a floating-point module.

    Parameters
    ----------
    fp_cls:
        Floating-point module class.

    variant:
        Requested execution variant.

        Typical values:
            - "prefill" : full-sequence execution (prompt processing).
            - "decode"  : single-token autoregressive decoding
            - "common"  : variant-independent implementation shared by
                          multiple execution modes

    Resolution Order
    ----------------
    Wrapper selection follows this priority:

        1. Exact variant match
           (e.g., "decode")

        2. "common" implementation
           Used for modules whose computation is identical across
           execution modes (e.g., Linear, LayerNorm, MLP).

    This design allows execution-specialized wrappers (e.g., decode-only
    attention) to coexist with variant-agnostic modules while keeping
    wrapper resolution deterministic.
    """
    _lazy_init()

    vmap = _WRAPPERS.get(fp_cls)
    if not vmap:
        return None

    return vmap.get(variant) or vmap.get("common")

## gemma4_gptq

A Gemma4 specific GPTQ implementation. It applies the same core GPTQ
algorithm as [`../gptq/`](../gptq/README.md) and
[`../qwen3_vl_gptq/`](../qwen3_vl_gptq/README.md), but restructures the
calibration and conversion flow around Gemma4's multimodal architecture.

### Why a separate implementation?

The generic `gptq/` quantizer assumes a single stack of decoder layers
(`model.model.layers`) and captures calibration inputs at the first decoder
layer. Gemma4 breaks those assumptions:

- **Two towers.** The model has a vision tower (patch embedder → vision
  encoder → pooler → multimodal embedder) and a text decoder, each needing its
  own layerwise pass.
- **Heterogeneous calibration batches.** Vision-language batches carry
  `pixel_values` plus image tokens; text-only batches don't. Vision stages must
  only see vision batches, while text layers consume all batches.
- **Per-Layer Embeddings (PLE).** Gemma4 uses PLE, which are handled internally
  by each text decoder layer. Unlike Qwen3-VL's deepstack, no special
  post-layer processing is needed during GPTQ re-forward.
- **Linear patch embedding.** Gemma4's vision patch embedder uses a `nn.Linear`
  (`input_proj`), not a `Conv3d` like Qwen3-VL.

### How it works

1. **`prepare(model, config)`** — replaces `model.forward` with a wrapper that
   caches the *raw* model inputs (args/kwargs) and returns `None` without
   running the model. Batches containing `pixel_values` are additionally stored
   in a separate vision cache.
2. **Calibration** — you call `model(...)` on your calibration batches (both
   vision-language and text-only). Nothing is executed; inputs are only
   recorded.
3. **`convert(model)`** — restores the original forward, resolves the Gemma4
   components via the config's attribute paths, then quantizes stage by stage:

   1. `vision.patch_embed` — raw replay of vision batches, hooks collect stats.
   2. `vision.blocks` — first-block entry inputs are captured by replaying
      vision batches, then each block is quantized and re-forwarded layerwise
      (as in classic GPTQ).
   3. `vision.pooler` — raw replay of vision batches.
   4. `vision.multimodal_embedder` — raw replay of vision batches.
   5. `text.layers` — first-layer entry inputs are captured from *all* batches,
      then each decoder layer is quantized and re-forwarded layerwise. No
      deepstack post-processing is needed (PLE is internal to each layer).
   6. `lm_head` (optional) — raw replay of all batches.

   If no vision batch was cached, vision stages are skipped with a warning.

As with the generic implementation, GPTQ performs **fake quantization** (weights
stay float, snapped to the grid), and the per-module `Quantizer` objects are
attached as `model.quantizers` (keyed by fully qualified module name) for reuse
by a subsequent real quantization step (e.g. `wrapq` / PTQ).

### Configuration (`Gemma4GPTQConfig`)

`Gemma4GPTQConfig` extends `GPTQConfig`, so all generic fields (`weight_bits`,
`weight_bits_overrides`, `perchannel`, `symmetric`, `mse`, `sensitivity`,
`percdamp`, `groupsize`, `actorder`, `static_groups`, `verbose`,
`show_progress`) work exactly as documented in
[`../gptq/README.md`](../gptq/README.md). For `weight_bits_overrides`, key
matching is full name → stage-local name → full-name suffix.

Gemma4 specific fields:

| Field | Default | Description |
|---|---|---|
| `quantize_vision` | `True` | Master switch for the vision tower. |
| `quantize_vision_patch_embed` | `True` | Quantize the patch embedder (`input_proj` Linear). |
| `quantize_vision_blocks` | `True` | Quantize the vision encoder blocks. |
| `quantize_vision_pooler` | `True` | Quantize the vision pooler (no-op if no Linear weights). |
| `quantize_multimodal_embedder` | `True` | Quantize the multimodal embedder (`embed_vision`). |
| `quantize_text` | `True` | Master switch for the text side. |
| `quantize_text_layers` | `True` | Quantize the text decoder layers. |
| `quantize_lm_head` | `False` | Quantize the output head. |
| `move_cache_to_cpu` | `False` | Store cached calibration inputs on CPU to reduce GPU memory pressure. |
| `cache_dtype` | `None` | Optional dtype for cached floating-point tensors (e.g. `torch.float16`). |
| `vision_tower_attr` … `lm_head_attr` | HF defaults | Dotted attribute paths used to locate the model components. Override these if the model structure differs. |

`validate()` enforces stage-switch consistency: sub-stage switches require
their master switch (e.g. `quantize_vision_blocks=True` requires
`quantize_vision=True`), and at least one stage must be enabled.

### How to use Gemma4GPTQQuantizer

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from tico.quantization import prepare, convert
from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig

model_id = "google/gemma-4-e2b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id)
model.eval()

# 1. Prepare: model.forward now only records inputs.
config = Gemma4GPTQConfig(
    weight_bits=4,
    move_cache_to_cpu=True,  # recommended for large calibration sets
)
prepare(model, config, inplace=True)

# 2. Calibration: mix of vision-language and text-only batches.
#    Vision batches (with pixel_values) calibrate the vision tower;
#    all batches calibrate the text decoder.
for messages in calibration_conversations:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model(**inputs)

# 3. Convert: stagewise GPTQ over vision and text.
convert(model, inplace=True)

print(model.quantizers.keys())
```

Text-only quantization (for `Gemma4ForCausalLM`):

```python
config = Gemma4GPTQConfig(
    quantize_vision=False,
    quantize_vision_patch_embed=False,
    quantize_vision_blocks=False,
    quantize_vision_pooler=False,
    quantize_multimodal_embedder=False,
    language_model_attr="model",
    text_layers_attr="model.layers",
    lm_head_attr="lm_head",
)
```

Mixed precision, e.g. keeping the patch embed at higher precision:

```python
config = Gemma4GPTQConfig(
    weight_bits=4,
    weight_bits_overrides={
        "input_proj": 8,  # suffix match
        "lm_head": 8,
    },
)
```

### Memory notes

- `prepare()` caches **raw model inputs** for every calibration batch; vision
  batches are cached twice (global + vision cache). Use `move_cache_to_cpu=True`
  and optionally `cache_dtype=torch.float16` to bound GPU/host memory.
- Raw-replay stages (`patch_embed`, `pooler`, `multimodal_embedder`, `lm_head`)
  run a full model forward per cached batch, so their cost scales with
  calibration set size. Blocks/layers use cached stage-entry inputs and only
  re-run the stage itself.

### Differences from `qwen3_vl_gptq/` at a glance

| | `qwen3_vl_gptq/` | `gemma4_gptq/` |
|---|---|---|
| Target | Qwen3-VL (vision + text) | Gemma4 E2B (vision + text) |
| Patch embedding | `Conv3d` (`patch_embed.proj`) | `Linear` (`patch_embedder.input_proj`) |
| Vision merger | `merger` + `deepstack_merger_list` | `pooler` + `embed_vision` (multimodal embedder) |
| Deepstack | Replays `_deepstack_process` after each text layer | Not present; PLE is internal to each layer |
| Vision detection | `pixel_values` or `pixel_values_videos` | `pixel_values` only |
| Stages | Patch embed, vision blocks, merger, deepstack mergers, text layers, `lm_head` | Patch embed, vision blocks, pooler, multimodal embedder, text layers, `lm_head` |
| Core GPTQ class | `qwen3_vl_gptq.gptq.GPTQ` (shared) | Re-exports same `GPTQ` class |

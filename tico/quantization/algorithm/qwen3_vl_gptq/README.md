## qwen3_vl_gptq

A Qwen3-VL specific GPTQ implementation. It applies the same core GPTQ algorithm as
[`../gptq/`](../gptq/README.md), but restructures the calibration and conversion flow
around Qwen3-VL's multimodal architecture, which the generic LLaMA-oriented quantizer
cannot handle.

### Why a separate implementation?

The generic `gptq/` quantizer assumes a single stack of decoder layers
(`model.model.layers`) and captures calibration inputs at the first decoder layer.
Qwen3-VL breaks those assumptions:

- **Two towers.** The model has a vision tower (patch embed → vision blocks →
  merger / deepstack mergers) and a text decoder, each needing its own
  layerwise pass.
- **Heterogeneous calibration batches.** Vision-language batches carry
  `pixel_values` (or `pixel_values_videos`) plus image tokens; text-only batches
  don't. Vision stages must only see vision batches, while text layers consume all
  batches.
- **Deepstack.** Qwen3-VL injects deepstack visual embeddings into early text
  decoder layers. When re-forwarding layer outputs for the next layer, the same
  `_deepstack_process` merging must be replayed, or the cached hidden states drift
  from the real forward pass.
- **Conv3d patch embed.** The vision patch embedding is a `Conv3d`; the bundled
  `gptq.py` adds Conv3d input unfolding for Hessian accumulation.

### How it works

1. **`prepare(model, config)`** — replaces `model.forward` with a wrapper that
   caches the *raw* model inputs (args/kwargs) and returns `None` without running
   the model. Batches containing `pixel_values` / `pixel_values_videos` are
   additionally stored in a separate vision cache.
2. **Calibration** — you call `model(...)` on your calibration batches (both
   vision-language and text-only). Nothing is executed; inputs are only recorded.
3. **`convert(model)`** — restores the original forward, resolves the Qwen3-VL
   components via the config's attribute paths, then quantizes stage by stage:

   1. `vision.patch_embed` — raw replay of vision batches, hooks collect stats.
   2. `vision.blocks` — first-block entry inputs are captured by replaying vision
      batches, then each block is quantized and re-forwarded layerwise (as in
      classic GPTQ).
   3. `vision.merger` — raw replay of vision batches.
   4. `vision.deepstack_merger[i]` — raw replay of vision batches, one merger at
      a time.
   5. `text.layers` — first-layer entry inputs are captured from *all* batches,
      then each decoder layer is quantized and re-forwarded layerwise, applying
      deepstack post-processing where applicable.
   6. `lm_head` (optional) — raw replay of all batches.

   If no vision batch was cached, vision stages are skipped with a warning.

As with the generic implementation, GPTQ performs **fake quantization** (weights
stay float, snapped to the grid), and the per-module `Quantizer` objects are
attached as `model.quantizers` (keyed by fully qualified module name) for reuse by
a subsequent real quantization step (e.g. `wrapq` / PTQ).

### Configuration (`Qwen3VLGPTQConfig`)

`Qwen3VLGPTQConfig` extends `GPTQConfig`, so all generic fields (`weight_bits`,
`weight_bits_overrides`, `perchannel`, `symmetric`, `mse`, `sensitivity`,
`percdamp`, `groupsize`, `actorder`, `static_groups`, `verbose`, `show_progress`)
work exactly as documented in [`../gptq/README.md`](../gptq/README.md). For
`weight_bits_overrides`, key matching is full name → stage-local name → full-name
suffix.

Qwen3-VL specific fields:

| Field | Default | Description |
|---|---|---|
| `quantize_vision` | `True` | Master switch for the vision tower. |
| `quantize_vision_patch_embed` | `True` | Quantize the patch embedding projection (`Conv3d`). |
| `quantize_vision_blocks` | `True` | Quantize the vision transformer blocks. |
| `quantize_vision_merger` | `True` | Quantize the final patch merger. |
| `quantize_vision_deepstack_mergers` | `True` | Quantize the auxiliary deepstack mergers. |
| `quantize_text` | `True` | Master switch for the text side. |
| `quantize_text_layers` | `True` | Quantize the text decoder layers. |
| `quantize_lm_head` | `False` | Quantize the output head (see the tied-embedding caveat in the base README). |
| `move_cache_to_cpu` | `False` | Store cached calibration inputs on CPU to reduce GPU memory pressure. |
| `cache_dtype` | `None` | Optional dtype for cached floating-point tensors (e.g. `torch.float16`). |
| `visual_attr` … `lm_head_attr` | HF defaults | Dotted attribute paths used to locate the model components (`model.visual`, `model.visual.blocks`, `model.visual.patch_embed.proj`, `model.visual.merger`, `model.visual.deepstack_merger_list`, `model.language_model`, `model.language_model.layers`, `lm_head`). Override these if the model structure differs. |

`validate()` enforces stage-switch consistency: sub-stage switches require their
master switch (e.g. `quantize_vision_blocks=True` requires `quantize_vision=True`),
and at least one stage must be enabled.

### How to use Qwen3VLGPTQQuantizer

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from tico.quantization import prepare, convert
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig

model_id = "Qwen/Qwen3-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id)
model.eval()

# 1. Prepare: model.forward now only records inputs.
config = Qwen3VLGPTQConfig(
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

Vision-only or text-only quantization:

```python
# Text decoder only
config = Qwen3VLGPTQConfig(
    quantize_vision=False,
    quantize_vision_patch_embed=False,
    quantize_vision_blocks=False,
    quantize_vision_merger=False,
    quantize_vision_deepstack_mergers=False,
)
```

Mixed precision, e.g. keeping the patch embed at higher precision:

```python
config = Qwen3VLGPTQConfig(
    weight_bits=4,
    weight_bits_overrides={
        "patch_embed.proj": 8,  # suffix match
        "lm_head": 8,
    },
)
```

### Memory notes

- `prepare()` caches **raw model inputs** for every calibration batch; vision
  batches are cached twice (global + vision cache). Use `move_cache_to_cpu=True`
  and optionally `cache_dtype=torch.float16` to bound GPU/host memory.
- Raw-replay stages (`patch_embed`, `merger`, `deepstack_mergers`, `lm_head`) run
  a full model forward per cached batch, so their cost scales with calibration set
  size. Blocks/layers use cached stage-entry inputs and only re-run the stage
  itself.

### Differences from `gptq/` at a glance

| | `gptq/` | `qwen3_vl_gptq/` |
|---|---|---|
| Target | LLaMA-like text decoders | Qwen3-VL (vision + text) |
| Capture point | First decoder layer inputs | Raw model inputs (per stage replay) |
| Calibration forward | Runs up to layer 0, then stops | Never executes the model |
| Stages | Decoder layers (+ optional `lm_head`) | Patch embed, vision blocks, mergers, deepstack mergers, text layers, `lm_head` |
| Vision/text split | — | Separate vision cache; vision stages skip text-only batches |
| Deepstack handling | — | Replays `_deepstack_process` after each text layer |
| `use_orig_model_inference` | Supported | Not supported |
| Cache controls | Always CPU | `move_cache_to_cpu`, `cache_dtype` |

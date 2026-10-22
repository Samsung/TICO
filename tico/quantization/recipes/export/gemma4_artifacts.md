# Gemma4 E2B quantized export artifacts

This document describes every file produced by the Gemma4 E2B quantize + export
workflow:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/gemma4_spinquant_group_r2_quantize.yaml \
  --model <path-or-id of gemma-4-E2B-it>

python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/gemma4_export.yaml \
  --checkpoint ./out/gemma4_quantized/quantized_model.pt \
  --model <path-or-id of gemma-4-E2B-it>
```

Sizes are measured from one E2B run (SpinQuant group R2 + GPTQ + PTQ, UINT4
linear weights, INT16 activations, UINT8 embeddings, `max_seq_len=2048`, vision
profile `e2b_66x36_264`, `export.vision.granularity: both`). The artifact tag
`q` marks quantized graphs; a floating-point export uses `f32` instead.

## Top level: `out/gemma4_quantized/`

| File | Size | Description |
|---|---|---|
| `quantized_model.pt` | 21.2 GB | PTQ-wrapped model checkpoint after SpinQuant (R2), GPTQ, and PTQ: float32 weights plus frozen observer qparams. Input of `export --checkpoint`. |
| `group_r2.pt` | 12.6 MB | SpinQuant result: learned R2 rotation per shared-KV group (`r2_map`, `groups`) and optimization reports. Already fused into the weights; not needed at runtime. |
| `effective_config.yaml` | 2.3 KB | Resolved configuration of the quantize run (`gemma4_spinquant_group_r2_quantize.yaml` plus CLI overrides). |
| `circle_layers/` | 5.0 GB | Export output directory, described below. |

## `circle_layers/`: text pipeline

| File | Size | Description |
|---|---|---|
| `token_embedding.q.circle` | 406 MB | Token embedding (scaled by sqrt(hidden)). Input `input_ids` `[1, S]` with dynamic `1 <= S <= 2048`, output `[1, S, 1536]`. Shared by prefill and decode, CPU. |
| `ple_embedding.q.pt` | 2.35 GB | Host-side PLE token lookup. UINT8 table `(262144, 8960)` with per-row scale/zero-point, `embed_scale`, and the four frozen observers replayed by the stage. Saved as `.pt` because the table exceeds the 2 GiB Circle flatbuffer limit. Load with `Gemma4PLEEmbeddingHostTable.from_artifact`; output `[1, S, 35, 256]`. Shared by prefill and decode. |
| `ple_projection_prefill.q.circle` | 7.0 MB | PLE projection, norm, and combine stage for the NPU. Inputs `inputs_embeds` `[1, 2048, 1536]` and `per_layer_token_inputs` `[1, 2048, 35, 256]`, output `per_layer_inputs` `[1, 2048, 35, 256]`. |
| `ple_projection_decode.q.circle` | 7.0 MB | Same graph traced with `S = 1`. |
| `ple_pipeline.json` | 4.4 KB | PLE manifest: chosen `ple_embedding` format (`pt`) and size estimate, projection input/output shapes, and the qparams of the three boundary observers (`per_layer_token_inputs`, `per_layer_projection`, `per_layer_inputs`). |
| `multimodal_fusion_prefill.q.circle` | 1.7 KB | Fixed-slot fusion that writes the 264 visual embeddings into the text sequence starting at `visual_start_idx`. No weights. |
| `decoder_layer_prefill_00..34.q.circle` | 18.4 MB each, 945 MB total | The 35 text decoder layers at `S = 2048`. Inputs: hidden states, attention mask, RoPE, `per_layer_input` `[1, 2048, 256]` (plus shared K/V for shared-KV consumers); outputs hidden states (plus K/V). |
| `decoder_layer_decode_00..34.q.circle` | 18.4 MB each, 945 MB total | Single-token decode graphs of the same layers with 2047 past K/V slots as input. |
| `lm_head.q.circle` | 406 MB | Final RMSNorm plus LM head. Input `[1, 1, 1536]`, output logits `[1, 1, 262144]`. Final logit softcapping is applied on the host. |

Each decoder layer `i` consumes `per_layer_inputs[:, :, i, :]` from the PLE
projection output. The `per_layer_inputs` qparam is identical on the projection
output and on every decoder layer `per_layer_input` input.

Inside `ple_projection_*`, `per_layer_model_projection_scale` and
`per_layer_input_scale` observe the two multiplication constants, and
`per_layer_sum` observes the projection-plus-token sum before the final scale.
These internal observers follow the configured activation policy; the existing
`per_layer_inputs` observer still owns the final output/decoder boundary. Both
prefill and decode reuse the same frozen observers and scale buffers.

Checkpoints created before these internal PLE observers were added must be
prepared and PTQ-calibrated again before quantization and export. Re-exporting an
older `quantized_model.pt` alone cannot supply the missing sum statistics and is
rejected with a recalibration message. Adding the intermediate fake-quant boundary
can change quantized outputs, so rerun accuracy validation after recalibration.

## KV cache boundaries

Gemma4 text attention has three independent observer roles:

| Role | Tensor and qparam owner |
|---|---|
| `new_k`, `new_v` | Current projected KV and the delta written to persistent storage. These producer observers own the cache input/output dtype, scale, zero-point, and granularity. |
| `past_key`, `past_value` | Actual existing KV read by this attention layer; a shared consumer reads its producer's supplied KV. Read overrides do not change storage. |
| `present_key`, `present_value` | Combined KV used by attention. Affine compute converts concat operands to this grid; sequence-axis MX V converts the complete assembled tensor. |

An ordinary empty-cache prefill applies `new_*`, returns the storage delta, and
applies `present_*` for attention. It never executes `past_*`, including in
calibration. Cached decode applies the producer storage boundary and then
`past_*` to the existing tokens before combining them with current `new_*` KV
on the compute branch. The current token is never collected as past.

The static host copies/slices prefill and decode deltas without requantizing
them. Decode cache inputs and shared consumer inputs therefore use the producer's
frozen `new_*` qparams. Independent read and compute policies remain internal
conversions, including when only scale or zero-point differs. Supported Circle
cache conversions use per-tensor UINT8/INT16 affine policies, or a consistently
unquantized cache path. An explicitly declared external float source can also
feed affine read/compute boundaries. MX storage/read policies, per-channel
cache grids, and unsupported affine-to-identity paths are rejected rather than
serialized with mismatched concat qparams. Disabling fake quantization on an
affine cache boundary is also rejected; use an explicit supported identity
policy for a float path.

With affine storage and affine reads, `present_value` can use MXINT8 on the
sequence axis (`axis: -2`). For cached decode, the compute branch first converts
current V to the independent `past_value` affine grid, concatenates it with the
already converted existing V, then applies MX to the full sequence. It does not
concatenate separately quantized partial MX blocks. Empty-cache prefill applies
MX directly to current V without executing `past_value`; a shared consumer
converts the actual supplied full storage tensor through its read and compute
boundaries. The affine `present_*` path retains its existing conversions.
Together with MXINT8 probabilities on the key axis (`axis: -1`), this supports
MXINT8 × MXINT8 attention with an INT16 result and INT16 persistent cache. See
the [policy example](../../examples/configs/PTQ_OVERRIDE_POLICIES.md#gemma4-int16-cache-with-mx-attention)
for the supported observer settings.

The full prepared text model binds each shared consumer to the corresponding
KV-owning layer of the same layer type. For standalone shared attention or
decoder use, call `bind_cache_source(producer_attention)` on the attention
wrapper before calibration. Export also accepts
`as_export_module(cache_source=producer_attention)` to supply or check the
binding, but first binding a calibrated consumer is rejected: reprepare and
bind before calibration so the default read policy is resolved correctly.
The source is a reference, not an owned child module, so consumer calibration
does not reset producer observers or traverse its weights.

Gemma4 Assistant receives KV from the FP target used by its existing assisted
generation adapter. Its backbone explicitly declares this external float cache
contract with `bind_external_float_cache()`; it does not guess a quantized
producer grid. The independent default past observers preserve that float
storage policy, and `present_*` keeps the assistant's attention conversion.

`cache_output_mode="delta"` keeps returning current storage KV.
`"present"`/`"full"` returns the updated HF cache object or the full **storage**
tuple. The tuple previously contained attention-quantized `present_*` values;
it now preserves the producer grid so it can safely be reused as cache input.
The shared-KV dictionary likewise carries storage tensors. Read-quantized
tensors are never written back into an HF cache.

The HF runtime path supports non-offloaded `DynamicCache` and `StaticCache`
full/sliding layers using their cache-layer APIs. Offloaded, quantized, and other
cache layer implementations are rejected explicitly.

For generation, calibrate through real cached decode with the existing
`calibration.decode_steps` setting (the quantize preset uses `8`; an active
PTQ stage's `decode_calibration_steps` takes precedence). `decode_steps: 0`
supports prefill-only use without inventing past statistics. Cached inference
or decode export with an unobserved affine read observer fails with guidance to
recalibrate using positive decode steps. Valid loaded frozen qparams and
observers that need no statistics remain usable without new decode calibration.

Save/load preserves read statistics and frozen qparams. Recreate the same
prepared model/configuration before loading a state dict so shared source
bindings are rebuilt. Checkpoints from before `past_*` was added require
re-prepare and PTQ recalibration; export cannot infer the missing read statistics.
No automatic checkpoint migration is provided.

## `circle_layers/`: vision pipeline

`granularity: both` writes the monolithic graph and the split pipeline.

| File | Size | Description |
|---|---|---|
| `vision_prefill.q.circle` | 98.9 MB | Monolithic vision tower plus `embed_vision` projection. Input `pixel_values` `[1, 2520, 768]`, output visual tokens `[264, 1536]`. |
| `vision_patch_embedder.q.circle` | 4.5 MB | Split stage 1: patch embedding plus the baked positional embedding, output `[1, 2520, 768]`. |
| `vision_encoder_layer_00..15.q.circle` | 4.9 MB each, 79 MB total | Split stage 2: the 16 vision encoder layers. Each takes `hidden_states` and the three tensors stored in `vision_context.pt`. |
| `vision_pooler.q.circle` | 1.4 MB | Split stage 3: pools 2520 patches into 280 soft tokens. |
| `vision_post_projection.q.circle` | 0.6 MB | Split stage 4: selects the 264 valid tokens and applies `embed_vision`, output `[264, 1536]`. |
| `vision_context.pt` | 26.7 MB | Shared inputs of the split encoder layers: additive `attention_mask` `(1, 1, 2520, 2520)` and RoPE `cos`/`sin` `(1, 2520, 64)`. Stored once instead of being embedded in 16 graphs. |
| `vision_pipeline.json` | 31.7 KB | Split pipeline manifest: stage order, tensor shapes, boundary observer qparams, and the `vision_context.pt` reference. |
| `vision_profile.json` | 656 B | Static image profile `e2b_66x36_264`: 1056x576 image, 36x66 patch grid, 2520 patches (2376 valid plus 144 padding), 264 visual tokens, and the `image_position_ids` hash. |
| `effective_config.yaml` | 591 B | Resolved configuration of the export run (`gemma4_export.yaml` plus `--checkpoint` and `--model`). |

If only the monolithic vision graph is used, the 20 split-vision files
(`vision_patch_embedder`, `vision_encoder_layer_*`, `vision_pooler`,
`vision_post_projection`, `vision_context.pt`, `vision_pipeline.json`) are not
needed; set `export.vision.granularity: monolithic` to skip them.

## Runtime data flow

```text
Prefill:
  token_embedding(input_ids)           -> text_embeds       [1, 2048, 1536]
  vision_prefill(pixel_values)         -> visual_embeds     [264, 1536]
  multimodal_fusion_prefill            -> inputs_embeds     [1, 2048, 1536]
  ple_embedding(input_ids)   (host)    -> per_layer_token_inputs [1, 2048, 35, 256]
  ple_projection_prefill               -> per_layer_inputs  [1, 2048, 35, 256]
  decoder_layer_prefill_00..34         (layer i gets per_layer_inputs[:, :, i, :])
  lm_head(last hidden)                 -> logits

Decode (one token):
  token_embedding(next_token)          -> inputs_embeds     [1, 1, 1536]
  ple_embedding(next_token)  (host)    -> per_layer_token_inputs [1, 1, 35, 256]
  ple_projection_decode                -> per_layer_inputs  [1, 1, 35, 256]
  decoder_layer_decode_00..34          (past K/V managed by the host)
  lm_head                              -> logits
```

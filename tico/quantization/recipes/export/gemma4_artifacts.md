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

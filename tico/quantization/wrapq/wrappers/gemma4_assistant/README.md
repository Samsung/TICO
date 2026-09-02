# Gemma4 Assistant (MTP Draft) PTQ and Static NPU Export

This package quantizes `Gemma4AssistantForCausalLM` — the single-position
multi-token-prediction (MTP) draft model used by Hugging Face assisted
generation — and exports a static-shape draft-one core for NPU execution.

## Architecture

The assistant consumes, per draft step:

- `inputs_embeds` `(1, 1, 2 * backbone_hidden_size)` — the target model's
  last-token embedding concatenated with the last target/assistant hidden
  state;
- `shared_kv_states` — the target model's `full_attention` and
  `sliding_attention` KV states (the assistant never projects or caches K/V);
- a constant `position_ids` value and the target padding `attention_mask`.

Wrapper layout (all shapes derived from the checkpoint config):

```text
PTQWrapper
└── QuantGemma4AssistantForCausalLM         (quant_for_causal_lm.py)
    ├── pre_projection                       QuantLinear
    ├── model: QuantGemma4AssistantBackbone  (quant_backbone.py)
    │   ├── layers[i]                        QuantGemma4TextDecoderLayer
    │   │   └── self_attn                    QuantGemma4TextAttention
    │   │                                    (shared-KV consumer, no K/V proj)
    │   └── norm                             QuantGemma4RMSNorm
    ├── post_projection                      QuantLinear
    ├── lm_head                              QuantLinear (tied-weight source)
    └── masked_embedding                     QuantGemma4AssistantMaskedEmbedder
        └── centroids                        QuantLinear
```

`lm_head.weight` is tied to `model.embed_tokens.weight`; the wrapper never
quantizes the embedding copy and the ordered sparse head reads the single
fake-quantized `lm_head` weight source.

## NPU / host partition

NPU (`gemma4_assistant_core.q.circle`, tensor-only ABI, batch=1, q_len=1):

```text
assistant_input → pre_projection → 4 shared-KV decoder layers → norm
                                        │
                                        ├→ post_projection → projected_state
                                        ├→ assistant_hidden
                                        └→ centroids → centroid_logits
```

Host (`gemma4_assistant_sparse_head.pt`, see `sparse_head.py`):

```text
centroid_logits → top-k centroids
token_ordering.view(num_centroids, vocab_per_centroid) → selected token ids
lm_head_weight[selected ids] → selected rows
assistant_hidden @ rows.T → selected logits → argmax → next draft token
```

The core graph contains no dictionaries, `.item()`, dynamic top-k/gather/
scatter, full-vocabulary tensors, or assistant-owned KV cache. The host head
never reconstructs full 262K logits except through the explicit
`full_logits()` debug helper.

## Static input ABI

`static_inputs.py` canonicalizes dynamic HF inputs to the fixed contract
(`GEMMA4_ASSISTANT_CORE_INPUT_NAMES` order):

| Input                    | Shape                                        |
|--------------------------|----------------------------------------------|
| `assistant_input`        | `(1, 1, 2 * backbone_hidden_size)`           |
| `full_key` / `full_value`| `(1, kv_heads, FULL_KV_LEN, global_head_dim)`|
| `sliding_key` / `sliding_value` | `(1, kv_heads, SLIDING_KV_LEN, head_dim)` |
| `full_attention_mask`    | `(1, 1, 1, FULL_KV_LEN)` additive            |
| `sliding_attention_mask` | `(1, 1, 1, SLIDING_KV_LEN)` additive         |
| `full_cos` / `full_sin`  | `(1, 1, global_head_dim)`                    |
| `sliding_cos` / `sliding_sin` | `(1, 1, head_dim)`                      |

Outputs: `projected_state (1,1,backbone_hidden)`, `assistant_hidden
(1,1,hidden)`, `centroid_logits (1,1,num_centroids)`.

Masking follows `Gemma4AssistantForCausalLM.create_attention_masks`: full
attention sees every valid shared-KV position; sliding attention sees only
the last `sliding_window + 1` valid positions (the HF bidirectional sliding
overlay is distance-inclusive). Therefore `sliding_kv_length` must be at
least `min(sliding_window + 1, full_kv_length)`; K/V are right-padded with
zeros and padded slots are masked with the bounded fill value
(`PTQConfig.attention_mask_fill_value`, default `-120.0`).

## Calibration

Calibration runs real Hugging Face assisted generation
(`Gemma4AssistantAdapter`): the FP target drafts with the *prepared*
assistant plugged in through `Gemma4AssistantGenerationAdapter`, so observers
stream over the exact `inputs_embeds` / shared-KV / mask / position
distribution of the MTP runtime. No Gaussian-only calibration is used.

## Commands

Quantize (PTQ `safe_w8a16`: int16 activations, uint8 weights) and export:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/gemma4_e2b_assistant_quantize.yaml
```

Export from a saved checkpoint:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/gemma4_e2b_assistant_export.yaml \
  --checkpoint ./out/gemma4_e2b_assistant_quantized/quantized_model.pt
```

Wrapper smoke checks (tiny synthetic model, no downloads) and Circle export:

```bash
python -m tico.quantization.examples.inspector \
  --mode wrapper-smoke --case gemma4_assistant_core --export circle \
  --output-dir ./out/assistant_smoke
python -m tico.quantization.examples.inspector \
  --mode wrapper-smoke --case gemma4_assistant_sparse_head
```

FP / PTQ / static parity debug runtime:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/static_gemma4_assistant_runtime.yaml \
  --mode static-gemma4-assistant-runtime
```

Checkpoint paths can be overridden with the `GEMMA4_ASSISTANT_PATH` and
`GEMMA4_TARGET_PATH` environment variables; the env-gated integration test in
`test/quantization/recipes/integration/test_gemma4_assistant_integration.py`
uses the same variables.

Artifacts written by the export step:

- `gemma4_assistant_core.q.circle` — static draft-one NPU core;
- `gemma4_assistant_sparse_head.pt` — integer LM-head weight + qparams +
  `token_ordering` metadata for the host sparse head;
- `gemma4_assistant_manifest.json` — I/O contract, static KV capacities,
  quantization profile, sparse-head location, and the token-ordering
  checksum.

## Weight profiles

- `safe_w8a16` (default): `activation: int16`, all weights `uint8`.
- `compact_w4a16` (experiment): set `linear_weight: uint4` in the PTQ stage
  and keep `projection_weight`, `centroid_weight`, and `lm_head_weight` at
  `uint8` to protect the sensitive boundaries.

## Out of scope

Target-side verify-K graphs, candidate acceptance logic, and target KV
commit/crop are runtime work tracked separately; this package covers the
assistant draft-one core only.

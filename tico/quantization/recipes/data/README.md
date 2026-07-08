# Data Builders Developer Guide

`recipes/data/` contains reusable calibration-data builders used by model
adapters and debug tools.

Data builders are intentionally kept outside `examples/` so that a dataset can
be reused by multiple workflows: GPTQ calibration, PTQ observer calibration,
SmoothQuant statistics collection, parity tracing, and smoke tests.

## Responsibilities

A data builder should:

- load or synthesize calibration/evaluation inputs;
- normalize those inputs into the format expected by a model adapter;
- keep dataset-specific dependencies and preprocessing details out of
  `examples/*.py`;
- expose small, typed functions that are easy to call from adapters.

A data builder should not:

- call `prepare()` or `convert()`;
- own quantization algorithm logic;
- own benchmark scoring logic;
- save quantized model artifacts;
- parse CLI arguments directly.

## Current layout

```text
recipes/data/
├── README.md
├── llm.py        # text-only calibration input builders
└── vlm.py        # vision-language calibration input builders
```

The split is by **input modality**, not by model name. For example, a future
Gemma or Mistral text model should usually reuse `llm.py` unless it needs a
different input shape. A future image-text model should usually reuse or extend
`vlm.py`.

## Input format conventions

Calibration builders should return a `list` of samples.

For text-only causal LMs, each sample is usually a tensor of token IDs:

```python
list[torch.Tensor]  # each tensor shape: [batch, seq_len]
```

For VLMs, each sample is usually a mapping that can be passed as keyword
arguments to the model:

```python
list[dict[str, torch.Tensor | Any]]
```

The adapter is responsible for moving samples to the target device before
forward execution. Data builders may create CPU tensors by default to avoid
holding unnecessary GPU memory.

## Recommended function shape

Use explicit keyword-only arguments. Avoid passing the full recipe config into a
data builder unless the dataset genuinely needs many sections of the config.

```python
def build_wikitext_calibration_inputs(
    *,
    tokenizer: Any,
    cache_dir: str | None,
    n_samples: int,
    seq_len: int,
    seed: int,
    device: torch.device | str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
) -> list[torch.Tensor]:
    ...
```

This style makes call sites readable and makes it clear which config fields the
builder actually uses.

## Adding a new calibration dataset

1. Decide whether the dataset belongs in an existing modality file or a new file.
   Add a new file only when the modality or preprocessing contract is different.
2. Add a function with explicit keyword-only parameters.
3. Return samples in the adapter’s expected input format.
4. Keep random sampling deterministic through a `seed` argument.
5. Avoid downloading or preprocessing more data than requested by `n_samples`.
6. Add or update a config preset under `examples/configs/`.
7. Add a smoke test with a tiny `n_samples` value.

Example:

```python
def build_c4_calibration_inputs(
    *,
    tokenizer: Any,
    cache_dir: str | None,
    n_samples: int,
    seq_len: int,
    seed: int,
    split: str = "train",
) -> list[torch.Tensor]:
    ...
```

Then call it from the relevant adapter:

```python
dataset = calib.get("dataset", "Salesforce/wikitext")
if dataset == "c4":
    return build_c4_calibration_inputs(...)
if dataset == "Salesforce/wikitext":
    return build_wikitext_calibration_inputs(...)
raise ValueError(f"Unsupported calibration dataset for llama: {dataset}")
```

## Synthetic data

Synthetic data is useful for fast smoke tests and wrapper-level integration
tests, but it should not be mixed with real benchmark presets.

Use naming that makes the intent obvious:

```text
synthetic_llm_smoke
synthetic_vlm_smoke
random_tokens
random_image_text_batch
```

When adding synthetic builders, document:

- supported shapes;
- value ranges;
- whether the generated inputs are valid model inputs or only wrapper inputs;
- whether the builder is safe for CPU-only CI.

## Dataset-specific dependencies

Do not add heavy dependencies at module import time unless they are already
required by the package. Prefer local imports inside the builder function:

```python
def build_some_dataset(...):
    from datasets import load_dataset
    ...
```

This keeps unrelated recipes importable even when a dataset dependency is not
installed.

## Caching and download behavior

Always expose `cache_dir` when using Hugging Face datasets or model processors.
Do not hard-code a local developer path.

Good:

```python
load_dataset(name, config, split=split, cache_dir=cache_dir)
```

Bad:

```python
load_dataset("/home/user/private_dataset")
```

## Error handling

Raise clear errors when the calibration request cannot be satisfied:

```python
if token_ids.shape[1] <= seq_len + 1:
    raise ValueError(
        f"Calibration corpus is too short for seq_len={seq_len}. "
        f"token_count={token_ids.shape[1]}"
    )
```

Prefer failing early over silently returning fewer samples than requested.

## Testing checklist

For a new data builder, test:

- `n_samples=1`;
- CPU execution;
- deterministic sampling with the same seed;
- a short sequence length or small image shape;
- helpful error messages for unsupported dataset names or invalid shapes;
- no unnecessary GPU allocation during data loading.

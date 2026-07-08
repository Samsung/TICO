# Evaluation Helpers Developer Guide

`recipes/evaluation/` contains reusable evaluation helpers used by
`examples/evaluate.py`, optional evaluation inside `examples/quantize.py`, and
debug workflows.

Evaluation helpers are kept outside examples so that metrics can be shared
across FP models, fake-quant models, checkpoint evaluation, and regression tests.

## Responsibilities

Evaluation helpers should:

- run benchmark-specific evaluation loops;
- compute and print metrics;
- hide benchmark-specific adapters or dataset plumbing;
- return structured results when practical.

Evaluation helpers should not:

- load the main model from Hugging Face;
- decide which model family is being evaluated;
- run quantization stages;
- save quantized model artifacts;
- parse CLI arguments directly.

The model adapter decides **which** evaluation helpers to call. The helper
decides **how** to evaluate a task.

## Current layout

```text
recipes/evaluation/
├── README.md
├── llm.py      # perplexity and text-only LM benchmark helpers
├── vlm.py      # VQA / COCO-style VLM benchmark helpers
└── mmlu.py     # MMLU wrapper helpers
```

The split is by benchmark/input type, not by model name.

## Result conventions

Prefer returning structured values in addition to printing readable summaries.

Examples:

```python
float                         # perplexity
dict[str, tuple[int, int]]    # task -> (correct, total)
dict[str, float]              # metric -> score
```

Printing is acceptable for user-facing examples, but the returned value should
remain useful for tests.

## Function shape

Use explicit keyword-only arguments:

```python
def evaluate_perplexity(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    cache_dir: str | None,
    max_seq_len: int,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> float:
    ...
```

This avoids coupling the evaluation helper to the full recipe config. The
adapter should translate config fields into function arguments.

## Adding a new benchmark

1. Decide whether the benchmark belongs in an existing file or a new file.
2. Write a helper with explicit keyword-only arguments.
3. Keep dataset loading and benchmark adapters inside the helper.
4. Return structured results.
5. Add a compact print helper if the metric table is non-trivial.
6. Call the helper from the relevant model adapter’s `evaluate()` method.
7. Add config fields under the `evaluation:` section.
8. Add a smoke test with a tiny sample count.

Example config section:

```yaml
evaluation:
  enabled: true
  hellaswag:
    enabled: true
    n_samples: 32
```

Example adapter call:

```python
hellaswag_cfg = eval_cfg.get("hellaswag", {})
if hellaswag_cfg.get("enabled", False):
    results = evaluate_hellaswag(
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        device=str(ctx.device),
        n_samples=int(hellaswag_cfg.get("n_samples", -1)),
        max_seq_len=max_seq_len,
    )
```

## Config naming rules

Use task names that match common benchmark names:

```text
perplexity
lm_eval_tasks
vlm_tasks
coco
mmlu
hellaswag
arc
gsm8k
```

Avoid model-family names inside benchmark keys. The adapter already knows the
model family.

Good:

```yaml
evaluation:
  mmlu:
    enabled: true
```

Bad:

```yaml
evaluation:
  qwen3_vl_mmlu:
    enabled: true
```

## Side effects and logging

Evaluation helpers may print compact metric summaries. They should not print
large model outputs, full prompts, or dataset examples by default.

For verbose inspection, add an explicit argument:

```python
verbose: bool = False
```

## Device and dtype

Evaluation helpers should not move the full model between devices unless the
function contract explicitly says so. The adapter or caller owns model placement.

It is fine to move batches or tokenized inputs to `device` inside the helper.

## Dealing with external benchmark packages

Some benchmarks require optional packages. Keep those imports local and fail with
a clear message:

```python
def evaluate_some_task(...):
    try:
        import some_benchmark
    except ImportError as exc:
        raise RuntimeError(
            "some_benchmark is required for evaluation.some_task. "
            "Install the optional evaluation dependencies."
        ) from exc
```

This keeps basic quantization recipes importable without all benchmark
dependencies installed.

## Testing checklist

For a new evaluation helper, test:

- `n_samples=1` or the smallest supported sample count;
- CPU execution when possible;
- checkpoint evaluation through `examples/evaluate.py`;
- optional evaluation through `examples/quantize.py`;
- clear behavior when the benchmark dependency is missing;
- structured results that can be asserted in tests.

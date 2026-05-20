# Model Adapter Developer Guide

Adapters isolate model-family-specific behavior from the generic quantization
pipeline.

Create or modify an adapter when the model family changes how any of the
following work:

- model/tokenizer/processor loading
- calibration input construction
- forward invocation during calibration
- PTQ config construction
- evaluation tasks or metrics
- export input shape or export module selection

Do **not** create a new example script just because a new model family needs a
slightly different loading path. Add an adapter and a config preset instead.

## Adapter contract

Every adapter implements `ModelAdapter`:

```python
class ModelAdapter(ABC):
    family: str

    def load_model(self, ctx: RecipeContext) -> RecipeContext: ...
    def build_calibration_inputs(self, ctx: RecipeContext) -> list[Any]: ...
    def forward_calibration(self, ctx, model, calibration_inputs, *, desc: str) -> None: ...
    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg) -> None: ...
    def build_ptq_config(self, ctx, stage_cfg): ...
    def evaluate(self, ctx: RecipeContext) -> None: ...
    def export(self, ctx: RecipeContext) -> None: ...
```

The adapter is selected from:

```yaml
model:
  family: llama
```

and registered in `recipes/adapters/__init__.py`.

## When to add a new adapter

Add a new adapter when the model family needs its own implementation for one or
more of these areas:

| Difference | Adapter responsibility |
|---|---|
| HF class differs | `load_model` |
| Uses `AutoProcessor` instead of tokenizer-only flow | `load_model`, `build_calibration_inputs` |
| Calibration sample is a dict instead of `input_ids` tensor | `build_calibration_inputs`, `forward_calibration` |
| Requires special model args for wrappers | `build_ptq_config` |
| Has VQA, captioning, or multimodal evaluation | `evaluate` |
| Needs special export module or dummy input | `export` |

Do not add a new adapter for only a different model checkpoint of the same
family. Add a config preset or use `--model`.

## Step-by-step: adding a new model family

Assume the new family is `gemma`.

### 1. Create the adapter

Create:

```text
tico/quantization/recipes/adapters/gemma.py
```

Skeleton:

```python
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.context import RecipeContext


class GemmaAdapter(ModelAdapter):
    family = "gemma"

    def load_model(self, ctx: RecipeContext) -> RecipeContext:
        model_cfg = ctx.cfg["model"]
        runtime_cfg = ctx.cfg.get("runtime", {})
        # Load tokenizer/model here.
        # Set ctx.model, ctx.tokenizer or ctx.processor, ctx.device, ctx.dtype.
        return ctx

    def build_calibration_inputs(self, ctx: RecipeContext) -> list[Any]:
        # Return samples consumed by forward_calibration.
        return []

    def forward_calibration(
        self,
        ctx: RecipeContext,
        model: torch.nn.Module,
        calibration_inputs: Sequence[Any],
        *,
        desc: str,
    ) -> None:
        model.eval()
        with torch.no_grad():
            for sample in calibration_inputs:
                model(sample.to(ctx.device))

    def calibrate_prepared_model(
        self,
        ctx: RecipeContext,
        prepared_model: torch.nn.Module,
        stage_cfg: Mapping[str, Any],
    ) -> None:
        self.forward_calibration(
            ctx,
            prepared_model,
            ctx.calibration_inputs,
            desc="PTQ calibration",
        )

    def build_ptq_config(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]):
        # Return the model-family PTQ config.
        raise NotImplementedError

    def evaluate(self, ctx: RecipeContext) -> None:
        if not ctx.cfg.get("evaluation", {}).get("enabled", False):
            return
        # Call reusable evaluation helpers here.

    def export(self, ctx: RecipeContext) -> None:
        if not ctx.cfg.get("export", {}).get("enabled", False):
            return
        # Call reusable export helpers here.
```

### 2. Register the adapter

Edit `recipes/adapters/__init__.py`:

```python
from tico.quantization.recipes.adapters.gemma import GemmaAdapter

_ADAPTERS = {
    "llama": LlamaAdapter(),
    "qwen3_vl": Qwen3VLAdapter(),
    "gemma": GemmaAdapter(),
}
```

### 3. Add config presets

Create at least one small smoke config and one representative full config:

```text
tico/quantization/examples/configs/gemma_ptq_only.yaml
tico/quantization/examples/configs/gemma_gptq_ptq.yaml
```

The smoke config should be cheap enough for CI or manual sanity checks.

### 4. Add tests

Recommended tests:

```text
tests/quantization/recipes/test_gemma_adapter_import.py
tests/quantization/recipes/test_gemma_config_load.py
tests/quantization/integration/test_gemma_ptq_smoke.py
```

The import/config tests should not require downloading a large checkpoint.

## Calibration input conventions

Adapters decide the shape and type of calibration samples.

Common forms:

```python
# LLM
list[torch.Tensor]  # each sample is input_ids with shape [batch, seq]

# VLM
list[dict[str, torch.Tensor]]  # processor output consumed as model(**sample)
```

The generic stages should not assume a sample shape. They should call:

```python
ctx.adapter.forward_calibration(ctx, model, ctx.calibration_inputs, desc="...")
```

or:

```python
ctx.adapter.calibrate_prepared_model(ctx, prepared_model, stage_cfg)
```

## PTQ config construction

`build_ptq_config` should translate config values into the concrete TICO PTQ
config object for that model family.

Keep these rules:

- Parse dtypes and qschemes using shared helpers in `recipes/utils.py`.
- Keep model-family wrapper names and model args in the adapter.
- Keep generic algorithm state in stages.
- Fail early if required model metadata is missing.

Example:

```python
def build_ptq_config(self, ctx, stage_cfg):
    return build_llm_ptq_config(
        model_type="gemma",
        num_hidden_layers=len(ctx.model.model.layers),
        activation_dtype=wrapq_dtype_from_name(stage_cfg.get("activation_dtype", "int16")),
        default_qscheme=qscheme_from_name(stage_cfg.get("default_qscheme", "per_tensor_symm")),
        linear_weight_bits=stage_cfg.get("linear_weight_bits"),
        strict_wrap=bool(stage_cfg.get("strict_wrap", True)),
    )
```

## Evaluation and export

Evaluation and export are adapter methods because the model family controls:

- tokenizer vs processor
- generation API
- benchmark compatibility
- maximum sequence length
- dummy inputs for export
- wrapper export module selection

However, reusable metric and export helpers should live in:

```text
recipes/evaluation/
recipes/export/
```

The adapter should orchestrate these helpers, not duplicate metric code.

## Adapter checklist

Before merging a new adapter:

- `family` is lower snake case.
- Adapter is registered in `recipes/adapters/__init__.py`.
- At least one config preset exists.
- Small calibration can run with `n_samples=1`.
- `forward_calibration` supports the sample type returned by
  `build_calibration_inputs`.
- `build_ptq_config` does not mutate global state.
- Evaluation and export are optional and respect `enabled: false`.
- No example imports this adapter-specific helper directly except through the
  recipe runner or debug CLI.

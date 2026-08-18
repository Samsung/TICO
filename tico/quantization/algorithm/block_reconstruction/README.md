# Activation block reconstruction

This package provides block reconstruction for WrapQ models. It optimizes
frozen per-tensor affine activation qparams against cached floating-point block
outputs while keeping model parameters and weight qparams fixed.

## Core reconstruction

- Cache both floating-point and quantized-prefix block inputs.
- Use floating-point block outputs as reconstruction targets.
- Optimize activation scale and, for asymmetric qschemes, zero-point.
- Use straight-through rounding and backend integer limits during optimization.
- Tie producer/consumer observers that represent one logical tensor domain.
- Support energy-normalized MSE and magnitude-normalized L1 objectives.
- Process model-specific blocks in execution order through a caller-provided
  adapter.

The package does not implement QDrop, adaptive weight rounding, Fisher
weighting, or task-aware detector losses. Those remain separate follow-up
stages.

## Generic local-only usage

The original PR 1 interface remains supported. Without a held-out evaluator,
the best full-cache local-loss state is committed as before.

```python
from tico.quantization.algorithm.block_reconstruction import (
    AffineObserverGroup,
    BlockReconstructionConfig,
    BlockReconstructor,
)

result = BlockReconstructor(
    BlockReconstructionConfig(steps=500)
).reconstruct(
    block_name="block_0",
    observer_model=quantized_model,
    block=executable_block,
    cache=reconstruction_cache,
    observer_groups=(
        AffineObserverGroup(
            "tensor_12",
            (
                "producer.act_out",
                "consumer.act_in",
            ),
        ),
    ),
)
```

## Held-out checkpoint selection and rollback

Pass a disjoint selection cache and an end-to-end evaluator to select qparams by
a model-level metric instead of local loss alone. Step zero is always a valid
checkpoint. If the selected state does not improve the entry state, or violates
an auxiliary-output tolerance, the original observers and qparams are restored.

```python
from tico.quantization.algorithm.block_reconstruction import (
    ValidationObjective,
)

result = reconstructor.reconstruct(
    block_name="block_0",
    observer_model=quantized_model,
    block=executable_block,
    cache=reconstruction_train_cache,
    selection_cache=reconstruction_selection_cache,
    observer_groups=observer_groups,
    selection_evaluator=evaluate_selection_subset,
    selection_objective=ValidationObjective(
        primary_output="regressors",
        primary_metric="mae",
        output_tolerances={"classifiers": 0.0},
    ),
)
```

`BlockReconstructionResult.accepted` reports whether learned qparams were
committed. `best_step`, `checkpoint_history`, `selected_qparams`, and
`acceptance_reason` make the selection decision auditable.

## Joint windows

Joint-window construction is model-adapter responsibility. A window should:

- contain consecutive semantic groups in model execution order;
- expose every live-in tensor produced outside the window;
- reconstruct every live-out tensor consumed outside the window;
- optimize all tied activation domains in the window atomically;
- commit or roll back the whole window as one unit.

Only per-tensor affine activation observers are supported. Tied observers must
start with the same dtype, qscheme, scale, zero-point, and fake-quant enabled
state.

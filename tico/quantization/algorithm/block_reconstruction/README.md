# Activation block reconstruction

This package provides the first block-reconstruction stage for WrapQ models.
It optimizes frozen per-tensor affine activation qparams against cached
floating-point block outputs while keeping model parameters and weight qparams
fixed.

## Scope of the first PR

- Cache both floating-point and quantized-prefix block inputs.
- Use floating-point block outputs as reconstruction targets.
- Optimize activation scale and, for asymmetric qschemes, zero-point.
- Use straight-through rounding and backend integer limits during optimization.
- Tie producer/consumer observers that represent one logical tensor domain.
- Minimize energy-normalized block-output MSE.
- Commit the best full-cache qparams back to the original observers.
- Process model-specific blocks in execution order through a caller-provided
  adapter.

The package does not yet implement QDrop, adaptive weight rounding, Fisher
weighting, or task-aware detector losses. Those are intentionally separate
follow-up stages.

## Generic usage

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

Only per-tensor affine activation observers are supported in this stage.
Tied observers must start with the same dtype, qscheme, scale, zero-point, and
fake-quant enabled state.

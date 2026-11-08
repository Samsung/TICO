## spinquant

SpinQuant is a rotation-based pre-quantization algorithm for large language models.
Its goal is to make a model more quantization-friendly by applying offline orthogonal
transformations to the weight space before downstream quantization.

> **Reference**: Liu et al., *"SpinQuant: LLM Quantization with Learned Rotations"*,
> arXiv:2405.16406, 2024. — R1 and R2 follow the rotation notation used in this paper.

In this implementation, SpinQuant is used as a **preprocessing / fusion step** rather
than as a standalone quantizer that directly emits low-bit weights. The quantizer
rewrites the model with SpinQuant-style rotations such that a later quantization stage
(for example GPTQ, PTQ, or wrapq-based fake quantization) can operate on the transformed
model.

This implementation is intentionally scoped to the subset currently needed by the PTQ
framework:

- global hidden-dimension rotation (**R1**)
- per-head-dimension rotations (**R2**)
- offline fusion of the corresponding rotations into linear weights where possible

It does **not** implement the full SpinQuant pipeline.

### Configuration

The `SpinQuantConfig` object holds the parameters for the LLaMA offline-fusion path
shown below. Pass this configuration to the public interface to select
`SpinQuantQuantizer`. Qwen3-VL uses a separate configuration and quantizer, described
under [Model assumptions and current support](#model-assumptions-and-current-support).

```python
from tico.quantization.config.spinquant import SpinQuantConfig

SpinQuantConfig(
    init_method="random",
    r1=None,
    r2_map=None,
    show_progress=True,
)
```

#### Parameters

- **init_method**  
  Strategy used to resolve rotation matrices.

  Supported values:
  - "random": generate random orthogonal matrices
  - "hadamard": generate randomized Hadamard-structured orthogonal matrices
  - "external": use user-provided matrices

- **r1**  
  Global hidden-dimension rotation matrix. Required when init_method="external".

- **r2_map**  
  Optional mapping from module keys to per-layer head-dimension rotation matrices.

  Example key:

  ```python
  "model.layers.0.self_attn.R2"
  ```

- **show_progress**
  Whether to show a progress bar while applying rotations (default: `True`).

### How to use SpinQuantQuantizer

Below is an example that demonstrates how to apply SpinQuant via the public interface:

```python
import torch
from transformers import AutoModelForCausalLM

from tico.quantization import prepare, convert
from tico.quantization.config.spinquant import SpinQuantConfig

model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

spinquant_config = SpinQuantConfig(
    init_method="hadamard",
)

model = prepare(model, spinquant_config)
model = convert(model)
```

### Customization in this implementation

This implementation includes one important customization compared with a straightforward
offline fusion approach:

**Tie-embedding-safe rotation handling**

For models with tied input embeddings and LM head weights, directly fusing the embedding-side
and LM-head-side rotations into the original weights would break weight tying or require
extra special-case handling. To avoid that, this implementation does **not** fuse those
two rotations into:

- `model.embed_tokens`
- `lm_head`

Instead, it adds two dedicated runtime rotation layers to the custom `SpinLlama` model:

- `model.model.rotate_embedding`
- `model.rotate_lm_head`

This preserves tied embedding behavior while still applying the same logical transforms
during inference.

### Using externally provided rotations

If you already have rotation matrices, you can provide them explicitly:

```python
import torch
from transformers import AutoModelForCausalLM

from tico.quantization import prepare, convert
from tico.quantization.config.spinquant import SpinQuantConfig

model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

hidden_size = model.config.hidden_size
head_dim = hidden_size // model.config.num_attention_heads

r1 = torch.eye(hidden_size)
r2_map = {
    "model.layers.0.self_attn.R2": torch.eye(head_dim),
    "model.layers.1.self_attn.R2": torch.eye(head_dim),
}

cfg = SpinQuantConfig(
    init_method="external",
    r1=r1,
    r2_map=r2_map,
)

model = prepare(model, cfg)
model = convert(model)
```

### Model assumptions and current support

`SpinQuantQuantizer`, selected by `SpinQuantConfig`, assumes a Hugging Face LLaMA
causal language model structure and validates that:

- the config is a Hugging Face PretrainedConfig
- config.model_type == "llama"
- the model exposes:
  - model
  - lm_head

This LLaMA implementation is designed around the custom `SpinLlamaForCausalLM`
conversion path and LLaMA-style decoder blocks.

The package also provides [`Qwen3VLSpinQuantQuantizer`](qwen3_vl_quantizer.py), selected
by [`Qwen3VLSpinQuantConfig`](../../config/qwen3_vl_spinquant.py). That path supports
tied Qwen3-VL word embeddings, text-side R1/R2 rotations, DeepStack output rotation
fusion, and optional vision-tower rotations. Vision R1 requires LayerNorm affine
folding and a rotation that preserves the all-ones direction; see the configuration
for its controls and validation requirements.

### Precautions and limitations

The following notes apply to the LLaMA examples above:

- It is an **offline fusion / preprocessing** step, not a complete standalone low-bit 
quantization algorithm.
- Embedding-side and LM-head-side rotations are intentionally **not** fused into the
original tied weights.
- The Hadamard-based path currently supports:
  - power-of-two sizes
  - 12 * 2^k sizes
- External rotations must match the exact expected shapes.
- If R2 is omitted in external mode for some layers, those layers simply skip the
layer-specific external R2 override.

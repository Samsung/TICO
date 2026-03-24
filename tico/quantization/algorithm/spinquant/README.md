## spinquant

SpinQuant is a rotation-based pre-quantization algorithm for large language models.
Its goal is to make a model more quantization-friendly by applying offline orthogonal
transformations to the weight space before downstream quantization.

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

The `SpinQuantConfig` object holds the parameters required for SpinQuant offline fusion.
When using the public interface functions, pass an instance of `SpinQuantConfig` so that
the framework dispatches the request to the SpinQuant-specific implementation.

```python
class SpinQuantConfig(
    init_method: Literal["random", "hadamard", "external"] = "random",
    r1: Optional[torch.Tensor] = None,
    r2_map: Optional[Dict[str, torch.Tensor]] = None,
)
```

**Parameters**

- init_method

Strategy used to resolve rotation matrices.

Supported values:
- "random": generate random orthogonal matrices
- "hadamard": generate randomized Hadamard-structured orthogonal matrices
- "external": use user-provided matrices

- r1

Global hidden-dimension rotation matrix. Required when init_method="external".

- r2_map

Optional mapping from module keys to per-layer head-dimension rotation matrices.

Example key:

```python
"model.layers.0.self_attn.R2"
```

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

SpinQuantQuantizer currently assumes a Hugging Face LLaMA causal language model
structure and validates that:

- the config is a Hugging Face PretrainedConfig
- config.model_type == "llama"
- the model exposes:
  - model
  - lm_head

The current implementation is designed around the custom SpinLlamaForCausalLM
conversion path and LLaMA-style decoder blocks.

### Precautions and limitations

- This implementation is currently limited to **LLaMA** models.
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

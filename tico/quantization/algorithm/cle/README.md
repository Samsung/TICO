## Cross-Layer Equalization (CLE)

Cross-Layer Equalization (CLE) is a **data-free preprocessing algorithm** for post-training quantization (PTQ).
It rescales adjacent layers to balance per-channel weight magnitudes, which improves quantization robustness—especially for low-bit weight quantization.

Unlike methods such as GPTQ, CLE does **not require calibration data** and operates directly on model weights.

---

### When to Use CLE

CLE is particularly useful in the following scenarios:

- **Before weight quantization (e.g., GPTQ, PTQ, UINT8/UINT4)**
  - Helps reduce per-channel scale imbalance
  - Improves quantization accuracy without extra data

- **Models with large channel-wise variance**
  - e.g., Transformer MLP layers (`up_proj → down_proj`)
  - CNN layers with uneven filter magnitudes

- **Data-free pipelines**
  - When calibration data is unavailable or expensive

---

### When NOT to Use

- If your model is already well-balanced (e.g., after SmoothQuant-like methods)
- If you rely heavily on activation-aware methods (CLE is weight-only)

---

### How CLE Works (Intuition)

CLE rescales two consecutive layers:

- First layer output channels are **scaled up/down**
- Second layer input channels are **scaled inversely**

This preserves the model's function algebraically only when the positive channel scales can
be moved through the operations between the two layers. For a diagonal scale `S`, an
intervening operation `f` must satisfy `f(Sx) = S f(x)`. ReLU satisfies this condition;
SiLU and GELU generally do not. The first layer's bias must also be scaled when it is
nonzero. Any other consumers of the scaled output need corresponding compensation.

The implementation checks layer types and channel counts, not these graph conditions.
Choose pairs whose complete dataflow satisfies them before applying CLE.

---

### Configuration

The `CLEConfig` object defines how CLE is applied.

```python
from tico.quantization.config.cle import CLEConfig

cle_config = CLEConfig(
    pairs=[
        ("model.layers.*.mlp.up_proj", "model.layers.*.mlp.down_proj"),
    ],
    method="absmax",  # or "range"
)
```

### Key Parameters

- **pairs**  
  List of layer pairs to equalize. Supports wildcard (`*`) patterns.

- **method** (`"absmax"` | `"range"`)  
  Defines how per-channel magnitude is measured (see below).

- **scale_range**  
  Clamp range for stability (default: `(1e-8, 1e8)`)

- **max_iter**  
  Number of times to apply CLE (usually 1 is sufficient)

- **equalize_bias**  
  Whether to scale the first layer's bias along with its weights (default: `True`).
  Disabling it with a nonzero bias generally breaks function preservation.

---

### How to Use CLEQuantizer

```python
from tico.quantization import prepare, convert
from tico.quantization.config.cle import CLEConfig

cle_config = CLEConfig(
    pairs=[
        ("model.layers.*.mlp.up_proj", "model.layers.*.mlp.down_proj"),
    ]
)

model = prepare(model, cle_config)
model = convert(model)
```

- `prepare()` does nothing (no calibration needed)
- `convert()` applies CLE directly to model weights

---

### `absmax` vs `range`

CLE relies on estimating **per-channel magnitude**.
This is where `method` becomes important.

#### 1. `absmax`

```
range = max(|w|)
```

**Meaning**  
Measures the largest absolute value in each channel.

**Characteristics**
- Robust to sign (handles both positive/negative well)
- Less sensitive to outliers
- Stable for most modern architectures

**When to use**
- Default choice for most cases
- Transformer models (LLMs, ViT, etc.)
- When weights are roughly symmetric around zero

#### 2. `range`

```
range = max(w) - min(w)
```

**Meaning**  
Measures full dynamic range of each channel.

**Characteristics**
- Sensitive to outliers
- Captures asymmetric distributions better
- Can be unstable if extreme values exist

**When to use**
- When weights are highly asymmetric
- CNN models with strong activation bias
- When using unsigned quantization (e.g., uint8 activations)

---

### Practical Recommendation

| Scenario | Recommended Method |
|--------|------------------|
| LLM (Transformer) | `absmax` |
| General use | `absmax` |
| Highly skewed weights | `range` |
| Experimental tuning | try both |

---

### Precautions

- CLE modifies weights **in-place**
- It is a **preprocessing step**, not a quantizer
- Must be followed by actual quantization (e.g., GPTQ, PTQ)

---

### Limitations

- Only supports **Conv2d and Linear layers**
- Requires matching channels:
  ```
  first.weight.shape[0] == second.weight.shape[1]
  ```
- Does not validate that the selected layers are connected, that intervening
  operations commute with channel scaling, or that every branch consuming the
  first layer's output is compensated. Those checks are the caller's responsibility.
- Does not handle:
  - Group convolution (yet)
  - BatchNorm folding explicitly

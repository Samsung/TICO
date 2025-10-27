## algorithm

The _algorithm_ module provides a collection of state-of-the-art quantization algorithms 
 for deep learning. These methods rewrites target graphs prior to the actual quantization 
step, ensuring minimal performance loss and imporved quantization accuracy.

### Design Philosophy

This module is desinged to be **self-contained** and **independent** of other components
 in the codebase. The primary reasons for this design are:

#### 1. External Dependencies

The _algorithm_ module relies on external libraries such as `transformers`, which may not
 be compatible with the minimal dependencies required by other parts of the project.

#### 2. Modular and Maintainable Code

Each algorithm in this module is implemented as a standalone component to:
- Avoid tight coupling with internal project code.
- Ensure ease of testing, maintenance, and future updates.

### Usage Guidelines

#### Do not Cross-Integrate

The _algorithm_ module should not be directly referenced by other modules or components
 within the _TICO_ codebase. Instead, this module is desinged to be used independently
for quantization and other algorithm-specific tasks.

#### Dependency Isolation

Ensure that any code or scripts utilizing this module explicitly install the required dependencies,
 such as `transformers`. Dependencies for this module should not be propagated to other project
components.

### Example Use Case

To utilize an algorithm from this module, such as SmoothQuant:

1. Install necessary dependencies:

```bash
pip install -r smooth_quant.txt
```

2. Import and use the specific algorithm:

```python
from tico.quantization.algorithm import smooth_quant

# Example usage of SmoothQuant
# ..
max_acts = smooth_quant.get_channelwise_activation_max(
    model, tokenizer, dataset, num_samples=10
)
smooth_quant.apply_smoothing(model, max_acts)
# ..
```

3. Ensure any output or data exchange between _algorithm_ and other modules is done via
 well-defined interfaces.

#### Contributing to the Module

- New algorithms should be implemented as standalone modules with minimal interdependencies.
- Avoid introducing code that requires circular imports or tight integration with internal
 project components.
- Document any external dependencies clearly in the module's README or requirements file.

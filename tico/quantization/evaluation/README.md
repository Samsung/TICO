## evaluation

The _evaluation_ module provides a convenient way to:

- Compile and run the Circle model on a specified backend to verify accuracy.
- Evaluate and optionally collect custom metrics.

All evaluation logic is encapsulated in the `evaluate()` function. It accepts a PyTorch
module, a `CircleModel`, a registered backend, input data, an output mode, and optional
custom metrics. The built-in backends are `BACKEND.CIRCLE` and `BACKEND.TRIV24`.

### Usage

#### Example: Basic Evaluation

The `evaluate()` function is the main entry point for comparing an existing PyTorch
module with its quantized Circle artifact on a chosen backend.

```python
from tico.quantization.evaluation.backend import BACKEND
from tico.quantization.evaluation.evaluate import evaluate
from tico.utils.model import CircleModel

circle_model = CircleModel.load("quantized_model.circle")

results = evaluate(
    torch_module=torch_module.eval(),
    circle_model=circle_model,
    backend=BACKEND.CIRCLE,  # Built-in reference runtime in fake-quantize mode
    # input_data=input_data,  # Otherwise, random floating-point inputs are generated
    mode="return",  # "return" or "plot"
    custom_metrics={},  # Optional mapping from metric names to callables
)

print("Evaluation results:", results)
```

`circle_model` must be a `CircleModel`, not a file path. You can also pass the object
returned by `tico.convert()`. Supply `input_data` when the model requires specific
input values or dtypes, such as integer token IDs.

`evaluate()` computes the built-in `diff`, `max_abs_diff`, `peir`, and `mse` metrics
plus any `custom_metrics`; it does not accept a `metrics` argument. Each custom metric
receives the PyTorch and Circle output tensors and returns a scalar. Custom names must
not duplicate built-in names. In `mode="return"`, each metric maps to a list containing
one value per output tensor. See the function docstring for the remaining parameters.

`BACKEND.CIRCLE` executes the quantized Circle model with TICO's reference runtime in
`ExecutionMode.FAKE_QUANTIZE` (see `tico/circle/README.md`, "Reference runtime"). This
reproduces the semantics of the former `onecc quantize --fake_quantize` evaluation
without any ONE installation: weights are dequantized, operators compute in FLOAT32,
and each quantized activation is rounded to its serialized grid. It measures the
quantization error of the exported parameters and is not a bit-exact emulation of an
integer backend; use `BACKEND.TRIV24` for backend-exact results.

### Adding a New Backend

#### Step 1: Create a Custom BackendExecutor

To introduce a new backend, create a class that inherits from the abstract
`BackendExecutor` in `tico/quantization/evaluation/executor/my_custom_backend.py`.
The following skeleton requires backend-specific compilation and inference bodies;
`run_inference()` must return the output tensors as NumPy arrays.

```python
from typing import List

import numpy as np
import torch

from tico.quantization.evaluation.executor.backend_executor import BackendExecutor
from tico.utils.model import CircleModel

class MyCustomBackend(BackendExecutor):
    def compile(self, circle_model: CircleModel) -> None:
        # Load the binary and perform any backend-specific compilation.
        raise NotImplementedError("Implement backend compilation")

    def run_inference(self, input_data: List[torch.Tensor]) -> List[np.ndarray]:
        # Execute the compiled model and return one NumPy array per output.
        raise NotImplementedError("Implement backend inference")
```

#### Step 2: Register the Backend

1. Add your new backend name to the `BACKEND` enum in
   `tico/quantization/evaluation/backend.py`.

```python
from enum import Enum

class BACKEND(Enum):
    CIRCLE = 1
    TRIV24 = 2
    MY_CUSTOM_BACKEND = 3
```

2. Import the executor in `tico/quantization/evaluation/evaluate.py` and add its
   mapping after the existing `BACKEND_TO_EXECUTOR` declaration.

```python
from tico.quantization.evaluation.executor.my_custom_backend import MyCustomBackend

BACKEND_TO_EXECUTOR[BACKEND.MY_CUSTOM_BACKEND] = MyCustomBackend
```

With these changes, calling `evaluate()` with `backend=BACKEND.MY_CUSTOM_BACKEND` will use 
 `MyCustomBackend` for compilation and inference.

```python
results = evaluate(
    torch_module=torch_module.eval(),
    circle_model=circle_model,
    backend=BACKEND.MY_CUSTOM_BACKEND,
    mode="return",
)
```

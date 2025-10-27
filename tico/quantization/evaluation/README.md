## evaluation

The _evaluation_ module provides a convenient way to:

- Compile and run the Circle model on a specified backend to measure performance or verify accuracy.
- Evaluate and optionally collect custom metrics.

All evaluation logic is encapsulated in the `evaluate()` function. This function accepts a PyTorch module,
 a Circle model, a backend (e.g., CPU, GPU, or custom), input data, evaluation modes, and various metrics 
to compare.

### Usage

#### Example: Basic Evaluation

The `evaluate()` function is the main entry point for running comparisons between your PyTorch module 
 and a quantized Circle model on a chosen backend.

```python
from evaluation import evaluate

"""
Suppose you already have:

1) A PyTorch module
2) A quantized Circle model
3) (optional) Some input_data for inference
4) A list of metrics you want to compute

Evaluate the two models (PyTorch vs. Circle on CPU, for example)
"""

results = evaluate(
    torch_module=torch_module,
    circle_model=circle_model_path,
    backend=BACKEND.CIRCLE,  # Use circle interperter
    # input_data=input_data, # Use random data if not specified
    mode='return',  # Could be 'return', 'plot', etc.
    metrics=['peir'],  # Built-in metrics (default: peir)
    custom_metrics=None  # or provide a dict of your own metrics
)

print("Evaluation results:", results)
```

A more detailed explanation of each parameter is available in the docstrings within the codebase.

### Adding a New Backend

#### Step 1: Create a Custom BackendExecutor

To introduce a new backend, create a class that inherits from the abstract `BackendExecutor`. This class 
 should live in `executor/` directory (e.g., circle_executor.py):

```python
from executor.backend_executor import BackendExecutor

class MyCustomBackend(BackendExecutor):
    """
    BackendExecutor for MyCustomBackend.

    This backend demonstrates how to compile and run inference for a circle model.
    """

    def compile(self, circle_model: CircleModel) -> None:
        # Perform any compilation steps required by this backend
        # e.g., load the binary, configure it for specialized hardware, etc.
        pass

    def run_inference(self, input_data: List[torch.Tensor]) -> List[np.ndarray]:
        # Run the model on your specialized backend
        # Return the inference result
        return output
```

#### Step 2: Register the Backend

1. Add your new backend name to the `BACKEND` enum (in backend.py).

```python
from enum import Enum

class BACKEND(Enum):
    CIRCLE = 1
    TRIV24 = 2
    # ... other existing backends ...
    MY_CUSTOM_BACKEND = 5  # Add your new backend
```

2. Map your backend name to your custom executor class (in evaluate.py).

```python
from .my_custom_backend import MyCustomBackend

BACKEND_TO_EXECUTOR: Dict[BACKEND, type[BackendExecutor]] = {
    BACKEND.CIRCLE: CircleExecutor,
    BACKEND.TRIV24: Triv24Executor,
    # ...
    BACKEND.MY_CUSTOM_BACKEND: MyCustomBackend,  # Add your mapping
}
```

With these changes, calling `evaluate()` with `backend=BACKEND.MY_CUSTOM_BACKEND` will use 
 `MyCustomBackend` for compilation and inference.

```python
results = evaluate(
    torch_module=torch_module,
    circle_model=circle_model_path,
    backend=BACKEND.MY_CUSTOM_BACKEND,
    mode='return',
)
```

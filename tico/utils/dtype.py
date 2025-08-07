import numpy as np
import torch

from circle_schema import circle

NUMPY_TO_TORCH_DTYPE_DICT = {
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("float16"): torch.float16,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
    np.dtype("int64"): torch.int64,
    np.dtype("int32"): torch.int32,
    np.dtype("int16"): torch.int16,
    np.dtype("int8"): torch.int8,
    np.dtype("uint8"): torch.uint8,
    np.dtype("bool"): torch.bool,
}

CIRCLE_TO_TORCH_DTYPE_DICT = {
    circle.TensorType.TensorType.FLOAT32: torch.float32,
    circle.TensorType.TensorType.FLOAT32: torch.float,
    circle.TensorType.TensorType.UINT8: torch.uint8,
    circle.TensorType.TensorType.INT8: torch.int8,
    circle.TensorType.TensorType.INT16: torch.int16,
    circle.TensorType.TensorType.INT16: torch.short,
    circle.TensorType.TensorType.INT32: torch.int32,
    circle.TensorType.TensorType.INT32: torch.int,
    circle.TensorType.TensorType.INT64: torch.int64,
    circle.TensorType.TensorType.BOOL: torch.bool,
}


def numpy_dtype_to_torch_dtype(np_dtype: np.dtype) -> torch.dtype:
    return NUMPY_TO_TORCH_DTYPE_DICT[np_dtype]


def circle_dtype_to_torch_dtype(circle_dtype: int) -> torch.dtype:
    assert isinstance(circle_dtype, int)
    if circle_dtype not in CIRCLE_TO_TORCH_DTYPE_DICT:
        raise RuntimeError(f"Unsupported dtype {circle_dtype}")

    torch_dtype = CIRCLE_TO_TORCH_DTYPE_DICT[circle_dtype]
    assert torch_dtype is not None
    return torch_dtype

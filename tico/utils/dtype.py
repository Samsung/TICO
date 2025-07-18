import numpy as np
import torch

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


def numpy_dtype_to_torch_dtype(np_dtype: np.dtype) -> torch.dtype:
    return NUMPY_TO_TORCH_DTYPE_DICT[np_dtype]

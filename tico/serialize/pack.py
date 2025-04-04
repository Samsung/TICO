import numpy as np


def pack_buffer(flat_data: np.ndarray, dtype: str) -> np.ndarray:
    assert len(flat_data.shape) == 1

    if dtype == "uint4":
        if flat_data.dtype != np.uint8:
            raise RuntimeError("uint4 data should be saved in uint8.")

        numel = flat_data.shape[0]
        packed = np.zeros((numel + 1) // 2, dtype=np.uint8)
        for i in range(numel):
            if i % 2 == 0:
                packed[i // 2] = flat_data[i] & 0x0F
            else:
                packed[i // 2] |= flat_data[i] << 4
        return packed
    else:
        raise RuntimeError(f"NYI dtype: {dtype}")

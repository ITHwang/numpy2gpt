from typing import Any, NewType, TypeAlias

import numpy as np

# NOTE: In the original PyTorch, when importing `torch`, dtypes are initialized.
# For simplicity, we just re-define the dtypes from numpy.
# Refer: https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_dtypes.cpp
int32 = NewType("int32", np.int32)  # type: ignore
int64 = NewType("int64", np.int64)  # type: ignore
float32 = NewType("float32", np.float32)  # type: ignore
float64 = NewType("float64", np.float64)  # type: ignore
int32.__module__ = "torch"
int64.__module__ = "torch"
float32.__module__ = "torch"
float64.__module__ = "torch"

TORCH_DTYPE: TypeAlias = type[int32] | type[int64] | type[float32] | type[float64]
NUMPY_DTYPE: TypeAlias = (
    type[np.int32] | type[np.int64] | type[np.float32] | type[np.float64]
)
NUMERIC_TYPE: TypeAlias = int | float | NUMPY_DTYPE
ARRAY_TYPE: TypeAlias = np.ndarray | list[NUMERIC_TYPE]
INPUT_TYPE: TypeAlias = NUMERIC_TYPE | ARRAY_TYPE
INPUT_TYPE_TUPLE = (
    int,
    float,
    list,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.ndarray,
)

np2torch: dict[NUMPY_DTYPE, TORCH_DTYPE] = {
    np.int32: int32,
    np.int64: int64,
    np.float32: float32,
    np.float64: float64,
}
torch2np = {v: k for k, v in np2torch.items()}


def type_np2torch(np_dtype: np.dtype) -> TORCH_DTYPE:
    """Cast a NumPy dtype object to the corresponding torch dtype object."""
    np_dtype_scalar: NUMPY_DTYPE = np_dtype.type

    if np_dtype_scalar not in np2torch:
        raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")

    return np2torch[np_dtype_scalar]


def type_torch2np(torch_dtype: TORCH_DTYPE) -> NUMPY_DTYPE:
    """Cast a torch dtype object to the corresponding NumPy dtype object."""
    if torch_dtype not in torch2np:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

    return torch2np[torch_dtype]

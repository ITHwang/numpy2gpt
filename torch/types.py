from typing import Any, NewType, TypeAlias

import numpy as np

NUMERIC_TYPE: TypeAlias = int | float | np.int32 | np.int64 | np.float32 | np.float64
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

TORCH_TYPE: TypeAlias = type[int32] | type[int64] | type[float32] | type[float64]

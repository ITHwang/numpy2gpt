from typing import Any, Callable

import numpy as np

import torch


def goldstein(x: torch.Tensor, y: torch.Tensor) -> Any:
    z = (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


def sphere(x: torch.Tensor, y: torch.Tensor) -> Any:
    return x**2 + y**2


def matyas(x: torch.Tensor, y: torch.Tensor) -> Any:
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def numerical_diff(
    f: Callable[
        [torch.INPUT_TYPE | torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, ...],
    ],
    x: torch.Tensor,
    eps: float = 1e-4,
) -> float:
    """Calculate the numerical gradient of a function.

    The central difference is more accurate than the forward difference.
    See: http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture27.pdf

    Args:
        f: A function that takes a Tensor and returns a Tensor.
        x: The input Tensor.
        eps: The epsilon value for numerical differentiation.
    """
    x_data: np.ndarray = x._data
    x0 = torch.tensor(x_data - eps)
    x1 = torch.tensor(x_data + eps)
    y0 = f(x0)
    y1 = f(x1)

    assert isinstance(y0, torch.Tensor), (
        "The case that the output is a tuple of Tensors is not considered yet. "
        "If you met the case, please implement."
    )
    assert isinstance(y1, torch.Tensor), (
        "The case that the output is a tuple of Tensors is not considered yet. "
        "If you met the case, please implement."
    )

    numerator = y1._data - y0._data
    if isinstance(numerator, np.ndarray) and numerator.size == 1:
        numerator = float(numerator[0])
    denominator = 2 * eps

    return float(numerator / denominator)

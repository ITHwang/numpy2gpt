from typing import Any

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

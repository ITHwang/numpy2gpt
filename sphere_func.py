from typing import Any

import numpy as np

import torch


def sphere(x: torch.Tensor, y: torch.Tensor) -> Any:
    return x**2 + y**2


x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0], requires_grad=True)

z = sphere(x, y)

z.backward()

print(x)
print(x.grad)
print(y)
print(y.grad)
print(z)

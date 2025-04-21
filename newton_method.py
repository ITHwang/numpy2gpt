import numpy as np

import torch

"""
Newton's method

x <- x - f'(x) / f''(x)
"""


def f(x: torch.Tensor) -> torch.Tensor:
    y = x**4 - 2 * x**2
    return y  # type: ignore


x = torch.tensor(2.0, requires_grad=True)
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.grad = None
    y.backward(create_graph=True)

    gx = x.grad
    x.grad = None
    gx.backward()  # type: ignore
    gx2 = x.grad

    with torch.no_grad():
        x -= gx / gx2

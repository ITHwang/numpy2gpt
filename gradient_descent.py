import torch
from torch import Tensor


def rosenbrock(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2

    assert isinstance(y, torch.Tensor)

    return y


if __name__ == "__main__":
    x0 = torch.tensor(0.0, requires_grad=True)
    x1 = torch.tensor(2.0, requires_grad=True)
    lr = 0.001
    iters = 1000

    for i in range(iters):
        print(x0, x1)

        y = rosenbrock(x0, x1)

        x0.grad = None
        x1.grad = None
        y.backward()

        with torch.no_grad():
            x0 -= lr * x0.grad  # type: ignore
            x1 -= lr * x1.grad  # type: ignore

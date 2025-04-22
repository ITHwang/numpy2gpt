import numpy as np

from .core import Function, Tensor
from .types import INPUT_TYPE


class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)

    def backward(self, gy: Tensor) -> Tensor:
        (x,) = self.inputs
        gx: Tensor = gy * cos(x)  # type: ignore

        return gx


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)

    def backward(self, gy: Tensor) -> Tensor:
        (x,) = self.inputs
        gx: Tensor = -gy * sin(x)  # type: ignore

        return gx


class Square(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        if len(xs) != 1:
            raise ValueError("Square must take one argument")

        x = xs[0]
        y = x**2

        return y

    def backward(self, *gys: Tensor) -> Tensor:
        if len(gys) != 1:
            raise ValueError("Square must take one argument")

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = 2 * x * gy  # type: ignore

        return gx


class Exp(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        if len(xs) != 1:
            raise ValueError("Exp must take one argument")

        x = xs[0]
        y = np.exp(x)

        return y

    def backward(self, *gys: Tensor) -> Tensor:
        if len(gys) != 1:
            raise ValueError("Exp must take one argument")

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = exp(x) * gy  # type: ignore

        return gx


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def backward(self, gy: Tensor) -> Tensor:
        y = self.outputs[0]()
        gx: Tensor = gy * (1 - y * y)

        return gx


def sin(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Sin()(x)


def cos(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Cos()(x)


def square(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Square()(x)


def exp(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Exp()(x)


def tanh(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Tanh()(x)

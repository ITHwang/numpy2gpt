from __future__ import annotations

import numpy as np

from .core import Function, Tensor
from .types import INPUT_TYPE


class Sin(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        assert len(xs) == 1, "Sin must take one argument"

        x = xs[0]

        return np.sin(x)

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Sin must take one argument"

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = gy * cos(x)

        return gx


class Cos(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        assert len(xs) == 1, "Cos must take one argument"

        x = xs[0]

        return np.cos(x)

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Cos must take one argument"

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = -gy * sin(x)

        return gx


class Square(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        assert len(xs) == 1, "Square must take one argument"

        x = xs[0]
        y = x**2

        return y

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Square must take one argument"

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = 2 * x * gy

        return gx


class Exp(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        assert len(xs) == 1, "Exp must take one argument"

        x = xs[0]
        y = np.exp(x)

        return y

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Exp must take one argument"

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = exp(x) * gy

        return gx


class Tanh(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        assert len(xs) == 1, "Tanh must take one argument"

        x = xs[0]

        return np.tanh(x)

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Tanh must take one argument"

        gy = gys[0]
        y = self.outputs[0]()
        gx = gy * (1 - y * y)

        return gx


def sin(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.sin.html"""
    result = Sin()(x)

    assert len(result) == 1, "sin must return a single Tensor"

    return result[0]


def cos(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.cos.html"""
    result = Cos()(x)

    assert len(result) == 1, "cos must return a single Tensor"

    return result[0]


def square(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.square.html"""
    result = Square()(x)

    assert len(result) == 1, "square must return a single Tensor"

    return result[0]


def exp(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.exp.html"""
    result = Exp()(x)

    assert len(result) == 1, "exp must return a single Tensor"

    return result[0]


def tanh(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.tanh.html"""
    result = Tanh()(x)

    assert len(result) == 1, "tanh must return a single Tensor"

    return result[0]

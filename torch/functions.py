from __future__ import annotations

import numpy as np

from .core import Function, Tensor, as_tensor
from .types import INPUT_TYPE


class Sin(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Sin must take one argument"

        if inplace:
            raise NotImplementedError(
                "Sin does not currently support the inplace operation"
            )

        x = xs[0]

        return np.sin(x)

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Sin must take one argument"

        if self.inplace:
            raise NotImplementedError(
                "Sin does not currently support the inplace operation"
            )

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = gy * cos(x)

        return gx


class Cos(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Cos must take one argument"

        if inplace:
            raise NotImplementedError(
                "Cos does not currently support the inplace operation"
            )

        x = xs[0]

        return np.cos(x)

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Cos must take one argument"

        if self.inplace:
            raise NotImplementedError(
                "Cos does not currently support the inplace operation"
            )

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = -gy * sin(x)

        return gx


class Square(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Square must take one argument"

        if inplace:
            raise NotImplementedError(
                "Square does not currently support the inplace operation"
            )

        x = xs[0]
        y = x**2

        return y

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Square must take one argument"

        if self.inplace:
            raise NotImplementedError(
                "Square does not currently support the inplace operation"
            )

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = 2 * x * gy

        return gx


class Exp(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Exp must take one argument"

        if inplace:
            raise NotImplementedError(
                "Exp does not currently support the inplace operation"
            )

        x = xs[0]
        y = np.exp(x)

        return y

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Exp must take one argument"

        if self.inplace:
            raise NotImplementedError(
                "Exp does not currently support the inplace operation"
            )

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = exp(x) * gy

        return gx


class Tanh(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Tanh must take one argument"

        if inplace:
            raise NotImplementedError(
                "Tanh does not currently support the inplace operation"
            )

        x = xs[0]

        return np.tanh(x)

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Tanh must take one argument"

        if self.inplace:
            raise NotImplementedError(
                "Tanh does not currently support the inplace operation"
            )

        gy = gys[0]
        y = self.outputs[0]()
        gx = gy * (1 - y * y)

        return gx


def sin(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.sin.html"""
    x = as_tensor(x)

    result = Sin()(x)

    assert isinstance(result, tuple), "sin isn't in-place operation"
    assert len(result) == 1, "sin must return a single Tensor"

    return result[0]


def cos(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.cos.html"""
    x = as_tensor(x)

    result = Cos()(x)

    assert isinstance(result, tuple), "cos isn't in-place operation"
    assert len(result) == 1, "cos must return a single Tensor"

    return result[0]


def square(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.square.html"""
    x = as_tensor(x)

    result = Square()(x)

    assert isinstance(result, tuple), "square must return a tuple"
    assert len(result) == 1, "square must return a single Tensor"

    return result[0]


def exp(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.exp.html"""
    x = as_tensor(x)

    result = Exp()(x)

    assert isinstance(result, tuple), "exp must return a tuple"
    assert len(result) == 1, "exp must return a single Tensor"

    return result[0]


def tanh(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.tanh.html"""
    x = as_tensor(x)

    result = Tanh()(x)

    assert isinstance(result, tuple), "tanh must return a tuple"
    assert len(result) == 1, "tanh must return a single Tensor"

    return result[0]

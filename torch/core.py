from __future__ import annotations

from typing import Any

import numpy as np


def as_array(x: Any) -> np.ndarray:
    """Convert a scalar or a list to a numpy.ndarray

    If x is a scalar, convert it to a numpy.ndarray with shape (1,).
    See: https://github.com/numpy/numpy/blob/v2.1.0/numpy/_core/numeric.py#L1937-L2021
    """
    if np.isscalar(x):
        return np.array(x)
    return x


def tensor(data: np.ndarray | Any) -> Tensor:
    """An imitation of torch.tensor in PyTorch

    Originally, torch.tensor is written in C++.
    Python binding code:
        https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_torch_functions_manual.cpp#L243-L267
    """
    return Tensor(as_array(data))


class Tensor:
    """An imitation of torch.Tensor in PyTorch

    fyi. in PyTorch,
    Every `torch.Tensor` is a `Variable`(exactly same), a internal C++ class.
    Each `Variable` has one unique `AutogradMeta` struct,
    which stores autograd metadata fields including `grad_`, `grad_fn`, etc.
    See: https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.h
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Args:
            data: numpy.ndarray
            grad: numpy.ndarray | None
            creator: when forwarding, memorize the function that created this tensor
                for backward propagation.
                Similarly, in PyTorch `Variable`s have the notion of a `gradient_edge`, which is the
                edge in the autograd graph that connects the variable to a particular input
                of the gradient function(`grad_fn` or `grad_accumulator`).
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Data must be a numpy.ndarray. But got {type(data)}")

        self.data = data
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None

    @property
    def creator(self) -> Function | None:
        return self._creator

    @creator.setter
    def creator(self, func: Function) -> None:
        self._creator = func

    def backward(self) -> None:
        """Backward propagation.

        Traverses the computational graph backwards:
        - Starts with the function that created this tensor
        - Computes gradients by calling each function's backward method
        - Propagates gradients to input tensors
        - Continues recursively through the entire graph
        """
        # if the gradient is not set, set it to 1.0
        # this is because dy/dy = 1.0
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        if self.creator is not None:
            funcs.append(self.creator)

        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Tensor) -> Tensor:
        x = input.data
        y = self.forward(x)
        # Sometimes the output of the forward method is a scalar,
        # when the input is a zero-dimensional array.
        output = Tensor(as_array(y))
        output.creator = self

        self.input: Tensor = input
        self.output: Tensor = output

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, gy: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: np.ndarray) -> np.ndarray:
    return Square()(x)


def exp(x: np.ndarray) -> np.ndarray:
    return Exp()(x)


if __name__ == "__main__":
    x = Tensor(np.array(0.5))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)

    # TODO: it shoud be possible
    # https://www.kaggle.com/code/reichenbch/tensor-playground
    b = Tensor(2.0)

from __future__ import annotations

from typing import TypeAlias

import numpy as np

T: TypeAlias = (
    int
    | float
    | list[int | float]
    | np.int32
    | np.int64
    | np.float32
    | np.float64
    | np.ndarray
)
T_TUPLE = (
    int,
    float,
    list,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.ndarray,
)


def tensor(data: T | None = None) -> Tensor:
    """An imitation of torch.tensor in PyTorch

    Originally, torch.tensor is written in C++.
    Python binding code:
        https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_torch_functions_manual.cpp#L243-L267
    """
    return Tensor(data)


class Tensor:
    """An imitation of torch.Tensor in PyTorch

    fyi. in PyTorch,
    Every `torch.Tensor` is a `Variable`(exactly same), a internal C++ class.
    Each `Variable` has one unique `AutogradMeta` struct,
    which stores autograd metadata fields including `grad_`, `grad_fn`, etc.
    See: https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.h
    """

    def __init__(self, data: T | None = None) -> None:
        """
        Args:
            data: numpy.ndarray | int | float | None
            grad: numpy.ndarray | None
            creator: when forwarding, memorize the function that created this tensor
                for backward propagation.
                Similarly, in PyTorch `Variable`s have the notion of a `gradient_edge`, which is the
                edge in the autograd graph that connects the variable to a particular input
                of the gradient function(`grad_fn` or `grad_accumulator`).
        """
        if data is not None:
            if isinstance(data, np.ndarray):
                pass
            elif isinstance(data, T_TUPLE):
                data = np.array(data)
            else:
                raise ValueError(f"Data has an invalid type: {type(data)}")

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

            # get gradients of outputs
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            # set gradients of inputs
            # If the input has a creator, it means that the input is an output of another function.
            # So, we need to add the creator to the list of functions to get another gradient.
            for x, gx in zip(f.inputs, gxs):
                # if the grad is set in the loop, accumulate it.
                if x.grad is None:
                    x.grad = gx
                else:
                    # DO NOT use +=, it is in-place operation(numpy) causing side effects.
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self) -> None:
        self.grad = None


class Function:
    def __call__(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Do forward propagation.

        Warning: Sometimes the output of the forward method is a scalar,
            when the input is a zero-dimensional array.

        Args:
            inputs: list[Tensor]

        Returns:
            Tensor | list[Tensor]
        """
        xs = tuple(input.data for input in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = tuple(Tensor(y) for y in ys)
        for output in outputs:
            output.creator = self

        self.inputs: tuple[Tensor, ...] = inputs
        self.outputs: tuple[Tensor, ...] = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        raise NotImplementedError

    def backward(self, *gys: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        raise NotImplementedError


class Add(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        if len(xs) != 2:
            raise ValueError("Add must take two arguments")

        x0, x1 = xs
        y = x0 + x1

        return y

    def backward(self, *gys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(gys) != 1:
            raise ValueError("Add must take one argument")

        gy = gys[0]

        return gy, gy


class Square(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        if len(xs) != 1:
            raise ValueError("Square must take one argument")

        x = xs[0]
        y = x**2

        return y

    def backward(self, *gys: np.ndarray) -> np.ndarray:
        if len(gys) != 1:
            raise ValueError("Square must take one argument")

        gy = gys[0]
        x = self.inputs[0].data
        gx = 2 * x * gy

        return gx


class Exp(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        if len(xs) != 1:
            raise ValueError("Exp must take one argument")

        x = xs[0]
        y = np.exp(x)

        return y

    def backward(self, *gys: np.ndarray) -> np.ndarray:
        if len(gys) != 1:
            raise ValueError("Exp must take one argument")

        gy = gys[0]
        x = self.inputs[0].data
        gx = np.exp(x) * gy

        return gx


def square(x: np.ndarray) -> np.ndarray:
    return Square()(x)


def exp(x: np.ndarray) -> np.ndarray:
    return Exp()(x)


def add(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return Add()(x0, x1)


if __name__ == "__main__":
    x0 = Tensor(np.array(2.0))

    y = add(add(x0, x0), x0)
    y.backward()
    print(y.data)
    print(x0.grad)

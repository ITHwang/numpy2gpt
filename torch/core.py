from __future__ import annotations

import weakref
from typing import TypeAlias

import numpy as np

from torch.priority_queue import PriorityQueue

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
        self.grad: numpy.ndarray | None
        self.creator: when forwarding, memorize the function that created this tensor
            for backward propagation.
            Similarly, in PyTorch `Variable`s have the notion of a `gradient_edge`, which is the
            edge in the autograd graph that connects the variable to a particular input
            of the gradient function(`grad_fn` or `grad_accumulator`).
        self.generation: the generation of the tensor.
            A function gets inputs and outputs.
            e.g., [input] -> [function] -> [output]

            The generation of the input will be the generation of the function.
            The generation of the output is the generation of the function plus 1.
            e.g., [input][gen=2] -> [function][gen=2] -> [output][gen=3]

            Generation is for determining the order of backward propagation.
            The backward propagation that has inputs with higher generation will be called first.
            If the generations of inputs are different, the higher generation will be the one of the function.

        Args:
            data: numpy.ndarray | int | float | None
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
        self.generation: int = 0

    @property
    def creator(self) -> Function | None:
        return self._creator

    @creator.setter
    def creator(self, func: Function) -> None:
        self._creator = func

    def backward(self, retain_grad: bool = False) -> None:
        """Backward propagation.

        Traverses the computational graph backwards:
        - Starts with the function that created this tensor
        - Computes gradients by calling each function's backward method
        - Propagates gradients to input tensors
        - Continues recursively through the entire graph

        Args:
            retain_grad: Whether to keep the gradients of the output tensors
                In most cases, we don't need to keep the gradients in the middle of the backpropagation.
                We finally use only the gradient of the "first" input tensor.(dy/dx)
                So, we set it to False by default.
        """
        # if the gradient is not set, set it to 1.0
        # this is because dy/dy = 1.0
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs: PriorityQueue[Function] = PriorityQueue()
        seen_set: set[Function] = set()

        def add_func(f: Function) -> None:
            """Push a func to the priority queue.

            If the func is already in the seen set, it will not be pushed,
            which prevent the same func from doing backward propagation multiple times.

            Args:
                f (Function): the function to push
            """
            if f not in seen_set:
                funcs.push(f, f.generation)
                seen_set.add(f)

        # push the creator which creates the final output
        if self.creator is None:
            raise ValueError("The final output should have a creator")
        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            # get gradients of outputs
            # NOTE: f.outputs is a tuple of weakref.ref[Tensor]
            gys = [output().grad for output in f.outputs]  # type: ignore
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            # set gradients of inputs
            for x, gx in zip(f.inputs, gxs):
                # if the grad is set in the loop, accumulate it.
                if x.grad is None:
                    x.grad = gx
                else:
                    # DO NOT use +=, it is in-place operation(numpy) causing side effects.
                    x.grad = x.grad + gx

                # If the input has a creator, it means that the input is an output of another function.
                # So, we need to add the creator to the list of functions to get another gradient.
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for output in f.outputs:
                    # NOTE: output is a weakref.ref[Tensor]
                    output().grad = None  # type: ignore

    def cleargrad(self) -> None:
        self.grad = None


class Function:
    def __call__(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Do forward propagation.
        self.generation: the generation of the function.
            A function gets inputs and outputs.
            e.g., [input] -> [function] -> [output]

            The generation of the input will be the generation of the function.
            The generation of the output is the generation of the function plus 1.
            e.g., [input][gen=2] -> [function][gen=2] -> [output][gen=3]

            Generation is for determining the order of backward propagation.
            The backward propagation that has inputs with higher generation will be called first.
            If the generations of inputs are different, the higher generation will be the one of the function.

        Warning: Sometimes the output of the forward method is a scalar,
            when the input is a zero-dimensional array.

        Args:
            inputs: list[Tensor]

        Returns:
            Tensor | list[Tensor]
        """
        # forward propagation
        xs = tuple(input.data for input in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = tuple(Tensor(y) for y in ys)

        # set the creator of the output
        for output in outputs:
            output.creator = self

        # If the generations of inputs are different, the higher generation will be the one of the function.
        self.generation = max([input.generation for input in inputs])

        # set the inputs and outputs
        # NOTE: For memory optimization we wraps output tensors with weakref,
        #       by removing the circular reference between the function and the output tensor.
        self.inputs: tuple[Tensor, ...] = inputs
        self.outputs: tuple[weakref.ref[Tensor], ...] = tuple(
            weakref.ref(output) for output in outputs
        )

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
    x0 = Tensor(np.array(1.0))
    x1 = Tensor(np.array(2.0))
    t = add(x0, x1)
    y = square(t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)

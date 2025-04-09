from __future__ import annotations

import contextlib
import weakref
from typing import ContextManager, Generator, TypeAlias

import numpy as np

from torch.priority_queue import PriorityQueue

INPUT_TYPE: TypeAlias = (
    int
    | float
    | list[int | float]
    | np.int32
    | np.int64
    | np.float32
    | np.float64
    | np.ndarray
)
INPUT_TYPE_TUPLE = (
    int,
    float,
    list,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.ndarray,
)


int32: TypeAlias = np.int32
int64: TypeAlias = np.int64
float32: TypeAlias = np.float32
float64: TypeAlias = np.float64

TORCH_TYPE: TypeAlias = int32 | int64 | float32 | float64


class Config:
    """Config for toggling backpropagation.

    When we only inference, we don't need to backpropagate gradients.
    So, we can save memory by disabling backpropagation and removing the data used in backpropagation.
    """

    enable_backprop: bool = True


@contextlib.contextmanager
def using_config(name: str, value: bool) -> Generator[None, None, None]:
    """Context manager to set a config value temporarily.

    Args:
        name: The name of the config to set
        value: The value to set the config to
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad() -> ContextManager[None]:
    """Context manager to disable gradient calculation."""
    return using_config("enable_backprop", False)


class Size:
    def __init__(self, *args: int) -> None:
        self.args = args

    def __len__(self) -> int:
        return len(self.args)

    def __getitem__(self, index: int) -> int:
        return self.args[index]

    def __repr__(self) -> str:
        return f"torch.Size([{', '.join(str(arg) for arg in self.args)}])"


def tensor(data: INPUT_TYPE | None = None) -> Tensor:
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

    # NOTE: set the priority of tensor to be higher than numpy array.
    # When calling special methods(`__add__`, `__mul__`, etc.) with a numpy array,
    # the methods of the tensor will be called.
    __array_priority__ = 200

    def __init__(
        self,
        data: INPUT_TYPE,
        name: str | None = None,
        dtype: TORCH_TYPE | None = None,
    ) -> None:
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
            name: You can give a name to the tensor. This is useful for debugging.
            dtype: The dtype of the tensor. If None, the dtype will be the same as the type of the data.
        """
        if isinstance(data, np.ndarray):
            pass
        elif isinstance(data, INPUT_TYPE_TUPLE):
            data = np.array(data)
        else:
            raise ValueError(f"Data has an invalid type: {type(data)}")

        if dtype is not None:
            data = data.astype(dtype)

        self.data = data
        self.name = name
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None
        self.generation: int = 0

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        p = str(self.data).replace("\n", "\n" + " " * 7)
        return f"tensor({p}, dtype={self.dtype})"

    @property
    def creator(self) -> Function | None:
        return self._creator

    @creator.setter
    def creator(self, func: Function) -> None:
        self._creator = func

    @property
    def ndim(self) -> int:
        return int(self.data.ndim)

    @property
    def shape(self) -> Size:
        return self.size()

    @property
    def dtype(self) -> TORCH_TYPE:
        return self.data.dtype

    def size(self) -> Size:
        return Size(*self.data.shape)

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


def as_tensor(obj: INPUT_TYPE | Tensor) -> Tensor:
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


class Function:
    def __call__(self, *input_args: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
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
            input_args: list[Tensor] is expected, but also any numeric or array object can be accepted.

        Returns:
            Tensor | list[Tensor]
        """
        inputs: tuple[Tensor, ...] = tuple(as_tensor(input) for input in input_args)

        # forward propagation
        xs = tuple(input.data for input in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = tuple(Tensor(y) for y in ys)

        # When only inferencing, we can skip the logic below which is needed for backpropagation.
        if Config.enable_backprop:
            # If the generations of inputs are different, the higher generation will be the one of the function.
            self.generation = max([input.generation for input in inputs])

            # set the creator of the output
            for output in outputs:
                output.creator = self

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


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1

        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy * x1
        gx1 = gy * x0

        return gx0, gx1


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return gy, -gy


class Div(Function):
    """Divide two tensors element-wise.

    Actually, PyTorch differentiates the division operator in two ways:
    - torch.div()
        - If both inputs are integers, it performs floor division.(like '//' in Python 3)
        - If either input is a float, it performs true division.
    - torch.true_divide()
        - Always performs true division, regardless of input types.
        - The '/' operator between tensors is equivalent to torch.true_divide().

    For simplicity, we only implement true division with the '/' operator.
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = -gy * x0 / x1**2

        return gx0, gx1


class Pow(Function):
    def __init__(self, c: int | float) -> None:
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**self.c

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        gx = self.c * x ** (self.c - 1) * gy

        return gx


def square(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Square()(x)


def exp(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Exp()(x)


def add(
    x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor
) -> Tensor | tuple[Tensor, ...]:
    return Add()(x0, x1)


def mul(
    x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor
) -> Tensor | tuple[Tensor, ...]:
    return Mul()(x0, x1)


def neg(x: INPUT_TYPE | Tensor) -> Tensor | tuple[Tensor, ...]:
    return Neg()(x)


def sub(
    x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor
) -> Tensor | tuple[Tensor, ...]:
    return Sub()(x0, x1)


def rsub(
    x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor
) -> Tensor | tuple[Tensor, ...]:
    return Sub()(x1, x0)


def div(
    x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor
) -> Tensor | tuple[Tensor, ...]:
    return Div()(x0, x1)


def rdiv(
    x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor
) -> Tensor | tuple[Tensor, ...]:
    return Div()(x1, x0)


def pow(x: INPUT_TYPE | Tensor, c: int | float) -> Tensor | tuple[Tensor, ...]:
    return Pow(c)(x)


Tensor.__add__ = add  # type: ignore
Tensor.__radd__ = add  # type: ignore
Tensor.__mul__ = mul  # type: ignore
Tensor.__rmul__ = mul  # type: ignore
Tensor.__neg__ = neg  # type: ignore
Tensor.__sub__ = sub  # type: ignore
Tensor.__rsub__ = rsub  # type: ignore
Tensor.__truediv__ = div  # type: ignore
Tensor.__rtruediv__ = rdiv  # type: ignore
Tensor.__pow__ = pow  # type: ignore

if __name__ == "__main__":
    x = Tensor(np.array(2.0))
    y = pow(x, 3)
    print(y)

from __future__ import annotations

import contextlib
import copy
import weakref
from typing import (
    ContextManager,
    Generator,
    Iterator,
    Literal,
    overload,
)

import numpy as np
from loguru import logger

from .priority_queue import PriorityQueue
from .types import (
    INPUT_TYPE,
    INPUT_TYPE_TUPLE,
    NUMERIC_TYPE,
    NUMPY_DTYPE,
    TORCH_DTYPE,
    torch2np,
    type_np2torch,
    type_torch2np,
)


class Function:
    inplace: bool = False

    def __call__(
        self, *inputs: Tensor, inplace: bool = False
    ) -> tuple[Tensor, ...] | None:
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
            *inputs: input Tensors
            inplace: Whether to perform the in-place operation.
        Returns:
            tuple[Tensor, ...] for normal operations.
            None for in-place operations.
        """
        assert isinstance(inputs, tuple) and all(
            isinstance(input, Tensor) for input in inputs
        )

        if inplace and any(input.is_leaf and input.requires_grad for input in inputs):
            raise RuntimeError(
                "a leaf Variable that requires grad is being used in an in-place operation."
            )

        # forward propagation
        xs = tuple(input._data for input in inputs)

        # NOTE: temporary if-branch for only add inplace operation.
        # After all inplace operations are implemented, remove the if-branch.
        if inplace:
            ys = self.forward(*xs, inplace=inplace)
        else:
            ys = self.forward(*xs)

        if not isinstance(ys, tuple):
            ys = (ys,)

        if inplace and len(ys) > 1:
            raise ValueError(
                "In-place operation is not supported for multiple outputs."
            )

        # If any input requires grad, the output should require grad.
        should_outputs_require_grad: bool = any(input.requires_grad for input in inputs)
        outputs = tuple(
            Tensor(y, requires_grad=should_outputs_require_grad) for y in ys
        )

        # When backpropagation is enabled, we need to set the creator(a.k.a. grad_fn) of the output.
        if Config.enable_backprop and should_outputs_require_grad:
            # If the generations of inputs are different, the higher generation will be the one of the function.
            self.generation = max([input.generation for input in inputs])

            self.inputs: tuple[Tensor, ...] = inputs

            if inplace:
                # For in-place operations, set this function as the gradient function (grad_fn)
                # of the first input tensor, which is being modified in-place
                # The input tensor may have multiple creators, stored in order of generation
                inputs[0].grad_fn = self
                inputs[0]._data = ys[0]
            else:
                # Use weak references to output tensors to avoid circular references
                # This optimizes memory usage by allowing garbage collection
                # NOTE: We don't need to create self.outputs in in-place operation
                # because the output is the same as the first input.
                self.outputs: tuple[weakref.ref[Tensor], ...] = tuple(
                    weakref.ref(output) for output in outputs
                )

                # Set this function as the gradient function for all output tensors
                for output in outputs:
                    output.grad_fn = self
        # When only inferencing and inplace is True,
        elif inplace:
            inputs[0]._data = ys[0]
            return None

        return outputs

    @overload
    def forward(
        self, *xs: np.ndarray, inplace: Literal[False] = False
    ) -> np.ndarray | tuple[np.ndarray, ...]: ...

    @overload
    def forward(self, *xs: np.ndarray, inplace: Literal[True]) -> np.ndarray: ...

    def forward(
        self, *xs: np.ndarray, inplace: bool = False
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        raise NotImplementedError

    def backward(self, *gys: Tensor) -> Tensor | tuple[Tensor, ...]:
        raise NotImplementedError


class Add(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 2, "Add must take two arguments"

        self.inplace = inplace

        x0, x1 = xs
        y = x0 + x1

        return y

    def backward(self, *gys: Tensor) -> tuple[Tensor, Tensor]:
        assert len(gys) == 1, "Add must take one argument"

        if self.inplace:
            # Revert the in-place operation.
            x0, x1 = self.inputs
            x0 = x0 - x1

            self.inputs[0]._data = x0._data

        gy = gys[0]

        return gy, gy


class Mul(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 2, "Mul must take two arguments"

        self.inplace = inplace

        x0, x1 = xs
        y = x0 * x1

        return y

    def backward(self, *gys: Tensor) -> tuple[Tensor, ...]:
        assert len(gys) == 1, "Mul must take one argument"

        if self.inplace:
            x0, x1 = self.inputs

            try:
                x0 /= x1
            except ZeroDivisionError:
                logger.warning(
                    "Zero division occurred in Mul.backward. Adding eps to the divisor."
                )
                eps = 1e-7
                x0 /= x1 + eps

            self.inputs[0]._data = x0._data

        gy = gys[0]
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0

        return gx0, gx1


class Neg(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Neg must take one argument"

        self.inplace = inplace

        x = xs[0]

        return -x

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Neg must take one argument"

        if self.inplace:
            # TODO
            ...

        gy = gys[0]

        return -gy


class Sub(Function):
    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 2, "Sub must take two arguments"

        self.inplace = inplace

        x0, x1 = xs

        return x0 - x1

    def backward(self, *gys: Tensor) -> tuple[Tensor, ...]:
        assert len(gys) == 1, "Sub must take one argument"

        if self.inplace:
            # Revert the in-place operation.
            x0, x1 = self.inputs
            x0 = x0 + x1

            self.inputs[0]._data = x0._data

        gy = gys[0]
        gx0 = gy
        gx1 = -gy

        return gx0, gx1


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

    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 2, "Div must take two arguments"

        self.inplace = inplace

        x0, x1 = xs

        return x0 / x1

    def backward(self, *gys: Tensor) -> tuple[Tensor, Tensor]:
        assert len(gys) == 1, "Div must take one argument"

        if self.inplace:
            # TODO
            ...

        gy = gys[0]
        x0, x1 = self.inputs

        gx0 = gy / x1
        gx1 = -gy * x0 / x1**2

        return gx0, gx1


class Pow(Function):
    def __init__(self, c: int | float) -> None:
        self.c = c

    def forward(self, *xs: np.ndarray, inplace: bool = False) -> np.ndarray:
        assert len(xs) == 1, "Pow must take one argument"

        self.inplace = inplace

        x = xs[0]

        return x**self.c

    def backward(self, *gys: Tensor) -> Tensor:
        assert len(gys) == 1, "Pow must take one argument"

        if self.inplace:
            # TODO
            ...

        gy = gys[0]
        x = self.inputs[0]
        gx: Tensor = self.c * x ** (self.c - 1) * gy

        return gx


def add(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.add.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    result = Add()(x0, x1)

    assert isinstance(result, tuple), "Add must return a tuple of Tensors"
    assert len(result) == 1, "Add must return a single Tensor"

    return result[0]


def add_(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> None:
    """https://pytorch.org/docs/stable/generated/torch.Tensor.add_.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    Add()(x0, x1, inplace=True)


def mul(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.mul.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    result = Mul()(x0, x1)

    assert isinstance(result, tuple), "Mul must return a tuple of Tensors"
    assert len(result) == 1, "Mul must return a single Tensor"

    return result[0]


def mul_(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> None:
    """https://pytorch.org/docs/stable/generated/torch.Tensor.mul_.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    Mul()(x0, x1, inplace=True)


def neg(x: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.neg.html"""
    x = as_tensor(x)

    result = Neg()(x)

    assert isinstance(result, tuple), "Neg must return a tuple of Tensors"
    assert len(result) == 1, "Neg must return a single Tensor"

    return result[0]


def sub(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.sub.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    result = Sub()(x0, x1)

    assert isinstance(result, tuple), "Sub must return a tuple of Tensors"
    assert len(result) == 1, "Sub must return a single Tensor"

    return result[0]


def sub_(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> None:
    """https://pytorch.org/docs/stable/generated/torch.Tensor.sub_.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    Sub()(x0, x1, inplace=True)


def rsub(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> Tensor:
    x0, x1 = as_tensor(x0), as_tensor(x1)

    result = Sub()(x1, x0)

    assert isinstance(result, tuple), "rsub must return a tuple of Tensors"
    assert len(result) == 1, "rsub must return a single Tensor"

    return result[0]


def div(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.div.html"""
    x0, x1 = as_tensor(x0), as_tensor(x1)

    result = Div()(x0, x1)

    assert isinstance(result, tuple), "div must return a tuple of Tensors"
    assert len(result) == 1, "div must return a single Tensor"

    return result[0]


def rdiv(x0: INPUT_TYPE | Tensor, x1: INPUT_TYPE | Tensor) -> Tensor:
    x0, x1 = as_tensor(x0), as_tensor(x1)

    result = Div()(x1, x0)

    assert isinstance(result, tuple), "rdiv must return a tuple of Tensors"
    assert len(result) == 1, "rdiv must return a single Tensor"

    return result[0]


def pow(x: INPUT_TYPE | Tensor, c: int | float) -> Tensor:
    """https://pytorch.org/docs/stable/generated/torch.pow.html"""
    x = as_tensor(x)

    result = Pow(c)(x)

    assert isinstance(result, tuple), "pow must return a tuple of Tensors"
    assert len(result) == 1, "pow must return a single Tensor"

    return result[0]


class Tensor:
    """An imitation of torch.Tensor in PyTorch

    fyi. in PyTorch,
    Every `torch.Tensor` is a `Variable`(exactly same), a internal C++ class.
    Each `Variable` has one unique `AutogradMeta` struct,
    which stores autograd metadata fields including `grad_`, `grad_fn`, etc.
    See: https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/variable.h
    """

    # See: https://pytorch.org/docs/stable/tensors.html
    __add__ = add
    __radd__ = add
    __mul__ = mul
    __rmul__ = mul
    __neg__ = neg
    __sub__ = sub
    __rsub__ = rsub
    __truediv__ = div
    __rtruediv__ = rdiv
    __pow__ = pow
    add = add
    mul = mul
    neg = neg
    sub = sub
    div = div
    pow = pow
    add_ = add_
    sub_ = sub_
    mul_ = mul_

    # NOTE: set the priority of tensor to be higher than numpy array.
    # When calling special methods(`__add__`, `__mul__`, etc.) with a numpy array,
    # the methods of the tensor will be called.
    __array_priority__ = 200

    def __init__(
        self,
        data: INPUT_TYPE | int | float | None,
        dtype: TORCH_DTYPE | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        """
        self.grad: numpy.ndarray | None

        Args:
            data: numpy.ndarray | int | float | None
            dtype: The dtype of the tensor. If None, the dtype will be the same as the type of the data.
            requires_grad: Whether to require gradients for the tensor.
            name: You can give a name to the tensor. This is useful for debugging and visualization.
        """
        if isinstance(data, INPUT_TYPE_TUPLE):
            if isinstance(data, np.ndarray):
                pass
            else:
                data = np.array(data)
        else:
            raise ValueError(f"Data has an invalid type: {type(data)}")

        if dtype is not None:
            # Convert data type using astype.
            # copy=False ensures that we don't create a copy if the dtype
            # already matches, which is crucial for detach() to work correctly
            # (i.e., share the underlying numpy array).
            data_type: NUMPY_DTYPE = type_torch2np(dtype)
            data = data.astype(data_type, copy=False)

        if (
            requires_grad
            and isinstance(data, np.ndarray)
            and data.dtype
            not in (
                np.float32,
                np.float64,
            )
        ):
            raise RuntimeError(
                "Only Tensors of floating point and complex dtype can require gradients"
            )

        self._data: np.ndarray = data
        # We can have multiple creators when the tensor is the input and the output(in-place operation)
        self._creators: PriorityQueue[Function] = PriorityQueue()
        self.name = name
        self.grad: Tensor | None = None
        self.requires_grad = requires_grad

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError(f"The tensor is a scalar: {self}")
        else:
            return len(self._data)

    def __repr__(self) -> str:
        p = str(self._data).replace("\n", "\n" + " " * 7)

        if self.grad_fn:
            return f"tensor({p}, dtype={self.dtype}, grad_fn={self.grad_fn})"
        elif self.requires_grad:
            return f"tensor({p}, dtype={self.dtype}, requires_grad=True)"
        else:
            return f"tensor({p}, dtype={self.dtype})"

    @property
    def data(self) -> Tensor:
        """Detach the tensor from the computational graph.

        Often used when updating parameters after backpropagation.

        In PyTorch, this method is implemented by `Tensor.data`.
        But This attribute is kinda legacy, and it is not recommended to use it in modern PyTorch.
        We should implement this method though.
        Also See: https://stackoverflow.com/questions/51743214/is-data-still-useful-in-pytorch

        For simplicity, we just call `detach()` here.
        """
        return self.detach()

    @data.setter
    def data(self, input_tensor: Tensor) -> None:
        if not isinstance(input_tensor, Tensor):
            raise ValueError(
                f"input_tensor must be a Tensor. Got {type(input_tensor)}."
            )

        self._data = input_tensor._data

    @property
    def generation(self) -> int:
        """The generation of the tensor.

        A function gets inputs and outputs.
        e.g., [input] -> [function] -> [output]

        The generation of the input will be the generation of the function.
        The generation of the output is the generation of the function plus 1.
        e.g., [input][gen=2] -> [function][gen=2] -> [output][gen=3]

        Generation is for determining the order of backward propagation.
        The backward propagation that has inputs with higher generation will be called first.
        If the generations of inputs are different, the higher generation will be the one of the function.

        The generation of the tensor is the generation of the function that created the tensor most recently.
        The function should have the youngest(biggest) generation among self._creators.
        """
        if (youngest_creator := self._creators.peek()) is not None:
            return youngest_creator.generation + 1
        else:
            return 0

    @property
    def grad_fn(self) -> Function | None:
        """The function that created this tensor.

        Returns the youngest(biggest) function in self._creators.
        Creators:
            When forwarding, memorize the function that created this tensor for backward propagation.
            Similarly, in PyTorch `Variable`s have the notion of a `gradient_edge`, which is the edge
            in the autograd graph that connects the variable to a particular input of the gradient function
            (`grad_fn` or `grad_accumulator`).
            The tensor which does in-place operation has multiple creators.
        """
        return self._creators.peek()

    @grad_fn.setter
    def grad_fn(self, func: Function) -> None:
        self._creators.push(func, func.generation)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the tensor.

        Alias for `dim()`

        https://pytorch.org/docs/stable/generated/torch.Tensor.ndim.html
        """
        return self.dim()

    @property
    def shape(self) -> Size:
        """The shape of the tensor.

        Alias for `size()`

        https://pytorch.org/docs/stable/generated/torch.Tensor.shape.html
        """
        return self.size()

    @property
    def dtype(self) -> TORCH_DTYPE:
        item_: np.ndarray = self._data

        return type_np2torch(item_.dtype)

    @dtype.setter
    def dtype(self, input_dtype: TORCH_DTYPE) -> None:
        data_type: NUMPY_DTYPE = type_torch2np(input_dtype)

        self._data = self._data.astype(data_type, copy=False)

    @property
    def is_leaf(self) -> bool:
        """Check if the tensor is a leaf tensor.

        https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html
        """
        if self.requires_grad:
            if self.grad_fn:
                return False
            else:
                return True
        else:
            return True

    def size(self) -> Size:
        """The size of the tensor.

        https://pytorch.org/docs/stable/generated/torch.Tensor.size.html
        """
        item_: np.ndarray = self._data

        return Size(*item_.shape)

    def item(self) -> NUMERIC_TYPE:
        data_: np.ndarray = self._data

        if data_.size > 1:
            raise ValueError(
                f"can only convert an array of size {data_.size} to a Python scalar"
            )

        item_ = data_.item()

        if isinstance(item_, int) or isinstance(item_, float):
            return item_

        match type(item_):
            case np.int32:
                return int(item_)
            case np.int64:
                return int(item_)
            case np.float32:
                return float(item_)
            case np.float64:
                return float(item_)
            case _:
                raise ValueError(f"Unsupported dtype: {type(item_)}")

    def detach(self) -> Tensor:
        """Detach the tensor from the computational graph.

        Like PyTorch, this method returns a new tensor with the same underlying data but without the computational graph.

        Refer: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TensorShape.cpp#L4504
        """
        return Tensor(data=self._data, dtype=self.dtype, requires_grad=False, name=None)

    def numpy(self, force: bool = False) -> np.ndarray:
        """Convert the tensor to a numpy array.

        https://pytorch.org/docs/stable/generated/torch.Tensor.numpy.html

        Args:
            force: Whether to force the conversion.
                If True, The ndarray may be a copy of the tensor instead of always sharing memory.
                If False, The returned ndarray and the tensor will share their storage.
                Defaults to False.
        """

        if force:
            return copy.deepcopy(self._data)
        else:
            return self._data

    def dim(self) -> int:
        """The number of dimensions of the tensor.

        https://pytorch.org/docs/stable/generated/torch.Tensor.dim.html
        """
        item_: np.ndarray = self._data

        return int(item_.ndim)

    def type(self, dtype: TORCH_DTYPE | None = None) -> str | Tensor:
        """Returns the type if dtype is not provided, else casts this object to the specified type.

        Returned tensor shares the same underlying storage(memory) with this tensor.

        https://pytorch.org/docs/stable/generated/torch.Tensor.type.html
        """
        if dtype:
            if dtype not in torch2np:
                raise ValueError(
                    f"dtype must be a torch.TORCH_DTYPE. Got {type(dtype)}."
                )

            if self.dtype == dtype:
                return self

            data_type: NUMPY_DTYPE = type_torch2np(dtype)
            self._data = self._data.astype(data_type, copy=False)

            return Tensor(
                data=copy.copy(self._data),
                dtype=dtype,
                requires_grad=self.requires_grad,
            )
        else:
            return str(self.dtype)

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool | None = None,
        create_graph: bool = False,
    ) -> None:
        """Backward propagation.

        Traverses the computational graph backwards:
        - Starts with the function that created this tensor
        - Computes gradients by calling each function's backward method
        - Propagates gradients to input tensors
        - Continues recursively through the entire graph

        See: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch-tensor-backward

        Args:
            gradient: The gradient of the function being differentiated w.r.t. self. This argument can be omitted if self is a scalar.
            retain_graph: Whether to keep the computational graph of the output tensors
                In most cases, we don't need to keep the gradients in the middle of the backpropagation.
                We finally use only the gradient of the "first" input tensor.(dy/dx)
                Defaults to the value of `create_graph`.
            create_graph: Whether to create a new graph for the backward pass.
                If True, the backward pass will create a new computational graph for the gradient.
                This is useful for computing higher-order derivatives.
                Defaults to False.
                See: https://pytorch.org/docs/stable/autograd.html#default-gradient-layouts
        """
        if retain_graph is None:
            retain_graph = create_graph

        if self.grad is None:
            if self.requires_grad:
                # if gradient is provided, use it as the gradient of self
                if gradient is None:
                    # if self is a scalar, the gradient is 1.0(dy/dy = 1.0)
                    # else, raise an error
                    if self.ndim == 0:
                        self.grad = ones_like(self, dtype=self.dtype)
                    else:
                        raise RuntimeError(
                            "grad can be implicitly created only for scalar outputs"
                        )
                else:
                    self.grad = gradient
            elif self.grad_fn is None:
                # The final output tensor of the computational graph does not require grad.
                raise RuntimeError(
                    "element 0 of tensors does not require grad and does not have a grad_fn"
                )
            else:
                # This is an anomaly.
                raise RuntimeError(
                    "This is the case that is not considered yet. MUST fix it."
                )

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

        if self.grad_fn:
            for creator in self._creators.items():
                add_func(creator)
        else:
            raise ValueError("The final output should have a grad_fn")

        while funcs:
            f = funcs.pop()

            # get gradients of outputs
            # If the function is in-place, pass the first input's grad as the output grad.
            gys = []
            if f.inplace:
                if (first_input_grad := f.inputs[0].grad) is not None:
                    gys.append(first_input_grad)
                    f.inputs[0].grad = None
                else:
                    raise ValueError(
                        "The first input of the inplace function must have a grad."
                    )
            else:
                for output in f.outputs:
                    output_tensor = output()

                    assert output_tensor is not None, (
                        f"The outputs of the function {f} must not be None."
                    )

                    gy = output_tensor.grad

                    assert gy is not None, (
                        f"The gradient of the output {output_tensor} of the function {f} must not be None."
                    )

                    gys.append(gy)

            # NOTE: f.backward(*gys) calls Function.__call__() which refers to the value of `enable_backprop`.
            # That's why we need to set the value of `enable_backprop` here.
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                # set gradients of inputs
                for x, gx in zip(f.inputs, gxs):
                    # if the grad is set in the loop, accumulate it.
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # DO NOT use `+=` which is an in-place operation of numpy causing side effects.
                        x.grad = x.grad + gx

                    # If the input has a creator, it means that the input is an output of another function.
                    # So, we need to add the creator to the list of functions to get another gradient.
                    if x.grad_fn is not None:
                        for creator in x._creators.items():
                            add_func(creator)

            if not retain_graph and not f.inplace:
                for output in f.outputs:
                    # NOTE: output is a weakref.ref[Tensor]
                    output_tensor = output()

                    assert output_tensor is not None, (
                        f"One of the outputs of the function {f} is None."
                    )

                    output_tensor.grad = None


class Config:
    """Config for toggling backpropagation.

    When we only inference, we don't need to backpropagate gradients.
    So, we can save memory by disabling backpropagation and removing the data used in backpropagation.
    """

    enable_backprop: bool = True


class Size:
    def __init__(self, *args: int) -> None:
        self.args = args

    def __len__(self) -> int:
        return len(self.args)

    def __getitem__(self, index: int) -> int:
        return self.args[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Size):
            return False
        return self.args == other.args

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over the dimensions."""
        return iter(self.args)

    def __repr__(self) -> str:
        return f"torch.Size([{', '.join(str(arg) for arg in self.args)}])"


def as_tensor(obj: INPUT_TYPE | Tensor) -> Tensor:
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj)


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


def tensor(
    data: INPUT_TYPE,
    dtype: TORCH_DTYPE | None = None,
    requires_grad: bool = False,
    name: str | None = None,
) -> Tensor:
    """An imitation of torch.tensor in PyTorch

    Originally, torch.tensor is written in C++.
    Python binding code:
        https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_torch_functions_manual.cpp#L243-L267
    """
    return Tensor(data, dtype, requires_grad, name)


def ones(
    *size: int,
    dtype: TORCH_DTYPE | None = None,
    requires_grad: bool = False,
    name: str | None = None,
) -> Tensor:
    """Create a tensor with all elements set to 1.

    https://pytorch.org/docs/stable/generated/torch.ones.html

    Args:
        size: The size of the tensor.
        dtype: The dtype of the tensor.
        requires_grad: Whether to require gradients for the tensor.
        name: The name of the tensor.
    """
    return Tensor(np.ones(size), dtype, requires_grad, name)


def ones_like(
    input: Tensor,
    *,
    dtype: TORCH_DTYPE | None = None,
    requires_grad: bool = False,
    name: str | None = None,
) -> Tensor:
    """Create a tensor with all elements set to 1.

    https://pytorch.org/docs/stable/generated/torch.ones_like.html

    Args:
        input: The input tensor.
    """
    if not isinstance(input, Tensor):
        raise ValueError(f"Input must be a Tensor. Got {type(input)}.")

    return ones(
        *input.size(),
        dtype=dtype if dtype else input.dtype,
        requires_grad=requires_grad,
        name=name,
    )

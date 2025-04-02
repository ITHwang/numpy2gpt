from typing import Callable

import numpy as np
import pytest

from torch import core


def numerical_diff(
    f: Callable[[core.Tensor], core.Tensor], x: core.Tensor, eps: float = 1e-4
) -> float:
    """Calculate the numerical gradient of a function.

    The central difference is more accurate than the forward difference.
    See: http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture27.pdf

    Args:
        f: A function that takes a Tensor and returns a Tensor.
        x: The input Tensor.
        eps: The epsilon value for numerical differentiation.
    """
    x0 = core.tensor(x.data - eps)
    x1 = core.tensor(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return float((y1.data - y0.data) / (2 * eps))


@pytest.mark.parametrize(  # type: ignore
    "input_data",
    [
        5,
        3.14,
        [1, 2, 3],
        [1.1, 2.2, 3.3],
        np.int32(5),
        np.int64(5),
        np.float32(5.0),
        np.float64(5.0),
        np.array([1, 2, 3]),
        np.array([1.1, 2.2, 3.3]),
    ],
)
def test_tensor_init(input_data: core.T) -> None:
    t = core.tensor(input_data)
    assert isinstance(t.data, np.ndarray)


def test_tensor_init_with_none() -> None:
    t = core.tensor()
    assert t.data is None


def test_tensor_init_with_invalid_type() -> None:
    with pytest.raises(ValueError):
        t = core.tensor("invalid")


def test_square_backward() -> None:
    x = core.tensor(np.array(2.0))
    y = core.square(x)
    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(core.square, x)

    # Check if they're close
    assert np.allclose(analytical_grad, numerical_grad)


def test_exp_backward() -> None:
    x = core.tensor(np.array(1.0))
    y = core.exp(x)
    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(core.exp, x)

    # Check if they're close
    assert np.allclose(analytical_grad, numerical_grad)


def test_multi_branch_backward() -> None:
    """Test the backward propagation of a function with multiple branches.

    (x) -> [square] -> (a) -> [square] -> (b) -> [add] -> (y)
                        |                          ^
                        |                          |
                        ----> [square] -> (c) ------
    """
    x = core.tensor(np.array(1.0))
    a = core.square(x)
    b = core.square(a)
    c = core.square(a)
    y = core.add(b, c)
    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    grad_b, grad_c = np.array(1), np.array(1)

    grad_a = (
        numerical_diff(core.square, a) * grad_b
        + numerical_diff(core.square, a) * grad_c
    )

    grad_x = numerical_diff(core.square, core.tensor(grad_a))

    assert np.allclose(analytical_grad, grad_x)

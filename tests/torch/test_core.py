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
def test_tensor_init(input_data: core.INPUT_TYPE) -> None:
    """If the input is numeric or array, it is converted to a numpy array."""
    t = core.tensor(input_data)
    assert isinstance(t.data, np.ndarray)


def test_tensor_init_with_invalid_type() -> None:
    """If the input is not numeric or array, it raises an error."""
    with pytest.raises(ValueError):
        t = core.tensor("invalid")


def test_square_backward() -> None:
    """Test the backward propagation of the square function."""
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
    """Test the backward propagation of the exp function."""
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


@pytest.mark.parametrize(  # type: ignore
    "retain_grad",
    [True, False],
)
def test_retain_grad(retain_grad: bool) -> None:
    """Test the retain_grad argument of the backward method.

    If retain_grad is True, the gradient of the tensor is not None.
    If retain_grad is False, the gradient of the tensor is None.
    """
    x0 = core.tensor(np.array(1.0))
    x1 = core.tensor(np.array(2.0))
    t = core.add(x0, x1)
    y = core.square(t)
    y.backward(retain_grad=retain_grad)

    if retain_grad:
        assert t.grad is not None
        assert y.grad is not None
    else:
        assert t.grad is None
        assert y.grad is None


def test_using_config() -> None:
    """Test the using_config context manager.

    When using the using_config context manager,
    the value of the config is temporarily changed.
    """
    # Save original value
    original_value = core.Config.enable_backprop

    # Test changing the value
    with core.using_config("enable_backprop", False):
        assert core.Config.enable_backprop is False

    # Test that the value is restored
    assert core.Config.enable_backprop == original_value

    # Test nested context
    with core.using_config("enable_backprop", False):
        assert core.Config.enable_backprop is False
        with core.using_config("enable_backprop", True):
            assert core.Config.enable_backprop is True
        assert core.Config.enable_backprop is False

    # Ensure original value is restored
    assert core.Config.enable_backprop == original_value


def test_no_grad() -> None:
    """Test the no_grad context manager.

    When using the no_grad context manager,
    the value of the config is temporarily changed to False.
    """
    # Save original value
    original_value = core.Config.enable_backprop

    # Test that no_grad sets enable_backprop to False
    with core.no_grad():
        assert core.Config.enable_backprop is False

    # Test that the value is restored
    assert core.Config.enable_backprop == original_value


def test_forward_with_no_grad() -> None:
    """Test forward propagation when using no_grad.

    When using no_grad, the creator of the tensor is None.
    When using gradients enabled, the creator of the tensor is not None.
    """
    x = core.tensor(np.array(2.0))

    with core.no_grad():
        y = core.square(x)

    # When no_grad is used, the creator is None
    assert y.creator is None

    # Now compute with gradients enabled
    y = core.square(x)
    y.backward()

    # Gradient should be computed
    assert x.grad is not None
    assert y.creator is not None
    assert y.creator.inputs is not None
    assert y.creator.outputs is not None
    assert np.allclose(x.grad, 4.0)

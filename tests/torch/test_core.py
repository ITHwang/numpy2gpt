from functools import partial
from typing import Callable

import numpy as np
import pytest

import torch
from torch import Tensor

from .complex_funcs import goldstein, matyas, sphere


def numerical_diff(
    f: Callable[
        [torch.INPUT_TYPE | torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, ...],
    ],
    x: torch.Tensor,
    eps: float = 1e-4,
) -> float:
    """Calculate the numerical gradient of a function.

    The central difference is more accurate than the forward difference.
    See: http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture27.pdf

    Args:
        f: A function that takes a Tensor and returns a Tensor.
        x: The input Tensor.
        eps: The epsilon value for numerical differentiation.
    """
    x_data: np.ndarray = x._data
    x0 = torch.tensor(x_data - eps)
    x1 = torch.tensor(x_data + eps)
    y0 = f(x0)
    y1 = f(x1)

    assert isinstance(y0, torch.Tensor), (
        "The case that the output is a tuple of Tensors is not considered yet. "
        "If you met the case, please implement."
    )
    assert isinstance(y1, torch.Tensor), (
        "The case that the output is a tuple of Tensors is not considered yet. "
        "If you met the case, please implement."
    )

    numerator = y1._data - y0._data
    if isinstance(numerator, np.ndarray) and numerator.size == 1:
        numerator = float(numerator[0])
    denominator = 2 * eps

    return float(numerator / denominator)


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
def test_tensor_init(input_data: torch.INPUT_TYPE) -> None:
    """If the input is numeric or array, it is converted to a numpy array."""
    t = torch.tensor(input_data)
    assert isinstance(t._data, np.ndarray)


def test_tensor_init_with_invalid_type() -> None:
    """If the input is not numeric or array, it raises an error."""
    with pytest.raises(ValueError):
        t = torch.tensor("invalid")


def test_square_backward() -> None:
    """Test the backward propagation of the square function."""
    x = torch.tensor(np.array(2.0), requires_grad=True)
    y = torch.square(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(torch.square, x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_exp_backward() -> None:
    """Test the backward propagation of the exp function."""
    x = torch.tensor(np.array(1.0), requires_grad=True)
    y = torch.exp(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(torch.exp, x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_mul_backward() -> None:
    """Test the backward propagation of the mul function."""
    x0 = torch.tensor(np.array(2.0), requires_grad=True)
    x1 = torch.tensor(np.array(3.0), requires_grad=True)

    y = torch.mul(x0, x1)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_x0_grad = x0.grad
    analytical_x1_grad = x1.grad

    # Numerical gradient
    numerical_x0_grad = numerical_diff(partial(torch.mul, x1), x0)
    numerical_x1_grad = numerical_diff(partial(torch.mul, x0), x1)

    # Check if they're close
    assert analytical_x0_grad is not None
    assert analytical_x1_grad is not None
    assert numerical_x0_grad is not None
    assert numerical_x1_grad is not None
    assert np.allclose(analytical_x0_grad, numerical_x0_grad)
    assert np.allclose(analytical_x1_grad, numerical_x1_grad)


def test_neg_backward() -> None:
    """Test the backward propagation of the neg function."""
    x = torch.tensor(np.array(1.0), requires_grad=True)
    y = torch.neg(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(torch.neg, x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_sub_backward() -> None:
    """Test the backward propagation of the sub function."""
    x = torch.tensor(np.array(100.0), requires_grad=True)
    y = x - np.array(200.0)
    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(partial(torch.sub, x1=np.array(200.0)), x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_rsub_backward() -> None:
    """Test the backward propagation of the rsub function."""
    x = torch.tensor(np.array(100.0), requires_grad=True)
    y = np.array(200.0) - x
    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(partial(torch.rsub, x1=np.array(200.0)), x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_div_backward() -> None:
    """Test the backward propagation of the div function.(true division)"""
    x0 = torch.tensor(np.array(4.0), requires_grad=True)
    x1 = torch.tensor(np.array(2.0), requires_grad=True)
    y = torch.div(x0, x1)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x0.grad

    # Numerical gradient
    numerical_grad = numerical_diff(partial(torch.div, x1=x1), x0)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_rdiv_backward() -> None:
    """Test the backward propagation of the rdiv function.(true division)"""
    x = torch.tensor(np.array(2.0), requires_grad=True)
    y = torch.rdiv(x, 4)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(partial(torch.rdiv, x1=4), x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_multi_branch_backward() -> None:
    """Test the backward propagation of a function with multiple branches.

    (x) -> [square] -> (a) -> [square] -> (b) -> [add] -> (y)
                        |                          ^
                        |                          |
                        ----> [square] -> (c) ------
    """
    x = torch.tensor(np.array(1.0), requires_grad=True)

    a = torch.square(x)
    assert isinstance(a, Tensor)

    b = torch.square(a)
    assert isinstance(b, Tensor)

    c = torch.square(a)
    assert isinstance(c, Tensor)

    y = torch.add(b, c)
    assert isinstance(y, Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    grad_b, grad_c = np.array(1), np.array(1)

    assert isinstance(a, torch.Tensor)
    grad_a = (
        numerical_diff(torch.square, a) * grad_b
        + numerical_diff(torch.square, a) * grad_c
    )

    grad_x = numerical_diff(torch.square, torch.tensor(grad_a))

    assert analytical_grad is not None
    assert grad_x is not None
    assert np.allclose(analytical_grad, grad_x)


def test_pow_backward() -> None:
    """Test the backward propagation of the pow function."""
    x = torch.tensor(np.array(2.0), requires_grad=True)
    y = torch.pow(x, 3)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(partial(torch.pow, c=3), x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


@pytest.mark.parametrize(  # type: ignore
    "retain_grad",
    [True, False],
)
def test_retain_grad(retain_grad: bool) -> None:
    """Test the retain_grad argument of the backward method.

    If retain_grad is True, the gradient of the tensor is not None.
    If retain_grad is False, the gradient of the tensor is None.
    """
    x0 = torch.tensor(np.array(1.0), requires_grad=True)
    x1 = torch.tensor(np.array(2.0), requires_grad=True)

    t = torch.add(x0, x1)
    assert isinstance(t, Tensor)

    y = torch.square(t)
    assert isinstance(y, Tensor)

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
    original_value = torch.Config.enable_backprop

    # Test changing the value
    with torch.using_config("enable_backprop", False):
        assert torch.Config.enable_backprop is False

    # Test that the value is restored
    assert torch.Config.enable_backprop == original_value

    # Test nested context
    with torch.using_config("enable_backprop", False):
        assert torch.Config.enable_backprop is False
        with torch.using_config("enable_backprop", True):
            assert torch.Config.enable_backprop is True
        assert torch.Config.enable_backprop is False

    # Ensure original value is restored
    assert torch.Config.enable_backprop == original_value


def test_no_grad() -> None:
    """Test the no_grad context manager.

    When using the no_grad context manager,
    the value of the config is temporarily changed to False.
    """
    # Save original value
    original_value = torch.Config.enable_backprop

    # Test that no_grad sets enable_backprop to False
    with torch.no_grad():
        assert torch.Config.enable_backprop is False

    # Test that the value is restored
    assert torch.Config.enable_backprop == original_value


def test_forward_with_no_grad() -> None:
    """Test forward propagation when using no_grad.

    When using no_grad, the creator of the tensor is None.
    When using gradients enabled, the creator of the tensor is not None.
    """
    x = torch.tensor(np.array(2.0), requires_grad=True)

    with torch.no_grad():
        y = torch.square(x)

    assert isinstance(y, torch.Tensor)

    # When no_grad is used, the creator is None
    assert y.creator is None

    # Now compute with gradients enabled
    y = torch.square(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Gradient should be computed
    assert x.grad is not None
    assert y.creator is not None
    assert y.creator.inputs is not None
    assert y.creator.outputs is not None
    assert np.allclose(x.grad, 4.0)


def test_item_single_element() -> None:
    """Test item() method returns the proper Python scalar for single-element tensors."""
    # Test int32
    t_int32 = torch.tensor(np.array(5, dtype=np.int32))
    assert t_int32.item() == 5
    assert isinstance(t_int32.item(), int)

    # Test int64
    t_int64 = torch.tensor(np.array(5, dtype=np.int64))
    assert t_int64.item() == 5
    assert isinstance(t_int64.item(), int)

    # Test float32
    t_float32 = torch.tensor(np.array(3.14, dtype=np.float32))
    assert t_float32.item() == pytest.approx(3.14)
    assert isinstance(t_float32.item(), float)

    # Test float64
    t_float64 = torch.tensor(np.array(3.14, dtype=np.float64))
    assert t_float64.item() == pytest.approx(3.14)
    assert isinstance(t_float64.item(), float)


def test_item_multi_element_error() -> None:
    """Test item() method raises ValueError for multi-element tensors."""
    t = torch.tensor(np.array([1, 2, 3]))
    with pytest.raises(ValueError) as excinfo:
        t.item()
    assert "can only convert an array of size" in str(excinfo.value)

    t2d = torch.tensor(np.array([[1, 2], [3, 4]]))
    with pytest.raises(ValueError) as excinfo:
        t2d.item()
    assert "can only convert an array of size" in str(excinfo.value)


def test_dtype_property() -> None:
    """Test the dtype property returns the correct torch type for different numpy dtypes."""
    # Test int32
    t_int32 = torch.tensor(np.array(5, dtype=np.int32))
    assert t_int32.dtype == torch.int32

    # Test int64
    t_int64 = torch.tensor(np.array(5, dtype=np.int64))
    assert t_int64.dtype == torch.int64

    # Test float32
    t_float32 = torch.tensor(np.array(3.14, dtype=np.float32))
    assert t_float32.dtype == torch.float32

    # Test float64
    t_float64 = torch.tensor(np.array(3.14, dtype=np.float64))
    assert t_float64.dtype == torch.float64


def test_dtype_cast() -> None:
    """Test that tensors can be created with specific dtypes."""
    # Create tensor with default dtype
    t_default = torch.tensor(np.array([1, 2, 3]))

    # Create tensor with specified dtype
    t_float32 = torch.tensor(np.array([1, 2, 3]), dtype=torch.float32)
    assert t_float32.dtype == torch.float32

    # Create tensor with int32 dtype
    t_int32 = torch.tensor(np.array([1.5, 2.5, 3.5]), dtype=torch.int32)
    assert t_int32.dtype == torch.int32
    # Check that values were truncated during casting
    assert np.array_equal(t_int32._data, np.array([1, 2, 3], dtype=np.int32))


def test_dtype_unsupported() -> None:
    """Test that unsupported dtypes raise ValueError when accessing the dtype property."""
    # Mock a tensor with an unsupported dtype
    t = torch.tensor(np.array([1, 2, 3]))

    # Monkey patch the data attribute with an unsupported dtype
    # Using bool dtype which isn't in the supported types
    t._data = np.array([True, False, True], dtype=np.bool_)

    with pytest.raises(ValueError) as excinfo:
        dtype = t.dtype

    assert "Unsupported NumPy dtype" in str(excinfo.value)


def test_requires_grad_with_non_float_dtype() -> None:
    """Test that creating tensors with non-floating point dtypes and requires_grad=True raises an error."""
    # Test with int32
    with pytest.raises(RuntimeError) as excinfo:
        t = torch.tensor(np.array([1, 2, 3], dtype=np.int32), requires_grad=True)
    assert (
        "Only Tensors of floating point and complex dtype can require gradients"
        in str(excinfo.value)
    )

    # Test with int64
    with pytest.raises(RuntimeError) as excinfo:
        t = torch.tensor(np.array([1, 2, 3], dtype=np.int64), requires_grad=True)
    assert (
        "Only Tensors of floating point and complex dtype can require gradients"
        in str(excinfo.value)
    )

    # Verify that floating point types work with requires_grad=True
    t_float32 = torch.tensor(
        np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True
    )
    assert t_float32.requires_grad is True

    t_float64 = torch.tensor(
        np.array([1.0, 2.0, 3.0], dtype=np.float64), requires_grad=True
    )
    assert t_float64.requires_grad is True


def test_backward_without_requires_grad() -> None:
    """Test that calling backward on a tensor without requires_grad raises an error."""
    # Create a tensor without requires_grad
    x = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=False)

    # Try calling backward on it
    with pytest.raises(RuntimeError) as excinfo:
        x.backward()

    assert (
        "element 0 of tensors does not require grad and does not have a grad_fn"
        in str(excinfo.value)
    )

    # Create a tensor that is the result of an operation but doesn't require grad
    with torch.no_grad():
        a = torch.tensor(np.array([1.0], dtype=np.float32))
        b = torch.tensor(np.array([2.0], dtype=np.float32))
        c = a + b  # This tensor has a creator but doesn't require grad

    # This should work because c has a creator (Function) even though requires_grad=False
    # The error about "This is the case that is not considered yet" won't be raised
    # because we're not hitting that case - no_grad() disables the creation of grad_fn
    assert c._creator is None

    with pytest.raises(RuntimeError) as excinfo:
        c.backward()

    assert (
        "element 0 of tensors does not require grad and does not have a grad_fn"
        in str(excinfo.value)
    )


def test_sphere_function_backward() -> None:
    """Test gradients of the sphere function using numerical differentiation."""
    # Create input tensors
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)

    # Forward pass
    z = sphere(x, y)

    # Backward pass (analytical gradient)
    z.backward()

    # Get analytical gradients
    dx_analytical = x.grad
    dy_analytical = y.grad

    # Calculate numerical gradients
    dx_numerical = numerical_diff(lambda x: sphere(x, y), x)
    dy_numerical = numerical_diff(lambda y: sphere(x, y), y)

    # Check if they match
    assert dx_analytical is not None
    assert dy_analytical is not None
    assert dx_numerical is not None
    assert dy_numerical is not None
    assert np.allclose(dx_analytical, dx_numerical)
    assert np.allclose(dy_analytical, dy_numerical)


def test_goldstein_function_backward() -> None:
    """Test gradients of the Goldstein-Price function using numerical differentiation."""
    # Create input tensors
    x = torch.tensor([-0.5], requires_grad=True)
    y = torch.tensor([0.5], requires_grad=True)

    # Forward pass
    z = goldstein(x, y)

    # Backward pass (analytical gradient)
    z.backward()

    # Get analytical gradients
    dx_analytical = x.grad
    dy_analytical = y.grad

    # Calculate numerical gradients
    dx_numerical = numerical_diff(lambda x: goldstein(x, y), x)
    dy_numerical = numerical_diff(lambda y: goldstein(x, y), y)

    # Check if they match
    assert dx_analytical is not None
    assert dy_analytical is not None
    assert dx_numerical is not None
    assert dy_numerical is not None
    assert np.allclose(dx_analytical, dx_numerical, rtol=1e-3, atol=1e-3)
    assert np.allclose(dy_analytical, dy_numerical, rtol=1e-3, atol=1e-3)


def test_matyas_function_backward() -> None:
    """Test gradients of the Matyas function using numerical differentiation."""
    # Create input tensors
    x = torch.tensor([1.5], requires_grad=True)
    y = torch.tensor([2.0], requires_grad=True)

    # Forward pass
    z = matyas(x, y)

    # Backward pass (analytical gradient)
    z.backward()

    # Get analytical gradients
    dx_analytical = x.grad
    dy_analytical = y.grad

    # Calculate numerical gradients
    dx_numerical = numerical_diff(lambda x: matyas(x, y), x)
    dy_numerical = numerical_diff(lambda y: matyas(x, y), y)

    # Check if they match
    assert dx_analytical is not None
    assert dy_analytical is not None
    assert dx_numerical is not None
    assert dy_numerical is not None
    assert np.allclose(dx_analytical, dx_numerical)
    assert np.allclose(dy_analytical, dy_numerical)


def test_sin_backward() -> None:
    """Test the backward propagation of the sin function."""
    x = torch.tensor(np.array(1.0), requires_grad=True)
    y = torch.sin(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(torch.sin, x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_cos_backward() -> None:
    """Test the backward propagation of the cos function."""
    x = torch.tensor(np.array(1.0), requires_grad=True)
    y = torch.cos(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(torch.cos, x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad, numerical_grad)


def test_tensor_data_getter() -> None:
    """Test the `data` property getter returns a detached tensor."""
    x_data = np.array([1.0, 2.0], dtype=np.float32)
    x = torch.tensor(x_data, requires_grad=True)
    y = torch.square(x)  # Create some history

    assert isinstance(y, Tensor)

    data_tensor = y.data

    assert isinstance(data_tensor, Tensor)
    # The data property should return a detached tensor
    assert data_tensor.requires_grad is False
    assert data_tensor.creator is None
    assert data_tensor.grad_fn is None
    # Check if the underlying numpy data is the same
    assert np.array_equal(data_tensor._data, y._data)
    # Ensure the original tensor is unchanged
    assert y.requires_grad is True
    assert y.creator is not None


def test_tensor_detach() -> None:
    """Test the `detach` method returns a new tensor detached from the graph."""
    x_data = np.array([1.0, 2.0], dtype=np.float32)
    x = torch.tensor(x_data, requires_grad=True)
    y = torch.square(x)  # y requires grad and has a creator

    assert isinstance(y, Tensor)

    detached_y = y.detach()

    # Check detached tensor properties
    assert isinstance(detached_y, Tensor)
    assert detached_y.requires_grad is False
    assert detached_y.creator is None
    assert detached_y.grad_fn is None
    assert detached_y.dtype == y.dtype
    assert np.array_equal(detached_y._data, y._data)

    # Ensure the original tensor is unchanged
    assert y.requires_grad is True
    assert y.creator is not None

    # Check that the underlying numpy data is shared (modifying detached affects original)
    # Note: This depends on the Tensor constructor not forcing a copy.
    # If the constructor copies, this part of the test might fail.
    detached_y._data[0] = 99.0
    assert y._data[0] == 99.0


def test_tensor_detach_data_sharing() -> None:
    """Verify if detach shares the underlying numpy array."""
    original_data = np.array([1.0, 2.0])
    t = torch.tensor(original_data, requires_grad=True)
    detached_t = t.detach()

    # Check if they reference the same numpy array object
    assert detached_t._data is t._data

    # Modify the detached tensor's data
    detached_t._data[0] = 100.0

    # Check if the original tensor's data is also modified
    assert t._data[0] == 100.0

    # Modify the original tensor's data
    t._data[1] = 200.0

    # Check if the detached tensor's data is also modified
    assert detached_t._data[1] == 200.0


def test_ones() -> None:
    """Test the `ones` function for creating tensors."""
    # Test basic creation
    t1 = torch.ones(2, 3)
    assert isinstance(t1, Tensor)
    assert t1.shape == torch.Size(2, 3)
    # Default dtype might vary, but check it's a float type if no dtype specified
    assert t1.dtype in (torch.float32, torch.float64)
    assert np.array_equal(t1._data, np.ones((2, 3), dtype=t1._data.dtype))
    assert t1.requires_grad is False

    # Test with specified dtype (integer)
    t2 = torch.ones(4, dtype=torch.int32)
    assert isinstance(t2, Tensor)
    assert t2.shape == torch.Size(
        4,
    )
    assert t2.dtype == torch.int32
    assert np.array_equal(t2._data, np.ones(4, dtype=np.int32))
    assert t2.requires_grad is False

    # Test with specified dtype (float) and requires_grad
    t3 = torch.ones(1, 2, dtype=torch.float32, requires_grad=True)
    assert isinstance(t3, Tensor)
    assert t3.shape == torch.Size(1, 2)
    assert t3.dtype == torch.float32
    assert np.array_equal(t3._data, np.ones((1, 2), dtype=np.float32))
    assert t3.requires_grad is True

    # Test with name
    t4 = torch.ones(5, name="test_ones_tensor")
    assert isinstance(t4, Tensor)
    assert t4.shape == torch.Size(
        5,
    )
    assert np.array_equal(t4._data, np.ones(5, dtype=t4._data.dtype))
    assert t4.requires_grad is False
    assert t4.name == "test_ones_tensor"


def test_ones_like() -> None:
    """Test the `ones_like` function."""
    # Input tensor with float dtype
    input1 = torch.tensor(
        np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=torch.float32, requires_grad=True
    )

    # Basic test: inherit shape, dtype, requires_grad
    t1 = torch.ones_like(input1)
    assert isinstance(t1, Tensor)
    assert t1.shape == input1.shape
    assert t1.dtype == input1.dtype
    # ones_like should default requires_grad to False unless overridden
    assert t1.requires_grad is False
    assert np.array_equal(t1._data, np.ones_like(input1._data))

    # Test overriding dtype (to int)
    t2 = torch.ones_like(input1, dtype=torch.int64)
    assert isinstance(t2, Tensor)
    assert t2.shape == input1.shape
    assert t2.dtype == torch.int64
    assert t2.requires_grad is False
    assert np.array_equal(t2._data, np.ones_like(input1._data, dtype=np.int64))

    # Test overriding requires_grad
    t3 = torch.ones_like(input1, requires_grad=True)
    assert isinstance(t3, Tensor)
    assert t3.shape == input1.shape
    assert t3.dtype == input1.dtype
    assert t3.requires_grad is True
    assert np.array_equal(t3._data, np.ones_like(input1._data))

    # Test overriding dtype and requires_grad
    t4 = torch.ones_like(input1, dtype=torch.float64, requires_grad=True)
    assert isinstance(t4, Tensor)
    assert t4.shape == input1.shape
    assert t4.dtype == torch.float64
    assert t4.requires_grad is True
    assert np.array_equal(t4._data, np.ones_like(input1._data, dtype=np.float64))

    # Test with name
    t5 = torch.ones_like(input1, name="test_ones_like")
    assert isinstance(t5, Tensor)
    assert t5.shape == input1.shape
    assert t5.dtype == input1.dtype
    assert t5.requires_grad is False
    assert t5.name == "test_ones_like"
    assert np.array_equal(t5._data, np.ones_like(input1._data))

    # Test with non-tensor input -> raises ValueError
    with pytest.raises(ValueError):
        torch.ones_like(np.array([1, 2, 3]))

    # Test with integer input tensor
    input_int = torch.tensor([5, 6], dtype=torch.int32)
    t6 = torch.ones_like(input_int)
    assert isinstance(t6, Tensor)
    assert t6.shape == input_int.shape
    assert t6.dtype == input_int.dtype
    assert t6.requires_grad is False  # Cannot require grad for int
    assert np.array_equal(t6._data, np.ones_like(input_int._data))

    # Test overriding requires_grad on int input (should fail if dtype not float)
    with pytest.raises(RuntimeError):
        torch.ones_like(input_int, requires_grad=True)

    # Test overriding requires_grad on int input with float dtype (should work)
    t7 = torch.ones_like(input_int, dtype=torch.float32, requires_grad=True)
    assert isinstance(t7, Tensor)
    assert t7.shape == input_int.shape
    assert t7.dtype == torch.float32
    assert t7.requires_grad is True
    assert np.array_equal(t7._data, np.ones_like(input_int._data, dtype=np.float32))

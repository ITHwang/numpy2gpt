import numpy as np

import torch

from .utils import goldstein, matyas, numerical_diff, sphere


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
    assert np.allclose(analytical_grad._data, numerical_grad)


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
    assert np.allclose(analytical_grad._data, numerical_grad)


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
    assert np.allclose(dx_analytical._data, dx_numerical)
    assert np.allclose(dy_analytical._data, dy_numerical)


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
    assert np.allclose(dx_analytical._data, dx_numerical, rtol=1e-3, atol=1e-3)
    assert np.allclose(dy_analytical._data, dy_numerical, rtol=1e-3, atol=1e-3)


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
    assert np.allclose(dx_analytical._data, dx_numerical)
    assert np.allclose(dy_analytical._data, dy_numerical)


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
    assert np.allclose(analytical_grad._data, numerical_grad)


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
    assert np.allclose(analytical_grad._data, numerical_grad)


def test_tanh_backward() -> None:
    """Test the backward propagation of the tanh function."""
    x = torch.tensor(np.array(5.0), requires_grad=True)
    y = torch.tanh(x)
    assert isinstance(y, torch.Tensor)

    y.backward()

    # Analytical gradient
    analytical_grad = x.grad

    # Numerical gradient
    numerical_grad = numerical_diff(torch.tanh, x)

    # Check if they're close
    assert analytical_grad is not None
    assert numerical_grad is not None
    assert np.allclose(analytical_grad._data, numerical_grad)

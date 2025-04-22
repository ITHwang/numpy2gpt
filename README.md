<p align="center">
  <img src='./asset/img/numpy2gpt-logo.png' alt='logo' width="30%" height="30%" style="display: block; margin-left: auto; margin-right: auto;"/>
</p>

<p align="center">
  <em>This is an educational implementation of GPT from scratch using numpy.</em>
</p>

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img alt="PyPI - Supported Python versions" src="https://img.shields.io/badge/python-3.12-blue">
  <img alt="build" src="https://github.com/ITHwang/numpy2gpt/actions/workflows/main.yml/badge.svg">
</p>

# Overview
- An educational implementation of GPT from scratch using numpy, heavily inspired by [deep-learning-from-scratch-3](https://github.com/oreilly-japan/deep-learning-from-scratch-3), [nanoGPT](https://github.com/karpathy/nanoGPT), and [pytorch](https://github.com/pytorch/pytorch).

# Concepts

## 1. Auto Differentiation

1. [Auto Differentiation](https://openreview.net/pdf?id=BJJsrmfCZ) is a core algorithm for neural networks to compute backpropagation by following the chain rule.
2. To implement the chain rule, we need to satisfy the following requirements:
  - During forward propagation,
    1. A function gets input tensors and returns output tensors.
    2. The function should refer to the input tensors and output tensors.
    3. The output tensor should also refer to the function.
  - During backward propagation,
    1. The output tensor brings the function that generated the tensor.
    2. The function computes the gradient of the output tensor with respect to the input tensor.
    3. If the function has multiple input tensors, the function should accumulate the gradients of the input tensors.
3. To meet the requirements, we define the following concepts:
  - `Creator`
    - A function that generates a tensor.
    - When backpropagating, each output tensor refers to its `Creator` so that the creator can compute the gradient of the output tensor.

    ```mermaid
    graph LR
    Tensor1[Tensor] --> Function[Function]
    Function --> Tensor2[Tensor]

    Function -. "inputs" .-> Tensor1
    Function -. "outputs" .-> Tensor2
    Tensor2 -. "creator" .-> Function
    ```

  - `Generation`
    - Each node(tensor or function) has its generation that means the order of the node being created.
    - Being started from 0, the generation number is incremented when the node is created.
    - When backpropagating, the backward computations are performed in the reverse order of the generation number.
    - Why is it needed?
      - Gradients of each generation should be accumulated before being passed back to the function at younger generation.
      - E.g., in the following diagram, the `dy/da` is the sum of `dy/db * db/da` and `dy/dc * dc/da`.
      - But there's a risk that the node `a` passes them one by one, which makes backpropagation being performed twice.
      - To avoid this, we need to compute gradients of tensors at older generation first and accumulate the gradients of the tensors that were inputted to multiple functions.

    ```mermaid
    graph LR
      x((x @ 0)) --> Func1[Func1 @ 0]
      Func1 --> a((a @ 1))
      a --> Func2[Func2 @ 1]
      a --> Func3[Func3 @ 1]
      Func2 --> b((b @ 2))
      Func3 --> c((c @ 2))
      b --> Func4[Func4 @ 2]
      c --> Func4[Func4 @ 2]
      Func4 --> y((y @ 3))
    ```
4. Memory Optimization
  - As you can see the diagram at `Creator`, the function and the output refer to each other.(circular reference)
  - To avoid this, we use `weakref.ref` for the outputs of the function.
  - Using weakref.ref for output tensors eliminates circular references, reducing memory usage by 72% (from 136.4 MiB to 38.2 MiB) in our test([measure_memory.py](./torch/measure_memory.py)).

## 2. Define-by-Run vs. Define-and-Run

There are two fundamental paradigms for implementing automatic differentiation and building neural networks:

### 2.1. Define-and-Run

In the Define-and-Run paradigm, the entire computational graph is constructed before execution.

- The computational graph is fully defined, analyzed, and compiled before any data flows through it
- Once defined, the same graph processes all input data without changing its structure
- Examples: TensorFlow 1.x, Theano, Caffe

**Key characteristics:**
- Clear separation between graph definition and execution phases
- Graph optimization happens during compilation, improving performance
- Easier to deploy to production environments and non-Python platforms
- Well-suited for distributed computing across multiple devices
- More difficult to debug as errors often occur in the compiled graph
- Less flexible for dynamic computational patterns

### 2.2. Define-by-Run

In the Define-by-Run paradigm, the computational graph is constructed dynamically during execution.

- The computational graph is created on-the-fly as operations are performed
- Each forward pass can potentially create a different graph structure
- Examples: PyTorch, TensorFlow Eager mode, Chainer

**Key characteristics:**
- No separate compilation step; operations execute immediately
- More Pythonic approach, using native control flow statements
- Easier to debug as errors occur at the point of operation
- More intuitive development experience with standard Python tools
- Supports dynamic models where graph structure depends on input data
- May sacrifice some optimization opportunities available in static graphs

### 2.3. Summary

| Feature | Define-and-Run | Define-by-Run |
|---------|---------------|---------------|
| Graph Construction | Pre-compiled static graph | Dynamic graph built during execution |
| Programming Style | Domain-specific language | Native Python code |
| Debugging | Harder (debugging compiled graph) | Easier (standard Python debugging) |
| Performance | Potentially better (global optimization) | Potentially worse (less optimization) |
| Flexibility | Less flexible for dynamic patterns | More flexible for dynamic patterns |
| Deployment | Easier to deploy to production/devices | More dependent on Python runtime |
| Learning Curve | Steeper | More intuitive for Python developers |
| Use Cases | Production deployment, mobile/edge devices | Research, rapid prototyping |
| Examples | TensorFlow 1.x, Theano | PyTorch, TensorFlow Eager |

This implementation follows the Define-by-Run paradigm, similar to PyTorch.

# Features

## 1. Tensor Operations

This library implements a subset of PyTorch's tensor operations from scratch using NumPy as the underlying computation engine. All operations support automatic differentiation.

### 1.1. Supported Operations

- **Creation**: `torch.tensor()` - Create tensors from Python scalars, lists, or NumPy arrays
- **Arithmetic**: Addition, Subtraction, Multiplication, Division, Negation, Power
- **Mathematical**: `square()`, `exp()`, `cos()`, `sin()`, `tanh()`
- **In-place Operations**: TBD

### 1.2. Autograd

- Autograd is a core algorithm for neural networks to compute backpropagation by following the chain rule.

```python
# Example of autograd
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 1  # Forward pass
y.backward()  # Backward pass
print(x.grad)  # Access gradients
```

- Using that, we can implement gradient descent.

  - Code

    ```python
    # Source: https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/step28.py

    import torch
    from torch import Tensor

    def f(x: torch.Tensor) -> torch.Tensor:
        """f(x) = x^4 - 2x^2"""
        y = x**4 - 2 * x**2

        return y

      x = torch.tensor(2.0, requires_grad=True)
      lr = 0.01
      iters = 1000

      for i in range(iters):
          print(i, x)

          y = f(x)

          x.grad = None
          y.backward()

          with torch.no_grad():
              x -= lr * x.grad
    ```
  
  - Output
    ```text
    0 tensor(2.0, dtype=torch.float64, requires_grad=True)
    1 tensor(1.76, dtype=torch.float64, requires_grad=True)
    2 tensor(1.61232896, dtype=torch.float64, requires_grad=True)
    3 tensor(1.5091654023014192, dtype=torch.float64, requires_grad=True)
    4 tensor(1.4320422081467723, dtype=torch.float64, requires_grad=True)
    5 tensor(1.3718537670818505, dtype=torch.float64, requires_grad=True)
    ...
    386 tensor(1.0000000000000036, dtype=torch.float64, requires_grad=True)
    387 tensor(1.0000000000000033, dtype=torch.float64, requires_grad=True)
    388 tensor(1.000000000000003, dtype=torch.float64, requires_grad=True)
    389 tensor(1.0000000000000029, dtype=torch.float64, requires_grad=True)
    390 tensor(1.0000000000000027, dtype=torch.float64, requires_grad=True)
    ...
    ```



## 2. Neural Network Visualization

The library provides robust tools for visualizing computational graphs, which helps understand the flow of tensors through operations:

- Generate DOT representations of computational graphs
- Visualize graphs in multiple formats (PNG, SVG, PDF, etc.)
- Display tensor shapes, names, and data types
- Follow the computation chain backward from outputs to inputs

### 2.1. Requirements

- Graphviz must be installed on your system for generating image files
  - [Installation instructions](https://www.graphviz.org/download)
  - macOS: `brew install graphviz`
  - Ubuntu/Debian: `apt-get install graphviz`
  - Windows: Download installer from the Graphviz website

### 2.2. Basic Usage

- Code

  ```python
  # Source: https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/step26.py

  import torch
  from torch.utils import plot_dot_graph
  from tests.torch.complex_funcs import goldstein

  x = torch.tensor(1.0, requires_grad=True, name="x")
  y = torch.tensor(1.0, requires_grad=True, name="y")
  z: torch.Tensor = goldstein(x, y)
  z.backward()

  x.name = "x"
  y.name = "y"
  z.name = "z"
  plot_dot_graph(z, "goldstein.png", verbose=True)
  ```

- Output

  ![visualization_goldstein](./asset/img/visualization_goldstein.png)

### 2.3. Details

- The visualization can be exported in any format supported by Graphviz:

  ```python
  # DOT file (raw graph definition)
  plot_dot_graph(z, "graph.dot", verbose=True)

  # Image formats
  plot_dot_graph(z, "graph.png", verbose=True)  # PNG
  plot_dot_graph(z, "graph.svg", verbose=True)  # SVG
  plot_dot_graph(z, "graph.pdf", verbose=True)  # PDF
  ```

- The `verbose` parameter controls the amount of information displayed:

  ```python
  # Minimal visualization (just the graph structure)
  plot_dot_graph(z, "minimal_graph.png", verbose=False)

  # Detailed visualization (with tensor names, shapes, and dtypes)
  plot_dot_graph(z, "detailed_graph.png", verbose=True)
  ```

- Understanding the Graph
  - **Orange nodes**: Tensor objects (inputs and intermediate values)
  - **Green nodes**: Named tensor objects (when verbose=True and tensor has a name)
  - **Blue boxes**: Operations (functions that transform tensors)
  - **Arrows**: Data flow direction between operations and tensors

## 3. Higher-order Derivatives

Higher-order derivatives involve taking the derivative of a function multiple times. For instance, the second derivative is the derivative of the first derivative, the third derivative is the derivative of the second derivative, and so on. These are crucial in various mathematical and scientific applications, including optimization algorithms and analyzing function curvature.

### 3.1. Implementation

To support higher-order differentiation, backpropagations should build a computational graph as well as forward propagations.

1. When we do `Tensor.backward(create_graph=True)`, the operations performed during gradient computations extends the computational graph. For that, we toggle the `enable_backprop` to `create_graph` by using the `using_config` context manager.
2. Gradients, nodes in the graph, should also be tensors, which allows us to call `backward()` again on the resulting gradient tensor.

### 3.2. Example 1: higher-order derivatives of sin function

Here are computational graphs illustrating the forward and backward propagations of `sin(x)`.:

- When `create_graph` is set to `False`(default),

```mermaid
graph LR
  subgraph x
    x_data((data))
    x_grad((grad))
  end
  subgraph y
    y_data((data))
    y_grad((grad))
  end
  subgraph gy
    gy_data((data))
    gy_grad((grad))
  end
  subgraph gx
    gx_data((data))
    gx_grad((grad))
  end

  x --> sin[sin / forward] --> y
  x_grad -.-> gx
  y_grad -.-> gy
```

- When `create_graph` is set to `True`,

```mermaid
graph LR
  subgraph x
    x_data((data))
    x_grad((grad))
  end
  subgraph y
    y_data((data))
    y_grad((grad))
  end
  subgraph cosine_of_x
    cos_data((data))
    cos_grad((grad))
  end
  subgraph "gx"
    gx_data((data))
    gx_grad((grad))
  end
  subgraph "gy"
    gy_data((data))
    gy_grad((grad))
  end

  x --> sin[sin / forward] --> y
  x --> cos[cos / backward] --> cosine_of_x
  cosine_of_x --> mul[mul / backward]
  gy --> mul
  mul --> gx
  x_grad -.-> gx
  y_grad -.-> gy
```

In the graphs:
- The forward pass computes `y = sin(x)`.
- Both graphs computes the derivativve `gx`($ f'(x) = \cos(x) \cdot \frac{dy}{dy} $).
- The only difference is that the `create_graph=True` graph also constructs the backward computational graph (the `cos` and `mul` operations shown) that represents this derivative calculation.

Base on that, we can implement higher-order derivatives of sin function.

- Code

  ```python
  # Source: https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/step34.py

  import matplotlib.pyplot as plt
  import numpy as np

  import torch

  x = torch.tensor(np.linspace(-7, 7, 200), requires_grad=True)
  y = torch.sin(x)
  y.backward(create_graph=True)

  logs = [y._data]

  for i in range(3):
      logs.append(x.grad._data)

      gx = x.grad

      x.grad = None
      gx.backward(create_graph=True)

  labels = ["y=sin(x)", "y'", "y''", "y'''"]

  for i, v in enumerate(logs):
      plt.plot(x._data, logs[i], label=labels[i])

  plt.legend(loc="lower right")
  plt.savefig("sin_derivatives.png")
  ```

- Output

  ![sin_derivatives](./asset/img/sin_derivatives.png)

#### 3.3. Example 2: Newton's Method for Optimization

Newton's method is an iterative optimization algorithm that uses second-order derivative information to find the minimum of a function. The update rule is:

$$ x_{new} = x_{old} - \frac{f'(x_{old})}{f''(x_{old})} $$

This requires computing both the first derivative $ f'(x) $ and the second derivative $ f''(x) $. Our implementation of higher-order derivatives allows this directly:

- Code

  ```python
  # Source: https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/steps/step33.py

  import torch


  def f(x: torch.Tensor) -> torch.Tensor:
      """f(x) = x^4 - 2x^2"""
      y = x**4 - 2 * x**2

      return y


  x = torch.tensor(2.0, requires_grad=True)
  iters = 10

  for i in range(iters):
      print(i, x)

      # Compute first derivative: f'(x)
      y = f(x)
      x.grad = None 
      y.backward(create_graph=True) # Keep graph for second derivative

      # Compute second derivative: f''(x)
      gx = x.grad # gx = f'(x)
      x.grad = None # Clear previous grad
      gx.backward() # Compute gradient of gx w.r.t x
      gx2 = x.grad # gx2 = f''(x)

      # Update x using Newton's method
      with torch.no_grad():
          x -= gx / gx2
  ```

- Output

  ```
  0 tensor(2.0, dtype=torch.float64, requires_grad=True)
  1 tensor(1.4545454545454546, dtype=torch.float64, requires_grad=True)
  2 tensor(1.1510467893775467, dtype=torch.float64, requires_grad=True)
  3 tensor(1.0253259289766978, dtype=torch.float64, requires_grad=True)
  4 tensor(1.0009084519430513, dtype=torch.float64, requires_grad=True)
  5 tensor(1.0000012353089454, dtype=torch.float64, requires_grad=True)
  6 tensor(1.000000000002289, dtype=torch.float64, requires_grad=True)
  7 tensor(1.0, dtype=torch.float64, requires_grad=True)
  8 tensor(1.0, dtype=torch.float64, requires_grad=True)
  9 tensor(1.0, dtype=torch.float64, requires_grad=True)
  ```
- Pro: As you can see above, Newton's method can converge faster than gradient descent.
- Cons: Computing the full [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix)(for multivariate functions) can be computationally expensive. For an input vector $ \textbf{x} \in \mathbb{R}^n $, the Hessian matrix takes $ \mathcal{O}(n^2) $ space for storing all elements, and has the $ \mathcal{O}(n^3) $ time complexity for computing the inverse of the matrix.
- Alternative: To mitigate the computational cost, Several [Quasi-Newton methods](https://en.wikipedia.org/wiki/Quasi-Newton_method) have been introduced, using approximations of the derivatives of the functions in place of exact derivatives. **L-BFGS**, one of the Quasi-Newton methods, approximates the Hessian or its inverse using only first-order gradient information, offering a balance between the rapid convergence of Newton's method and the lower computational cost of first-order methods. PyTorch provides the implementation in [torch.optim.LBFGS](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html).

# References
- [PyTorch Pocket Reference](https://www.oreilly.com/library/view/pytorch-pocket-reference/9781492089995)
- [모두를 위한 컨벡스 최적화 - 18. Quasi-Newton Methods](https://convex-optimization-for-all.github.io/contents/chapter18)
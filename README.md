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
- An educational implementation of GPT from scratch using numpy, inspired by [deep-learning-from-scratch-3](https://github.com/oreilly-japan/deep-learning-from-scratch-3) and [nanoGPT](https://github.com/karpathy/nanoGPT).

# Core Concepts

## Auto Differentiation
1. Auto Differentiation is a core algorithm for neural networks to compute backpropagation by following the chain rule.
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

4. `Memory Optimization`: As you can see the diagram at `Creator`, the function and the output refer to each other.(circular reference)

# Core Features

## Tensor Operations
- See: [Tensor Playground](https://www.kaggle.com/code/reichenbch/tensor-playground)
- TBD

# References
- [pytorch](https://github.com/pytorch/pytorch)
- [PyTorch Pocket Reference](https://www.oreilly.com/library/view/pytorch-pocket-reference/9781492089995)
- Paszke, Adam, et al. "Automatic differentiation ni pytorch." (2017). [link](https://openreview.net/pdf?id=BJJsrmfCZ)

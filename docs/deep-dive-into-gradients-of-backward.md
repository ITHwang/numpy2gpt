# Deep dive into gradients of backward

## TL;DR

- PyTorch's `backward()` uses **reverse-mode automatic differentiation (AD)**, which is highly efficient for neural network training where you typically have many input parameters and a single scalar loss output.
- Reverse-mode AD calculates gradients using the **Vector-Jacobian Product (VJP)**. For a scalar output (like loss), it computes all gradients w.r.t. inputs in one backward pass.
- **Forward-mode AD** uses the **Jacobian-Vector Product (JVP)** and is more efficient when the number of outputs is much larger than the number of inputs.
- The preference for reverse-mode explains why `backward()` works implicitly on scalar tensors or requires an explicit `gradient` argument for non-scalar tensors to maintain efficiency.

## Understanding `torch.Tensor.backward()`

- The `backward()` method is a key part of [PyTorch's automatic differentiation engine](https://pytorch.org/blog/overview-of-pytorch-autograd-engine), `torch.autograd`, which is essential for training neural networks.
- This method initiates the gradient computation process when called on a tensor, provided certain conditions are met (we'll explore the reasons for these conditions later):
    1. The tensor must have `requires_grad=True`.
    2. If called with arguments, it needs both the input tensor and a gradient tensor of the same shape.
    3. Alternatively, if the tensor is a scalar, it can be called without arguments.
- The method then traverses the computational graph backward, applying the chain rule of calculus:

$$
\frac{\partial y}{\partial x}=\Bigl(\frac{\partial y}{\partial b}\frac{\partial b}{\partial a}\Bigr)\frac{\partial a}{\partial x}\tag{1}
$$

- But **why not apply the chain rule in the forward direction?** Mathematically, both approaches seem equivalent:

$$
\frac{\partial y}{\partial x}=\frac{\partial y}{\partial b}\Bigl(\frac{\partial b}{\partial a}\frac{\partial a}{\partial x}\Bigr)\tag{2}
$$

## Forward Mode and Reverse Mode of Automatic Differentiation

- Automatic differentiation has two primary modes: forward mode (corresponding to equation (2)) and reverse mode (corresponding to equation (1)).
- Forward mode calculates gradients in the same order as the forward pass through the computation graph, while reverse mode works backward from the output.
- As noted earlier, for scalar variables, both modes yield the same result with similar efficiency.
- However, when dealing with vector inputs and outputs, specifically $ x\in\mathbb{R}^n $ and $ y\in\mathbb{R}^m $, the choice between modes significantly impacts efficiency.
- PyTorch primarily uses reverse mode, stating it's more efficient for typical neural network training scenarios. [link](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/#what-is-autograd)
- **So, why is reverse mode generally preferred?**

## Vector-Jacobian Product(VJP) and Jacobian-Vector Product(JVP)
- The concepts of [Vector-Jacobian Product (VJP)](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#vector-jacobian-products-vjps-aka-reverse-mode-autodiff) and [Jacobian-Vector Product (JVP)](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#jacobian-vector-products-jvps-aka-forward-mode-autodiff) help explain the efficiency difference.
- Given a function $ f: \mathbb{R}^n \to \mathbb{R}^m $ and an input vector $ \mathbf{x}\in\mathbb{R}^n $, the Jacobian matrix $ J $ of $ f(x) $ has dimensions:

$$
J\in\mathbb{R}^{m\times n}
$$

### JVP and Forward Mode AD

- The JVP is calculated by multiplying the Jacobian $ J $ by a vector $ v\in\mathbb{R}^n $. Often, $v$ is chosen as a standard basis vector (like a one-hot vector) to extract a single column of the Jacobian:

$$
J\,v=\begin{bmatrix}\dfrac{\partial y_{1}}{\partial x_{1}} & \cdots & \dfrac{\partial y_{1}}{\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial y_{m}}{\partial x_{1}} & \cdots & \dfrac{\partial y_{m}}{\partial x_{n}}
\end{bmatrix}
\begin{bmatrix}
1 \\[6pt]
0 \\[3pt]
\vdots \\[3pt]
0
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\partial y_{1}}{\partial x_{1}} \\[3pt]
\vdots \\[3pt]
\dfrac{\partial y_{m}}{\partial x_{1}}
\end{bmatrix}
$$

- Computing a JVP effectively yields one column of the Jacobian matrix.
- To obtain the full Jacobian, forward mode AD needs to compute $ n $ JVPs, resulting in a time complexity proportional to $ O(n) $.
- Therefore, forward mode is more efficient when the input dimension $ n $ is much smaller than the output dimension $ m $ ($ n \ll m $).

### VJP and Reverse Mode AD

- The VJP is calculated by multiplying a vector $ v\in\mathbb{R}^m $ (often a standard basis vector) by the Jacobian $ J $:

$$
v^T\,J =
\begin{bmatrix}
1 & 0 & \cdots & 0
\end{bmatrix}
\begin{bmatrix}
\dfrac{\partial y_{1}}{\partial x_{1}} & \cdots & \dfrac{\partial y_{1}}{\partial x_{n}} \\[6pt]
\vdots & \ddots & \vdots \\[3pt]
\dfrac{\partial y_{m}}{\partial x_{1}} & \cdots & \dfrac{\partial y_{m}}{\partial x_{n}}
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\partial y_{1}}{\partial x_{1}} \\[3pt]
\vdots \\[3pt]
\dfrac{\partial y_{1}}{\partial x_{n}}
\end{bmatrix}
$$

- Computing a VJP yields one row of the Jacobian matrix.
- To obtain the full Jacobian, reverse mode AD needs to compute $ m $ VJPs, resulting in a time complexity proportional to $ O(m) $.
- Therefore, reverse mode is more efficient when the output dimension $ m $ is much smaller than the input dimension $ n $ ($ n \gg m $).

### Why is the reverse mode more efficient?

- In machine learning, the condition $ n \gg m $ is very common. Inputs often have high dimensions (e.g., pixels in an image, words in text), while the final output used for training is typically a single scalar loss value ($ m=1 $).
- When $ m=1 $, the Jacobian becomes a row vector ($ J\in\mathbb{R}^{1\times n} $), and reverse mode AD requires only a single VJP computation to get all gradients with respect to the inputs. Its complexity effectively becomes $ O(1) $ in terms of passes (though the computation cost per pass depends on the network size).
- This significant efficiency advantage is why PyTorch defaults to reverse mode AD for `backward()`, although it also provides support for forward mode AD. [link](https://pytorch.org/docs/stable/autograd.html#forward-mode-automatic-differentiation)
- [This article](https://leimao.github.io/blog/PyTorch-Automatic-Differentiation) provides a helpful performance comparison between forward and reverse mode AD in PyTorch.

## The parameter requirements of `backward()`

- Returning to the requirements for calling `backward()`, we can now see the rationale.
- If `backward()` is called on a tensor without a `gradient` argument, PyTorch checks if the tensor is a scalar.
- If it is scalar ($m=1$), PyTorch implicitly uses a gradient of `1`, aligning perfectly with the efficiency benefits of reverse mode AD for scalar outputs.
- If the tensor is non-scalar and no gradient is provided, PyTorch raises `RuntimeError: grad can be implicitly created only for scalar outputs`. This error occurs because applying reverse mode naively to a non-scalar output (where $m > 1$) would require multiple VJPs, which is less efficient than a single pass, especially if $m$ is large. The API requires explicit guidance (the `gradient` argument, often representing $v^T$) in such cases.
- Consequently, for efficient gradient computation using reverse mode, it's best practice to either call `backward()` on a scalar tensor (like the final loss) or provide an explicit `gradient` argument (often a tensor of ones matching the output tensor's shape if you need gradients w.r.t. an intermediate multi-output tensor).

## Conclusion

- In summary, PyTorch's `backward()` function leverages the power of reverse-mode automatic differentiation, which is highly optimized for the common machine learning task of minimizing a scalar loss function derived from high-dimensional inputs.
- The efficiency stems from the use of Vector-Jacobian Products (VJPs), allowing the computation of all necessary gradients with respect to the input parameters in a single backward pass when the output is scalar.
- This contrasts with forward-mode AD (using Jacobian-Vector Products - JVPs), which is more efficient when the input dimension is smaller than the output dimension.
- Understanding the interplay between VJPs, JVPs, and the dimensionality of inputs and outputs clarifies why `backward()` has specific requirements for scalar outputs or explicit gradient arguments, ensuring computational efficiency during the training process.

## References

- [What do I do with these equations to create a Jacobian matrix?](https://math.stackexchange.com/questions/951917/what-do-i-do-with-these-equations-to-create-a-jacobian-matrix)
- [Reverse mode differentiation vs. forward mode differentiation - where are the benefits?](https://math.stackexchange.com/questions/2195377/reverse-mode-differentiation-vs-forward-mode-differentiation-where-are-the-be)
- [자코비안(Jacobian) 행렬의 기하학적 의미](https://angeloyeo.github.io/2020/07/24/Jacobian.html)
- [Nick McGreivy - A Tutorial on Automatic Differentiation for Scientific
Design: Practical, Elegant, and Powerful](https://nickmcgreivy.scholar.princeton.edu/sites/g/files/toruqf5041/files/ast558_seminar_tutorial_on_automatic_differentiation-2.pdf)
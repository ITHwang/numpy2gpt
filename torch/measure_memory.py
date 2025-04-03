import numpy as np
from memory_profiler import profile

from torch.core import square, tensor


@profile  # type: ignore
def main() -> None:
    """Measure memory usage of tensor operations

    - In neural networks, the high memory usage often becomes a bottleneck.
    - To optimize the memory usage, we removed the circular reference between the tensor and the function.
    - And we tested how much memory is saved by this change.
        - Before: 136.4 MiB
        - After: 38.2 MiB
    - For more details, refer to `concepts` in README.md
    """

    for _ in range(10000):
        x = tensor(np.random.randn(10000))
        y = square(square(square(x)))
        y.backward()


if __name__ == "__main__":
    main()

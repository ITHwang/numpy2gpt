from torch.core_simple import (
    Config,
    Function,
    Tensor,
    add,
    as_tensor,
    div,
    exp,  # to be moved to functions.py
    mul,
    neg,
    no_grad,
    pow,
    rdiv,
    rsub,
    setup_tensor,
    square,  # to be moved to functions.py
    sub,
    tensor,
    using_config,
)

# from torch.core import (
#     Config,
#     Function,
#     Tensor,
#     add,
#     as_tensor,
#     div,
#     exp,  # to be moved to functions.py
#     mul,
#     neg,
#     no_grad,
#     pow,
#     rdiv,
#     rsub,
#     setup_tensor,
#     square,  # to be moved to functions.py
#     sub,
#     tensor,
#     using_config,
# )
from torch.types import (
    ARRAY_TYPE,
    INPUT_TYPE,
    INPUT_TYPE_TUPLE,
    NUMERIC_TYPE,
    TORCH_TYPE,
    float32,
    float64,
    int32,
    int64,
)

setup_tensor()

__all__ = [
    "Config",
    "Function",
    "Tensor",
    "add",
    "as_tensor",
    "div",
    "exp",
    "mul",
    "neg",
    "no_grad",
    "pow",
    "rdiv",
    "rsub",
    "setup_tensor",
    "square",
    "sub",
    "tensor",
    "using_config",
    "ARRAY_TYPE",
    "INPUT_TYPE",
    "INPUT_TYPE_TUPLE",
    "NUMERIC_TYPE",
    "TORCH_TYPE",
    "float32",
    "float64",
    "int32",
    "int64",
]

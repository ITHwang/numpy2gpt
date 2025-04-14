import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger

from torch import Function, Tensor

logger.remove()
logger.add(sys.stdout, level="INFO")


def _dot_var(v: Tensor, verbose: bool = False) -> str:
    """Get a dot graph of a variable.

    Args:
        v: The variable to get the dot graph of.
        verbose: Whether to print verbose information of the tensor.

    Returns:
        A dot graph of the variable.
    """
    label = ""

    if verbose:
        if v.name:
            label += f"{v.name}\n"
        label += f"{v.shape}\n{v.dtype}"

    if verbose and v.name:
        color = "lightgreen"
    else:
        color = "orange"

    return f'{id(v)} [label="{label}", color={color}, style=filled]\n'


def _dot_func(f: Function) -> str:
    """Get a dot graph of a function.

    Args:
        f: The function to get the dot graph of.

    Returns:
        A dot graph of the function.
    """

    txt = f'{id(f)} [label="{f.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    for x in f.inputs:
        txt += f"{id(x)} -> {id(f)}\n"

    for y in f.outputs:
        # NOTE: y is a weakref.ref[Tensor]
        txt += f"{id(f)} -> {id(y())}\n"

    return txt


def get_dot_graph(output: Tensor, verbose: bool = False) -> str:
    """Get a dot graph of the computation graph.

    The way to build a dot graph is very similar to backward method of `Tensor`,
        except that we don't need to calculate gradients of inputs.
    Following the path of the chain rule, we can trace back the computation graph from the final output.

    Args:
        output: The output tensor of the computation graph.
        verbose: Whether to print verbose information of tensors.

    Returns:
        A dot graph of the computation graph.
    """
    txt = ""
    funcs: list[Function] = []
    seen_set: set[Function] = set()

    def add_func(f: Function) -> None:
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    if output.creator is None:
        raise ValueError("output tensor has no creator")

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func: Function = funcs.pop()
        txt += _dot_func(func)

        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return f"digraph g {{\n{txt}}}"


def plot_dot_graph(
    output: Tensor, file_path: Path | str, verbose: bool = False
) -> None:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    dot_graph = get_dot_graph(output, verbose)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)

    file_extension = file_path.suffix[1:]

    if file_extension == "dot":
        with open(file_path, "w") as f:
            f.write(dot_graph)

        logger.info(f"Dot graph saved to {file_path} successfully.")
    else:
        # Temporarily use a temp dir to save the dot file
        with tempfile.TemporaryDirectory() as temp_dir:
            dot_file_path = Path(temp_dir) / "graph.dot"
            with open(dot_file_path, "w") as f:
                f.write(dot_graph)

            cmd = f"dot {dot_file_path} -T {file_extension} -o {file_path}"
            subprocess.run(cmd, shell=True)

        logger.info(f"Dot graph saved to {file_path} successfully.")

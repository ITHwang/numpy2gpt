import os
import tempfile
from pathlib import Path

import torch
from torch.utils import get_dot_graph, plot_dot_graph

from .utils import goldstein


def test_get_dot_graph() -> None:
    # Setup test tensors
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)
    z: torch.Tensor = goldstein(x, y)
    z.backward()

    # Set names for visualization
    x.name = "x"
    y.name = "y"
    z.name = "z"

    # Get dot graph
    dot_graph = get_dot_graph(z, verbose=True)

    # Basic assertions
    assert "digraph g {" in dot_graph
    assert "z" in dot_graph
    assert "x" in dot_graph
    assert "y" in dot_graph

    # Check for tensor shapes and data types
    assert str(z.shape) in dot_graph
    assert str(z.dtype) in dot_graph


def test_plot_dot_graph() -> None:
    # Setup test tensors
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)
    z: torch.Tensor = goldstein(x, y)
    z.backward()

    # Set names for visualization
    x.name = "x"
    y.name = "y"
    z.name = "z"

    # Create temporary directory and file paths for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test dot file output
        dot_file_path = Path(temp_dir) / "test_graph.dot"
        plot_dot_graph(z, dot_file_path, verbose=True)

        assert dot_file_path.exists()

        # Verify content of dot file
        with open(dot_file_path, "r") as f:
            dot_content = f.read()
            assert "digraph g {" in dot_content
            assert "z" in dot_content

        # Test png file output
        png_file_path = Path(temp_dir) / "test_graph.png"
        plot_dot_graph(z, png_file_path, verbose=True)

        assert png_file_path.exists()
        assert os.path.getsize(png_file_path) > 0  # Ensure file has content

        # Test with file path as string instead of Path
        svg_file_path = str(Path(temp_dir) / "test_graph.svg")
        plot_dot_graph(z, svg_file_path, verbose=True)

        assert Path(svg_file_path).exists()
        assert os.path.getsize(svg_file_path) > 0  # Ensure file has content

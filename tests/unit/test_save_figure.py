import pytest
import matplotlib.pyplot as plt
import os
from wine_quality_tools import save_figure

def test_save_figure_directory_creation(tmp_path):
    # Simple case: Check if os.makedirs actually works
    fig, _ = plt.subplots()
    # tmp_path is a pytest tool to keep your real project clean
    nested_path = tmp_path / "new_folder" / "deep_folder" / "test.png"
    
    save_figure(fig, str(nested_path))
    
    assert os.path.exists(str(nested_path))
    assert os.path.exists(os.path.dirname(str(nested_path)))

def test_save_figure_standard(tmp_path):
    # Simple case 2: Save to an existing directory
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    output_path = tmp_path / "simple_plot.png"
    save_figure(fig, str(output_path))
    assert output_path.exists()

def test_save_figure_different_extension(tmp_path):
    # Edge case: Different extension (JPG)
    fig, _ = plt.subplots()
    output_path = tmp_path / "test.jpg"
    save_figure(fig, str(output_path))
    assert output_path.exists()

def test_save_figure_none_input():
    # Error case: Passing None instead of a Figure
    with pytest.raises(ValueError, match="Figure object is None"):
        save_figure(None, "test.png")

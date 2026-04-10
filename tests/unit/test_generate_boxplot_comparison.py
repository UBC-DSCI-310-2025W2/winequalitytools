import pytest
import pandas as pd
import matplotlib.pyplot as plt
from wine_quality_tools import generate_boxplot_comparison

@pytest.fixture
def sample_wine_data():
    return pd.DataFrame({'alcohol': [10, 11], 'sugar': [1, 2], 'quality': [5, 6]})

def test_generate_boxplot_subplots(sample_wine_data):
    # Coverage: Ensure it creates exactly 2 subplots for 2 y_cols
    y_vars = ['alcohol', 'sugar']
    fig = generate_boxplot_comparison(sample_wine_data, 'quality', y_vars)
    assert len(fig.axes) == 2 
    plt.close(fig)

def test_generate_boxplot_title_formatting(sample_wine_data):
    # Simple Case: Check that underscores are replaced by spaces in subplot titles
    fig = generate_boxplot_comparison(sample_wine_data, 'quality', ['alcohol'])
    ax_title = fig.axes[0].get_title()
    assert "_" not in ax_title
    assert "Alcohol" in ax_title
    plt.close(fig)

def test_generate_boxplot_different_x(sample_wine_data):
    # Simple Case: Use a categorical string as x
    sample_wine_data['type'] = ['red', 'white']
    
    fig = generate_boxplot_comparison(sample_wine_data, 'type', ['alcohol'])
    assert fig.axes[0].get_xlabel() == 'type'
    plt.close(fig)

def test_generate_boxplot_many_y_cols(sample_wine_data):
    # Edge Case: Multiple subplots
    df = pd.DataFrame({
        'x': [1, 2], 'y1': [1, 2], 'y2': [3, 4], 'y3': [5, 6], 'y4': [7, 8]
    })
    fig = generate_boxplot_comparison(df, 'x', ['y1', 'y2', 'y3', 'y4'])
    assert len(fig.axes) == 4
    plt.close(fig)

def test_generate_boxplot_missing_column(sample_wine_data):
    # Error Case: Key missing from DF
    with pytest.raises(KeyError, match="not found in DataFrame"):
        generate_boxplot_comparison(sample_wine_data, 'quality', ['non_existent'])

import pytest
import pandas as pd
import matplotlib.pyplot as plt
from wine_quality_tools import generate_correlation_heatmap

@pytest.fixture
def mixed_data():
    return pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [3, 2, 1],
        'text': ['a', 'b', 'c'] # Should be ignored
    })



def test_generate_heatmap_ignores_text(mixed_data):
    fig = generate_correlation_heatmap(mixed_data)
    ax = fig.gca()
    # Simple Case: Check that labels only include the numeric columns
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert 'num1' in labels
    assert 'text' not in labels
    plt.close(fig)

def test_generate_heatmap_values(mixed_data):
    # Simple Case: num1 vs num2 is perfectly negatively correlated (-1.00)
    fig = generate_correlation_heatmap(mixed_data)
    ax = fig.gca()
    annotation_texts = [t.get_text() for t in ax.texts]
    assert "-1.00" in annotation_texts
    plt.close(fig)

def test_generate_heatmap_custom_title(mixed_data):
    # Simple Case: Custom title
    fig = generate_correlation_heatmap(mixed_data, title="Wine Correlations")
    assert fig.axes[0].get_title() == "Wine Correlations"
    plt.close(fig)

def test_generate_heatmap_single_column():
    # Edge Case: Only one numeric column
    df = pd.DataFrame({'num': [1, 2, 3], 'text': ['a', 'b', 'c']})
    fig = generate_correlation_heatmap(df)
    ax = fig.gca()
    # Should only have one label on the axis
    assert len(ax.get_xticklabels()) == 1
    plt.close(fig)

def test_generate_heatmap_no_numeric_error():
    # Error Case: No numbers available
    df_only_text = pd.DataFrame({'a': ['x', 'y']})
    with pytest.raises(ValueError, match="No numeric columns available"):
        generate_correlation_heatmap(df_only_text)

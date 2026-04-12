import pytest
import pandas as pd
from wine_quality_tools import clean_data


# simple case 1

def test_clean_data_simple_basic(tmp_path):
    """Basic case: small datasets are cleaned and merged correctly."""

    red_df = pd.DataFrame({"Fixed Acidity": [1], "pH": [3.5]})
    white_df = pd.DataFrame({"Fixed Acidity": [2], "pH": [3.0]})

    red_path = tmp_path / "red.csv"
    white_path = tmp_path / "white.csv"
    output_path = tmp_path / "output.csv"

    red_df.to_csv(red_path, index=False)
    white_df.to_csv(white_path, index=False)

    result = clean_data(red_path, white_path, output_path)

    assert result.shape[0] == 2
    assert list(result.columns) == ["fixed_acidity", "ph", "wine_type"]
    assert list(result["wine_type"]) == ["red", "white"]
    assert output_path.exists()


# Simple case 2


def test_clean_data_multiple_rows(tmp_path):
    """Normal case: multiple rows per dataset."""

    red_df = pd.DataFrame({"Fixed Acidity": [1, 2], "pH": [3.5, 3.6]})
    white_df = pd.DataFrame({"Fixed Acidity": [3, 4], "pH": [3.0, 3.1]})

    red_path = tmp_path / "red.csv"
    white_path = tmp_path / "white.csv"
    output_path = tmp_path / "output.csv"

    red_df.to_csv(red_path, index=False)
    white_df.to_csv(white_path, index=False)

    result = clean_data(red_path, white_path, output_path)

    # 4 rows total
    assert result.shape[0] == 4

    # check order and labels
    assert list(result["wine_type"]) == ["red", "red", "white", "white"]


# Edge case

def test_clean_data_empty_inputs(tmp_path):
    """Edge case: empty datasets."""

    red_df = pd.DataFrame({"dummy": []})
    white_df = pd.DataFrame({"dummy": []})

    red_path = tmp_path / "red.csv"
    white_path = tmp_path / "white.csv"
    output_path = tmp_path / "output.csv"

    red_df.to_csv(red_path, index=False)
    white_df.to_csv(white_path, index=False)

    result = clean_data(red_path, white_path, output_path)

    assert result.empty
    assert "wine_type" in result.columns


# Error case

def test_clean_data_invalid_path():
    """Error case: invalid file paths."""

    with pytest.raises(ValueError):
        clean_data("bad_red.csv", "bad_white.csv", "out.csv")

import pytest
import pandas as pd
import os
from pathlib import Path
from wine_quality_tools import download_data


# Simple case 1

def test_download_data_basic(tmp_path):
    """Downloads datasets and saves them correctly."""

    red_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    white_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    red_df, white_df, red_path, white_path = download_data(
        red_url, white_url, tmp_path, sep=","
    )

    assert isinstance(red_df, pd.DataFrame)
    assert isinstance(white_df, pd.DataFrame)
    assert Path(red_path).exists()
    assert Path(white_path).exists()


# Simple case 2

def test_download_data_shape(tmp_path):
    """Downloaded data has expected content."""

    red_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    white_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    red_df, white_df, _, _ = download_data(
        red_url, white_url, tmp_path, sep=","
    )

    assert red_df.shape[0] > 0
    assert white_df.shape[0] > 0


# Edge case


def test_download_data_custom_separator(tmp_path):
    """Handles custom separator input."""

    red_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    white_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    red_df, _, _, _ = download_data(
        red_url, white_url, tmp_path, sep=","
    )

    assert "sepal_length" in red_df.columns


# Error case

def test_download_data_invalid_url(tmp_path):
    """Invalid URL should raise ValueError."""

    with pytest.raises(Exception):
        download_data("invalid_url", "invalid_url", tmp_path)

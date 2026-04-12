import pytest
import pandas as pd
import sys
import os
import numpy as np

# Import from preprocessing.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wine_quality_tools import build_preprocessor

## Setup data
##============

# Input test data
numeric_df = pd.DataFrame({
    'age': [18, 25, 40, 65],
    'income': [30000, 50000, 80000, 120000]
})

categorical_df = pd.DataFrame({
    'city': ['Van', 'Vic', 'Van', 'Cal'],
    'gender': ['F', 'M', 'F', 'M']
})

mixed_df = pd.DataFrame({
    'age': [18, 25, 40, 65],
    'city': ['Van', 'Vic', 'Van', 'Cal'],
    'income': [30000, 50000, 80000, 120000]
})

empty_cols_df = pd.DataFrame({
    'age': [18, 25, 40, 65],
    'city': ['Van', 'Vic', 'Van', 'Cal']
})

missing_col_df = pd.DataFrame({
    'age': [18, 25, 40, 65],
    'city': ['Van', 'Vic', 'Van', 'Cal']
})

non_numeric_df = pd.DataFrame({
    'age': [18, 'twenty-five', 40, 65],
    'city': ['Van', 'Vic', 'Van', 'Cal']
})

numeric_in_categorical_df = pd.DataFrame({
    'age': [18, 25, 40, 65],
    'zip_code': [11111, 22222, 11111, 33333]
})

##============
## Tests
##============

def test_numeric_only_expected_values():
    """Tests numeric standardization against known expected values."""
    df = pd.DataFrame({
        'age': [10, 20, 30],
        'income': [100, 200, 300]
    })

    preprocessor = build_preprocessor(df, [], ['age', 'income'])
    X_trans = preprocessor.fit_transform(df)

    expected = np.array([
        [-1.22474487, -1.22474487],
        [ 0.        ,  0.        ],
        [ 1.22474487,  1.22474487]
    ])

    assert np.allclose(X_trans, expected)

def test_categorical_only_expected_encoding():
    """Tests categorical encoding against expected output structure."""
    df = pd.DataFrame({
        'city': ['Van', 'Vic', 'Van'],
        'gender': ['F', 'M', 'F']
    })

    preprocessor = build_preprocessor(df, ['city', 'gender'], [])
    X_trans = preprocessor.fit_transform(df)

    expected = np.array([
        [1., 0., 0.],
        [0., 1., 1.],
        [1., 0., 0.]
    ])

    assert X_trans.shape == expected.shape
    assert np.allclose(X_trans, expected)

def test_mixed_columns():
    """Tests numeric scaling + categorical encoding with known expected output."""

    df = pd.DataFrame({
        'age': [10, 20, 30],
        'city': ['A', 'B', 'A']
    })

    preprocessor = build_preprocessor(df, ['city'], ['age'])
    X_trans = preprocessor.fit_transform(df)

    # Expected:
    # age standardized → [-1.2247, 0, 1.2247]
    # city (binary, drop='if_binary') → A=0, B=1
    expected = np.array([
        [-1.22474487, 0.],
        [ 0.        , 1.],
        [ 1.22474487, 0.]
    ])

    assert X_trans.shape == expected.shape
    assert np.allclose(X_trans, expected)

def test_empty_column_lists():
    """Edge case: both column lists empty."""
    with pytest.raises(ValueError, match="At least one non-empty column"):
        build_preprocessor(empty_cols_df, [], [])

def test_missing_column():
    """Error case: missing column."""
    with pytest.raises(ValueError, match="Columns not found"):
        build_preprocessor(missing_col_df, ['missing_city'], [])

def test_non_numeric_data():
    """Error case: non-numeric in numeric column."""
    with pytest.raises(ValueError, match="Non-numeric values found"):
        build_preprocessor(non_numeric_df, [], ['age'])

def test_numeric_in_categorical_warns(capsys):
    """Tests warning prints for numeric categorical data."""
    build_preprocessor(numeric_in_categorical_df, ['zip_code'], [])
    
    # Check warning was printed
    captured = capsys.readouterr()
    assert "WARNING: Numeric data detected" in captured.out

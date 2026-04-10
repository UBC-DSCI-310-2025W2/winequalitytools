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

def test_numeric_only():
    """Tests numeric standardization with multiple numeric columns."""
    preprocessor = build_preprocessor(numeric_df, [], ['age', 'income'])
    X_trans = preprocessor.fit_transform(numeric_df)
    assert X_trans.shape == (4, 2)
    assert not np.isnan(X_trans).any()

def test_categorical_only():
    """Tests categorical one-hot encoding with multiple categorical columns."""
    preprocessor = build_preprocessor(categorical_df, ['city', 'gender'], [])
    X_trans = preprocessor.fit_transform(categorical_df)
    assert X_trans.shape[0] == 4
    assert not np.isnan(X_trans).any()

def test_mixed_columns():
    """Tests using both numeric and categorical columns together."""
    preprocessor = build_preprocessor(mixed_df, ['city'], ['age', 'income'])
    X_trans = preprocessor.fit_transform(mixed_df)
    assert X_trans.shape[0] == 4
    assert not np.isnan(X_trans).any()

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

"""
Functions for the Wine Quality Tools package.
This module provides utilities for data acquisition, cleaning, 
preprocessing, and visualization of wine quality data.
"""

import os
import pandas as pd
from pandas.errors import EmptyDataError
from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
import pandas.api.types as pd_types

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import seaborn as sns

def _fetch_csv(url, sep=";"):
    """
    Fetch a CSV file from a URL and return a DataFrame.
    """
    try:
        return pd.read_csv(url, sep=sep)
    except Exception as e:
        raise ValueError(f"Error downloading data: {e}")

def _validate_and_convert_to_string(value, name):
    """
    Validate that a value is a non-empty string or Path, and return it as a string.
    """
    if isinstance(value, Path):
        value = str(value)

    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{name} must be a non-empty string.")

    return value
  
def download_data(red_url, white_url, output_dir, sep=";"):
    """
    Download wine datasets from URLs and save them locally.

    Parameters
    ----------
    red_url : str or Path
        URL for red wine dataset
    white_url : str or Path
        URL for white wine dataset
    output_dir : str or Path
        Directory to save downloaded CSV files
    sep : str
        Separator used in CSV files (default ';')

    Returns
    -------
    tuple
        (red_df, white_df, red_path, white_path)

    Raises
    ------
    ValueError
        If inputs are invalid or download fails
    """
    red_url = _validate_and_convert_to_string(red_url, "red_url")
    white_url = _validate_and_convert_to_string(white_url, "white_url")
    output_dir = _validate_and_convert_to_string(output_dir, "output_dir")

    os.makedirs(output_dir, exist_ok=True)

    red_df = _fetch_csv(red_url, sep)
    white_df = _fetch_csv(white_url, sep)

    red_path = os.path.join(output_dir, "winequality-red.csv")
    white_path = os.path.join(output_dir, "winequality-white.csv")

    red_df.to_csv(red_path, index=False)
    white_df.to_csv(white_path, index=False)

    return red_df, white_df, red_path, white_path

def load_data(filepath):
  """
    Load a dataset from a CSV file.

    This function reads a CSV file from the given filepath and returns it
    as a pandas DataFrame. It performs input validation to ensure the
    filepath is valid and exists before attempting to load the file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file to be loaded.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    ValueError
        If an error occurs while reading the file.

    Examples
    --------
    >>> df = load_data("data/wine.csv")
    """
   try:
        return pd.read_csv(filepath)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
      
def _clean_column_names(df):
    """
    Standardize column names.
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return df
  
def _add_wine_type(df, wine_type):
    """
    Add wine_type column to dataframe
    """
    df["wine_type"] = wine_type
    return df
  
def _merge_datasets(red_df, white_df):
    """
    Merge red and white wine datasets.
    """
    return pd.concat([red_df, white_df], ignore_index=True)
  

def save_data(df, filepath):
    """
    Save DataFrame to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
    filepath : str
    
    Examples
    --------
    >>> save_data(df, "output/cleaned.csv")
    """
    df.to_csv(filepath, index=False)

def clean_data(red_input, white_input, output_file):

    """
    Load, clean, merge, and save wine datasets.

    Parameters
    ----------
    red_input : str or Path
        Path to red wine CSV file
    white_input : str or Path
        Path to white wine CSV file
    output_file : str or Path
        Path to save cleaned dataset

    Returns
    -------
    pd.DataFrame
        Cleaned and merged dataset

    Raises
    ------
    ValueError
        If inputs are invalid or files cannot be read
    """
    
    # ---- Convert Path to string ----
    red_input = _validate_and_convert_to_string(red_input, "red_input")
    white_input = _validate_and_convert_to_string(white_input, "white_input")
    output_file = _validate_and_convert_to_string(output_file, "output_file")

    # ---- Load data ----
    red = load_data(red_input)
    white = load_data(white_input)

    # ---- Clean column names ----
    red = _clean_column_names(red)
    white = _clean_column_names(white)

    # ---- Add wine type ----
    red = _add_wine_type(red, "red")
    white = _add_wine_type(white, "white")

    # ---- Merge datasets ----
    combined = _merge_datasets(red, white)

    # ---- Save output ----
    try:
        combined.to_csv(output_file, index=False)
    except Exception as e:
        raise ValueError(f"Error saving file: {e}")

    return combined


def stratified_split(df, target_col, test_size=0.3, random_state=42):
    """
    Perform stratified train-test split.

    Parameters
    ----------
    df : pandas.DataFrame
    target_col : str
    test_size : float
    random_state : int

    Returns
    -------
    (train_df, test_df)
    """
    if target_col not in df.columns:
        raise KeyError(f"{target_col} not found in dataframe")

    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )

    return train, test


def build_preprocessor(df, categorical_cols, numeric_cols):
    """
    Build a preprocessing transformer for numeric and categorical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the columns to check.
    categorical_cols : list of str
        List of categorical column names to one-hot encode.
    numeric_cols : list of str
        List of numeric column names to standardize.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A transformer pipeline that:
        - Applies StandardScaler to numeric columns
        - Applies OneHotEncoder(drop='if_binary') to categorical columns

    Raises
    ------
    ValueError
        - If both `numeric_cols` and `categorical_cols` are empty
        - If specified columns don't exist in `df`
        - If numeric columns contain non-numeric data
        
    Warnings
    --------
    Prints warning if categorical columns contain numeric data
    (suggests moving them to numeric_cols).
    
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [25, 30], 'city': ['A', 'B']})
    >>> preprocessor = build_preprocessor(df, ['city'], ['age'])
    >>> X_transformed = preprocessor.fit_transform(df)
    """
    
    if not numeric_cols and not categorical_cols:
        raise ValueError("At least one non-empty column must be provided.")

    # Check that all provided columns exist in df
    missing_cols = [col for col in numeric_cols + categorical_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")

    # Validate numeric columns contain only numeric data (ERROR if not)
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Non-numeric values found in numeric column '{col}'")

    # Warn about numeric data in categorical columns (can still proceed)
    for col in categorical_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(
                f"WARNING: Numeric data detected in categorical column '{col}'. "
                f"Consider moving it to numeric_cols."
            )

    # Build and return column transformer
    transformer = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(drop="if_binary"), categorical_cols))

    return transformer


def generate_boxplot_comparison(data, x, y_cols, title=None):
    """
    Creates side-by-side boxplots for multiple y-variables against a single x-variable.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe containing the variables to plot.
    x : str
        The column name to be used for the x-axis (usually a categorical variable like 'quality').
    y_cols : list of str
        A list of column names for the y-axis (numerical features like 'alcohol' or 'pH').
    title : str, optional
        The main title for the entire figure. Defaults to "Comparison across {x}".

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the generated boxplots.

    Raises
    ------
    KeyError
        If any of the specified columns are not present in the dataframe.
    """
    for col in [x] + y_cols:
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

    fig, axes = plt.subplots(1, len(y_cols), figsize=(15, 6))
    if len(y_cols) == 1: axes = [axes]

    for i, y in enumerate(y_cols):
        sns.boxplot(data=data, x=x, y=y, ax=axes[i])
        axes[i].set_title(f"{y.replace('_', ' ').title()} vs {x.title()}")

    plt.suptitle(title or f"Comparison across {x}", fontsize=16)
    return fig


def generate_correlation_heatmap(data, title="Correlation Matrix"):
    """
    Generates a heatmap of Pearson correlation coefficients for all numeric columns.

    This function automatically filters for numeric data types, calculates the 
    correlation matrix, and visualizes it with a 'coolwarm' color map and annotations.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe to analyze.
    title : str, optional
        The title for the heatmap. Defaults to "Correlation Matrix".

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the correlation heatmap.

    Raises
    ------
    ValueError
        If the dataframe contains no numeric columns (float64 or int64).
    """
    numeric_df = data.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError("No numeric columns available to correlate.")

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    return fig
  

def generate_histograms(data, columns, is_categorical=False, title=None):
    """
    Creates distribution plots (histograms or countplots) for specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe containing the data to visualize.
    columns : str or list of str
        The column name(s) to plot. If `is_categorical` is True, a single 
        string or the first element of a list is used.
    is_categorical : bool, optional
        If True, generates a Seaborn countplot (best for strings/grades). 
        If False, generates Seaborn histplots (best for numeric features). 
        Defaults to False.
    title : str, optional
        The main title for the figure. Defaults to "Distribution Plot".

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the generated plots.

    Raises
    ------
    ValueError
        If the provided DataFrame is empty or if 'columns' is not 
        contained within the DataFrame.
    TypeError
        If 'data' is not a pandas DataFrame.
    """
   
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("The provided DataFrame is empty.")
    
    # Ensure columns is a list for consistency
    cols_to_plot = [columns] if isinstance(columns, str) else columns
    
    missing_cols = [c for c in cols_to_plot if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    num_cols = 1 if is_categorical else len(cols_to_plot)
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5), squeeze=False)
    sns.set_theme(style="whitegrid")

    if is_categorical:
        col = cols_to_plot[0]
        sns.countplot(data=data, x=col, palette='viridis', ax=axes[0, 0])
        axes[0, 0].set_title(f"Count of {col}")
    else:
        for i, col in enumerate(cols_to_plot):
            sns.histplot(data=data, x=col, bins=20, color='steelblue', ax=axes[0, i])
            axes[0, i].set_title(f"Distribution of {col}")

    plt.suptitle(title or "Distribution Plot", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def save_figure(fig, output_path):
    """
    Saves a Matplotlib figure object to a file, handling directory creation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    output_path : str
        The full file path (including filename and extension) where the image 
        should be saved (e.g., 'results/figures/plot.png').

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the figure object provided is None.
    """
    if fig is None:
        raise ValueError("Figure object is None.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

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

    # ---- Convert Path to string ----
    for name, value in [("red_url", red_url), ("white_url", white_url), ("output_dir", output_dir)]:
        if isinstance(value, Path):
            value = str(value)

        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"{name} must be a non-empty string.")

    red_url = str(red_url)
    white_url = str(white_url)
    output_dir = str(output_dir)

    # ---- Create directory ----
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    try:
        red_df = pd.read_csv(red_url, sep=sep)
        white_df = pd.read_csv(white_url, sep=sep)
    except Exception as e:
        raise ValueError(f"Error downloading data: {e}")

    # ---- Save files ----
    red_path = os.path.join(output_dir, "winequality-red.csv")
    white_path = os.path.join(output_dir, "winequality-white.csv")

    red_df.to_csv(red_path, index=False)
    white_df.to_csv(white_path, index=False)

    return red_df, white_df, red_path, white_path


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
    if isinstance(red_input, Path):
        red_input = str(red_input)
    if isinstance(white_input, Path):
        white_input = str(white_input)
    if isinstance(output_file, Path):
        output_file = str(output_file)

    # ---- Validate inputs ----
    for name, path in [("red_input", red_input), ("white_input", white_input), ("output_file", output_file)]:
        if not isinstance(path, str) or path.strip() == "":
            raise ValueError(f"{name} must be a non-empty string.")

    # ---- Load data ----
    try:
        red = pd.read_csv(red_input)
    except EmptyDataError:
        red = pd.DataFrame()
    except Exception as e:
        raise ValueError(f"Error reading red_input: {e}")

    try:
        white = pd.read_csv(white_input)
    except EmptyDataError:
        white = pd.DataFrame()
    except Exception as e:
        raise ValueError(f"Error reading white_input: {e}")

    # ---- Clean column names ----
    red.columns = red.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    white.columns = white.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # ---- Add wine type ----
    red["wine_type"] = "red"
    white["wine_type"] = "white"

    # ---- Merge datasets ----
    combined = pd.concat([red, white], ignore_index=True)

    # ---- Save output ----
    try:
        combined.to_csv(output_file, index=False)
    except Exception as e:
        raise ValueError(f"Error saving file: {e}")

    return combined


def load_data(filepath):
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(filepath)


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


def save_data(df, filepath):
    """
    Save DataFrame to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
    filepath : str
    """
    df.to_csv(filepath, index=False)


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
    numeric_df = data.select_dtypes(include=['float64', 'int64'])
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
        The column(s) to plot. If is_categorical is True, only the first column is used.
    is_categorical : bool, optional
        If True, generates a Seaborn countplot. If False, generates Matplotlib 
        histograms for numeric data. Defaults to False.
    title : str, optional
        The main title for the figure. Defaults to "Distribution Plot".

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the distribution plots.

    Raises
    ------
    ValueError
        If the provided dataframe is empty.
    """
    if data.empty:
        raise ValueError("The provided DataFrame is empty.")
    
    fig = plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")

    if is_categorical:
        # If columns is a list, take the first one for countplot
        col = columns[0] if isinstance(columns, list) else columns
        sns.countplot(data=data, x=col, palette='viridis', hue=col, legend=False)
    else:
        data[columns].hist(bins=20, figsize=(15, 10), color='steelblue', edgecolor='black')
        fig = plt.gcf()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.suptitle(title or "Distribution Plot", fontsize=16)
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

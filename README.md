# Welcome to Wine Quality Tools

|        |        |
|--------|--------|
| Run Tests | [![Run Tests](https://github.com/UBC-DSCI-310-2025W2/winequalitytools/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/UBC-DSCI-310-2025W2/winequalitytools/actions/workflows/test.yml) |

**DSCI 310 Group 13 Authors:** Sungha Choi, Karen Siem, Alexis Widjaja, Siluni Jayarathne

`wine_quality_tools` is a Python package designed to streamline the analysis of [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). It provides automated tools for data acquisition, stratified splitting, standardized preprocessing, and common exploratory visualizations (histograms, boxplots, and correlation heatmaps).


**The Package Ecosystem**

While general libraries like `pandas` and `seaborn` provide the foundation for data manipulation and plotting, `wine_quality_tools` specializes these tools specifically for the UCI Wine Quality dataset, reducing boilerplate code for cleaning and standardized preprocessing.

## Get started

You can install the package directly from GitHub using:

```bash
$ pip install git+https://github.com/UBC-DSCI-310-2025W2/winequalitytools.git
```

To use wine-quality-tools in your code:

```python
import wine_quality_tools as wqt

# 1. Automated Data Acquisition & Cleaning
# Merges red and white wine CSVs and handles missing data internally
df = wqt.clean_data(
    red_input="data/raw/winequality-red.csv", 
    white_input="data/raw/winequality-white.csv", 
    output_file="data/processed/cleaned_wine.csv"
)

# 2. Robust Data Splitting
# Performs stratified splitting to preserve class balance in quality scores
train, test = wqt.stratified_split(df, target_col="quality", test_size=0.2)

# 3. Exploratory Visualization
# Generates a correlation heatmap focused on chemical properties
fig = wqt.generate_correlation_heatmap(df, title="Chemical Correlation Matrix")
wqt.save_figure(fig, "results/figures/heatmap.png")

# 4. Standardized Preprocessing
# Automatically builds transformers for numeric and categorical wine features
preprocessor = wqt.build_preprocessor(
    df, 
    categorical_cols=["wine_type"], 
    numeric_cols=["alcohol", "ph", "sulphates"]
)
```

## Documentation

- [Contributing](./CONTRIBUTING.md): Guidelines for contributing to this project.
- [Code of Conduct](./CODE_OF_CONDUCT.md): Expectations for team and community interactions.
- [License](./LICENSE): MIT License information.


## Copyright

- Copyright © 2026 UBC DSCI 310 Group 13.
- Free software distributed under the [MIT License](./LICENSE).

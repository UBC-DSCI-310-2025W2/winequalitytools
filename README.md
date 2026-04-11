# Welcome to Wine Quality Tools

|        |        |
|--------|--------|
| Run Tests | [![Run Tests](https://github.com/UBC-DSCI-310-2025W2/winequalitytools/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/UBC-DSCI-310-2025W2/winequalitytools/actions/workflows/test.yml) |


Python package designed to streamline the analysis of UCI Wine Quality datasets. It provides automated tools for data acquisition, stratified splitting, standardized preprocessing, and common exploratory visualizations (histograms, boxplots, and correlation heatmaps).


While general libraries like `pandas` and `seaborn` provide the foundation for data manipulation and plotting, `wine_quality_tools` specializes these tools specifically for the UCI Wine Quality dataset, reducing boilerplate code for cleaning and standardized preprocessing.

## Get started

Install the package directly from GitHub:

```bash
$ pip install git+https://github.com/UBC-DSCI-310-2025W2/winequalitytools.git
```

To use wine-quality-tools in your code:

```python
import wine_quality_tools as wqt

# Load and clean the data (ensure raw files are in your data folder)
df = wqt.clean_data(
    red_input="data/winequality-red.csv", 
    white_input="data/winequality-white.csv", 
    output_file="data/cleaned_wine.csv"
)

# Generate and save a correlation heatmap
fig = wqt.generate_correlation_heatmap(df)
wqt.save_figure(fig, "results/heatmap.png")v
```

## Copyright

- Copyright © 2026 UBC DSCI 310 Group 13.
- Free software distributed under the [MIT License](./LICENSE).

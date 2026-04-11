# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Upcoming features and fixes

## [0.1.0] - (2026-04-11)

- First release

### Added

- Initial package structure for `wine_quality_tools`.
- `clean_data` function for merging and cleaning UCI wine datasets.
- `stratified_split` function to ensure class balance during data partitioning.
- `generate_histograms`, `generate_boxplot_comparison`, and `generate_correlation_heatmap` for EDA.
- Integrated `build_preprocessor` which utilizes `scikit-learn` transformers to handle numeric scaling and categorical encoding for wine features.
- Comprehensive test suite with 36 test cases and code coverage reporting.

### Fixed

- Improved input validation in plotting functions to handle non-DataFrame inputs gracefully.

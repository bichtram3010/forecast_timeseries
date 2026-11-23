# Legacy pipeline (before the rebuild)

This folder keeps the first version of the project for reference. Its scores are not valid: see
"How this project evolved" in the main README. Do not use these files for results.

- `timeseries-forecast.ipynb`: the old single notebook (EDA, data prep, five-model training,
  recursive forecast, reconciliation).
- `figures/`: 14 EDA figures from that notebook. Some panels of `03`, `06` and `08` are wrong
  because of EDA aggregation bugs.
- `output/eda_summary.csv`: the EDA summary, with the same known bugs.
- `mmap/feature_cols.json`: the 44 features of the old pipeline.

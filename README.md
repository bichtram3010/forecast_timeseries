# forecast_timeseries

Global gradient-boosting forecasts for the **M5 Forecasting (Walmart) dataset**: 28-day-ahead unit sales for 30,490 item x store series, built to fit in a 30 GB Kaggle notebook.

> **Status:** work in progress. The pipeline runs end to end and produces a submission file, but the validation setup has known issues (see [Known limitations](#known-limitations)). Please read that section before quoting any number from this repo.

## What is in this repo

| Path | What it is |
|---|---|
| `timeseries-forecast (1).ipynb` | The whole pipeline as one Kaggle notebook (7 cells, saved with outputs) |
| `figures/01_...14_*.png` | 14 EDA figures (distribution, trends, seasonality, events/SNAP, prices, sparsity, outliers, hierarchy, data quality, correlations, decomposition, lag correlation) |
| `output/eda_summary.csv` | Summary statistics produced by the EDA cell |
| `mmap/feature_cols.json` | The 49 feature names fed to the models, in order |
| `LICENSE` | License |

Raw data, parquet files and memmaps are **not** committed (see `.gitignore`).

## Data

- M5 Forecasting - Accuracy (Walmart): 3,049 items, 10 stores in CA / TX / WI, 3 categories, 7 departments, so 30,490 bottom-level series.
- About 68% of item-day sales are zero (intermittent demand). Mean sales per item-day is about 1.12 with heavy right skew (skewness about 18).
- The notebook reads CSVs prepared in a Kaggle dataset (`/kaggle/input/datasets/minhduylebao/dataeda`, files `train.csv`, `val.csv`, `test.csv`). You need to point `DATA_DIR` to your own copy.

## Pipeline (notebook cells)

| Cell | Step | Notes |
|---|---|---|
| 0 | `pip install` | catboost, xgboost, lightgbm, torch, etc. |
| 1 | EDA | Streams the data in 1M-row chunks, writes the 14 figures and `eda_summary.csv` |
| 2 | Data prep and features | CSV to Snappy Parquet in 500k-row chunks, per-store feature engineering written to `np.memmap` so RAM stays low. This cell also contains a first training attempt, but its saved output ends right after a LightGBM GPU error and CPU fallback; the training results come from cell 3 |
| 3 | Training (v3) | Random 40% subsample of the training rows (22.6M of 56.6M), LightGBM (Tweedie 1.1) and XGBoost (Tweedie 1.1), GPU first with CPU fallback |
| 4 | Recursive forecast | Predicts day by day for 28 days, feeding predictions back into lag/rolling features |
| 5 | Multi-loss ensemble | Adds LightGBM with Tweedie 1.5, Poisson and RMSE objectives, inverse-RMSE weighted blend, then blended 50/50 with the recursive output |
| 6 | Hierarchical reconciliation (experimental) | Forecasts state and store totals with an exponential moving average plus weekday profile, then rescales the bottom-level forecasts toward them |

### Features (49)

- Calendar: `wday`, `month`, `year`, `quarter`, `is_weekend`, `day_of_month`, `is_payday`
- Events and SNAP: `event_type_1_enc`, `event_type_2_enc`, `has_event`, `snap_CA/TX/WI` and their interactions with events and weekday
- Price: `sell_price`, `price_ratio`, `price_norm`
- Label-encoded ids: item, dept, cat, store, state
- Lags and rolling stats: `lag_1/7/14/28`, `rmean_7/14/28`, `rstd_28`, `rmax_7`, `ewm_01`, `ewm_05`, `lag1_diff`, `trend_7`, `zero_streak`
- Aggregates: `item_gmean`, `store_day_mean`, `dept_day_mean`, `cat_day_mean`, `global_day_mean`, `vs_store`, `vs_item`

## Results

Metrics below are on the 28-day validation block (853,720 rows = 28 days x 30,490 series), **at the bottom level (item x store) only**, as printed in the notebook.

| Model | MAE | RMSE | WAPE % | Bias % |
|---|---|---|---|---|
| LightGBM, Tweedie 1.1 | 0.5924 | 1.6533 | 42.73 | +1.11 |
| LightGBM, Tweedie 1.5 | 0.5908 | 1.6509 | 42.62 | +0.94 |
| LightGBM, Poisson | 0.5950 | 1.6774 | 42.91 | +1.36 |
| LightGBM, RMSE | 0.6151 | 1.6727 | 44.37 | +2.70 |
| XGBoost, Tweedie 1.1 | 0.5906 | 1.6571 | 42.60 | +0.91 |
| Ensemble (LGB + XGB) | 0.5907 | 1.6494 | 42.61 | +1.01 |
| Ensemble (5 models) | 0.5953 | 1.6508 | 42.93 | +1.40 |

Not available yet in this repo: WRMSSE (the official M5 metric), WAPE at total / state / store level, and a comparison against LSTM or other baselines.

## How to run

1. Create a Kaggle notebook with a GPU (optional) and about 30 GB of RAM.
2. Attach a dataset containing the M5 `train.csv`, `val.csv`, `test.csv` in the layout the notebook expects, and edit `DATA_DIR` in cells 1 and 2.
3. Run the cells in order. Approximate wall times from the saved outputs: training (cell 3) about 30 min, recursive forecast (cell 4) about 18 min, multi-loss ensemble (cell 5) about 73 min, reconciliation (cell 6) under 1 min.
4. Outputs are written to `/kaggle/working/` (`figures/`, `output/`, `models/`, `preds/`, and the `submission_*.csv` files).

There is no `requirements.txt` yet; the packages are installed by the first cell.

## Known limitations

These are open issues found while reviewing the saved notebook. They affect how far the numbers above can be trusted.

1. **Possible target leakage in validation.** `store_day_mean`, `dept_day_mean`, `cat_day_mean` and `global_day_mean` are the mean of the *same day's actual sales*, computed on train + validation rows (cell 2). They are filled with 0 on the test days. Validation scores are therefore likely optimistic, and the train/test mismatch can push test forecasts down. `item_gmean` also includes validation rows.
2. **Recursive forecasts look too low.** The mean predicted value per series-day is about 0.26 to 0.34, against about 1.12 in the history.
3. **Reconciliation step is unreliable.** In cell 6 the bottom-up state totals are only about 25% of the top-level estimates, and the correction factor is clipped at 2.0 for all three states, so the two levels are not actually reconciled. The top level is an exponential moving average, not a trained model.
4. **EDA summary has wrong values** (event lift 0%, SNAP impact N/A, date coverage 2.52%, stores per item 1.0) caused by bugs in the EDA aggregation.
5. **Not easy to reproduce:** hard-coded Kaggle paths, several scripts pasted into one notebook, GPU failure with CPU fallback, no pinned dependencies, no saved submission or leaderboard score.
6. CatBoost and LSTM code appears in the early pipeline cell and in older scripts, but no CatBoost or LSTM results are reported here.

## Planned next steps

- Remove the same-day aggregate features, or lag them, and re-run validation.
- Compute WRMSSE and WAPE at every hierarchy level on a proper rolling-origin hold-out.
- Fix the reconciliation (for example bottom-up with MinT) and the EDA summary.
- Split the notebook into scripts, add `requirements.txt`, and commit the submission file and score.

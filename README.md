# forecast_timeseries

Global gradient-boosting forecasts for the **M5 Forecasting (Walmart) dataset**: 28-day-ahead unit sales for 30,490 item x store series, built to fit in a 30 GB Kaggle notebook.

> **Status:** work in progress. The leakage and recursive-evaluation bugs found in review have been fixed in the code (see [Changes after review](#changes-after-review)), but the notebook **has not been re-run yet**, so there are no valid scores in this repo at the moment. Please read [Results](#results) and [Known limitations](#known-limitations) before quoting any number.

## What is in this repo

| Path | What it is |
|---|---|
| `timeseries-forecast (1).ipynb` | The whole pipeline as one Kaggle notebook (7 cells). Outputs of the changed cells were cleared; only the `pip install` cell keeps its old output |
| `figures/01_...14_*.png` | 14 EDA figures (distribution, trends, seasonality, events/SNAP, prices, sparsity, outliers, hierarchy, data quality, correlations, decomposition, lag correlation) |
| `output/eda_summary.csv` | Summary statistics produced by the EDA cell |
| `mmap/feature_cols.json` | The 44 feature names fed to the models, in order |
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
| 2 | Data prep and features | CSV to Snappy Parquet in 500k-row chunks, feature engineering, `X_tr` / `X_val` / `y` written to `np.memmap` so RAM stays low. Data prep only, no training |
| 3 | Training (v3) | Random 40% subsample of the training rows (22.6M of 56.6M), LightGBM (Tweedie 1.1) and XGBoost (Tweedie 1.1), GPU first with CPU fallback. Saves models and one-step validation metrics only |
| 4 | Multi-loss models | Adds LightGBM with Tweedie 1.5, Poisson and RMSE objectives, and writes inverse-RMSE ensemble weights for all 5 models to `preds/ensemble_weights.json` |
| 5 | Recursive forecast (val + test) | Hides validation sales and forecasts the 28 validation days recursively with the 5-model ensemble (the reported score), then forecasts the 28 test days the same way. Writes `submission_recursive.csv` and predictions with their ids/days in (series, day) order |
| 6 | Hierarchical reconciliation (experimental) | Forecasts state and store totals with an exponential moving average plus weekday profile, then rescales the recursive bottom-level forecasts toward them |

"One-step" metrics (cells 3 and 4) use the true validation sales as lag inputs, so they only measure a 1-day-ahead forecast. They are used for early stopping and ensemble weights, not for reporting. The 28-day score is in `output/metrics_recursive_val.csv` from cell 5.

### Features (44)

- Calendar: `wday`, `month`, `year`, `quarter`, `is_weekend`, `day_of_month`, `is_payday`
- Events and SNAP: `event_type_1_enc`, `event_type_2_enc`, `has_event`, `snap_CA/TX/WI` and their interactions with events and weekday
- Price: `sell_price`, `price_ratio`, `price_norm`
- Label-encoded ids: item, dept, cat, store, state
- Lags and rolling stats: `lag_1/7/14/28`, `rmean_7/14/28`, `rstd_28`, `rmax_7`, `ewm_01`, `ewm_05`, `lag1_diff`, `trend_7`, `zero_streak`. All of them use sales up to the previous day only
- Aggregates: `item_gmean` (training rows only), `vs_item`

## Results

**No valid results yet.** The notebook has to be re-run on Kaggle after the fixes below; the 28-day recursive validation score will then be in `output/metrics_recursive_val.csv`.

The table below is kept only for reference. It came from the old pipeline and **overstates accuracy**: the validation lags used the true validation sales (a 1-day-ahead score, not a 28-day one), `zero_streak` included the target day's own sales, and four features were same-day sales means. Do not quote these numbers.

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
3. Run the cells in order, starting from an empty `/kaggle/working/` (cells 3 and 4 skip training when model and prediction files already exist, so stale files from an older run would be reused). Approximate wall times from the old run: training (cell 3) about 30 min, multi-loss models (cell 4) about 70 min, recursive forecast (cell 5) about 36 min (18 min for validation, 18 min for test; estimate, not measured yet), reconciliation (cell 6) under 1 min.
4. Outputs are written to `/kaggle/working/` (`figures/`, `output/`, `models/`, `preds/`, `output/metrics_recursive_val.csv`, `output/submission_recursive.csv` and `output/submission_reconciled.csv`).

There is no `requirements.txt` yet; the packages are installed by the first cell.

## Changes after review

Fixed in the code, **not yet verified by a run**:

1. `zero_streak` counted the target day's own sales, so it told the model whether that day had any sales. It now counts zero days up to the previous day. On test days the old version was always at least 1, which is the likely cause of the too-low recursive forecasts (mean 0.26 to 0.34 against about 1.12 in history).
2. The validation score was a 1-day-ahead score, because validation lags used the true validation sales. Cell 5 now hides the validation sales and forecasts all 28 days recursively.
3. The old multi-loss cell blended two prediction arrays stored in different orders (series-then-day against day-then-series), mixing up series and days. That blend is gone; predictions are now saved together with their ids and days.
4. Test forecasts from the training cells were not recursive (lags from day 2 onward were filled with 0). They are no longer produced; test is only forecast recursively, with all 5 models.
5. Removed the same-day aggregates (`store_day_mean`, `dept_day_mean`, `cat_day_mean`, `global_day_mean`, `vs_store`). `item_gmean` now uses training rows only.
6. Removed the first training attempt (including CatBoost) from the data prep cell.

## Known limitations

1. **Not re-run yet.** All scores must be regenerated before anything is quoted.
2. **Early stopping and ensemble weights use one-step validation**, on the same validation block that is later scored recursively, so the recursive score can still be slightly optimistic.
3. **Reconciliation step is unreliable.** In cell 6 the top level is an exponential moving average, not a trained model, and in the old run the correction factor was clipped at 2.0 for every state. Needs to be checked again once the bottom-level forecasts are fixed.
4. **EDA summary has wrong values** (event lift 0%, SNAP impact N/A, date coverage 2.52%, stores per item 1.0) caused by bugs in the EDA aggregation. The EDA sample is also much larger than intended (about 14M rows instead of 250k).
5. **Not easy to reproduce:** hard-coded Kaggle paths, several scripts pasted into one notebook, GPU failure with CPU fallback, no pinned dependencies, no saved submission or leaderboard score.
6. No CatBoost or LSTM results are reported here.

## Planned next steps

- Re-run the notebook on Kaggle and fill in the recursive validation results.
- Compute WRMSSE and WAPE at every hierarchy level on a proper rolling-origin hold-out.
- Fix the reconciliation (for example bottom-up with MinT) and the EDA summary.
- Split the notebook into scripts, add `requirements.txt`, and commit the submission file and score.

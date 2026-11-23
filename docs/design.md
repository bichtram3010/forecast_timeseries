# M5 Forecasting Rebuild: Design

Status: draft for review · 2026-11-23

## 1. Goal

Rebuild this repository into a portfolio project whose every reported number is real, reproducible and traceable to a commit and a Kaggle run. A reviewer should understand the project in two minutes from the README and be able to rerun it from the original M5 files.

**Success criteria**

1. The leakage test and all other tests pass in GitHub Actions on every push.
2. Four Kaggle notebooks run end to end from one pinned commit, starting from the official M5 files.
3. `results/` holds validation and holdout scores for every model, at every hierarchy level and every forecast week, plus run metadata (commit SHA, package versions, Kaggle notebook versions).
4. The README tables and figures are generated from `results/`, never typed by hand.

No target score is promised. Whatever the holdout says is what gets reported, including models that lose.

**Out of scope:** hyperparameter search, reconciliation methods other than bottom-up (for example MinT), probabilistic or quantile forecasts, store- or department-level aggregate features, external data, any deployment or API.

## 2. Data

- Source: the official Kaggle competition files `sales_train_evaluation.csv` (30,490 series, days d_1 to d_1941), `calendar.csv` and `sell_prices.csv` from `m5-forecasting-accuracy`.
- The wide sales table is kept as `(series x day)` numpy arrays, with calendar and prices as separate arrays and tables (not melted to long format); dtypes are downcast and everything is cached as `.npy` plus Parquet.
- Rows where `sell_price` is missing (item not on sale yet) are dropped.
- Training rows start at d_1100. Earlier days are kept only as history for lags and rolling windows.

## 3. Evaluation protocol

| Block | Days | Use |
|---|---|---|
| Train | d_1100 to d_1885 | Fit models |
| Validation | d_1886 to d_1913 (origin T = 1885) | Early stopping, feature and setting choices, ensemble weights |
| Holdout | d_1914 to d_1941 (origin T = 1913) | Scored **once**, after every choice is frozen |

- Horizon `h = d − T`, from 1 to 28. Week `k = ceil(h / 7)`, from 1 to 4.
- Before the holdout, every model is refit on d_1100 to d_1913 with the iteration or epoch count chosen on validation.
- After the holdout is scored, nothing is tuned further. If a bug is found later, the fix and the rerun are recorded in the README changelog with both numbers.

**Metrics**

- **WAPE per level**: `sum |y − ŷ| / sum |y|` over all series and days of a level, after bottom-up aggregation. Levels: total (1 series), state (3), store (10), category (3), department (7), item × store (30,490).
- **WRMSSE**: the official 12-level M5 metric. Scales from one-step naive errors over each series' history (from its first non-zero sale to the end of training); weights from dollar sales over the last 28 training days. Training end is d_1885 for validation and d_1913 for holdout.
- **Per-week breakdown**: WAPE per level for weeks 1 to 4.
- **Baselines that every model must be compared with**: seasonal naive (repeat the last 7 known days four times) and MA28 (mean of the last 28 known days).

## 4. Features

For the week-k model, every sales-history feature uses only data at least `L = 7k` days older than the target day. For any target day in week k this means data up to the origin T and nothing after it.

| Group | Features |
|---|---|
| Sales history | `lag_L`, `lag_{L+7}`, `lag_{L+14}`, `lag_{L+21}`; rolling mean and std over 7, 28, 90 and 180 days of the series shifted by L |
| Price | `sell_price`; change vs the previous week; ratio to the item-store mean price over the previous 365 days. Prices are published in advance in M5, so the target day's price is known |
| Calendar | weekday, month, year, day of month, `event_name_1/2`, `event_type_1/2`, SNAP flag of the series' state |
| IDs | `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` as categoricals |

**Leakage test**: on synthetic series, change the sales of every day closer than L days to the target day, rebuild the features, and assert that no feature of the week-k model changed. Run for k = 1 to 4.

## 5. Models

| Model | Setup |
|---|---|
| Seasonal naive, MA28 | Baselines, no training |
| LightGBM × 4 weeks | Tweedie loss (variance power 1.1), CPU, fixed parameters in code, early stopping on the validation days of week k |
| CatBoost × 4 weeks | Tweedie loss (variance power 1.1), GPU, same features, early stopping as above |
| LSTM (one model) | PyTorch. Input: last 112 days of sales divided by the window mean plus 1, ID embeddings (item, department, store), and calendar, price and SNAP for the 28 future days. 2-layer LSTM encoder (hidden 128) and an MLP head with 28 non-negative outputs, rescaled. Tweedie deviance loss. Training windows sampled from origins inside the training period; early stopping on the validation origin |
| Ensemble | Weighted mean of LightGBM, CatBoost and LSTM. Weights on a 0.1-step simplex grid (66 combinations), chosen by validation WRMSSE, then frozen |

Seeds are fixed. Each model writes bottom-level forecasts as `(id, d, forecast)` Parquet files, so the order of rows can never mix series.

## 6. Code layout

```text
src/m5/
  data.py        load official files, keep (series x day) arrays, downcast, .npy + Parquet cache
  splits.py      day constants and window helpers
  features.py    week-k features (lag >= 7k)
  baselines.py   seasonal naive, MA28
  gbdt.py        LightGBM and CatBoost: fit, refit, predict, per week
  lstm.py        window sampling, model, training loop
  ensemble.py    weight search on validation
  hierarchy.py   bottom-up aggregation to the 6 WAPE levels and 12 WRMSSE levels
  metrics.py     WAPE per level and week, WRMSSE
  report.py      results/ to README tables and figures
  cli.py         m5 prepare | train-gbdt | train-lstm | evaluate | report
notebooks/kaggle/
  01_lightgbm.ipynb  02_catboost.ipynb  03_lstm.ipynb  04_evaluate.ipynb
  make_notebooks.py  generates the four notebooks
tests/              pytest on small synthetic M5-shaped data
results/            CSV scores and metadata.json
figures/            generated from results/
legacy/             previous notebook, EDA figures and outputs, with a short note
docs/design.md      this document
pyproject.toml      minimum dependency versions; exact versions of each run go to results/metadata.json
.github/workflows/ci.yml   ruff and pytest
```

Each Kaggle notebook installs the package from GitHub at a pinned commit with `pip install --no-deps`
(Kaggle's preinstalled libraries are used and their versions are recorded), runs `m5 prepare`, and
calls one CLI command. Notebooks 01 to 03 write forecasts; notebook 04 takes their outputs as inputs.
If a notebook cannot reach the internet, the repository is attached as a Kaggle dataset instead.

## 7. Tests

- Feature leakage test (section 4).
- WAPE and WRMSSE against hand-computed values on a tiny example.
- Bottom-up aggregation sums correctly to every level.
- Baseline forecasts on a known series.
- Split helpers return the documented day ranges.
- Ensemble weight search picks the known best weights on a constructed case.
- Smoke tests: LightGBM, CatBoost and LSTM fit and predict on a few hundred rows on CPU, with output shape and non-negativity checks.

Tests run locally (uv-managed Python) and in GitHub Actions.

## 8. Running on Kaggle

1. The repository owner signs in to Kaggle and accepts the M5 competition rules.
2. Notebooks 01 to 03 run (01 on CPU, 02 and 03 on GPU, possibly at the same time), then 04. Budget: about 6 to 10 hours of CPU and GPU time, within one week of the free GPU quota.
3. `results/` and `figures/` are downloaded into the repository, and the README is regenerated with `m5 report`.

LightGBM on 4 CPU cores may not finish four weekly models in one 12-hour session. `m5 train-gbdt` saves each finished week as a part file, so the four weeks can run across two sessions with `--weeks` and `--resume-from` (notebook 01 has the two variables for this).

If LightGBM on 4 CPU cores is too slow, the fallback is to train on a fixed 50% row sample; this is recorded in the README if used.

## 9. Repository changes

- All work happens on branch `rebuild` and lands in `main` through one pull request.
- The old notebook, `figures/`, `output/` and `mmap/` move to `legacy/`. The README keeps the self-audit story in a "How this project evolved" section.

## 10. Risks

| Risk | Mitigation |
|---|---|
| LightGBM training too slow on Kaggle CPU | 50% row sample, recorded in README |
| Kaggle notebook without internet | Attach the repository as a Kaggle dataset |
| GPU quota runs out mid-week | Notebooks save outputs after each model, so a run can resume |
| Holdout scores worse than baselines for some model | Reported as is; the README explains it |
| Kaggle editor is hard to read from the browser automation | Read progress from screenshots and from saved output files |

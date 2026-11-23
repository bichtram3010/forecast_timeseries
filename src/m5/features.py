"""Features for the week-k model. Sales history only uses data at least 7k days old."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import EVENT_COLS, M5Data
from .splits import week_lag

LAG_STEPS = (0, 7, 14, 21)
ROLL_WINDOWS = (7, 28, 90, 180)
ID_FEATURES = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
CALENDAR_FEATURES = ("wday", "month", "year", "mday", *EVENT_COLS, "snap")
PRICE_FEATURES = ("sell_price", "price_change_1w", "price_rel_365")
CATEGORICAL_FEATURES = (*ID_FEATURES, *EVENT_COLS)


def history_feature_names(week: int) -> list[str]:
    lag = week_lag(week)
    return (
        [f"lag_{lag + s}" for s in LAG_STEPS]
        + [f"rmean_{w}" for w in ROLL_WINDOWS]
        + [f"rstd_{w}" for w in ROLL_WINDOWS]
    )


def feature_names(week: int) -> list[str]:
    return [*ID_FEATURES, *CALENDAR_FEATURES, *PRICE_FEATURES, *history_feature_names(week)]


def shift_right(a: np.ndarray, n: int) -> np.ndarray:
    """out[:, j] = a[:, j - n]; the first n columns are NaN."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    if n < a.shape[1]:
        out[:, n:] = a[:, : a.shape[1] - n]
    return out


def _trailing_sums(a: np.ndarray, w: int, partial: bool = False) -> np.ndarray:
    """Sum of columns j-w+1..j for every column j (float64). NaN for short windows unless partial."""
    n, t = a.shape
    cs = np.zeros((n, t + 1), dtype=np.float64)
    cs[:, 1:] = np.cumsum(a, axis=1, dtype=np.float64)
    end = np.arange(1, t + 1)
    out = cs[:, end] - cs[:, np.maximum(end - w, 0)]
    if not partial:
        out[:, : w - 1] = np.nan
    return out


def rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    return (_trailing_sums(a, w) / w).astype(np.float32)


def rolling_std(a: np.ndarray, w: int) -> np.ndarray:
    mean = _trailing_sums(a, w) / w
    mean_sq = _trailing_sums(a.astype(np.float64) ** 2, w) / w
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0)).astype(np.float32)


def history_features(sales: np.ndarray, week: int) -> dict[str, np.ndarray]:
    lag = week_lag(week)
    feats = {f"lag_{lag + s}": shift_right(sales, lag + s) for s in LAG_STEPS}
    for w in ROLL_WINDOWS:
        feats[f"rmean_{w}"] = shift_right(rolling_mean(sales, w), lag)
        feats[f"rstd_{w}"] = shift_right(rolling_std(sales, w), lag)
    return feats


def price_features(prices: np.ndarray) -> dict[str, np.ndarray]:
    """Prices are published in advance in M5, so the target day's price is known."""
    valid = ~np.isnan(prices)
    sums = _trailing_sums(np.where(valid, prices, 0.0), 365, partial=True)
    counts = _trailing_sums(valid.astype(np.float64), 365, partial=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_365 = np.where(counts > 0, sums / counts, np.nan)
        change = prices / shift_right(prices, 7) - 1.0
        rel = prices / mean_365
    return {
        "sell_price": prices.astype(np.float32),
        "price_change_1w": change.astype(np.float32),
        "price_rel_365": rel.astype(np.float32),
    }


def snap_matrix(ids: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """SNAP flag of each series' state, per day."""
    states = ids["state_id"].astype(str).to_numpy()
    out = np.zeros((len(ids), len(calendar)), dtype=np.int8)
    for state in np.unique(states):
        out[states == state] = calendar[f"snap_{state}"].to_numpy(dtype=np.int8)
    return out


def build_frame(
    data: M5Data, week: int, days: np.ndarray, drop_not_on_sale: bool = True
) -> pd.DataFrame:
    """One row per (day, series): week-k features and the `sales` target.

    Rows are ordered day by day, series fastest. Rows with no price are dropped by default.
    """
    days = np.asarray(days, dtype=np.int64)
    if days.min() < 1 or days.max() > data.n_days:
        raise ValueError(f"days must be within 1..{data.n_days}")
    cols = days - 1
    n = data.n_series
    series_idx = np.tile(np.arange(n, dtype=np.int32), len(days))

    def pick(mat: np.ndarray) -> np.ndarray:
        return mat[:, cols].ravel(order="F")

    frame: dict[str, object] = {"series_idx": series_idx, "d": np.repeat(days.astype(np.int16), n)}
    for col in ID_FEATURES:
        cat = data.ids[col].cat
        frame[col] = pd.Categorical.from_codes(
            cat.codes.to_numpy()[series_idx], categories=cat.categories
        )
    cal = data.calendar.iloc[cols]
    for col in ("wday", "month", "year", "mday"):
        frame[col] = np.repeat(cal[col].to_numpy(), n)
    for col in EVENT_COLS:
        frame[col] = pd.Categorical.from_codes(
            np.repeat(cal[col].cat.codes.to_numpy(), n),
            categories=data.calendar[col].cat.categories,
        )
    frame["snap"] = pick(snap_matrix(data.ids, data.calendar))
    for name, mat in price_features(data.prices).items():
        frame[name] = pick(mat)
    for name, mat in history_features(data.sales, week).items():
        frame[name] = pick(mat)
    frame["sales"] = pick(data.sales)
    df = pd.DataFrame(frame)
    if drop_not_on_sale:
        df = df[df["sell_price"].notna()].reset_index(drop=True)
    return df

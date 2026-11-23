"""WAPE per hierarchy level and week, and the official M5 WRMSSE."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchy import WAPE_LEVELS, WRMSSE_LEVELS, aggregate


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    if denom == 0:
        raise ValueError("WAPE is undefined when every actual value is zero")
    return float(np.abs(actual - forecast).sum() / denom)


def wape_table(actual: np.ndarray, forecast: np.ndarray, ids: pd.DataFrame) -> pd.DataFrame:
    """WAPE per level for all days (week 0) and for each 7-day week."""
    rows = []
    n_weeks = actual.shape[1] // 7
    for level, by in WAPE_LEVELS.items():
        a = aggregate(actual, ids, by)
        f = aggregate(forecast, ids, by)
        rows.append({"level": level, "week": 0, "value": wape(a, f)})
        for k in range(1, n_weeks + 1):
            cols = slice(7 * (k - 1), 7 * k)
            rows.append({"level": level, "week": k, "value": wape(a[:, cols], f[:, cols])})
    return pd.DataFrame(rows, columns=["level", "week", "value"])


def _naive_scale(hist: np.ndarray) -> np.ndarray:
    """Mean squared one-step naive error, from each series' first non-zero value to the end."""
    diffs = np.diff(hist, axis=1) ** 2
    nonzero = hist != 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), hist.shape[1])
    mask = np.arange(diffs.shape[1])[None, :] >= first[:, None]
    counts = mask.sum(axis=1)
    return np.where(counts > 0, (diffs * mask).sum(axis=1) / np.maximum(counts, 1), 0.0)


class WRMSSE:
    """Official M5 metric. History arrays cover day 1 to the last training day (the origin)."""

    def __init__(self, sales_history: np.ndarray, prices_history: np.ndarray, ids: pd.DataFrame):
        self.ids = ids
        dollars = np.nan_to_num(sales_history[:, -28:] * prices_history[:, -28:]).sum(
            axis=1, keepdims=True
        )
        self.levels: dict[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]] = {}
        for level, by in WRMSSE_LEVELS.items():
            scale = _naive_scale(aggregate(sales_history, ids, by))
            weight = aggregate(dollars, ids, by)[:, 0]
            self.levels[level] = (by, scale, weight / weight.sum())

    def per_level(self, actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
        out = {}
        for level, (by, scale, weight) in self.levels.items():
            err = aggregate(actual, self.ids, by) - aggregate(forecast, self.ids, by)
            mse = (err**2).mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rmsse = np.where(scale > 0, np.sqrt(mse / scale), 0.0)
            out[level] = float((weight * rmsse).sum())
        return out

    def score(self, actual: np.ndarray, forecast: np.ndarray) -> float:
        return float(np.mean(list(self.per_level(actual, forecast).values())))

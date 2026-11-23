"""Day constants. Days are 1-based as in the M5 files (d_1 ... d_1941)."""

from __future__ import annotations

import numpy as np

TRAIN_START = 1100
VAL_ORIGIN = 1885
HOLDOUT_ORIGIN = 1913
HORIZON = 28
N_WEEKS = 4
SPLIT_ORIGINS = {"validation": VAL_ORIGIN, "holdout": HOLDOUT_ORIGIN}


def week_lag(week: int) -> int:
    """Minimum sales lag (in days) the week-k model may use."""
    if week not in range(1, N_WEEKS + 1):
        raise ValueError(f"week must be 1..{N_WEEKS}, got {week}")
    return 7 * week


def target_days(origin: int, week: int | None = None) -> np.ndarray:
    """Days forecast from `origin` (the last known day): all 28, or the 7 of one week."""
    if week is None:
        return np.arange(origin + 1, origin + HORIZON + 1)
    week_lag(week)
    start = origin + 7 * (week - 1) + 1
    return np.arange(start, start + 7)


def fit_days(origin: int, start: int = TRAIN_START) -> np.ndarray:
    """Target days a model may train on when the last known day is `origin`."""
    return np.arange(start, origin + 1)

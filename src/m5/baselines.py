"""Forecasts that every model must beat."""

from __future__ import annotations

import numpy as np

from .splits import HORIZON


def seasonal_naive(sales: np.ndarray, origin: int, horizon: int = HORIZON) -> np.ndarray:
    """Repeat the last 7 known days (origin - 6 .. origin)."""
    last_week = sales[:, origin - 7 : origin]
    reps = -(-horizon // 7)
    return np.tile(last_week, (1, reps))[:, :horizon].astype(np.float32)


def moving_average(
    sales: np.ndarray, origin: int, window: int = 28, horizon: int = HORIZON
) -> np.ndarray:
    """Mean of the last `window` known days, flat over the horizon."""
    mean = sales[:, origin - window : origin].mean(axis=1, keepdims=True)
    return np.repeat(mean, horizon, axis=1).astype(np.float32)

"""Weighted mean of model forecasts, weights chosen on validation."""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np


def simplex_grid(n_models: int, step: float = 0.1) -> np.ndarray:
    k = round(1 / step)
    combos = [c for c in itertools.product(range(k + 1), repeat=n_models) if sum(c) == k]
    return np.array(combos, dtype=np.float64) / k


def combine(forecasts: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * forecasts[name] for name in weights).astype(np.float32)


def search_weights(
    forecasts: dict[str, np.ndarray],
    score_fn: Callable[[np.ndarray], float],
    step: float = 0.1,
) -> tuple[dict[str, float], float]:
    """Try every grid point; lower score is better; the first best point wins ties."""
    names = sorted(forecasts)
    best_weights: dict[str, float] = {}
    best_score = np.inf
    for row in simplex_grid(len(names), step):
        weights = {name: float(w) for name, w in zip(names, row)}
        score = score_fn(combine(forecasts, weights))
        if score < best_score - 1e-12:
            best_weights, best_score = weights, score
    return best_weights, float(best_score)

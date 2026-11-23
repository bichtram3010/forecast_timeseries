"""Score every model and the ensemble on validation and holdout."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import moving_average, seasonal_naive
from .data import M5Data
from .ensemble import combine, search_weights
from .metrics import WRMSSE, wape_table
from .splits import HOLDOUT_ORIGIN, HORIZON, SPLIT_ORIGINS, VAL_ORIGIN, target_days

MEMBERS = ("lgb", "cb", "lstm")


def evaluate_all(
    data: M5Data, forecasts: dict[tuple[str, str], np.ndarray]
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    models = sorted({model for model, _ in forecasts})
    missing = [(m, s) for m in models for s in SPLIT_ORIGINS if (m, s) not in forecasts]
    if missing:
        raise ValueError(f"missing forecast files: {missing}")

    per_split: dict[str, dict[str, np.ndarray]] = {}
    for split, origin in SPLIT_ORIGINS.items():
        fcs = {
            "seasonal_naive": seasonal_naive(data.sales, origin),
            "ma28": moving_average(data.sales, origin),
        }
        fcs.update({m: fc for (m, s), fc in forecasts.items() if s == split})
        per_split[split] = fcs

    scorers = {
        split: WRMSSE(data.sales[:, :origin], data.prices[:, :origin], data.ids)
        for split, origin in SPLIT_ORIGINS.items()
    }

    members = [m for m in MEMBERS if m in models]
    weights: dict[str, float] = {}
    if len(members) >= 2:
        actual_val = data.sales[:, VAL_ORIGIN : VAL_ORIGIN + HORIZON]
        weights, _ = search_weights(
            {m: per_split["validation"][m] for m in members},
            lambda fc: scorers["validation"].score(actual_val, fc),
        )
        for split in SPLIT_ORIGINS:
            per_split[split]["ensemble"] = combine(
                {m: per_split[split][m] for m in members}, weights
            )

    rows = []
    for split, origin in SPLIT_ORIGINS.items():
        actual = data.sales[:, origin : origin + HORIZON]
        for model, fc in per_split[split].items():
            rows.append(
                {"split": split, "model": model, "metric": "wrmsse", "level": "all", "week": 0,
                 "value": scorers[split].score(actual, fc)}
            )
            for r in wape_table(actual, fc, data.ids).to_dict("records"):
                rows.append({"split": split, "model": model, "metric": "wape", **r})
    scores = pd.DataFrame(rows, columns=["split", "model", "metric", "level", "week", "value"])

    total = pd.DataFrame(
        {
            "d": target_days(HOLDOUT_ORIGIN),
            "actual": data.sales[:, HOLDOUT_ORIGIN : HOLDOUT_ORIGIN + HORIZON].sum(axis=0),
        }
    )
    for model, fc in per_split["holdout"].items():
        total[model] = fc.sum(axis=0)
    return scores, weights, total

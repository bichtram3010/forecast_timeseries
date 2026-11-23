"""Forecast files: long (id, d, forecast) Parquet, so rows can never mix series."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import HORIZON

_PATTERN = re.compile(r"^fc_(?P<model>[a-z0-9_]+)_(?P<split>validation|holdout)\.parquet$")


def to_long(forecast: np.ndarray, ids: pd.DataFrame, origin: int) -> pd.DataFrame:
    n, h = forecast.shape
    return pd.DataFrame(
        {
            "id": np.repeat(ids["id"].astype(str).to_numpy(), h),
            "d": np.tile(np.arange(origin + 1, origin + h + 1), n).astype(np.int64),
            "forecast": forecast.ravel().astype(np.float32),
        }
    )


def from_long(
    df: pd.DataFrame, ids: pd.DataFrame, origin: int, horizon: int = HORIZON
) -> np.ndarray:
    wide = df.assign(d=df["d"].astype(np.int64)).pivot(index="id", columns="d", values="forecast")
    days = np.arange(origin + 1, origin + horizon + 1)
    wide = wide.reindex(index=ids["id"].astype(str).to_numpy(), columns=days)
    if wide.isna().to_numpy().any():
        raise ValueError("forecast file is missing some (id, d) pairs")
    return wide.to_numpy(dtype=np.float32)


def forecast_path(directory: Path, model: str, split: str) -> Path:
    return Path(directory) / f"fc_{model}_{split}.parquet"


def save_forecast(path: Path, forecast: np.ndarray, ids: pd.DataFrame, origin: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_long(forecast, ids, origin).to_parquet(path, index=False)


def load_forecast(path: Path, ids: pd.DataFrame, origin: int) -> np.ndarray:
    return from_long(pd.read_parquet(path), ids, origin)


def discover_forecasts(root: Path) -> dict[tuple[str, str], Path]:
    """Find every fc_{model}_{split}.parquet under root. Duplicates are an error."""
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(Path(root).rglob("fc_*.parquet")):
        match = _PATTERN.match(path.name)
        if not match:
            continue
        key = (match["model"], match["split"])
        if key in found:
            raise ValueError(f"duplicate forecast files for {key}: {found[key]} and {path}")
        found[key] = path
    return found


def zero_when_not_on_sale(forecast: np.ndarray, prices: np.ndarray, origin: int) -> np.ndarray:
    on_sale = ~np.isnan(prices[:, origin : origin + forecast.shape[1]])
    return np.where(on_sale, forecast, 0.0).astype(np.float32)

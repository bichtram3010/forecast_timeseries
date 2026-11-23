"""Load the official M5 files into wide arrays and cache them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
N_DAYS = 1941
SALES_FILE = "sales_train_evaluation.csv"


@dataclass
class M5Data:
    """All inputs, one row per series. Column j of `sales` and `prices` is day j + 1."""

    ids: pd.DataFrame
    sales: np.ndarray
    prices: np.ndarray
    calendar: pd.DataFrame

    @property
    def n_series(self) -> int:
        return self.sales.shape[0]

    @property
    def n_days(self) -> int:
        return self.sales.shape[1]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "sales.npy", self.sales)
        np.save(directory / "prices.npy", self.prices)
        self.ids.to_parquet(directory / "ids.parquet")
        self.calendar.to_parquet(directory / "calendar.parquet")

    @classmethod
    def load(cls, directory: Path) -> "M5Data":
        directory = Path(directory)
        return cls(
            ids=pd.read_parquet(directory / "ids.parquet"),
            sales=np.load(directory / "sales.npy"),
            prices=np.load(directory / "prices.npy"),
            calendar=pd.read_parquet(directory / "calendar.parquet"),
        )


def find_data_dir(root: Path) -> Path:
    """Return the folder under `root` that contains the M5 sales file."""
    matches = sorted(Path(root).rglob(SALES_FILE))
    if not matches:
        raise FileNotFoundError(f"{SALES_FILE} not found under {root}")
    return matches[0].parent


def load_m5(data_dir: Path, n_days: int = N_DAYS) -> M5Data:
    data_dir = Path(data_dir)
    day_cols = [f"d_{i}" for i in range(1, n_days + 1)]
    wide = pd.read_csv(
        data_dir / SALES_FILE,
        usecols=ID_COLS + day_cols,
        dtype={c: "float32" for c in day_cols},
    )
    ids = wide[ID_COLS].astype("category").reset_index(drop=True)
    sales = wide[day_cols].to_numpy(dtype=np.float32)
    calendar = _load_calendar(data_dir / "calendar.csv", n_days)
    prices = _price_matrix(data_dir / "sell_prices.csv", ids, calendar)
    return M5Data(ids=ids, sales=sales, prices=prices, calendar=calendar)


def _load_calendar(path: Path, n_days: int) -> pd.DataFrame:
    cal = pd.read_csv(path).iloc[:n_days].reset_index(drop=True)
    if cal["d"].tolist() != [f"d_{i}" for i in range(1, n_days + 1)]:
        raise ValueError("calendar.csv rows are not d_1 .. d_n in order")
    out = cal[["wm_yr_wk", "wday", "month", "year"]].astype(
        {"wm_yr_wk": "int32", "wday": "int8", "month": "int8", "year": "int16"}
    )
    out["mday"] = pd.to_datetime(cal["date"]).dt.day.astype("int8")
    for col in EVENT_COLS:
        out[col] = cal[col].astype("category")
    for col in [c for c in cal.columns if c.startswith("snap_")]:
        out[col] = cal[col].astype("int8")
    return out


def _price_matrix(path: Path, ids: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    prices = pd.read_csv(
        path,
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
    )
    key = (ids["item_id"].astype(str) + "|" + ids["store_id"].astype(str)).to_numpy()
    series_index = pd.Series(np.arange(len(ids)), index=key)
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    week_index = pd.Series(np.arange(len(weeks)), index=weeks)
    prices = prices[prices["wm_yr_wk"].isin(weeks)]
    rows = series_index.reindex((prices["item_id"] + "|" + prices["store_id"]).to_numpy()).to_numpy()
    keep = ~np.isnan(rows.astype(np.float64))
    cols = week_index.reindex(prices["wm_yr_wk"].to_numpy()).to_numpy()
    by_week = np.full((len(ids), len(weeks)), np.nan, dtype=np.float32)
    by_week[rows[keep].astype(np.int64), cols[keep]] = prices["sell_price"].to_numpy()[keep]
    day_week = week_index.reindex(calendar["wm_yr_wk"].to_numpy()).to_numpy()
    return by_week[:, day_week]

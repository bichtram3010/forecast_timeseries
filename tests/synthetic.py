"""Small synthetic data in the official M5 file format, for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("CA", "TX")
N_DAYS_SALES = 1941
N_DAYS_CALENDAR = 1969
DEPTS = (("FOODS", "FOODS_1"), ("FOODS", "FOODS_2"), ("HOBBIES", "HOBBIES_1"))


def make_ids(n_items_per_dept: int = 2) -> pd.DataFrame:
    rows = []
    for cat, dept in DEPTS:
        for j in range(1, n_items_per_dept + 1):
            item = f"{dept}_{j:03d}"
            for state in STATES:
                store = f"{state}_1"
                rows.append(
                    {
                        "id": f"{item}_{store}_evaluation",
                        "item_id": item,
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": store,
                        "state_id": state,
                    }
                )
    return pd.DataFrame(rows)


def make_calendar(n_days: int = N_DAYS_CALENDAR) -> pd.DataFrame:
    dates = pd.date_range("2011-01-29", periods=n_days, freq="D")
    wday = ((dates.dayofweek + 2) % 7) + 1  # M5: 1 = Saturday ... 7 = Friday
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + np.arange(n_days) // 7,
            "weekday": dates.day_name(),
            "wday": wday,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, n_days + 1)],
            "event_name_1": None,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
        }
    )
    cal.loc[cal.index % 50 == 0, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    cal.loc[cal.index % 90 == 0, ["event_name_2", "event_type_2"]] = ["Easter", "Cultural"]
    for state in ("CA", "TX", "WI"):
        cal[f"snap_{state}"] = (dates.day <= 10).astype(int)
    return cal


def make_sales(ids: pd.DataFrame, n_days: int = N_DAYS_SALES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    weekly = 1.0 + 0.3 * np.sin(2 * np.pi * t / 7)
    base = rng.uniform(0.5, 4.0, size=(len(ids), 1))
    sales = rng.poisson(base * weekly[None, :]).astype(np.int64)
    sales[0, :35] = 0  # series 0 launches on day 36
    return sales


def make_prices(ids: pd.DataFrame, calendar: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    rows = []
    for k, row in ids.iterrows():
        price = round(float(rng.uniform(1, 10)), 2)
        start = 5 if k == 0 else 0  # series 0 is not on sale in weeks 1-5 (days 1-35)
        for week in weeks[start:]:
            rows.append((row.store_id, row.item_id, int(week), price))
    return pd.DataFrame(rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


def write_m5_files(directory: Path, seed: int = 0) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = make_ids()
    calendar = make_calendar()
    sales = make_sales(ids, seed=seed)
    day_cols = [f"d_{i}" for i in range(1, N_DAYS_SALES + 1)]
    wide = pd.concat([ids, pd.DataFrame(sales, columns=day_cols)], axis=1)
    wide.to_csv(directory / "sales_train_evaluation.csv", index=False)
    calendar.to_csv(directory / "calendar.csv", index=False)
    make_prices(ids, calendar).to_csv(directory / "sell_prices.csv", index=False)
    return directory

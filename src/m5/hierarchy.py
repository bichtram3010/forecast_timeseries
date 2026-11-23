"""Bottom-up aggregation of series to hierarchy levels."""

from __future__ import annotations

import numpy as np
import pandas as pd

WAPE_LEVELS: dict[str, tuple[str, ...]] = {
    "total": (),
    "state": ("state_id",),
    "store": ("store_id",),
    "category": ("cat_id",),
    "department": ("dept_id",),
    "item_store": ("item_id", "store_id"),
}

WRMSSE_LEVELS: dict[str, tuple[str, ...]] = {
    "total": (),
    "state": ("state_id",),
    "store": ("store_id",),
    "category": ("cat_id",),
    "department": ("dept_id",),
    "state_category": ("state_id", "cat_id"),
    "state_department": ("state_id", "dept_id"),
    "store_category": ("store_id", "cat_id"),
    "store_department": ("store_id", "dept_id"),
    "item": ("item_id",),
    "item_state": ("item_id", "state_id"),
    "item_store": ("item_id", "store_id"),
}


def group_codes(ids: pd.DataFrame, by: tuple[str, ...]) -> np.ndarray:
    """Group number 0..G-1 of every series, groups sorted by key."""
    if not by:
        return np.zeros(len(ids), dtype=np.int64)
    return ids.groupby(list(by), observed=True, sort=True).ngroup().to_numpy()


def aggregate(values: np.ndarray, ids: pd.DataFrame, by: tuple[str, ...]) -> np.ndarray:
    """Sum the rows of `values` (one per series) within each group."""
    codes = group_codes(ids, by)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    return np.add.reduceat(values[order].astype(np.float64), starts, axis=0)

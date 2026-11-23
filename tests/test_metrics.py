import numpy as np
import pandas as pd
import pytest

from m5.hierarchy import WAPE_LEVELS, WRMSSE_LEVELS, aggregate
from m5.metrics import WRMSSE, _naive_scale, wape, wape_table


def test_aggregate_levels(m5data):
    vals = np.arange(12 * 3, dtype=np.float64).reshape(12, 3)
    np.testing.assert_allclose(aggregate(vals, m5data.ids, ()), vals.sum(axis=0, keepdims=True))
    stores = m5data.ids["store_id"].astype(str).to_numpy()
    by_store = aggregate(vals, m5data.ids, ("store_id",))
    assert by_store.shape == (2, 3)
    np.testing.assert_allclose(by_store[0], vals[stores == "CA_1"].sum(axis=0))
    assert aggregate(vals, m5data.ids, ("item_id", "store_id")).shape == (12, 3)


def test_level_sizes(m5data):
    sizes = {name: aggregate(np.ones((12, 1)), m5data.ids, by).shape[0] for name, by in WAPE_LEVELS.items()}
    assert sizes == {"total": 1, "state": 2, "store": 2, "category": 2, "department": 3, "item_store": 12}
    assert len(WRMSSE_LEVELS) == 12


def test_wape_simple():
    assert wape(np.array([[1.0, 3.0]]), np.array([[2.0, 1.0]])) == pytest.approx(3 / 4)
    with pytest.raises(ValueError):
        wape(np.zeros((1, 2)), np.ones((1, 2)))


def test_wape_table(m5data):
    actual = m5data.sales[:, 1913:1941]
    table = wape_table(actual, actual, m5data.ids)
    assert list(table.columns) == ["level", "week", "value"]
    assert len(table) == 6 * 5
    assert (table["value"] == 0).all()
    worse = wape_table(actual, actual + 1, m5data.ids)
    assert (worse["value"] > 0).all()


def test_naive_scale_starts_at_first_nonzero():
    np.testing.assert_allclose(_naive_scale(np.array([[0, 0, 2, 4, 4.0]])), [2.0])
    np.testing.assert_allclose(_naive_scale(np.zeros((1, 5))), [0.0])


def test_wrmsse_perfect_is_zero(m5data):
    scorer = WRMSSE(m5data.sales[:, :1913], m5data.prices[:, :1913], m5data.ids)
    actual = m5data.sales[:, 1913:1941]
    assert scorer.score(actual, actual) == 0.0


def _reference_wrmsse(hist, prices, actual, fc, ids):
    total = 0.0
    for by in WRMSSE_LEVELS.values():
        if by:
            key = ids[list(by)].astype(str).agg("|".join, axis=1)
        else:
            key = pd.Series("all", index=ids.index)
        parts = []
        for group in sorted(key.unique()):
            m = (key == group).to_numpy()
            h = hist[m].sum(axis=0)
            nz = np.flatnonzero(h != 0)
            h = h[nz[0]:] if len(nz) else h[:0]
            scale = np.mean(np.diff(h) ** 2) if len(h) > 1 else 0.0
            err = np.mean((actual[m].sum(axis=0) - fc[m].sum(axis=0)) ** 2)
            rmsse = np.sqrt(err / scale) if scale > 0 else 0.0
            parts.append((np.nansum(hist[m, -28:] * prices[m, -28:]), rmsse))
        weight_sum = sum(w for w, _ in parts)
        total += sum(w * r for w, r in parts) / weight_sum
    return total / len(WRMSSE_LEVELS)


def test_wrmsse_matches_reference(m5data):
    origin = 1913
    hist, prices = m5data.sales[:, :origin], m5data.prices[:, :origin]
    actual = m5data.sales[:, origin : origin + 28]
    fc = actual * np.random.default_rng(0).uniform(0.5, 1.5, actual.shape)
    score = WRMSSE(hist, prices, m5data.ids).score(actual, fc)
    assert score == pytest.approx(_reference_wrmsse(hist, prices, actual, fc, m5data.ids), rel=1e-6)


def test_wrmsse_handles_all_zero_history(m5data):
    hist = m5data.sales[:, :400].copy()
    hist[2] = 0.0
    actual = m5data.sales[:, 400:428]
    score = WRMSSE(hist, m5data.prices[:, :400], m5data.ids).score(actual, actual + 1)
    assert np.isfinite(score)

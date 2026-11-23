import numpy as np
import pytest

from m5.forecasts import (
    discover_forecasts,
    forecast_path,
    from_long,
    load_forecast,
    save_forecast,
    to_long,
    zero_when_not_on_sale,
)


def _fc():
    return np.arange(12 * 28, dtype=np.float32).reshape(12, 28)


def test_roundtrip(m5data, tmp_path):
    path = forecast_path(tmp_path, "lgb", "holdout")
    assert path.name == "fc_lgb_holdout.parquet"
    save_forecast(path, _fc(), m5data.ids, origin=1913)
    np.testing.assert_array_equal(load_forecast(path, m5data.ids, origin=1913), _fc())


def test_row_order_does_not_matter(m5data):
    long = to_long(_fc(), m5data.ids, origin=1885)
    shuffled = long.sample(frac=1.0, random_state=0)
    np.testing.assert_array_equal(from_long(shuffled, m5data.ids, origin=1885), _fc())


def test_missing_pairs_raise(m5data):
    long = to_long(_fc(), m5data.ids, origin=1885).iloc[1:]
    with pytest.raises(ValueError, match="missing"):
        from_long(long, m5data.ids, origin=1885)


def test_discover_and_duplicates(m5data, tmp_path):
    save_forecast(forecast_path(tmp_path / "a", "lgb", "validation"), _fc(), m5data.ids, 1885)
    save_forecast(forecast_path(tmp_path / "b", "lstm", "holdout"), _fc(), m5data.ids, 1913)
    assert set(discover_forecasts(tmp_path)) == {("lgb", "validation"), ("lstm", "holdout")}
    save_forecast(forecast_path(tmp_path / "c", "lgb", "validation"), _fc(), m5data.ids, 1885)
    with pytest.raises(ValueError, match="duplicate"):
        discover_forecasts(tmp_path)


def test_zero_when_not_on_sale(m5data):
    fc = np.ones((12, 28), dtype=np.float32)
    out = zero_when_not_on_sale(fc, m5data.prices, origin=20)  # days 21-48
    assert (out[0, :15] == 0).all()  # series 0 has no price on days 21-35
    assert (out[0, 15:] == 1).all()
    assert (out[1:] == 1).all()

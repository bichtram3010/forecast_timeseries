import numpy as np
import pytest

from m5.baselines import moving_average, seasonal_naive
from m5.splits import (
    HOLDOUT_ORIGIN,
    HORIZON,
    SPLIT_ORIGINS,
    VAL_ORIGIN,
    fit_days,
    target_days,
    week_lag,
)


def test_constants_chain():
    assert VAL_ORIGIN + HORIZON == HOLDOUT_ORIGIN
    assert HOLDOUT_ORIGIN + HORIZON == 1941
    assert SPLIT_ORIGINS == {"validation": 1885, "holdout": 1913}


def test_target_days():
    assert target_days(1885).tolist() == list(range(1886, 1914))
    assert target_days(1913, 1).tolist() == list(range(1914, 1921))
    assert target_days(1913, 4).tolist() == list(range(1935, 1942))


def test_week_lag():
    assert [week_lag(k) for k in (1, 2, 3, 4)] == [7, 14, 21, 28]
    for bad in (0, 5):
        with pytest.raises(ValueError):
            week_lag(bad)


def test_fit_days():
    days = fit_days(1885)
    assert days[0] == 1100 and days[-1] == 1885 and len(days) == 786


def test_seasonal_naive_repeats_last_week():
    sales = np.arange(1, 31, dtype=np.float32).reshape(1, 30)  # day d has value d
    fc = seasonal_naive(sales, origin=21, horizon=10)
    assert fc[0].tolist() == [15, 16, 17, 18, 19, 20, 21, 15, 16, 17]


def test_moving_average():
    sales = np.arange(1, 31, dtype=np.float32).reshape(1, 30)
    fc = moving_average(sales, origin=28, window=28, horizon=3)
    np.testing.assert_allclose(fc, 14.5)
    assert fc.shape == (1, 3)

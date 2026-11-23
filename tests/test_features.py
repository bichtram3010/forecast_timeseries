import numpy as np
import pandas as pd
import pytest

from m5.features import (
    CATEGORICAL_FEATURES,
    build_frame,
    feature_names,
    history_feature_names,
    history_features,
    price_features,
    rolling_mean,
    rolling_std,
    shift_right,
)


def test_shift_right():
    a = np.arange(10, dtype=np.float32).reshape(2, 5)
    out = shift_right(a, 2)
    assert np.isnan(out[:, :2]).all()
    np.testing.assert_array_equal(out[:, 2:], a[:, :3])
    np.testing.assert_array_equal(shift_right(a, 0), a)


@pytest.mark.parametrize("w", [1, 3, 7])
def test_rolling_matches_pandas(w):
    a = np.random.default_rng(0).poisson(3, size=(3, 40)).astype(np.float32)
    frame = pd.DataFrame(a.T)
    np.testing.assert_allclose(rolling_mean(a, w), frame.rolling(w).mean().T.to_numpy(), atol=1e-5)
    np.testing.assert_allclose(rolling_std(a, w), frame.rolling(w).std(ddof=0).T.to_numpy(), atol=1e-4)


def test_lag_and_rolling_values(m5data):
    feats = history_features(m5data.sales, week=2)
    t = 500
    assert feats["lag_14"][3, t - 1] == m5data.sales[3, t - 14 - 1]
    assert feats["lag_35"][3, t - 1] == m5data.sales[3, t - 35 - 1]
    expected = m5data.sales[3, t - 14 - 7 : t - 14].mean()
    assert feats["rmean_7"][3, t - 1] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("week", [1, 2, 3, 4])
def test_history_features_ignore_recent_days(m5data, week):
    lag, t = 7 * week, 600
    perturbed = m5data.sales.copy()
    perturbed[:, t - lag :] = 1000.0  # every day after t - lag
    base = history_features(m5data.sales, week)
    pert = history_features(perturbed, week)
    for name in history_feature_names(week):
        np.testing.assert_array_equal(base[name][:, t - 1], pert[name][:, t - 1], err_msg=name)


@pytest.mark.parametrize("week", [1, 2, 3, 4])
def test_lag_boundary_is_tight(m5data, week):
    lag, t = 7 * week, 600
    perturbed = m5data.sales.copy()
    perturbed[:, t - lag - 1] += 1000.0  # day t - lag itself is allowed
    feats = history_features(perturbed, week)
    np.testing.assert_array_equal(feats[f"lag_{lag}"][:, t - 1], perturbed[:, t - lag - 1])


def test_price_features():
    prices = np.array([[np.nan, 2, 2, 2, 2, 2, 2, 2, 3]], dtype=np.float32)
    f = price_features(prices)
    assert np.isnan(f["price_change_1w"][0, 7])  # day 8 vs day 1, which has no price
    assert f["price_change_1w"][0, 8] == pytest.approx(0.5)  # 3 / 2 - 1
    assert f["price_rel_365"][0, 8] == pytest.approx(3 / (17 / 8))  # mean of days 1-9 = 17/8


def test_build_frame_layout(m5data):
    df = build_frame(m5data, week=1, days=np.arange(400, 410))
    assert len(df) == 12 * 10
    assert list(df.columns[:2]) == ["series_idx", "d"]
    assert set(feature_names(1)) <= set(df.columns)
    row = df[(df.series_idx == 5) & (df.d == 405)].iloc[0]
    assert row["sales"] == m5data.sales[5, 404]
    assert row["lag_7"] == m5data.sales[5, 405 - 7 - 1]
    assert row["store_id"] == m5data.ids.loc[5, "store_id"]


def test_build_frame_drops_not_on_sale(m5data):
    df = build_frame(m5data, week=1, days=np.arange(30, 41))
    assert df.loc[df.series_idx == 0, "d"].min() == 36
    assert (df.groupby("series_idx").size().drop(0) == 11).all()


def test_categories_identical_across_date_ranges(m5data):
    a = build_frame(m5data, 1, np.arange(400, 410))
    b = build_frame(m5data, 1, np.arange(1900, 1910))
    for col in CATEGORICAL_FEATURES:
        assert list(a[col].cat.categories) == list(b[col].cat.categories), col


def test_snap_follows_state(m5data):
    df = build_frame(m5data, 1, np.arange(400, 402))
    for _, row in df.iterrows():
        expected = m5data.calendar.loc[int(row["d"]) - 1, f"snap_{row['state_id']}"]
        assert row["snap"] == expected


def test_build_frame_rejects_out_of_range(m5data):
    with pytest.raises(ValueError):
        build_frame(m5data, 1, np.arange(1940, 1945))

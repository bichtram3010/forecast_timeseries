import numpy as np
import pandas as pd
import pytest
from synthetic import make_ids, make_sales, write_m5_files

from m5.data import M5Data, find_data_dir


def test_shapes_and_values(m5data):
    assert m5data.sales.shape == (12, 1941)
    assert m5data.prices.shape == (12, 1941)
    assert len(m5data.calendar) == 1941
    assert m5data.sales.dtype == np.float32
    np.testing.assert_array_equal(m5data.sales, make_sales(make_ids()).astype(np.float32))


def test_prices_missing_before_launch(m5data):
    assert np.isnan(m5data.prices[0, :35]).all()
    assert not np.isnan(m5data.prices[0, 35:]).any()
    assert not np.isnan(m5data.prices[1:]).any()


def test_calendar_columns(m5data):
    cal = m5data.calendar
    expected = [
        "wm_yr_wk", "wday", "month", "year", "mday",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    assert all(c in cal.columns for c in expected)
    assert cal["event_name_1"].dtype == "category"
    assert cal.loc[0, "mday"] == 29


def test_ids_are_categorical(m5data):
    assert all(m5data.ids[c].dtype == "category" for c in m5data.ids.columns)


def test_save_load_roundtrip(m5data, tmp_path):
    m5data.save(tmp_path / "cache")
    back = M5Data.load(tmp_path / "cache")
    np.testing.assert_array_equal(back.sales, m5data.sales)
    np.testing.assert_array_equal(back.prices, m5data.prices)
    pd.testing.assert_frame_equal(back.ids, m5data.ids)
    pd.testing.assert_frame_equal(back.calendar, m5data.calendar)


def test_find_data_dir_searches_nested(tmp_path):
    nested = write_m5_files(tmp_path / "competitions" / "m5-forecasting-accuracy")
    assert find_data_dir(tmp_path) == nested


def test_find_data_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_data_dir(tmp_path)

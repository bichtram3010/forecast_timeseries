import pandas as pd
from synthetic import N_DAYS_CALENDAR, N_DAYS_SALES, make_ids


def test_files_have_official_layout(m5_dir):
    sales = pd.read_csv(m5_dir / "sales_train_evaluation.csv")
    assert list(sales.columns[:6]) == ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    assert sales.columns[-1] == f"d_{N_DAYS_SALES}"
    assert len(sales) == len(make_ids()) == 12
    cal = pd.read_csv(m5_dir / "calendar.csv")
    assert len(cal) == N_DAYS_CALENDAR
    assert cal.loc[0, "wday"] == 1 and cal.loc[0, "weekday"] == "Saturday"
    prices = pd.read_csv(m5_dir / "sell_prices.csv")
    assert set(prices.columns) == {"store_id", "item_id", "wm_yr_wk", "sell_price"}


def test_package_imports():
    import m5

    assert m5.__version__ == "1.0.0"

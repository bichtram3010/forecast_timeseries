import numpy as np
import pytest

from m5.features import build_frame
from m5.gbdt import GBDTConfig, fit, forecast_split, predict
from m5.splits import target_days


@pytest.fixture(scope="module")
def frames(m5data):
    train = build_frame(m5data, 1, np.arange(1500, 1886))
    valid = build_frame(m5data, 1, target_days(1885, 1))
    return train, valid


@pytest.mark.parametrize("kind", ["lgb", "cb"])
def test_fit_with_early_stopping(frames, kind):
    train, valid = frames
    model, best = fit(train, valid, 1, GBDTConfig(kind=kind, max_rounds=30))
    assert 1 <= best <= 30
    pred = predict(model, valid, 1, kind)
    assert pred.shape == (len(valid),)
    assert (pred >= 0).all()
    assert 0.33 < pred.mean() / valid["sales"].mean() < 3


@pytest.mark.parametrize("kind", ["lgb", "cb"])
def test_refit_runs_given_rounds(frames, kind):
    train, _ = frames
    model, rounds = fit(train, None, 1, GBDTConfig(kind=kind), rounds=7)
    assert rounds == 7
    n_trees = model.num_trees() if kind == "lgb" else model.tree_count_
    assert n_trees == 7


def test_unknown_kind_raises(frames):
    with pytest.raises(ValueError):
        fit(frames[0], None, 1, GBDTConfig(kind="xgb"), rounds=1)


def test_forecast_split_zero_when_not_on_sale(m5data):
    train = build_frame(m5data, 1, np.arange(500, 900))
    model, _ = fit(train, None, 1, GBDTConfig(kind="lgb"), rounds=10)
    out = forecast_split({1: model}, m5data, origin=28, kind="lgb")
    assert out.shape == (12, 28)
    assert (out[0, :7] == 0).all()  # days 29-35: series 0 not on sale
    assert (out[1:, :7] > 0).all()
    assert (out[:, 7:] == 0).all()  # weeks without a model stay zero

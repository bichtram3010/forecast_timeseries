import dataclasses

import numpy as np
import pytest
import torch

from m5.lstm import (
    FUTURE_FEATURES,
    HIST,
    LSTMConfig,
    future_covariates,
    model_inputs,
    model_targets,
    predict_lstm,
    train_lstm,
    tweedie_loss,
)

SMALL = LSTMConfig(hidden=8, epochs=2, patience=1, batch_size=8, origins_per_epoch=1)


def test_windows_alignment(m5data):
    cov = future_covariates(m5data)
    idx = np.arange(12)
    hist, fut, scale = model_inputs(m5data.sales, cov, origin=500, series_idx=idx)
    assert hist.shape == (12, HIST) and fut.shape == (12, 28, len(FUTURE_FEATURES))
    np.testing.assert_allclose(hist[:, -1] * scale, m5data.sales[:, 499], rtol=1e-6)  # day 500
    target = model_targets(m5data.sales, 500, idx, scale)
    np.testing.assert_allclose(target[:, 0] * scale, m5data.sales[:, 500], rtol=1e-6)  # day 501


def test_future_covariates_values(m5data):
    cov = future_covariates(m5data)
    assert cov.shape == (12, 1941, len(FUTURE_FEATURES))
    assert cov[0, 0, 0] == pytest.approx(np.sin(2 * np.pi * 1 / 7))  # day 1 is wday 1
    state = str(m5data.ids.loc[3, "state_id"])
    np.testing.assert_array_equal(cov[3, :, 4], m5data.calendar[f"snap_{state}"].to_numpy())


def test_tweedie_loss_prefers_the_target():
    target = torch.tensor([[1.0, 2.0, 3.0]])
    assert tweedie_loss(target, target) < tweedie_loss(target * 2, target)
    assert tweedie_loss(target, target) < tweedie_loss(target * 0.5, target)


def test_train_and_predict_smoke(m5data):
    model, best_epoch = train_lstm(m5data, SMALL, fit_origin=1885, valid_origin=1885, train_start=1700)
    assert best_epoch in (1, 2)
    fc = predict_lstm(model, m5data, 1885, SMALL)
    assert fc.shape == (12, 28) and np.isfinite(fc).all() and (fc >= 0).all()


def test_refit_runs_fixed_epochs(m5data):
    _, n = train_lstm(m5data, SMALL, fit_origin=1913, epochs=1, train_start=1800)
    assert n == 1


def test_not_on_sale_gets_zero(m5data):
    data = dataclasses.replace(m5data, prices=m5data.prices.copy())
    data.prices[0, 1885:1913] = np.nan  # series 0 off sale on days 1886-1913
    model, _ = train_lstm(data, SMALL, fit_origin=1913, epochs=1, train_start=1800)
    fc = predict_lstm(model, data, 1885, SMALL)
    assert (fc[0] == 0).all() and (fc[1:] > 0).all()


def test_too_short_training_range_raises(m5data):
    with pytest.raises(ValueError):
        train_lstm(m5data, SMALL, fit_origin=1200, epochs=1, train_start=1190)


def test_training_ignores_sales_after_fit_origin(m5data):
    cfg = LSTMConfig(hidden=8, epochs=1, patience=1, batch_size=8, origins_per_epoch=1)
    sales = m5data.sales.copy()
    sales[:, 1913:] = 1000
    altered = dataclasses.replace(m5data, sales=sales)
    model_a, _ = train_lstm(m5data, cfg, fit_origin=1913, epochs=1, train_start=1800)
    model_b, _ = train_lstm(altered, cfg, fit_origin=1913, epochs=1, train_start=1800)
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), key

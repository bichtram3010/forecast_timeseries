"""One global LSTM: 112 days of history in, 28 days out."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import M5Data
from .features import price_features, snap_matrix
from .forecasts import zero_when_not_on_sale
from .splits import HOLDOUT_ORIGIN, HORIZON, TRAIN_START, VAL_ORIGIN

HIST = 112
FUTURE_FEATURES = (
    "wday_sin", "wday_cos", "month_sin", "month_cos", "snap", "has_event", "price_rel_365",
)
TWEEDIE_POWER = 1.1


@dataclass
class LSTMConfig:
    hidden: int = 128
    epochs: int = 20
    patience: int = 3
    batch_size: int = 1024
    lr: float = 1e-3
    origins_per_epoch: int = 16
    device: str = "cpu"
    seed: int = 42


def future_covariates(data: M5Data) -> np.ndarray:
    """Known-in-advance inputs per series and day: (n_series, n_days, 7)."""
    cal = data.calendar
    wday = cal["wday"].to_numpy(np.float32)
    month = cal["month"].to_numpy(np.float32)
    out = np.empty((data.n_series, data.n_days, len(FUTURE_FEATURES)), dtype=np.float32)
    out[:, :, 0] = np.sin(2 * np.pi * wday / 7)[None, :]
    out[:, :, 1] = np.cos(2 * np.pi * wday / 7)[None, :]
    out[:, :, 2] = np.sin(2 * np.pi * month / 12)[None, :]
    out[:, :, 3] = np.cos(2 * np.pi * month / 12)[None, :]
    out[:, :, 4] = snap_matrix(data.ids, cal)
    out[:, :, 5] = cal["event_name_1"].notna().to_numpy(np.float32)[None, :]
    out[:, :, 6] = np.nan_to_num(price_features(data.prices)["price_rel_365"], nan=1.0)
    return out


def model_inputs(sales: np.ndarray, cov: np.ndarray, origin: int, series_idx: np.ndarray):
    """History of days origin-111..origin scaled by its mean + 1; covariates of the next 28 days."""
    hist = sales[series_idx, origin - HIST : origin].astype(np.float32)
    scale = (hist.mean(axis=1) + 1.0).astype(np.float32)
    fut = cov[series_idx, origin : origin + HORIZON]
    return hist / scale[:, None], fut, scale


def model_targets(sales: np.ndarray, origin: int, series_idx: np.ndarray, scale: np.ndarray):
    return sales[series_idx, origin : origin + HORIZON] / scale[:, None]


class Seq2Seq(nn.Module):
    def __init__(self, n_items: int, n_depts: int, n_stores: int, hidden: int = 128):
        super().__init__()
        self.item = nn.Embedding(n_items, 16)
        self.dept = nn.Embedding(n_depts, 4)
        self.store = nn.Embedding(n_stores, 4)
        self.encoder = nn.LSTM(1, hidden, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + 24 + HORIZON * len(FUTURE_FEATURES), 256),
            nn.ReLU(),
            nn.Linear(256, HORIZON),
        )

    def forward(self, hist, fut, item, dept, store):
        _, (h, _) = self.encoder(hist.unsqueeze(-1))
        z = torch.cat(
            [h[-1], self.item(item), self.dept(dept), self.store(store), fut.flatten(1)], dim=1
        )
        return nn.functional.softplus(self.head(z))


def tweedie_loss(pred: torch.Tensor, target: torch.Tensor, power: float = TWEEDIE_POWER):
    pred = pred.clamp_min(1e-6)
    return (
        -target * pred.pow(1 - power) / (1 - power) + pred.pow(2 - power) / (2 - power)
    ).mean()


def _tensor(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)


def _id_codes(data: M5Data, device: str):
    return tuple(
        torch.as_tensor(data.ids[c].cat.codes.to_numpy(np.int64), device=device)
        for c in ("item_id", "dept_id", "store_id")
    )


def _new_model(data: M5Data, cfg: LSTMConfig) -> Seq2Seq:
    sizes = [len(data.ids[c].cat.categories) for c in ("item_id", "dept_id", "store_id")]
    return Seq2Seq(*sizes, hidden=cfg.hidden).to(cfg.device)


def _batches(n: int, size: int, rng: np.random.Generator | None = None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield idx[start : start + size]


def _forward(model, hist, fut, codes, sel, device):
    sel_t = torch.as_tensor(sel, device=device)
    item, dept, store = (c[sel_t] for c in codes)
    return model(_tensor(hist[sel], device), _tensor(fut[sel], device), item, dept, store)


def _loss_at(model, data, cov, origin, codes, cfg) -> float:
    model.eval()
    idx = np.arange(data.n_series)
    hist, fut, scale = model_inputs(data.sales, cov, origin, idx)
    target = model_targets(data.sales, origin, idx, scale)
    total = 0.0
    with torch.no_grad():
        for sel in _batches(data.n_series, cfg.batch_size):
            pred = _forward(model, hist, fut, codes, sel, cfg.device)
            total += float(tweedie_loss(pred, _tensor(target[sel], cfg.device))) * len(sel)
    return total / data.n_series


def train_lstm(
    data: M5Data,
    cfg: LSTMConfig,
    fit_origin: int,
    valid_origin: int | None = None,
    epochs: int | None = None,
    train_start: int = TRAIN_START,
    cov: np.ndarray | None = None,
) -> tuple[Seq2Seq, int]:
    """Train on origins whose 28 targets end by `fit_origin`. With `valid_origin`, keep the
    best epoch by validation loss (early stopping). Returns the model and its epoch count."""
    low, high = max(train_start - 1, HIST), fit_origin - HORIZON
    if high < low:
        raise ValueError("no training origins: fit_origin is too close to train_start")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    cov = future_covariates(data) if cov is None else cov
    codes = _id_codes(data, cfg.device)
    model = _new_model(data, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_epochs = cfg.epochs if epochs is None else epochs
    all_idx = np.arange(data.n_series)
    best_loss, best_epoch, best_state, bad = np.inf, 0, None, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for origin in rng.integers(low, high + 1, size=cfg.origins_per_epoch):
            hist, fut, scale = model_inputs(data.sales, cov, int(origin), all_idx)
            target = model_targets(data.sales, int(origin), all_idx, scale)
            for sel in _batches(data.n_series, cfg.batch_size, rng):
                pred = _forward(model, hist, fut, codes, sel, cfg.device)
                loss = tweedie_loss(pred, _tensor(target[sel], cfg.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        if valid_origin is None:
            continue
        vloss = _loss_at(model, data, cov, valid_origin, codes, cfg)
        print(f"[lstm] epoch {epoch}: validation loss {vloss:.5f}", flush=True)
        if vloss < best_loss:
            best_loss, best_epoch, best_state, bad = vloss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if valid_origin is None:
        return model, n_epochs
    model.load_state_dict(best_state)
    return model, best_epoch


def predict_lstm(
    model: Seq2Seq, data: M5Data, origin: int, cfg: LSTMConfig, cov: np.ndarray | None = None
) -> np.ndarray:
    cov = future_covariates(data) if cov is None else cov
    codes = _id_codes(data, cfg.device)
    idx = np.arange(data.n_series)
    hist, fut, scale = model_inputs(data.sales, cov, origin, idx)
    out = np.empty((data.n_series, HORIZON), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for sel in _batches(data.n_series, cfg.batch_size):
            out[sel] = _forward(model, hist, fut, codes, sel, cfg.device).cpu().numpy()
    return zero_when_not_on_sale(out * scale[:, None], data.prices, origin)


def run_lstm(
    data: M5Data, cfg: LSTMConfig, train_start: int = TRAIN_START
) -> tuple[dict[str, np.ndarray], int]:
    cov = future_covariates(data)
    model, best_epoch = train_lstm(
        data, cfg, VAL_ORIGIN, valid_origin=VAL_ORIGIN, train_start=train_start, cov=cov
    )
    validation = predict_lstm(model, data, VAL_ORIGIN, cfg, cov)
    model, _ = train_lstm(
        data, cfg, HOLDOUT_ORIGIN, epochs=best_epoch, train_start=train_start, cov=cov
    )
    holdout = predict_lstm(model, data, HOLDOUT_ORIGIN, cfg, cov)
    return {"validation": validation, "holdout": holdout}, best_epoch

"""Four weekly LightGBM or CatBoost models (week k uses lag >= 7k)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import M5Data
from .features import CATEGORICAL_FEATURES, build_frame, feature_names
from .splits import (
    HOLDOUT_ORIGIN,
    HORIZON,
    N_WEEKS,
    TRAIN_START,
    VAL_ORIGIN,
    fit_days,
    target_days,
)

LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "max_bin": 255,
    "verbose": -1,
}
CB_PARAMS = {"learning_rate": 0.05, "depth": 8, "verbose": 200, "allow_writing_files": False}
EARLY_STOPPING_ROUNDS = 100


@dataclass
class GBDTConfig:
    kind: str  # "lgb" or "cb"
    max_rounds: int = 3000
    sample_frac: float = 1.0
    device: str = "cpu"  # "cpu" or "gpu"; only CatBoost uses the GPU
    seed: int = 42
    cb_loss: str = "Tweedie:variance_power=1.1"


def _xy(frame: pd.DataFrame, week: int) -> tuple[pd.DataFrame, np.ndarray]:
    return frame[feature_names(week)], frame["sales"].to_numpy(np.float32)


def _cat_codes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].cat.codes.astype(np.int32)
    return X


def _pool(X: pd.DataFrame, y: np.ndarray | None = None):
    from catboost import Pool

    return Pool(_cat_codes(X), y, cat_features=list(CATEGORICAL_FEATURES))


def fit(
    frame_train: pd.DataFrame,
    frame_valid: pd.DataFrame | None,
    week: int,
    cfg: GBDTConfig,
    rounds: int | None = None,
) -> tuple[object, int]:
    """With `frame_valid`: early stopping, returns the best iteration count.
    Without it: trains exactly `rounds` iterations."""
    if cfg.kind not in ("lgb", "cb"):
        raise ValueError(f"unknown model kind {cfg.kind!r}")
    if frame_valid is None and rounds is None:
        raise ValueError("rounds is required when there is no validation frame")
    if cfg.sample_frac < 1.0:
        frame_train = frame_train.sample(frac=cfg.sample_frac, random_state=cfg.seed)
    X, y = _xy(frame_train, week)
    valid = _xy(frame_valid, week) if frame_valid is not None else None
    if cfg.kind == "lgb":
        return _fit_lgb(X, y, valid, cfg, rounds)
    return _fit_cb(X, y, valid, cfg, rounds)


def _fit_lgb(X, y, valid, cfg: GBDTConfig, rounds):
    import lightgbm as lgb

    params = {**LGB_PARAMS, "seed": cfg.seed}
    train_set = lgb.Dataset(X, y, categorical_feature=list(CATEGORICAL_FEATURES))
    if valid is None:
        model = lgb.train(params, train_set, num_boost_round=rounds)
        return model, rounds
    valid_set = lgb.Dataset(valid[0], valid[1], reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.max_rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(100)],
    )
    return model, int(model.best_iteration)


def _fit_cb(X, y, valid, cfg: GBDTConfig, rounds):
    from catboost import CatBoostRegressor

    params = {
        **CB_PARAMS,
        "loss_function": cfg.cb_loss,
        "random_seed": cfg.seed,
        "task_type": "GPU" if cfg.device == "gpu" else "CPU",
    }
    if valid is None:
        model = CatBoostRegressor(iterations=rounds, **params)
        model.fit(_pool(X, y))
        return model, rounds
    model = CatBoostRegressor(
        iterations=cfg.max_rounds,
        od_type="Iter",
        od_wait=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
        **params,
    )
    model.fit(_pool(X, y), eval_set=_pool(*valid))
    return model, int(model.get_best_iteration()) + 1


def predict(model, frame: pd.DataFrame, week: int, kind: str) -> np.ndarray:
    X = frame[feature_names(week)]
    if kind == "lgb":
        pred = model.predict(X)
    else:
        pred = model.predict(_pool(X), prediction_type="Exponent")
    return np.clip(pred, 0, None).astype(np.float32)


def forecast_split(models: dict[int, object], data: M5Data, origin: int, kind: str) -> np.ndarray:
    """28-day forecast from `origin`. Weeks without a model and items not on sale stay 0."""
    out = np.zeros((data.n_series, HORIZON), dtype=np.float32)
    for week, model in models.items():
        frame = build_frame(data, week, target_days(origin, week))
        cols = frame["d"].to_numpy().astype(np.int64) - origin - 1
        out[frame["series_idx"].to_numpy(), cols] = predict(model, frame, week, kind)
    return out


def run_gbdt(
    data: M5Data, cfg: GBDTConfig, train_start: int = TRAIN_START
) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    """Per week: fit with early stopping (origin 1885), forecast validation,
    refit on data up to 1913 with the same iteration count, forecast holdout."""
    forecasts = {
        "validation": np.zeros((data.n_series, HORIZON), dtype=np.float32),
        "holdout": np.zeros((data.n_series, HORIZON), dtype=np.float32),
    }
    best: dict[int, int] = {}
    for week in range(1, N_WEEKS + 1):
        train = build_frame(data, week, fit_days(VAL_ORIGIN, train_start))
        valid = build_frame(data, week, target_days(VAL_ORIGIN, week))
        model, best[week] = fit(train, valid, week, cfg)
        del train, valid
        forecasts["validation"] += forecast_split({week: model}, data, VAL_ORIGIN, cfg.kind)
        refit = build_frame(data, week, fit_days(HOLDOUT_ORIGIN, train_start))
        model, _ = fit(refit, None, week, cfg, rounds=best[week])
        del refit
        forecasts["holdout"] += forecast_split({week: model}, data, HOLDOUT_ORIGIN, cfg.kind)
        print(f"[{cfg.kind}] week {week}: best iteration {best[week]}", flush=True)
    return forecasts, best

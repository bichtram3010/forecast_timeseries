import re

import numpy as np
import pytest

from m5.evaluate import evaluate_all
from m5.provenance import run_metadata
from m5.splits import SPLIT_ORIGINS


def _forecasts(m5data):
    fcs = {}
    for split, origin in SPLIT_ORIGINS.items():
        actual = m5data.sales[:, origin : origin + 28]
        fcs[("lgb", split)] = actual.copy()
        fcs[("cb", split)] = actual * 1.5
        fcs[("lstm", split)] = actual * 2.0
    return fcs


def test_evaluate_all(m5data):
    scores, weights, total = evaluate_all(m5data, _forecasts(m5data))
    assert weights == {"cb": 0.0, "lgb": 1.0, "lstm": 0.0}
    assert set(scores["model"]) == {"seasonal_naive", "ma28", "lgb", "cb", "lstm", "ensemble"}
    assert list(scores.columns) == ["split", "model", "metric", "level", "week", "value"]
    hold = scores[(scores.split == "holdout") & (scores.metric == "wrmsse")].set_index("model")["value"]
    assert hold["lgb"] == 0.0 and hold["ensemble"] == 0.0
    assert hold["seasonal_naive"] > 0 and hold["cb"] > 0
    per_model = scores.groupby(["split", "model"]).size()
    assert (per_model == 1 + 6 * 5).all()  # 1 WRMSSE + 6 levels x 5 weeks
    assert len(total) == 28 and (total["lgb"] == total["actual"]).all()


def test_evaluate_all_requires_both_splits(m5data):
    with pytest.raises(ValueError, match="missing"):
        evaluate_all(m5data, {("lgb", "validation"): np.zeros((12, 28), np.float32)})


def test_run_metadata():
    meta = run_metadata({"model": "lgb"})
    assert {"package_version", "commit", "python", "utc_time", "libraries", "model"} <= set(meta)
    assert meta["commit"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", meta["commit"])
    assert meta["libraries"]["numpy"]

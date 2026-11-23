import json

import pandas as pd
import pytest

from m5.evaluate import evaluate_all
from m5.report import (
    RESULTS_END,
    RESULTS_START,
    level_table,
    make_figures,
    render_results,
    update_readme,
)
from m5.splits import SPLIT_ORIGINS


@pytest.fixture()
def results_dir(m5data, tmp_path):
    fcs = {}
    for split, origin in SPLIT_ORIGINS.items():
        actual = m5data.sales[:, origin : origin + 28]
        fcs[("lgb", split)] = actual * 1.1
        fcs[("cb", split)] = actual * 0.8
        fcs[("lstm", split)] = actual * 1.3
    scores, weights, total = evaluate_all(m5data, fcs)
    out = tmp_path / "results"
    out.mkdir()
    scores.to_csv(out / "scores.csv", index=False)
    total.to_csv(out / "total_holdout.csv", index=False)
    (out / "ensemble_weights.json").write_text(json.dumps(weights))
    (out / "metadata.json").write_text(json.dumps({"evaluate": {"commit": "abcdef1234567"}}))
    return out


def test_level_table(results_dir):
    table = level_table(pd.read_csv(results_dir / "scores.csv"), "holdout")
    lines = table.splitlines()
    assert lines[0].startswith("| Model | WRMSSE | WAPE Total")
    assert len(lines) == 2 + 6  # header, separator, 6 models
    assert "Seasonal naive" in table and "%" in table


def test_render_results(results_dir):
    block = render_results(results_dir)
    assert "### Holdout" in block and "### Validation" in block
    assert "Ensemble weights" in block and "abcdef1" in block


def test_update_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\nbefore\n{RESULTS_START}\nold\n{RESULTS_END}\nafter\n", encoding="utf-8")
    update_readme(readme, "NEW BLOCK")
    text = readme.read_text(encoding="utf-8")
    assert "NEW BLOCK" in text and "old" not in text
    assert text.startswith("# T\nbefore\n") and text.endswith("after\n")


def test_update_readme_requires_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# no markers\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme(readme, "x")


def test_make_figures(results_dir, tmp_path):
    paths = make_figures(results_dir, tmp_path / "fig")
    assert sorted(p.name for p in paths) == ["total_holdout.png", "wape_by_level.png", "wape_by_week.png"]
    assert all(p.stat().st_size > 1000 for p in paths)

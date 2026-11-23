import json

import pandas as pd
import pytest
from synthetic import write_m5_files

from m5.cli import main
from m5.report import RESULTS_END, RESULTS_START


def test_full_pipeline(tmp_path):
    write_m5_files(tmp_path / "raw" / "nested")
    prep, fc, res = tmp_path / "prep", tmp_path / "fc", tmp_path / "results"
    main(["prepare", "--data-dir", str(tmp_path / "raw"), "--out", str(prep)])
    main(["train-gbdt", "--data", str(prep), "--model", "lgb", "--out", str(fc / "lgb"),
          "--max-rounds", "10"])
    main(["train-gbdt", "--data", str(prep), "--model", "cb", "--out", str(fc / "cb"),
          "--max-rounds", "10", "--device", "cpu"])
    main(["train-lstm", "--data", str(prep), "--out", str(fc / "lstm"), "--epochs", "1",
          "--origins-per-epoch", "1", "--hidden", "8", "--batch-size", "8"])
    main(["evaluate", "--data", str(prep), "--forecasts", str(fc), "--results", str(res)])

    scores = pd.read_csv(res / "scores.csv")
    assert set(scores["model"]) == {"seasonal_naive", "ma28", "lgb", "cb", "lstm", "ensemble"}
    assert scores["value"].notna().all()
    meta = json.loads((res / "metadata.json").read_text())
    assert set(meta["models"]) == {"lgb", "cb", "lstm"}
    assert "best_iterations" in meta["models"]["lgb"]
    assert "best_epoch" in meta["models"]["lstm"]

    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n{RESULTS_START}\nSTALE-PLACEHOLDER\n{RESULTS_END}\n", encoding="utf-8")
    main(["report", "--results", str(res), "--readme", str(readme), "--figures", str(tmp_path / "fig")])
    text = readme.read_text(encoding="utf-8")
    assert "### Holdout" in text and "STALE-PLACEHOLDER" not in text
    assert len(list((tmp_path / "fig").glob("*.png"))) == 3


def _fc_content(folder):
    return {
        p.name: pd.read_parquet(p)
        for p in sorted(folder.glob("fc_lgb_*.parquet"))
    }


def test_train_gbdt_resumes_across_two_runs(tmp_path, capsys):
    write_m5_files(tmp_path / "raw")
    prep = tmp_path / "prep"
    main(["prepare", "--data-dir", str(tmp_path / "raw"), "--out", str(prep)])
    base = ["train-gbdt", "--data", str(prep), "--model", "lgb", "--max-rounds", "5"]

    main([*base, "--out", str(tmp_path / "s1"), "--weeks", "1", "2"])
    assert sorted(p.name for p in (tmp_path / "s1" / "parts").glob("*.npz")) == [
        "lgb_w1.npz", "lgb_w2.npz"]
    assert not list((tmp_path / "s1").glob("fc_*")) and not list((tmp_path / "s1").glob("meta_*"))
    assert "still missing: [3, 4]" in capsys.readouterr().out

    main([*base, "--out", str(tmp_path / "s2"), "--weeks", "3", "4",
          "--resume-from", str(tmp_path / "s1")])
    meta = json.loads((tmp_path / "s2" / "meta_lgb.json").read_text())
    assert sorted(meta["best_iterations"]) == ["1", "2", "3", "4"]

    main([*base, "--out", str(tmp_path / "full")])
    resumed, full = _fc_content(tmp_path / "s2"), _fc_content(tmp_path / "full")
    assert set(resumed) == set(full) and len(full) == 2
    for name in full:
        pd.testing.assert_frame_equal(resumed[name], full[name])


def test_resume_from_two_dirs_with_same_week_raises(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "lgb_w1.npz").write_bytes(b"x")
    prep = tmp_path / "prep"
    write_m5_files(tmp_path / "raw")
    main(["prepare", "--data-dir", str(tmp_path / "raw"), "--out", str(prep)])
    with pytest.raises(ValueError):
        main(["train-gbdt", "--data", str(prep), "--model", "lgb", "--out", str(tmp_path / "o"),
              "--max-rounds", "5", "--weeks", "2", "--resume-from", str(tmp_path / "a"),
              str(tmp_path / "b")])

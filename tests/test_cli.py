import json

import pandas as pd
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

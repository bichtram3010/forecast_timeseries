import json
import runpy
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle"
EXPECTED = {
    "01_lightgbm.ipynb": "m5 train-gbdt --data /kaggle/working/m5data --model lgb",
    "02_catboost.ipynb": "m5 train-gbdt --data /kaggle/working/m5data --model cb",
    "03_lstm.ipynb": "m5 train-lstm --data /kaggle/working/m5data",
    "04_evaluate.ipynb": "m5 evaluate --data /kaggle/working/m5data --forecasts /kaggle/input",
}


def test_generated_notebooks_are_current():
    runpy.run_path(str(NB_DIR / "make_notebooks.py"), run_name="__main__")
    for name, command in EXPECTED.items():
        nb = json.loads((NB_DIR / name).read_text(encoding="utf-8"))
        source = "".join("".join(c["source"]) for c in nb["cells"])
        assert 'M5_REF = "' in source
        assert "pip install -q --no-deps" in source
        assert "m5 prepare --data-dir /kaggle/input --out /kaggle/working/m5data" in source
        assert command in source
        assert nb["nbformat"] == 4

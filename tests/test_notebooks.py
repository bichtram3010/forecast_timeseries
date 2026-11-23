import importlib.util
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle"
EXPECTED = {
    "01_lightgbm.ipynb": "m5 train-gbdt --data /kaggle/working/m5data --model lgb",
    "02_catboost.ipynb": "m5 train-gbdt --data /kaggle/working/m5data --model cb",
    "03_lstm.ipynb": "m5 train-lstm --data /kaggle/working/m5data",
    "04_evaluate.ipynb": "m5 evaluate --data /kaggle/working/m5data --forecasts /kaggle/input",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("make_notebooks", NB_DIR / "make_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_committed_notebooks_match_the_generator(tmp_path):
    _load_generator().main(out_dir=tmp_path)
    for name, command in EXPECTED.items():
        assert (tmp_path / name).read_bytes() == (NB_DIR / name).read_bytes(), f"{name} is stale"
        source = _source(tmp_path / name)
        assert 'M5_REF = "' in source
        assert "pip install -q --no-deps" in source
        assert "m5 prepare --data-dir /kaggle/input --out /kaggle/working/m5data" in source
        assert command in source
    assert "--weeks" in _source(tmp_path / "01_lightgbm.ipynb")
    assert "--weeks" in _source(tmp_path / "02_catboost.ipynb")
    assert "--device gpu" in _source(tmp_path / "02_catboost.ipynb")
    assert "--device cuda" in _source(tmp_path / "03_lstm.ipynb")

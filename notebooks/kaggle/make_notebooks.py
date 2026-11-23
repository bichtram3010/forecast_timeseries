"""Write the four thin Kaggle notebooks. Run: python notebooks/kaggle/make_notebooks.py"""

import json
from pathlib import Path

REPO = "https://github.com/bichtram3010/forecast_timeseries"
HERE = Path(__file__).resolve().parent

INSTALL = [
    "# Set M5_REF to the exact commit SHA being run, so every result is traceable.\n",
    'M5_REF = "rebuild"\n',
    f'!pip install -q --no-deps "git+{REPO}@{{M5_REF}}"\n',
    "!m5 --help",
]
PREPARE = ["!m5 prepare --data-dir /kaggle/input --out /kaggle/working/m5data"]
CLEANUP = ["!rm -rf /kaggle/working/m5data"]

NOTEBOOKS = {
    "01_lightgbm.ipynb": (
        "M5 rebuild 1/4: LightGBM",
        "Settings: Accelerator None, Internet on. Input: the m5-forecasting-accuracy competition.",
        [
            INSTALL,
            PREPARE,
            ["!m5 train-gbdt --data /kaggle/working/m5data --model lgb --out /kaggle/working/forecasts"],
            CLEANUP,
        ],
    ),
    "02_catboost.ipynb": (
        "M5 rebuild 2/4: CatBoost",
        "Settings: Accelerator GPU T4, Internet on. Input: the m5-forecasting-accuracy competition.",
        [
            INSTALL,
            PREPARE,
            ["!m5 train-gbdt --data /kaggle/working/m5data --model cb --device gpu "
             "--out /kaggle/working/forecasts"],
            CLEANUP,
        ],
    ),
    "03_lstm.ipynb": (
        "M5 rebuild 3/4: LSTM",
        "Settings: Accelerator GPU T4, Internet on. Input: the m5-forecasting-accuracy competition.",
        [
            INSTALL,
            PREPARE,
            ["!m5 train-lstm --data /kaggle/working/m5data --device cuda --out /kaggle/working/forecasts"],
            CLEANUP,
        ],
    ),
    "04_evaluate.ipynb": (
        "M5 rebuild 4/4: evaluate",
        "Settings: Accelerator None, Internet on. Inputs: the competition and the outputs of "
        "notebooks 1 to 3.",
        [
            INSTALL,
            PREPARE,
            ["!m5 evaluate --data /kaggle/working/m5data --forecasts /kaggle/input "
             "--results /kaggle/working/results"],
            CLEANUP,
            ["!ls -la /kaggle/working/results"],
        ],
    ),
}


def notebook(title: str, intro: str, code_cells: list[list[str]]) -> dict:
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", "\n", intro]}]
    for source in code_cells:
        cells.append(
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": source}
        )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for name, (title, intro, cells) in NOTEBOOKS.items():
        path = HERE / name
        path.write_text(json.dumps(notebook(title, intro, cells), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

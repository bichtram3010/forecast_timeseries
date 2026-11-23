"""Command line entry point: m5 prepare | train-gbdt | train-lstm | evaluate | report."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .data import M5Data, find_data_dir, load_m5
from .evaluate import evaluate_all
from .forecasts import discover_forecasts, forecast_path, load_forecast, save_forecast
from .gbdt import GBDTConfig, completed_weeks, run_gbdt
from .provenance import run_metadata
from .report import make_figures, render_results, update_readme
from .splits import SPLIT_ORIGINS


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _save_all(out: Path, model: str, forecasts: dict, data: M5Data) -> None:
    for split, origin in SPLIT_ORIGINS.items():
        save_forecast(forecast_path(out, model, split), forecasts[split], data.ids, origin)


def cmd_prepare(args: argparse.Namespace) -> None:
    source = find_data_dir(args.data_dir)
    print(f"reading M5 files from {source}", flush=True)
    data = load_m5(source)
    data.save(args.out)
    print(f"saved {data.n_series} series x {data.n_days} days to {args.out}")


def _collect_parts(resume_dirs: list[Path], parts: Path, kind: str) -> None:
    """Copy part files of earlier sessions into `parts`; never overwrite, reject duplicates."""
    found: dict[str, Path] = {}
    for folder in resume_dirs:
        for path in sorted(Path(folder).rglob(f"{kind}_w[1-4].npz")):
            if path.name in found:
                raise ValueError(f"{path.name} found twice: {found[path.name]} and {path}")
            found[path.name] = path
    parts.mkdir(parents=True, exist_ok=True)
    for name, path in found.items():
        if not (parts / name).exists():
            shutil.copy2(path, parts / name)


def cmd_train_gbdt(args: argparse.Namespace) -> None:
    data = M5Data.load(args.data)
    cfg = GBDTConfig(
        kind=args.model,
        max_rounds=args.max_rounds,
        sample_frac=args.sample_frac,
        device=args.device,
        cb_loss=args.cb_loss,
    )
    parts = args.out / "parts"
    _collect_parts(args.resume_from or [], parts, args.model)
    forecasts, best = run_gbdt(data, cfg, weeks=tuple(args.weeks), parts_dir=parts)
    done = completed_weeks(parts, args.model)
    if done != [1, 2, 3, 4]:
        missing = [w for w in (1, 2, 3, 4) if w not in done]
        print(
            f"weeks done: {done}; still missing: {missing}; "
            f"rerun with --resume-from {args.out} --weeks {' '.join(map(str, missing))}"
        )
        return
    _save_all(args.out, args.model, forecasts, data)
    meta = run_metadata({"model": args.model, "config": asdict(cfg), "best_iterations": best})
    _write_json(args.out / f"meta_{args.model}.json", meta)


def cmd_train_lstm(args: argparse.Namespace) -> None:
    from .lstm import LSTMConfig, run_lstm

    data = M5Data.load(args.data)
    cfg = LSTMConfig(
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        origins_per_epoch=args.origins_per_epoch,
        device=args.device,
    )
    forecasts, best_epoch = run_lstm(data, cfg)
    _save_all(args.out, "lstm", forecasts, data)
    meta = run_metadata({"model": "lstm", "config": asdict(cfg), "best_epoch": best_epoch})
    _write_json(args.out / "meta_lstm.json", meta)


def cmd_evaluate(args: argparse.Namespace) -> None:
    data = M5Data.load(args.data)
    paths = discover_forecasts(args.forecasts)
    forecasts = {
        key: load_forecast(path, data.ids, SPLIT_ORIGINS[key[1]]) for key, path in paths.items()
    }
    scores, weights, total = evaluate_all(data, forecasts)
    args.results.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.results / "scores.csv", index=False)
    total.to_csv(args.results / "total_holdout.csv", index=False)
    _write_json(args.results / "ensemble_weights.json", weights)
    models = {
        p.stem.removeprefix("meta_"): json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(args.forecasts).rglob("meta_*.json"))
    }
    evaluate_meta = run_metadata()
    commits = {name: meta.get("commit") for name, meta in models.items()}
    consistent = set(commits.values()) <= {evaluate_meta["commit"]}
    if not consistent:
        print(f"WARNING: runs used different commits: {commits} (evaluate: {evaluate_meta['commit']})")
    _write_json(
        args.results / "metadata.json",
        {"evaluate": evaluate_meta, "models": models, "commits_consistent": consistent},
    )
    print(scores[scores["metric"] == "wrmsse"].to_string(index=False))


def cmd_report(args: argparse.Namespace) -> None:
    update_readme(args.readme, render_results(args.results))
    for path in make_figures(args.results, args.figures):
        print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m5", description="M5 forecasting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="read the official M5 files and cache them")
    p.add_argument("--data-dir", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train-gbdt", help="train 4 weekly LightGBM or CatBoost models")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", choices=["lgb", "cb"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--cb-loss", default="Tweedie:variance_power=1.1")
    p.add_argument("--weeks", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    p.add_argument("--resume-from", nargs="*", type=Path, default=None)
    p.set_defaults(func=cmd_train_gbdt)

    p = sub.add_parser("train-lstm", help="train the LSTM")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--origins-per-epoch", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.set_defaults(func=cmd_train_lstm)

    p = sub.add_parser("evaluate", help="score all forecasts found under --forecasts")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--forecasts", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("report", help="write README tables and figures from results/")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--readme", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

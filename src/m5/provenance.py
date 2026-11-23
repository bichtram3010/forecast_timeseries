"""Where a result came from: code commit, library versions, time."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

LIBRARIES = ("numpy", "pandas", "pyarrow", "lightgbm", "catboost", "torch")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _commit() -> str:
    try:
        direct = importlib.metadata.distribution("m5-forecast").read_text("direct_url.json")
        if direct:
            commit = json.loads(direct).get("vcs_info", {}).get("commit_id")
            if commit:
                return commit
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_metadata(extra: dict | None = None) -> dict:
    meta = {
        "package_version": __version__,
        "commit": _commit(),
        "python": platform.python_version(),
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "libraries": {name: _version(name) for name in LIBRARIES},
    }
    if extra:
        meta.update(extra)
    return meta

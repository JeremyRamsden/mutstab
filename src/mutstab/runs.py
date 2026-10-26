"""Run records: one JSON per run, committed to git.

This is deliberately the whole of the tracking setup. A results JSON holding
the config, the git commit, the split file used and the metrics answers the
question that actually comes up later — "which code and which data produced
this number?" — without any service to sign into. Add Weights & Biases on top
if you want live curves; it does not replace this file.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mutstab import paths
from mutstab.config import Config

__all__ = ["git_sha", "git_dirty", "save_run", "load_run", "list_runs"]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=paths.repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_sha() -> str | None:
    """Current commit, or None outside a git checkout (e.g. a fresh Colab VM)."""
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool | None:
    """True when tracked files differ from the commit.

    A dirty run is not reproducible from the SHA alone. Worth recording rather
    than preventing — in Colab you will edit and rerun constantly — but treat
    any headline number from a dirty run as provisional.
    """
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def save_run(
    config: Config,
    metrics: Mapping[str, Any],
    *,
    split_file: str | Path | None = None,
    n_train: int | None = None,
    n_eval: int | None = None,
    notes: str = "",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write `results/<run_id>.json` and return its path."""
    record: dict[str, Any] = {
        "run_id": config.run_id,
        "name": config.name,
        "config_hash": config.hash,
        "config": config.to_dict(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "python": platform.python_version(),
        "colab": paths.in_colab(),
        "split_file": str(split_file) if split_file else None,
        "n_train": n_train,
        "n_eval": n_eval,
        "notes": notes,
        "metrics": dict(metrics),
    }
    if extra:
        record["extra"] = dict(extra)

    out_dir = paths.results_dir(create=True)
    out_path = out_dir / f"{config.run_id}.json"
    with out_path.open("w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return out_path


def load_run(run_id: str) -> dict[str, Any]:
    path = paths.results_dir() / f"{run_id}.json"
    with path.open() as fh:
        return json.load(fh)


def list_runs() -> list[dict[str, Any]]:
    """Every run record, newest first. Feed to pandas to compare experiments."""
    records = []
    for path in sorted(paths.results_dir().glob("*.json")):
        try:
            with path.open() as fh:
                records.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return records

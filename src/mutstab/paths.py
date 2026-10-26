"""Where things live, resolved once so nothing else has to care.

Three environments share this code: Colab (data in mounted Drive), a laptop
(data in a local folder), and CI (data absent, tests use tmp dirs). Every path
derives from one root:

    MUTSTAB_DATA_ROOT   explicit override, wins everywhere
    Colab               /content/drive/MyDrive/mutstab
    otherwise           <repo>/data

The other thing this module handles is Drive being slow. Drive is mounted over
FUSE, so many small reads are painfully slow, while one big sequential read is
fine. `stage()` copies a file to Colab's local disk once and hands back the
local path; read from that in your training loop. Write checkpoints back to
Drive so a disconnect doesn't cost you the run.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

__all__ = [
    "in_colab",
    "repo_root",
    "data_root",
    "raw_dir",
    "processed_dir",
    "cache_dir",
    "checkpoints_dir",
    "splits_dir",
    "results_dir",
    "scratch_root",
    "stage",
    "mount_drive",
    "describe",
]

DRIVE_DEFAULT = Path("/content/drive/MyDrive/mutstab")
COLAB_SCRATCH = Path("/content/mutstab-local")


def in_colab() -> bool:
    """True when running inside Google Colab."""
    return importlib.util.find_spec("google.colab") is not None


def repo_root() -> Path:
    """Repository root, found by walking up from this file to the marker."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Installed non-editable: no repo on disk. Fall back to the cwd.
    return Path.cwd()


def data_root() -> Path:
    """Root of all data. Everything else below hangs off this."""
    override = os.environ.get("MUTSTAB_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if in_colab():
        return DRIVE_DEFAULT
    return repo_root() / "data"


def _sub(name: str, create: bool) -> Path:
    path = data_root() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(create: bool = False) -> Path:
    """Downloads exactly as they arrived. Treat as read-only."""
    return _sub("raw", create)


def processed_dir(create: bool = False) -> Path:
    """Cleaned, filtered datasets. Parquet, one file per dataset version."""
    return _sub("processed", create)


def cache_dir(create: bool = False) -> Path:
    """Expensive derived artifacts: embedding stores, structure features."""
    return _sub("cache", create)


def checkpoints_dir(create: bool = False) -> Path:
    """Model checkpoints. Always on Drive, so a disconnect is survivable."""
    return _sub("checkpoints", create)


def splits_dir(create: bool = False) -> Path:
    """Split definitions.

    These live in the *repo*, not in Drive: they are small, they are the thing
    that makes a number comparable across runs, and they belong in git history.
    """
    path = repo_root() / "splits"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(create: bool = False) -> Path:
    """Run records and figures. Also in the repo, also committed."""
    path = repo_root() / "results"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def scratch_root() -> Path:
    """Fast local disk. Contents vanish when the Colab VM goes away."""
    override = os.environ.get("MUTSTAB_SCRATCH")
    if override:
        return Path(override).expanduser()
    if in_colab():
        return COLAB_SCRATCH
    # Off Colab there is no slow-mount problem, so staging is a no-op copy
    # into a sibling dir rather than something clever.
    return data_root() / "_scratch"


def stage(path: str | Path, force: bool = False) -> Path:
    """Copy `path` to fast local disk and return the local path.

    Call this once per session for each dataset you will read repeatedly::

        df = pd.read_parquet(paths.stage(paths.processed_dir() / "dtm_v1.parquet"))

    A file already staged is reused unless `force=True`. Off Colab, where the
    data is already on local disk, this returns the original path untouched.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"nothing to stage at {src}")

    if not in_colab() and not os.environ.get("MUTSTAB_SCRATCH"):
        return src

    root = data_root()
    try:
        rel = src.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(src.name)

    dest = scratch_root() / rel
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        # Copy to a temp name then rename, so an interrupted copy does not
        # leave a half-file that looks complete to the next session.
        tmp = dest.with_suffix(dest.suffix + ".partial")
        shutil.copy2(src, tmp)
        tmp.replace(dest)
    return dest


def mount_drive() -> Path:
    """Mount Drive if in Colab and not already mounted. Returns `data_root()`."""
    if in_colab() and not Path("/content/drive").is_dir():
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive")
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "processed", "cache", "checkpoints"):
        (root / name).mkdir(exist_ok=True)
    return root


def describe() -> str:
    """One-glance summary. Print this at the top of a notebook."""
    lines = [
        f"colab:       {in_colab()}",
        f"repo_root:   {repo_root()}",
        f"data_root:   {data_root()}  (exists={data_root().exists()})",
        f"scratch:     {scratch_root()}",
        f"splits:      {splits_dir()}",
        f"results:     {results_dir()}",
    ]
    return "\n".join(lines)

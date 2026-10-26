"""Split definitions as a versioned artifact.

The rule this module exists to enforce: no protein appears in more than one
fold. Splitting by mutation lets a model see other variants of the same
protein during training, and because a protein's mutations share a mean dTm,
that alone produces a respectable correlation. Published numbers have been
inflated this way for years.

Splits are written to a JSON file in the repo and committed. They are not
regenerated at training time. Two runs quoting the same split file are
comparable; two runs that each called a shuffling function are not.

Grouping by protein_id is the floor, not the ceiling. Homologs with different
accessions still leak information. When you have the sequences, cluster them
at 30 percent identity with mmseqs2 and pass the cluster ids as `groups` —
`make_protein_folds` does not care what the group key means, only that it is
the unit kept whole.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from mutstab import paths

__all__ = [
    "make_protein_folds",
    "save_splits",
    "load_splits",
    "fold_indices",
    "assert_no_group_leakage",
]


def make_protein_folds(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    group_column: str = "protein_id",
    groups: Sequence[Any] | None = None,
    holdout_frac: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Assign whole groups to folds, balancing rows per fold.

    Groups are placed largest-first into whichever fold currently has fewest
    rows. With a handful of dominant proteins this balances far better than
    random assignment, and it is deterministic given `seed`.

    `holdout_frac` carves off groups reserved for a final test, touched once at
    the end. Worth doing: cross-validation gets used for model selection, and
    after enough iterations those numbers are optimistic too.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    key = pd.Series(groups, index=df.index) if groups is not None else df[group_column]
    if key.isna().any():
        raise ValueError(
            f"{int(key.isna().sum())} rows have no group key; "
            "fix or drop them before splitting"
        )

    sizes = key.value_counts()
    rng = np.random.default_rng(seed)

    # Shuffle first so equal-sized groups do not order by name.
    names = list(sizes.index)
    rng.shuffle(names)
    names.sort(key=lambda g: int(sizes[g]), reverse=True)

    holdout: list[Any] = []
    if holdout_frac > 0:
        target = holdout_frac * len(df)
        taken = 0
        # Walk from the smallest groups up, so the holdout is several distinct
        # proteins rather than one giant one.
        for name in reversed(names):
            if taken >= target:
                break
            holdout.append(name)
            taken += int(sizes[name])
        names = [g for g in names if g not in set(holdout)]

    fold_rows = [0] * n_folds
    fold_groups: list[list[Any]] = [[] for _ in range(n_folds)]
    for name in names:
        target_fold = int(np.argmin(fold_rows))
        fold_groups[target_fold].append(name)
        fold_rows[target_fold] += int(sizes[name])

    spec: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "group_column": group_column if groups is None else "custom",
        "n_folds": n_folds,
        "seed": seed,
        "n_rows": int(len(df)),
        "n_groups": int(len(sizes)),
        "source_file": df.attrs.get("source_file"),
        "folds": {
            str(i): sorted(str(g) for g in fold_groups[i]) for i in range(n_folds)
        },
        "holdout": sorted(str(g) for g in holdout),
        "rows_per_fold": {str(i): fold_rows[i] for i in range(n_folds)},
        "rows_in_holdout": int(sum(int(sizes[g]) for g in holdout)),
    }
    return spec


def save_splits(spec: Mapping[str, Any], name: str) -> Path:
    """Write a split spec to `splits/<name>.json`. Commit the result."""
    out_dir = paths.splits_dir(create=True)
    path = out_dir / (name if name.endswith(".json") else f"{name}.json")
    with path.open("w") as fh:
        json.dump(spec, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_splits(name: str | Path) -> dict[str, Any]:
    """Load a split spec by name or path."""
    path = Path(name)
    if not path.exists():
        candidate = paths.splits_dir() / (
            path.name if path.suffix == ".json" else f"{path.name}.json"
        )
        if not candidate.exists():
            raise FileNotFoundError(f"no split file at {name!r} or {candidate}")
        path = candidate
    with path.open() as fh:
        spec = json.load(fh)
    spec["_path"] = str(path)
    return spec


def fold_indices(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    fold: int,
    *,
    groups: Sequence[Any] | None = None,
    exclude_reverse_from_eval: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) positional indices for one fold.

    Holdout groups are excluded from both. Synthetic reverse mutations are
    dropped from the validation side by default — scoring against labels you
    created by negation measures your own arithmetic, not the model.
    """
    group_column = spec.get("group_column", "protein_id")
    key = (
        pd.Series(groups, index=df.index).astype("string")
        if groups is not None
        else df[group_column].astype("string")
    )

    val_groups = set(spec["folds"][str(fold)])
    holdout = set(spec.get("holdout", []))

    is_val = key.isin(val_groups).to_numpy()
    is_holdout = key.isin(holdout).to_numpy()

    train_mask = ~is_val & ~is_holdout
    val_mask = is_val & ~is_holdout

    if exclude_reverse_from_eval and "is_reverse" in df.columns:
        val_mask &= ~df["is_reverse"].fillna(False).to_numpy(dtype=bool)

    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)


def assert_no_group_leakage(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    groups: Sequence[Any] | None = None,
) -> None:
    """Raise if any group appears in two folds, or in a fold and the holdout.

    Call this in a test and after any change to split generation. It is three
    lines of checking that protects every number in `results/`.
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for fold_name, members in spec["folds"].items():
        for member in members:
            if member in seen:
                collisions.append(f"{member}: folds {seen[member]} and {fold_name}")
            seen[member] = fold_name
    for member in spec.get("holdout", []):
        if member in seen:
            collisions.append(f"{member}: fold {seen[member]} and holdout")
    if collisions:
        raise AssertionError("group leakage across folds:\n  " + "\n  ".join(collisions))

    group_column = spec.get("group_column", "protein_id")
    key = (
        pd.Series(groups, index=df.index).astype("string")
        if groups is not None
        else df[group_column].astype("string")
    )
    assigned = set(seen) | set(spec.get("holdout", []))
    missing = set(key.dropna().unique()) - assigned
    if missing:
        shown = ", ".join(sorted(missing)[:5])
        raise AssertionError(
            f"{len(missing)} groups in the data are in no fold (e.g. {shown}); "
            "the split file is stale — regenerate it from this dataset"
        )


def fold_summary(df: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    """Rows, groups and label spread per fold — check folds are comparable."""
    group_column = spec.get("group_column", "protein_id")
    rows: list[dict[str, Any]] = []
    parts: Iterable[tuple[str, list[str]]] = list(spec["folds"].items())
    if spec.get("holdout"):
        parts = [*parts, ("holdout", spec["holdout"])]
    for name, members in parts:
        subset = df[df[group_column].astype("string").isin(set(members))]
        rows.append(
            {
                "fold": name,
                "n_rows": len(subset),
                "n_groups": len(members),
                "dtm_mean": round(float(subset["dtm"].mean()), 2) if len(subset) else None,
                "dtm_std": round(float(subset["dtm"].std()), 2) if len(subset) else None,
            }
        )
    return pd.DataFrame(rows)

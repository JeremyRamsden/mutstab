#!/usr/bin/env python
"""Generate a split file, once, and commit it.

    python scripts/make_splits.py --config baseline

Refuses to overwrite an existing split file unless given --force. That is the
point of the script: a split file that changes underneath you makes every
previous number in results/ incomparable, and the failure is invisible.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mutstab import paths
from mutstab.config import load_config
from mutstab.data.splits import (
    assert_no_group_leakage,
    fold_summary,
    make_protein_folds,
    save_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_name = cfg.get_path("split.file")
    out_path = paths.splits_dir(create=True) / f"{split_name}.json"

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists.\n"
            "Reuse it, or pass --force if you really mean to invalidate every\n"
            "result that cites it (and give the new one a new name instead)."
        )

    data_path = paths.processed_dir() / cfg.get_path("data.processed_file")
    if not data_path.exists():
        raise SystemExit(f"no processed data at {data_path}; run prepare_data.py first")

    df = pd.read_parquet(data_path)
    df.attrs["source_file"] = str(data_path)

    spec = make_protein_folds(
        df,
        n_folds=int(cfg.get_path("split.n_folds", 5)),
        holdout_frac=float(cfg.get_path("split.holdout_frac", 0.0)),
        seed=int(cfg.get_path("split.seed", 0)),
    )
    assert_no_group_leakage(df, spec)

    written = save_splits(spec, split_name)
    print(f"wrote {written}")
    print(f"  {spec['n_groups']} proteins over {spec['n_folds']} folds")
    print(f"  holdout: {len(spec['holdout'])} proteins, {spec['rows_in_holdout']:,} rows")
    print()
    print(fold_summary(df, spec).to_string(index=False))
    print("\nCommit this file.")


if __name__ == "__main__":
    main()

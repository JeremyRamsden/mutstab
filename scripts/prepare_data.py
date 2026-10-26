#!/usr/bin/env python
"""Raw export -> filtered Parquet. Run once per dataset version.

    python scripts/prepare_data.py --config baseline

Writes to paths.processed_dir() and prints the filter log, so you can see
exactly where rows were lost. Parquet rather than CSV: smaller, typed, and
much faster to read off Drive.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mutstab import paths
from mutstab.config import load_config
from mutstab.data.load import filter_dtm, load_fireprotdb, summarise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--raw-file", default=None, help="override the config's raw file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_name = args.raw_file or cfg.get_path("data.raw_file")
    raw_path = paths.raw_dir() / raw_name
    if not raw_path.exists():
        raise SystemExit(
            f"no raw export at {raw_path}\n"
            "Download the FireProtDB export and put it there "
            f"(data root is {paths.data_root()})."
        )

    print(f"reading {raw_path}")
    df = load_fireprotdb(raw_path)
    print(f"  {len(df):,} rows, {df['protein_id'].nunique():,} proteins")

    filters = cfg.get_path("data.filters", {}) or {}
    clean = filter_dtm(
        df,
        ph_range=tuple(filters["ph_range"]) if filters.get("ph_range") else None,
        abs_dtm_max=filters.get("abs_dtm_max"),
        drop_duplicates=filters.get("drop_duplicates", "median"),
    )

    print("\nfilter log")
    print(pd.DataFrame(clean.attrs["filter_log"]).to_string(index=False))

    out_path = paths.processed_dir(create=True) / cfg.get_path("data.processed_file")
    clean.to_parquet(out_path, index=False)
    print(f"\nwrote {len(clean):,} rows to {out_path}")

    table = summarise(clean)
    print(f"\n{clean['protein_id'].nunique()} proteins; top 10 by count:")
    print(table.head(10).to_string())
    top_share = float(table.head(5)["n"].sum() / len(clean))
    print(f"\ntop 5 proteins hold {top_share:.0%} of the data")
    if top_share > 0.5:
        print(
            "  -> heavily concentrated. Protein-level splits are essential, and\n"
            "     per-protein metrics are the only ones worth quoting."
        )


if __name__ == "__main__":
    main()

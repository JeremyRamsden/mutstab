#!/usr/bin/env python
"""Fetch UniProt sequences, verify positions, rewrite the processed dataset.

    python scripts/fetch_sequences.py --config baseline

Run this between prepare_data.py and embed.py. It is resumable — the cache
means a rerun after a disconnect costs nothing — and it is the step that
catches numbering problems before they turn into silently wrong features.

The report it prints is worth reading rather than skimming. "offset" proteins
had numbering shifted by a constant, which is fixed here. "unresolved"
proteins have numbering that no single shift reconciles, usually a different
isoform or a wrong accession; by default they are dropped, because features
read from the wrong residue are worse than missing data.

Pass --dry-run to see the report without writing anything.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mutstab import paths
from mutstab.config import load_config
from mutstab.data.sequences import (
    attach_sequences,
    fetch_uniprot_sequences,
    reconcile_positions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument(
        "--keep-unresolved",
        action="store_true",
        help="keep proteins whose numbering cannot be reconciled (not advised)",
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="ask UniProt again about accessions that previously returned nothing",
    )
    parser.add_argument("--max-offset", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = paths.processed_dir() / cfg.get_path("data.processed_file")
    if not data_path.exists():
        raise SystemExit(f"no processed data at {data_path}; run prepare_data.py first")

    df = pd.read_parquet(data_path)
    print(f"{len(df):,} rows, {df['protein_id'].nunique():,} proteins")

    # --- fetch ------------------------------------------------------
    cache_name = cfg.get_path("data.sequence_cache", "uniprot_sequences.json")
    cache_path = paths.cache_dir(create=True) / cache_name
    sequences = fetch_uniprot_sequences(
        df["protein_id"].dropna().unique(),
        cache_path,
        retry_missing=args.retry_missing,
    )
    print(f"{len(sequences)} sequences available; cache at {cache_path}")

    df = attach_sequences(df, sequences)
    have = int(df["sequence"].notna().sum())
    print(f"{have:,}/{len(df):,} rows now have a sequence")

    # --- verify -----------------------------------------------------
    out, report = reconcile_positions(
        df,
        max_offset=args.max_offset,
        drop_unresolved=not args.keep_unresolved,
    )

    print("\nposition check, by protein")
    columns = [
        "protein_id",
        "n_variants",
        "frac_match",
        "best_offset",
        "frac_match_after",
        "verdict",
        "verdict_after",
    ]
    print(report[columns].head(25).to_string(index=False))

    counts = report["verdict"].value_counts().to_dict()
    print(f"\nverdicts: {counts}")

    shifted = report[report["verdict"] == "offset"]
    if len(shifted):
        print(
            f"\n{len(shifted)} proteins had shifted numbering, now corrected "
            f"(offsets {sorted(set(shifted['best_offset'].astype(int)))})"
        )
        print(
            "  These would have produced features read from the wrong residue.\n"
            "  Worth spot-checking one against its UniProt entry."
        )

    still_bad = report[report["verdict_after"].isin(["unresolved", "no_sequence"])]
    if len(still_bad):
        action = "kept" if args.keep_unresolved else "dropped"
        rows_lost = len(df) - len(out)
        print(
            f"\n{len(still_bad)} proteins could not be reconciled and were {action}"
            f"{f' ({rows_lost:,} rows)' if not args.keep_unresolved else ''}"
        )

    print(f"\n{len(out):,} rows, {out['protein_id'].nunique():,} proteins remain")

    # --- write ------------------------------------------------------
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    report_path = paths.results_dir(create=True) / "position_check.csv"
    report.to_csv(report_path, index=False)
    out.to_parquet(data_path, index=False)
    print(f"\nwrote {data_path}")
    print(f"wrote {report_path}  (commit this — it records what was changed)")

    if len(out) != len(df):
        print(
            "\nRow count changed, so your split file is now stale.\n"
            "Regenerate it with a NEW name and leave the old results citing the old one."
        )


if __name__ == "__main__":
    main()

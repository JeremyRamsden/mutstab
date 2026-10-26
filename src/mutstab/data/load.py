"""Loading and filtering mutation-stability records.

Written against FireProtDB CSV exports, but the column aliasing means a
ThermoMutDB or ProThermDB export needs only a few more entries in
`COLUMN_ALIASES` rather than a separate loader.

Two things to know before trusting any number that comes out of here:

1. dTm and ddG are different measurements. They correlate but are not
   convertible: a mutation can raise Tm while barely changing ddG at 25 C,
   and the relation involves the unfolding enthalpy, which is usually not
   reported. This module keeps them in separate columns and never derives one
   from the other.
2. dTm sign conventions vary between source papers. The convention here is
   mutant minus wild type, so positive dTm means stabilising. Spot-check a
   handful of entries against their original papers before you train on them;
   a flipped subset is the single easiest way to get a quietly broken model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "COLUMN_ALIASES",
    "load_fireprotdb",
    "filter_dtm",
    "add_reverse_mutations",
    "summarise",
]

# What the rest of the codebase expects to exist after loading.
CANONICAL_COLUMNS = [
    "protein_id",   # grouping key for splits — UniProt accession where available
    "protein_name",
    "pdb_id",
    "chain",
    "position",     # 1-based position in the reference sequence
    "wt_aa",        # single-letter wild-type residue
    "mut_aa",       # single-letter mutant residue
    "dtm",          # Tm(mutant) - Tm(wild type), degrees C
    "ddg",          # kept alongside for multi-task / auxiliary use, kcal/mol
    "tm_wt",
    "ph",
    "method",
    "sequence",
    "source",
]

# Export headers vary by database and by release. Add to this map rather than
# renaming columns by hand in a notebook.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "protein_id": ("uniprot_id", "uniprot", "uniprot_ac", "accession"),
    "protein_name": ("protein_name", "protein", "name"),
    "pdb_id": ("pdb_id", "pdb", "pdb_code"),
    "chain": ("chain", "pdb_chain", "chain_id"),
    "position": ("position", "pos", "residue_number", "resnum", "wt_position"),
    "wt_aa": ("wild_type", "wt_res", "wt", "wild_type_residue", "wt_aa"),
    "mut_aa": ("mutation", "mut_res", "mutant", "mutated_residue", "mut_aa"),
    "dtm": ("dtm", "d_tm", "delta_tm", "ddtm", "tm_change"),
    "ddg": ("ddg", "d_dg", "delta_delta_g", "ddg_exp"),
    "tm_wt": ("tm", "tm_wt", "tm_wild_type"),
    "ph": ("ph", "p_h"),
    "method": ("method", "technique", "measurement_method"),
    "sequence": ("sequence", "seq", "protein_sequence"),
    "source": ("source", "reference", "pubmed_id", "pmid", "doi"),
}

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _normalise_header(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")


def load_fireprotdb(
    path: str | Path,
    *,
    extra_aliases: Mapping[str, Iterable[str]] | None = None,
    keep_unmapped: bool = True,
) -> pd.DataFrame:
    """Read an export and rename columns to the canonical names.

    Unrecognised columns are kept as-is when `keep_unmapped` is True, so
    nothing is silently dropped. Canonical columns the export lacks are added
    as all-NA, which keeps downstream code from needing per-source branches.
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")

    raw.columns = [_normalise_header(c) for c in raw.columns]

    aliases = {k: tuple(v) for k, v in COLUMN_ALIASES.items()}
    for key, values in (extra_aliases or {}).items():
        aliases[key] = tuple(values) + aliases.get(key, ())

    rename: dict[str, str] = {}
    for canonical, options in aliases.items():
        for option in options:
            option = _normalise_header(option)
            if option in raw.columns and option not in rename:
                rename[option] = canonical
                break

    df = raw.rename(columns=rename)
    if not keep_unmapped:
        df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]

    for column in CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    # A missing accession would collapse distinct proteins into one split
    # group, so fall back through the other identifiers before giving up.
    df["protein_id"] = (
        df["protein_id"]
        .astype("string")
        .fillna(df["pdb_id"].astype("string"))
        .fillna(df["protein_name"].astype("string"))
    )

    for column in ("wt_aa", "mut_aa"):
        df[column] = df[column].astype("string").str.strip().str.upper()
    df["position"] = pd.to_numeric(df["position"], errors="coerce").astype("Int64")
    for column in ("dtm", "ddg", "tm_wt", "ph"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["variant"] = (
        df["wt_aa"].astype("string")
        + df["position"].astype("string")
        + df["mut_aa"].astype("string")
    )
    df.attrs["source_file"] = str(path)
    return df


def filter_dtm(
    df: pd.DataFrame,
    *,
    ph_range: tuple[float, float] | None = (5.0, 9.0),
    abs_dtm_max: float | None = 30.0,
    require_sequence: bool = False,
    drop_duplicates: str | None = "median",
) -> pd.DataFrame:
    """Keep rows usable as dTm training data.

    Defaults are deliberately mild; every step is reported in
    `result.attrs["filter_log"]` so the notebook can show where rows went.

    `ph_range` guards against pooling measurements taken under conditions that
    are not comparable. `abs_dtm_max` removes the handful of implausible
    magnitudes that usually turn out to be unit or sign errors in the source.
    `drop_duplicates="median"` collapses repeated measurements of the same
    variant to their median, which stops heavily re-measured variants from
    dominating the loss; pass None to keep replicates.
    """
    log: list[dict[str, Any]] = []
    out = df.copy()

    def record(step: str, before: int) -> None:
        log.append({"step": step, "before": before, "after": len(out), "dropped": before - len(out)})

    before = len(out)
    out = out[out["dtm"].notna()]
    record("has dtm", before)

    before = len(out)
    out = out[out["position"].notna()]
    out = out[out["wt_aa"].isin(_AA) & out["mut_aa"].isin(_AA)]
    record("valid single substitution", before)

    before = len(out)
    out = out[out["wt_aa"] != out["mut_aa"]]
    record("not synonymous", before)

    before = len(out)
    out = out[out["protein_id"].notna()]
    record("has protein_id", before)

    if ph_range is not None:
        low, high = ph_range
        before = len(out)
        # Keep NA pH: dropping it discards a lot of otherwise fine data.
        out = out[out["ph"].isna() | out["ph"].between(low, high)]
        record(f"ph in [{low}, {high}] or missing", before)

    if abs_dtm_max is not None:
        before = len(out)
        out = out[out["dtm"].abs() <= abs_dtm_max]
        record(f"|dtm| <= {abs_dtm_max}", before)

    if require_sequence:
        before = len(out)
        out = out[out["sequence"].notna()]
        record("has sequence", before)

    if drop_duplicates:
        before = len(out)
        keys = ["protein_id", "position", "wt_aa", "mut_aa"]
        if drop_duplicates == "median":
            agg: dict[str, Any] = {c: "first" for c in out.columns if c not in keys}
            agg["dtm"] = "median"
            if "ddg" in out.columns:
                agg["ddg"] = "median"
            counts = out.groupby(keys, dropna=False).size().rename("n_measurements")
            out = out.groupby(keys, dropna=False).agg(agg).join(counts).reset_index()
        elif drop_duplicates == "first":
            out = out.drop_duplicates(subset=keys, keep="first")
        else:
            raise ValueError("drop_duplicates must be 'median', 'first' or None")
        record(f"collapse duplicate variants ({drop_duplicates})", before)

    out = out.reset_index(drop=True)
    out.attrs["filter_log"] = log
    out.attrs["source_file"] = df.attrs.get("source_file")
    return out


def add_reverse_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """Append the hypothetical reverse of every variant with dTm negated.

    Most measured mutations are destabilising, so a model trained on the raw
    data learns "predict negative" as a strong prior and scores respectably
    while having learned little. Adding reverses makes the label distribution
    symmetric by construction and gives the antisymmetry check in
    `mutstab.eval` something to measure.

    The reverse rows are synthetic, not measurements. Use them in training
    only — `is_reverse` marks them so evaluation can exclude them.
    """
    forward = df.copy()
    forward["is_reverse"] = False

    reverse = df.copy()
    reverse["is_reverse"] = True
    reverse["wt_aa"], reverse["mut_aa"] = df["mut_aa"].copy(), df["wt_aa"].copy()
    reverse["dtm"] = -df["dtm"]
    if "ddg" in reverse.columns:
        reverse["ddg"] = -df["ddg"]
    # Tm of the "wild type" in a reverse pair is the mutant's Tm.
    if "tm_wt" in reverse.columns:
        reverse["tm_wt"] = df["tm_wt"] + df["dtm"]
    reverse["variant"] = (
        reverse["wt_aa"].astype("string")
        + reverse["position"].astype("string")
        + reverse["mut_aa"].astype("string")
    )

    out = pd.concat([forward, reverse], ignore_index=True)
    out.attrs.update(df.attrs)
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-protein counts and dTm spread — the bias check, in one table.

    Read the top of this output before modelling. If a few proteins hold most
    of the rows, a random split will flatter any model you build, and
    per-protein metrics are the only ones worth quoting.
    """
    grouped = df.groupby("protein_id", dropna=False)["dtm"]
    table = pd.DataFrame(
        {
            "n": grouped.size(),
            "dtm_mean": grouped.mean().round(2),
            "dtm_std": grouped.std().round(2),
            "dtm_min": grouped.min().round(2),
            "dtm_max": grouped.max().round(2),
            "frac_stabilising": grouped.apply(lambda s: float(np.mean(s > 0))).round(3),
        }
    )
    table = table.sort_values("n", ascending=False)
    table["cum_frac_of_data"] = (table["n"].cumsum() / len(df)).round(3)
    return table

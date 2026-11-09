"""Fetch reference sequences from UniProt, then check the positions agree.

Two jobs, and the second is the important one.

Fetching is routine: ask UniProt for FASTA by accession, cache the result so a
Colab disconnect costs nothing, move on.

Checking is not. A mutation record says "position 42, wild type is alanine",
but position 42 *in which numbering*? Stability datasets inherit numbering
from their source papers, which often follow the PDB file, the mature protein
after signal-peptide cleavage, or a construct with an expression tag. Any of
those differs from the UniProt sequence by a constant offset, sometimes by
tens of residues. Nothing warns you. The embedding is read from the wrong
residue, the one-hot encoding of the wild type disagrees with the sequence,
and the model trains on quietly corrupted features.

So every sequence that arrives here is checked: does `sequence[position - 1]`
equal the recorded wild-type residue? When it does not, `check_positions`
searches for a constant offset that makes it agree. A protein where one offset
fixes nearly every variant had an offset problem. A protein where no offset
helps has something else wrong — a different isoform, or the wrong accession —
and is better dropped than guessed at.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "UNIPROT_ACCESSION_RE",
    "looks_like_accession",
    "parse_fasta",
    "fetch_uniprot_sequences",
    "attach_sequences",
    "check_positions",
    "apply_offsets",
    "reconcile_positions",
]

# Official UniProt accession pattern. Used to skip ids that are clearly
# something else — a PDB code, a protein name — rather than asking UniProt
# about them and recording a confusing miss.
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

_REST_BATCH = "https://rest.uniprot.org/uniprotkb/accessions"
_REST_SINGLE = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
# A generic client name. UniProt asks automated clients to identify
# themselves; it does not need anyone's email address.
_USER_AGENT = "mutstab/0.1 (research script)"


def looks_like_accession(value: Any) -> bool:
    """True for strings shaped like a UniProt accession, isoform suffix allowed."""
    if value is None or not isinstance(value, str):
        return False
    base = value.strip().upper().split("-", 1)[0]
    return bool(UNIPROT_ACCESSION_RE.match(base))


def parse_fasta(text: str) -> dict[str, str]:
    """Parse UniProt FASTA into {accession: sequence}.

    Handles the `>sp|P12345|NAME_ORG ...` and `>tr|...` forms, and falls back
    to the first whitespace-delimited token for anything else.
    """
    out: dict[str, str] = {}
    accession: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if accession and chunks:
            out[accession] = "".join(chunks).upper()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            chunks = []
            header = line[1:]
            parts = header.split("|")
            accession = parts[1].strip() if len(parts) >= 3 else header.split()[0]
        else:
            chunks.append(line)
    flush()
    return out


def _http_get(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def _fetch_batch(accessions: Sequence[str], timeout: float) -> dict[str, str]:
    """One request for many accessions, falling back to one request each.

    The batch endpoint is much faster but fails the whole batch if any single
    accession is rejected, so a failure retries individually to isolate the
    bad one rather than losing the other 99.
    """
    query = ",".join(accessions)
    url = f"{_REST_BATCH}?accessions={query}&format=fasta"
    try:
        return parse_fasta(_http_get(url, timeout))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        pass

    out: dict[str, str] = {}
    for accession in accessions:
        try:
            out.update(parse_fasta(_http_get(_REST_SINGLE.format(acc=accession), timeout)))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            continue
    return out


def fetch_uniprot_sequences(
    accessions: Iterable[str],
    cache_path: str | Path | None = None,
    *,
    batch_size: int = 100,
    timeout: float = 30.0,
    pause: float = 0.2,
    retry_missing: bool = False,
    fetcher: Callable[[Sequence[str], float], Mapping[str, str]] | None = None,
    progress: bool = True,
) -> dict[str, str]:
    """Fetch sequences for `accessions`, caching to a JSON file.

    Resumable: accessions already in the cache are not requested again. An
    accession UniProt had nothing for is cached as an empty string so repeated
    runs do not keep asking; `retry_missing=True` asks again (worth doing
    after a network problem, not after a genuine 404).

    `fetcher` exists so tests can run without network access.

    Returns only the accessions that resolved to a sequence.
    """
    wanted: list[str] = []
    seen: set[str] = set()
    skipped: list[str] = []
    for value in accessions:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if looks_like_accession(key):
            wanted.append(key)
        else:
            skipped.append(key)

    if skipped and progress:
        print(
            f"skipping {len(skipped)} ids that are not UniProt accessions "
            f"(e.g. {', '.join(skipped[:3])})"
        )

    cache: dict[str, str] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        try:
            with cache_file.open() as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                cache = {str(k): str(v) for k, v in loaded.items()}
        except (OSError, json.JSONDecodeError):
            # A truncated cache should not stop the run; refetch instead.
            cache = {}

    todo = [
        accession
        for accession in wanted
        if accession not in cache or (retry_missing and not cache[accession])
    ]
    if progress:
        print(f"{len(wanted)} accessions; {len(todo)} to fetch, {len(wanted) - len(todo)} cached")

    call = fetcher if fetcher is not None else _fetch_batch

    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        found = dict(call(chunk, timeout))
        for accession in chunk:
            # Empty string marks "asked, nothing there" so it is not retried.
            cache[accession] = str(found.get(accession, ""))

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(cache_file.suffix + ".partial")
            with tmp.open("w") as fh:
                json.dump(cache, fh, indent=0, sort_keys=True)
            tmp.replace(cache_file)

        if progress:
            done = min(start + batch_size, len(todo))
            print(f"  {done}/{len(todo)}", flush=True)
        if pause and start + batch_size < len(todo):
            time.sleep(pause)

    resolved = {k: v for k, v in cache.items() if v}
    missing = [a for a in wanted if not cache.get(a)]
    if missing and progress:
        print(
            f"{len(missing)} accessions returned nothing "
            f"(e.g. {', '.join(missing[:3])}) — obsolete or merged entries"
        )
    return resolved


def attach_sequences(
    df: pd.DataFrame, sequences: Mapping[str, str], *, overwrite: bool = False
) -> pd.DataFrame:
    """Fill the `sequence` column from a {accession: sequence} mapping.

    Existing sequences are kept unless `overwrite=True` — an export that ships
    its own sequences has already matched them to its own numbering, and
    replacing them silently changes what the positions mean.
    """
    out = df.copy()
    mapped = out["protein_id"].astype("string").map(lambda k: sequences.get(str(k), ""))
    mapped = mapped.replace("", pd.NA)

    if overwrite or "sequence" not in out.columns:
        out["sequence"] = mapped
    else:
        out["sequence"] = out["sequence"].astype("string").fillna(mapped)
    return out


def check_positions(
    df: pd.DataFrame, *, max_offset: int = 30, group_column: str = "protein_id"
) -> pd.DataFrame:
    """Per protein: do the recorded positions match the sequence?

    For each protein, compares `sequence[position - 1]` against `wt_aa`, then
    searches offsets in [-max_offset, +max_offset] for the one that agrees with
    the most variants. Returns one row per protein with:

        n_variants          variants with both a position and a sequence
        frac_match          agreement at the recorded numbering
        best_offset         offset with the highest agreement (0 if none beats it)
        frac_match_at_best  agreement at that offset
        verdict             ok | offset | unresolved

    `verdict` is the column to act on. "offset" means the numbering is
    systematically shifted and `apply_offsets` can fix it. "unresolved" means
    no single shift works — usually the wrong isoform or a bad accession — and
    those proteins are safer dropped than guessed at.
    """
    rows: list[dict[str, Any]] = []

    for protein, subset in df.groupby(group_column, dropna=False):
        sequences = subset["sequence"].dropna().astype("string").unique()
        if len(sequences) == 0:
            rows.append(
                {
                    group_column: protein,
                    "n_variants": 0,
                    "frac_match": None,
                    "best_offset": None,
                    "frac_match_at_best": None,
                    "verdict": "no_sequence",
                }
            )
            continue

        sequence = str(sequences[0])
        usable = subset[subset["position"].notna() & subset["wt_aa"].notna()]
        if usable.empty:
            rows.append(
                {
                    group_column: protein,
                    "n_variants": 0,
                    "frac_match": None,
                    "best_offset": None,
                    "frac_match_at_best": None,
                    "verdict": "no_positions",
                }
            )
            continue

        # astype first: a pandas nullable Int64 column does not convert
        # straight to a numpy int array.
        positions = usable["position"].astype("int64").to_numpy()
        wild_type = usable["wt_aa"].astype("string").to_numpy()

        def agreement(offset: int) -> float:
            index = positions - 1 + offset
            inside = (index >= 0) & (index < len(sequence))
            if not inside.any():
                return 0.0
            observed = np.array([sequence[i] for i in index[inside]])
            # Divide by the full count, not just in-range ones: an offset that
            # pushes variants off the end of the sequence is not a good offset.
            return float((observed == wild_type[inside]).sum() / len(positions))

        base = agreement(0)
        best_offset, best = 0, base
        for offset in range(-max_offset, max_offset + 1):
            if offset == 0:
                continue
            score = agreement(offset)
            if score > best:
                best_offset, best = offset, score

        if base >= 0.95:
            verdict = "ok"
        elif best >= 0.95 and best > base + 0.1:
            verdict = "offset"
        else:
            verdict = "unresolved"

        rows.append(
            {
                group_column: protein,
                "n_variants": int(len(positions)),
                "frac_match": round(base, 3),
                "best_offset": int(best_offset),
                "frac_match_at_best": round(best, 3),
                "verdict": verdict,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("n_variants", ascending=False).reset_index(drop=True)
    return table


def apply_offsets(
    df: pd.DataFrame,
    report: pd.DataFrame,
    *,
    group_column: str = "protein_id",
    only_verdict: str = "offset",
) -> pd.DataFrame:
    """Shift positions for proteins the report found a consistent offset for.

    Adds `position_original` so the change is visible and reversible. Only
    rows whose protein has the given verdict are touched.
    """
    offsets = {
        row[group_column]: int(row["best_offset"])
        for _, row in report.iterrows()
        if row.get("verdict") == only_verdict and row.get("best_offset")
    }
    if not offsets:
        return df.copy()

    out = df.copy()
    if "position_original" not in out.columns:
        out["position_original"] = out["position"]
    shift = out[group_column].map(lambda k: offsets.get(k, 0)).fillna(0).astype(int)
    out["position"] = (out["position"].astype("Int64") + shift).astype("Int64")
    return out


def reconcile_positions(
    df: pd.DataFrame,
    *,
    max_offset: int = 30,
    drop_unresolved: bool = True,
    group_column: str = "protein_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check, fix what is fixable, drop what is not. Returns (df, report).

    The usual entry point. With `drop_unresolved=True`, proteins whose
    numbering cannot be reconciled are removed — they would otherwise
    contribute features read from the wrong residues, which is worse than
    having less data.
    """
    report = check_positions(df, max_offset=max_offset, group_column=group_column)
    out = apply_offsets(df, report, group_column=group_column)

    # Re-check: applying an offset should move those proteins to "ok", and if
    # one does not, the report should say so rather than the caller assuming.
    report_after = check_positions(out, max_offset=max_offset, group_column=group_column)
    report = report.merge(
        report_after[[group_column, "frac_match", "verdict"]].rename(
            columns={"frac_match": "frac_match_after", "verdict": "verdict_after"}
        ),
        on=group_column,
        how="left",
    )

    if drop_unresolved:
        bad = set(
            report.loc[
                report["verdict_after"].isin(["unresolved", "no_sequence", "no_positions"]),
                group_column,
            ]
        )
        if bad:
            out = out[~out[group_column].isin(bad)].reset_index(drop=True)

    out.attrs.update(df.attrs)
    return out, report

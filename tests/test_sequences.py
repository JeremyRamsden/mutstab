"""Sequence fetching and, more importantly, position verification.

No network: `fetch_uniprot_sequences` takes a `fetcher` so these tests run
offline and deterministically.

The offset tests are the point of the module. A constant numbering shift
produces no error anywhere — it just reads features from the wrong residue —
so it has to be caught by an explicit check.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mutstab.data.sequences import (
    apply_offsets,
    attach_sequences,
    check_positions,
    fetch_uniprot_sequences,
    looks_like_accession,
    parse_fasta,
    reconcile_positions,
)

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def sequence() -> str:
    """A non-repeating sequence.

    Deliberately not a repeated motif: with a periodic sequence a shifted
    position can land on the same residue by coincidence, and the offset tests
    would pass without the code working.
    """
    rng = np.random.default_rng(42)
    return "".join(rng.choice(list(AA), size=80))


def variants_for(sequence: str, protein: str, true_positions, shift: int = 0) -> pd.DataFrame:
    """Variant rows whose recorded positions are `shift` away from the truth."""
    rows = []
    for position in true_positions:
        wt = sequence[position - 1]
        rows.append(
            {
                "protein_id": protein,
                "position": position + shift,
                "wt_aa": wt,
                "mut_aa": "A" if wt != "A" else "G",
                "dtm": -1.0,
                "sequence": sequence,
            }
        )
    return pd.DataFrame(rows)


# --- parsing ---------------------------------------------------------
def test_parse_fasta_handles_uniprot_headers():
    text = (
        ">sp|P12345|TEST_HUMAN Some protein OS=Homo sapiens\n"
        "ACDEFG\n"
        "HIKL\n"
        ">tr|Q67890|OTHER_MOUSE Another\n"
        "MNPQ\n"
    )
    out = parse_fasta(text)
    assert out == {"P12345": "ACDEFGHIKL", "Q67890": "MNPQ"}


def test_parse_fasta_falls_back_to_first_token():
    assert parse_fasta(">P99999 no pipes here\nACDE\n") == {"P99999": "ACDE"}


def test_parse_fasta_of_nothing():
    assert parse_fasta("") == {}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("P12345", True),
        ("Q6GZX4", True),
        ("A0A023GPI8", True),
        ("P12345-2", True),   # isoform suffix
        ("p12345", True),     # case is normalised
        ("1ABC", False),      # PDB code
        ("lysozyme", False),
        ("", False),
        (None, False),
        (12345, False),
    ],
)
def test_looks_like_accession(value, expected):
    assert looks_like_accession(value) is expected


# --- fetching --------------------------------------------------------
def test_fetch_uses_cache_and_does_not_refetch(tmp_path):
    cache = tmp_path / "seqs.json"
    calls: list[list[str]] = []

    def fake(accessions, timeout):
        calls.append(list(accessions))
        return {a: "ACDEFG" for a in accessions}

    first = fetch_uniprot_sequences(["P12345", "Q67890"], cache, fetcher=fake, progress=False)
    assert first == {"P12345": "ACDEFG", "Q67890": "ACDEFG"}
    assert len(calls) == 1

    # Second run: everything cached, so the fetcher is not called again.
    second = fetch_uniprot_sequences(["P12345", "Q67890"], cache, fetcher=fake, progress=False)
    assert second == first
    assert len(calls) == 1


def test_misses_are_cached_but_retryable(tmp_path):
    cache = tmp_path / "seqs.json"
    calls: list[list[str]] = []

    def nothing(accessions, timeout):
        calls.append(list(accessions))
        return {}

    assert fetch_uniprot_sequences(["P12345"], cache, fetcher=nothing, progress=False) == {}
    # Recorded as an empty string so a rerun does not ask again.
    assert json.loads(cache.read_text()) == {"P12345": ""}

    fetch_uniprot_sequences(["P12345"], cache, fetcher=nothing, progress=False)
    assert len(calls) == 1

    fetch_uniprot_sequences(["P12345"], cache, fetcher=nothing, retry_missing=True, progress=False)
    assert len(calls) == 2


def test_non_accessions_are_never_requested(tmp_path):
    seen: list[str] = []

    def fake(accessions, timeout):
        seen.extend(accessions)
        return {a: "ACDE" for a in accessions}

    out = fetch_uniprot_sequences(
        ["P12345", "1LYZ", "lysozyme", None], tmp_path / "c.json", fetcher=fake, progress=False
    )
    assert seen == ["P12345"]
    assert set(out) == {"P12345"}


def test_batches_respect_batch_size(tmp_path):
    accessions = [f"P{i:05d}" for i in range(1, 8)]
    sizes: list[int] = []

    def fake(chunk, timeout):
        sizes.append(len(chunk))
        return {a: "ACDE" for a in chunk}

    fetch_uniprot_sequences(
        accessions, tmp_path / "c.json", batch_size=3, pause=0, fetcher=fake, progress=False
    )
    assert sizes == [3, 3, 1]


def test_corrupt_cache_is_ignored(tmp_path):
    cache = tmp_path / "seqs.json"
    cache.write_text("{not json")

    def fake(accessions, timeout):
        return {a: "ACDE" for a in accessions}

    out = fetch_uniprot_sequences(["P12345"], cache, fetcher=fake, progress=False)
    assert out == {"P12345": "ACDE"}


# --- attaching -------------------------------------------------------
def test_attach_keeps_existing_sequences_by_default():
    df = pd.DataFrame(
        {"protein_id": ["P12345", "Q67890"], "sequence": ["ORIGINAL", None]}
    )
    out = attach_sequences(df, {"P12345": "FETCHED", "Q67890": "FETCHED"})
    assert list(out["sequence"]) == ["ORIGINAL", "FETCHED"]

    forced = attach_sequences(df, {"P12345": "FETCHED", "Q67890": "FETCHED"}, overwrite=True)
    assert list(forced["sequence"]) == ["FETCHED", "FETCHED"]


# --- position verification -------------------------------------------
def test_correct_numbering_is_ok(sequence):
    df = variants_for(sequence, "P00001", range(1, 31))
    report = check_positions(df)
    assert len(report) == 1
    assert report.loc[0, "verdict"] == "ok"
    assert report.loc[0, "frac_match"] == 1.0
    assert report.loc[0, "best_offset"] == 0


def test_constant_shift_is_detected(sequence):
    # Numbering 10 higher than the sequence, as with an expression tag.
    df = variants_for(sequence, "P00001", range(1, 31), shift=10)
    report = check_positions(df)
    assert report.loc[0, "verdict"] == "offset"
    assert report.loc[0, "best_offset"] == -10
    assert report.loc[0, "frac_match"] < 0.5
    assert report.loc[0, "frac_match_at_best"] == 1.0


def test_negative_shift_is_detected(sequence):
    df = variants_for(sequence, "P00001", range(21, 51), shift=-15)
    report = check_positions(df)
    assert report.loc[0, "verdict"] == "offset"
    assert report.loc[0, "best_offset"] == 15


def test_inconsistent_numbering_is_unresolved(sequence):
    df = variants_for(sequence, "P00001", range(1, 31))
    # Wild-type residues that match nothing at any single offset.
    df["wt_aa"] = "W"
    report = check_positions(df)
    assert report.loc[0, "verdict"] == "unresolved"


def test_missing_sequence_is_reported_not_guessed():
    df = pd.DataFrame(
        {
            "protein_id": ["P00001"],
            "position": [10],
            "wt_aa": ["A"],
            "mut_aa": ["V"],
            "dtm": [1.0],
            "sequence": [None],
        }
    )
    report = check_positions(df)
    assert report.loc[0, "verdict"] == "no_sequence"


def test_offset_beyond_search_range_is_not_claimed(sequence):
    df = variants_for(sequence, "P00001", range(1, 21), shift=10)
    report = check_positions(df, max_offset=3)
    assert report.loc[0, "verdict"] == "unresolved"


# --- applying and reconciling ----------------------------------------
def test_apply_offsets_shifts_and_records_the_original(sequence):
    df = variants_for(sequence, "P00001", range(1, 31), shift=10)
    report = check_positions(df)
    out = apply_offsets(df, report)

    assert "position_original" in out.columns
    assert (out["position_original"] - out["position"] == 10).all()
    # After the shift, positions agree with the sequence.
    assert check_positions(out).loc[0, "verdict"] == "ok"


def test_apply_offsets_leaves_ok_proteins_alone(sequence):
    df = variants_for(sequence, "P00001", range(1, 31))
    out = apply_offsets(df, check_positions(df))
    assert list(out["position"]) == list(df["position"])


def test_reconcile_fixes_one_protein_and_drops_another(sequence):
    good = variants_for(sequence, "P00001", range(1, 31))
    shifted = variants_for(sequence, "P00002", range(1, 31), shift=12)
    broken = variants_for(sequence, "P00003", range(1, 31))
    broken["wt_aa"] = "W"
    df = pd.concat([good, shifted, broken], ignore_index=True)

    out, report = reconcile_positions(df)

    remaining = set(out["protein_id"])
    assert remaining == {"P00001", "P00002"}

    by_protein = report.set_index("protein_id")
    assert by_protein.loc["P00002", "best_offset"] == -12
    assert by_protein.loc["P00002", "verdict_after"] == "ok"
    assert by_protein.loc["P00003", "verdict_after"] == "unresolved"

    # The corrected protein's positions now point at the right residues.
    fixed = out[out["protein_id"] == "P00002"]
    assert all(sequence[int(p) - 1] == w for p, w in zip(fixed["position"], fixed["wt_aa"]))


def test_reconcile_can_keep_unresolved(sequence):
    broken = variants_for(sequence, "P00003", range(1, 31))
    broken["wt_aa"] = "W"
    out, _ = reconcile_positions(broken, drop_unresolved=False)
    assert len(out) == len(broken)


def test_reconcile_works_with_nullable_int_positions(sequence):
    df = variants_for(sequence, "P00001", range(1, 31), shift=10)
    df["position"] = df["position"].astype("Int64")
    out, report = reconcile_positions(df)
    assert report.loc[0, "verdict_after"] == "ok"
    assert len(out) == len(df)

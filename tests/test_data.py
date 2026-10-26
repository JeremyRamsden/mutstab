"""Loading, filtering and reverse-mutation tests."""

from __future__ import annotations

import pandas as pd

from mutstab.data.load import add_reverse_mutations, filter_dtm, load_fireprotdb, summarise


def test_aliases_map_to_canonical_columns(tmp_path):
    raw = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P2"],
            "wild_type": ["A", "G"],
            "mutation": ["V", "W"],
            "residue_number": [10, 20],
            "delta_tm": [1.5, -2.0],
            "p_h": [7.0, 7.4],
        }
    )
    path = tmp_path / "export.csv"
    raw.to_csv(path, index=False)

    df = load_fireprotdb(path)
    assert list(df["protein_id"]) == ["P1", "P2"]
    assert list(df["wt_aa"]) == ["A", "G"]
    assert list(df["mut_aa"]) == ["V", "W"]
    assert list(df["dtm"]) == [1.5, -2.0]
    # Canonical columns absent from the export still exist, as NA.
    assert "ddg" in df.columns and df["ddg"].isna().all()
    assert list(df["variant"]) == ["A10V", "G20W"]


def test_filter_drops_bad_rows_and_logs_them(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "dtm"] = None            # no label
    df.loc[1, "mut_aa"] = "X"          # not a standard residue
    df.loc[2, "mut_aa"] = df.loc[2, "wt_aa"]  # synonymous
    df.loc[3, "dtm"] = 500.0           # implausible magnitude

    clean = filter_dtm(df, drop_duplicates=None)
    assert len(clean) == len(df) - 4
    log = pd.DataFrame(clean.attrs["filter_log"])
    assert log["dropped"].sum() == 4
    assert clean["dtm"].notna().all()


def test_missing_ph_is_kept(synthetic_df):
    df = synthetic_df.copy()
    df.loc[:5, "ph"] = None
    clean = filter_dtm(df, ph_range=(6.0, 8.0), drop_duplicates=None)
    assert clean["ph"].isna().sum() == 6


def test_duplicate_variants_collapse_to_median(synthetic_df):
    row = synthetic_df.iloc[[0]].copy()
    duplicate = row.copy()
    duplicate["dtm"] = row["dtm"].iloc[0] + 4.0
    df = pd.concat([synthetic_df, duplicate], ignore_index=True)

    clean = filter_dtm(df, drop_duplicates="median")
    keys = ["protein_id", "position", "wt_aa", "mut_aa"]
    assert not clean.duplicated(subset=keys).any()
    assert "n_measurements" in clean.columns
    assert clean["n_measurements"].max() >= 2


def test_reverse_mutations_negate_the_label(synthetic_df):
    out = add_reverse_mutations(synthetic_df)
    assert len(out) == 2 * len(synthetic_df)

    forward = out[~out["is_reverse"]].reset_index(drop=True)
    reverse = out[out["is_reverse"]].reset_index(drop=True)

    assert (reverse["wt_aa"].to_numpy() == forward["mut_aa"].to_numpy()).all()
    assert (reverse["mut_aa"].to_numpy() == forward["wt_aa"].to_numpy()).all()
    assert ((reverse["dtm"] + forward["dtm"]).abs() < 1e-9).all()
    # The augmented set is symmetric by construction.
    assert abs(float(out["dtm"].mean())) < 1e-9


def test_summarise_flags_concentration(synthetic_df):
    table = summarise(synthetic_df)
    assert table.index[0] == "P00001"
    assert table["n"].iloc[0] == 120
    assert table["cum_frac_of_data"].iloc[-1] == 1.0

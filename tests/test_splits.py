"""The split tests are the ones that matter most.

A leaking split does not raise, does not look wrong, and inflates every
number downstream. These tests are the only thing standing between that and
a results file you believe.
"""

from __future__ import annotations

import numpy as np
import pytest

from mutstab.data.splits import (
    assert_no_group_leakage,
    fold_indices,
    make_protein_folds,
)


def test_no_protein_in_two_folds(synthetic_df):
    spec = make_protein_folds(synthetic_df, n_folds=3, seed=0)
    assert_no_group_leakage(synthetic_df, spec)

    seen: set[str] = set()
    for members in spec["folds"].values():
        assert not (seen & set(members))
        seen |= set(members)


def test_train_and_val_proteins_are_disjoint(synthetic_df):
    spec = make_protein_folds(synthetic_df, n_folds=3, seed=0)
    for fold in range(3):
        train_idx, val_idx = fold_indices(synthetic_df, spec, fold)
        train_proteins = set(synthetic_df.iloc[train_idx]["protein_id"])
        val_proteins = set(synthetic_df.iloc[val_idx]["protein_id"])
        assert not (train_proteins & val_proteins), f"fold {fold} leaks"
        assert not np.intersect1d(train_idx, val_idx).size


def test_holdout_excluded_from_every_fold(synthetic_df):
    spec = make_protein_folds(synthetic_df, n_folds=3, holdout_frac=0.1, seed=0)
    assert spec["holdout"], "expected a non-empty holdout"
    holdout = set(spec["holdout"])
    for fold in range(3):
        train_idx, val_idx = fold_indices(synthetic_df, spec, fold)
        used = set(synthetic_df.iloc[np.concatenate([train_idx, val_idx])]["protein_id"])
        assert not (used & holdout)


def test_every_row_is_assigned_somewhere(synthetic_df):
    spec = make_protein_folds(synthetic_df, n_folds=3, seed=0)
    covered = set()
    for fold in range(3):
        _, val_idx = fold_indices(synthetic_df, spec, fold)
        covered |= set(val_idx.tolist())
    assert covered == set(range(len(synthetic_df)))


def test_folds_are_deterministic(synthetic_df):
    first = make_protein_folds(synthetic_df, n_folds=3, seed=7)
    second = make_protein_folds(synthetic_df, n_folds=3, seed=7)
    assert first["folds"] == second["folds"]


def test_stale_split_is_detected(synthetic_df):
    spec = make_protein_folds(synthetic_df, n_folds=3, seed=0)
    spec["folds"]["0"] = [g for g in spec["folds"]["0"] if g != "P00001"]
    spec["folds"]["1"] = [g for g in spec["folds"]["1"] if g != "P00001"]
    spec["folds"]["2"] = [g for g in spec["folds"]["2"] if g != "P00001"]
    with pytest.raises(AssertionError, match="in no fold"):
        assert_no_group_leakage(synthetic_df, spec)


def test_duplicate_group_is_detected(synthetic_df):
    spec = make_protein_folds(synthetic_df, n_folds=3, seed=0)
    first = spec["folds"]["0"][0]
    spec["folds"]["1"] = [*spec["folds"]["1"], first]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_group_leakage(synthetic_df, spec)


def test_missing_group_key_rejected(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "protein_id"] = None
    with pytest.raises(ValueError, match="no group key"):
        make_protein_folds(df, n_folds=3)


def test_custom_groups_are_respected(synthetic_df):
    # Pretend P00001 and P00002 are homologs that clustered together.
    clusters = synthetic_df["protein_id"].replace({"P00002": "P00001"})
    spec = make_protein_folds(synthetic_df, n_folds=2, groups=clusters, seed=0)
    assert_no_group_leakage(synthetic_df, spec, groups=clusters)
    for members in spec["folds"].values():
        assert "P00002" not in members

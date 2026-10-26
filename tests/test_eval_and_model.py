"""Metrics behave as claimed, and the CV loop runs end to end."""

from __future__ import annotations

import numpy as np
import pytest

from mutstab.data.splits import make_protein_folds
from mutstab.eval.metrics import (
    antisymmetry_error,
    constant_baseline,
    per_protein_metrics,
    regression_metrics,
    summarise_fold_metrics,
)
from mutstab.features.variant import build_feature_matrix, one_hot_aa
from mutstab.models.baseline import build_model, cross_validate


def test_perfect_prediction_scores_perfectly():
    y = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    out = regression_metrics(y, y)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["auc_stabilising"] == pytest.approx(1.0)


def test_constant_prediction_gives_undefined_correlation():
    y = np.array([-3.0, -1.0, 0.5, 2.0])
    out = regression_metrics(y, np.full_like(y, -1.0))
    assert out["spearman"] is None
    assert out["rmse"] > 0


def test_pooled_can_flatter_a_model_that_only_knows_protein_identity():
    # Two proteins with different means; within each, predictions are noise.
    rng = np.random.default_rng(0)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    y = np.concatenate([rng.normal(-6, 1, 40), rng.normal(6, 1, 40)])
    pred = np.where(groups == "A", -6.0, 6.0) + rng.normal(0, 1, 80)

    pooled = regression_metrics(y, pred)
    per_protein = per_protein_metrics(y, pred, groups)

    # This is the trap the per-protein view exists to expose: a respectable
    # pooled number from a model whose within-protein ranking is noise.
    assert pooled["spearman"] > 0.7
    assert abs(per_protein["median_spearman"]) < 0.3


def test_small_proteins_are_excluded_from_per_protein():
    y = np.arange(10.0)
    groups = ["A"] * 7 + ["B"] * 3
    out = per_protein_metrics(y, y, groups, min_per_group=5)
    assert out["n_proteins"] == 1
    assert out["per_protein"][0]["protein_id"] == "A"


def test_antisymmetry_detects_a_one_sided_model():
    forward = np.array([-2.0, -3.0, -1.0])
    # A model that calls both directions destabilising.
    reverse = np.array([-2.0, -3.0, -1.0])
    out = antisymmetry_error(forward, reverse)
    assert out["rmse_antisymmetry"] > 2.0
    assert out["mean_signed_sum"] < 0

    perfect = antisymmetry_error(forward, -forward)
    assert perfect["rmse_antisymmetry"] == pytest.approx(0.0)


def test_constant_baseline_uses_the_training_mean():
    out = constant_baseline([1.0, 3.0], [2.0, 2.0])
    assert out["prediction"] == pytest.approx(2.0)
    assert out["rmse"] == pytest.approx(0.0)


def test_fold_summary_reports_spread():
    out = summarise_fold_metrics([{"rmse": 1.0}, {"rmse": 3.0}])
    assert out["rmse_mean"] == pytest.approx(2.0)
    assert out["rmse_std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert out["n_folds"] == 2


def test_one_hot_handles_unknown_residues():
    out = one_hot_aa(["A", "X", "W"])
    assert out[0].sum() == 1
    assert out[1].sum() == 0  # unknown -> all zero, not a wrong guess
    assert out[2].sum() == 1


def test_feature_matrix_without_embeddings(synthetic_df):
    X, names, valid = build_feature_matrix(synthetic_df, use_embeddings=False)
    assert X.shape[0] == len(synthetic_df)
    assert X.shape[1] == len(names)
    assert valid.all()
    assert "delta_hydropathy" in names


def test_delta_features_are_the_difference(synthetic_df):
    X, names, _ = build_feature_matrix(synthetic_df, use_embeddings=False)
    wt = X[:, names.index("wt_volume")]
    mut = X[:, names.index("mut_volume")]
    delta = X[:, names.index("delta_volume")]
    assert np.allclose(delta, mut - wt)


def test_cross_validate_runs_and_beats_nothing_silently(synthetic_df):
    X, _, valid = build_feature_matrix(synthetic_df, use_embeddings=False)
    df = synthetic_df.loc[valid].reset_index(drop=True)
    spec = make_protein_folds(df, n_folds=3, seed=0)

    result = cross_validate(df, X, spec, {"kind": "ridge", "alpha": 10.0})

    assert len(result["folds"]) == 3
    assert result["pooled_oof"]["n"] > 0
    # Labels here are pure noise, so the model should not beat the baseline by
    # much. The point of the assertion is that both numbers are produced and
    # comparable at all.
    assert result["baseline_oof"]["rmse"] > 0
    assert np.isfinite(result["oof_predictions"][result["scored_mask"]]).all()


def test_misaligned_features_are_rejected(synthetic_df):
    X, _, _ = build_feature_matrix(synthetic_df, use_embeddings=False)
    spec = make_protein_folds(synthetic_df, n_folds=3, seed=0)
    with pytest.raises(ValueError, match="rows but X has"):
        cross_validate(synthetic_df.iloc[:-5], X, spec, {"kind": "ridge"})


def test_build_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model({"kind": "transformer"})

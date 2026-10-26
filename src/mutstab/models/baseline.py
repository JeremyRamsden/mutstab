"""Baseline models and the cross-validation loop.

Ridge over embeddings plus residue properties is the baseline worth beating.
It trains in seconds, has one hyperparameter, and on small, biased datasets it
is a genuinely hard target — regularised linear models are hard to beat when
there are a few thousand labels and a thousand features.

The loop is scikit-learn only, deliberately. Once a torch model is warranted,
keep this signature — take fold indices, return per-fold metrics — so the
comparison stays apples to apples.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mutstab.data.splits import fold_indices
from mutstab.eval.metrics import (
    constant_baseline,
    per_protein_metrics,
    regression_metrics,
    summarise_fold_metrics,
)

__all__ = ["build_model", "cross_validate"]


def build_model(spec: Mapping[str, Any]) -> Pipeline:
    """Build a model from a config block.

    Expected shape::

        model:
          kind: ridge          # ridge | hgb | rf
          alpha: 10.0          # ridge only
    """
    kind = str(spec.get("kind", "ridge")).lower()
    params = {k: v for k, v in spec.items() if k != "kind"}

    if kind == "ridge":
        # Scaling matters here: embedding dimensions and volume in cubic
        # angstroms are on wildly different scales, and a single alpha has to
        # serve both.
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=float(params.get("alpha", 10.0)))),
            ]
        )

    if kind in {"hgb", "hist_gradient_boosting"}:
        return Pipeline(
            [
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=int(params.get("max_iter", 300)),
                        learning_rate=float(params.get("learning_rate", 0.05)),
                        max_depth=params.get("max_depth"),
                        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
                        l2_regularization=float(params.get("l2_regularization", 1.0)),
                        early_stopping=bool(params.get("early_stopping", True)),
                        random_state=int(params.get("random_state", 0)),
                    ),
                )
            ]
        )

    if kind in {"rf", "random_forest"}:
        return Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(params.get("n_estimators", 400)),
                        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
                        max_features=params.get("max_features", "sqrt"),
                        n_jobs=int(params.get("n_jobs", -1)),
                        random_state=int(params.get("random_state", 0)),
                    ),
                )
            ]
        )

    raise ValueError(f"unknown model kind {kind!r}; expected ridge, hgb or rf")


def cross_validate(
    df: pd.DataFrame,
    X: np.ndarray,
    split_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    *,
    target: str = "dtm",
    group_column: str = "protein_id",
    min_per_group: int = 5,
    stabilising_threshold: float = 1.0,
    on_fold: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train and score one model across every fold of a split spec.

    `df` and `X` must be row-aligned: featurise first, then subset `df` by the
    returned valid mask, so a dropped row cannot silently shift the labels.

    Returns per-fold metrics, their across-fold summary, and the constant
    baseline for comparison. Also returns out-of-fold predictions, which are
    what you want for residual plots — each prediction comes from a model that
    never saw that protein.
    """
    if len(df) != X.shape[0]:
        raise ValueError(
            f"df has {len(df)} rows but X has {X.shape[0]}; "
            "subset df by the valid mask from build_feature_matrix"
        )

    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    folds: Sequence[str] = sorted(split_spec["folds"], key=int)

    oof = np.full(len(df), np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []

    for fold_name in folds:
        fold = int(fold_name)
        train_idx, val_idx = fold_indices(df, split_spec, fold)

        # Rows whose label is missing can still be featurised; drop them from
        # training rather than letting NaN poison the fit.
        train_idx = train_idx[np.isfinite(y[train_idx])]
        val_idx = val_idx[np.isfinite(y[val_idx])]
        if train_idx.size == 0 or val_idx.size == 0:
            continue

        model = build_model(model_spec)
        model.fit(X[train_idx], y[train_idx])
        prediction = model.predict(X[val_idx])
        oof[val_idx] = prediction

        groups = df[group_column].astype("string").to_numpy()
        metrics = {
            "fold": fold,
            "n_train": int(train_idx.size),
            **regression_metrics(
                y[val_idx], prediction, stabilising_threshold=stabilising_threshold
            ),
        }
        protein_view = per_protein_metrics(
            y[val_idx], prediction, groups[val_idx], min_per_group=min_per_group
        )
        metrics["median_spearman_per_protein"] = protein_view["median_spearman"]
        metrics["n_proteins_scored"] = protein_view["n_proteins"]
        metrics["baseline_rmse"] = constant_baseline(y[train_idx], y[val_idx])["rmse"]

        fold_metrics.append(metrics)
        if on_fold is not None:
            on_fold(fold, metrics)

    scored = np.isfinite(oof) & np.isfinite(y)
    groups = df[group_column].astype("string").to_numpy()

    pooled = regression_metrics(
        y[scored], oof[scored], stabilising_threshold=stabilising_threshold
    )
    protein_view = per_protein_metrics(
        y[scored], oof[scored], groups[scored], min_per_group=min_per_group
    )

    return {
        "folds": fold_metrics,
        "across_folds": summarise_fold_metrics(fold_metrics),
        "pooled_oof": pooled,
        "per_protein_oof": protein_view,
        "baseline_oof": constant_baseline(y[scored], y[scored]),
        "oof_predictions": oof,
        "scored_mask": scored,
    }

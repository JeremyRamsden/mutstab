"""Metrics for dTm prediction.

Report per-protein numbers, not just pooled ones. A pooled correlation over a
dataset where proteins differ in mean dTm rewards a model for ranking
proteins against each other, which is easy, and says nothing about ranking
mutations within a protein, which is the thing anyone designing a variant
actually needs. The two can differ enormously: pooled Spearman of 0.6
alongside a median per-protein Spearman near 0.2 is a common and
uncomfortable result.

Always compare against `constant_baseline`. If predicting each protein's mean
scores close to your model, the model has learned protein identity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "regression_metrics",
    "per_protein_metrics",
    "constant_baseline",
    "antisymmetry_error",
    "summarise_fold_metrics",
]


def _clean(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {pred.shape}")
    keep = np.isfinite(true) & np.isfinite(pred)
    return true[keep], pred[keep]


def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    stabilising_threshold: float = 1.0,
) -> dict[str, Any]:
    """Pooled metrics over all predictions.

    `stabilising_threshold` (degrees C) defines the "is this mutation
    stabilising?" classification. 1 C is a reasonable floor given typical
    measurement error; below that, the label is mostly noise.
    """
    true, pred = _clean(y_true, y_pred)
    n = int(true.size)
    out: dict[str, Any] = {"n": n}
    if n < 2:
        return out | {
            "rmse": None, "mae": None, "pearson": None,
            "spearman": None, "auc_stabilising": None,
        }

    residual = pred - true
    out["rmse"] = float(np.sqrt(np.mean(residual**2)))
    out["mae"] = float(np.mean(np.abs(residual)))
    out["bias"] = float(np.mean(residual))

    # Zero variance on either side makes correlation undefined, which happens
    # with a model that collapsed to a constant — report None, not a crash.
    if np.std(true) > 0 and np.std(pred) > 0:
        out["pearson"] = float(stats.pearsonr(true, pred).statistic)
        out["spearman"] = float(stats.spearmanr(true, pred).statistic)
    else:
        out["pearson"] = None
        out["spearman"] = None

    labels = true >= stabilising_threshold
    if labels.any() and not labels.all():
        order = np.argsort(pred)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        n_pos = int(labels.sum())
        n_neg = n - n_pos
        out["auc_stabilising"] = float(
            (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        )
        out["frac_stabilising"] = float(n_pos / n)
    else:
        out["auc_stabilising"] = None
        out["frac_stabilising"] = float(labels.mean())

    return out


def per_protein_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[Any],
    *,
    min_per_group: int = 5,
) -> dict[str, Any]:
    """Within-protein correlations, aggregated.

    Proteins with fewer than `min_per_group` measurements are excluded: a
    Spearman over three points is noise, and averaging it in makes the summary
    unstable. The median is the headline; the spread tells you how much to
    trust it.
    """
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=float),
            "p": np.asarray(y_pred, dtype=float),
            "g": pd.Series(list(groups)).astype("string").to_numpy(),
        }
    ).dropna()

    rows: list[dict[str, Any]] = []
    for group, subset in frame.groupby("g"):
        if len(subset) < min_per_group:
            continue
        if subset["y"].std() == 0 or subset["p"].std() == 0:
            continue
        rows.append(
            {
                "protein_id": group,
                "n": len(subset),
                "spearman": float(stats.spearmanr(subset["y"], subset["p"]).statistic),
                "pearson": float(stats.pearsonr(subset["y"], subset["p"]).statistic),
                "rmse": float(np.sqrt(np.mean((subset["p"] - subset["y"]) ** 2))),
            }
        )

    if not rows:
        return {
            "n_proteins": 0,
            "median_spearman": None,
            "mean_spearman": None,
            "iqr_spearman": None,
            "frac_positive_spearman": None,
            "per_protein": [],
        }

    table = pd.DataFrame(rows)
    spearman = table["spearman"].to_numpy()
    return {
        "n_proteins": int(len(table)),
        "median_spearman": float(np.median(spearman)),
        "mean_spearman": float(np.mean(spearman)),
        "iqr_spearman": [
            float(np.percentile(spearman, 25)),
            float(np.percentile(spearman, 75)),
        ],
        "frac_positive_spearman": float(np.mean(spearman > 0)),
        "median_rmse": float(table["rmse"].median()),
        "per_protein": table.sort_values("n", ascending=False).to_dict("records"),
    }


def constant_baseline(
    y_train: Sequence[float],
    y_true: Sequence[float],
    *,
    groups_train: Sequence[Any] | None = None,
    groups_eval: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Score of predicting the training mean for everything.

    With protein-level splits the evaluated proteins are unseen, so there is no
    per-protein mean to fall back on — the global mean is the honest floor.
    Any model not clearly beating this RMSE has learned nothing useful.
    """
    train = np.asarray(y_train, dtype=float)
    train = train[np.isfinite(train)]
    if train.size == 0:
        raise ValueError("no finite training labels")
    prediction = float(np.mean(train))
    true = np.asarray(y_true, dtype=float)
    true = true[np.isfinite(true)]
    return {
        "prediction": prediction,
        "rmse": float(np.sqrt(np.mean((true - prediction) ** 2))),
        "mae": float(np.mean(np.abs(true - prediction))),
        "n": int(true.size),
    }


def antisymmetry_error(
    forward_pred: Sequence[float], reverse_pred: Sequence[float]
) -> dict[str, Any]:
    """How far the model is from f(A->B) == -f(B->A).

    Physics requires this exactly. A model trained only on measured mutations
    usually violates it badly, which is a direct read on how much it has
    learned "most mutations are destabilising" rather than anything about the
    substitution. Needs predictions for both directions of the same variants,
    in the same order.
    """
    forward = np.asarray(forward_pred, dtype=float)
    reverse = np.asarray(reverse_pred, dtype=float)
    if forward.shape != reverse.shape:
        raise ValueError(f"shape mismatch: {forward.shape} vs {reverse.shape}")
    keep = np.isfinite(forward) & np.isfinite(reverse)
    forward, reverse = forward[keep], reverse[keep]
    if forward.size == 0:
        return {"n": 0, "mean_signed_sum": None, "rmse_antisymmetry": None, "bias": None}
    total = forward + reverse  # should be 0
    return {
        "n": int(forward.size),
        "mean_signed_sum": float(np.mean(total)),
        "rmse_antisymmetry": float(np.sqrt(np.mean(total**2))),
        # A negative bias means the model calls both directions destabilising.
        "bias": float(np.mean(forward) + np.mean(reverse)) / 2.0,
    }


def summarise_fold_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of each scalar metric across folds.

    The standard deviation is the part to look at. With protein-level splits
    and a few dominant proteins, fold-to-fold spread is often large enough
    that a small improvement in the mean means nothing.
    """
    if not fold_metrics:
        return {}
    keys = [
        key
        for key in fold_metrics[0]
        if isinstance(fold_metrics[0][key], (int, float)) and not isinstance(fold_metrics[0][key], bool)
    ]
    out: dict[str, Any] = {"n_folds": len(fold_metrics)}
    for key in keys:
        values = [
            float(m[key])
            for m in fold_metrics
            if isinstance(m.get(key), (int, float)) and m.get(key) is not None
        ]
        if not values:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return out

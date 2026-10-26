#!/usr/bin/env python
"""Train and score one config, then write a run record.

    python scripts/train.py --config baseline
    python scripts/train.py --config baseline --no-embeddings   # quick check

Everything it needs comes from the config, and everything it learned goes into
results/<run_id>.json. The run id contains the config hash, so two different
configs cannot overwrite each other's results.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mutstab import paths, runs
from mutstab.config import load_config
from mutstab.data.load import add_reverse_mutations
from mutstab.data.splits import assert_no_group_leakage, load_splits
from mutstab.features.variant import build_feature_matrix
from mutstab.models.baseline import cross_validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="properties only — fast sanity check with no embedding store",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"run: {cfg.run_id}")

    # --- data --------------------------------------------------------
    data_path = paths.processed_dir() / cfg.get_path("data.processed_file")
    if not data_path.exists():
        raise SystemExit(f"no processed data at {data_path}; run prepare_data.py first")
    # Stage off Drive onto local disk: one sequential copy beats many FUSE reads.
    df = pd.read_parquet(paths.stage(data_path))
    print(f"{len(df):,} rows, {df['protein_id'].nunique():,} proteins")

    split_spec = load_splits(cfg.get_path("split.file"))
    assert_no_group_leakage(df, split_spec)

    if cfg.get_path("data.add_reverse", False):
        df = add_reverse_mutations(df)
        print(f"with reverse mutations: {len(df):,} rows")

    # --- features ----------------------------------------------------
    use_embeddings = bool(cfg.get_path("features.use_embeddings", True)) and not args.no_embeddings
    store = None
    try:
        if use_embeddings:
            from mutstab.features.embeddings import EmbeddingStore

            store_path = paths.cache_dir() / cfg.get_path("features.embedding_store")
            if not store_path.exists():
                raise SystemExit(
                    f"no embedding store at {store_path}; run scripts/embed.py first\n"
                    "(or pass --no-embeddings for a properties-only baseline)"
                )
            store = EmbeddingStore(paths.stage(store_path), mode="r")
            print(f"embeddings: {len(store)} proteins, dim {store.dim}")

        X, names, valid = build_feature_matrix(
            df,
            store=store,
            use_embeddings=use_embeddings,
            include_mean_embedding=bool(
                cfg.get_path("features.include_mean_embedding", False)
            ),
            extra_columns=cfg.get_path("features.extra_columns", []) or [],
        )
    finally:
        if store is not None:
            store.close()

    dropped = int((~valid).sum())
    if dropped:
        print(f"dropped {dropped:,} rows that could not be featurised")
    df = df.loc[valid].reset_index(drop=True)
    print(f"X: {X.shape[0]:,} x {X.shape[1]:,}")

    # --- train -------------------------------------------------------
    def report(fold: int, metrics: dict) -> None:
        spearman = metrics.get("spearman")
        per_protein = metrics.get("median_spearman_per_protein")
        print(
            f"  fold {fold}: n={metrics['n']:,} rmse={metrics['rmse']:.2f} "
            f"(baseline {metrics['baseline_rmse']:.2f}) "
            f"spearman={spearman if spearman is None else round(spearman, 3)} "
            f"per-protein={per_protein if per_protein is None else round(per_protein, 3)}"
        )

    print("\ncross-validating")
    result = cross_validate(
        df,
        X,
        split_spec,
        cfg.get_path("model", {}),
        target=cfg.get_path("data.target", "dtm"),
        min_per_group=int(cfg.get_path("eval.min_per_group", 5)),
        stabilising_threshold=float(cfg.get_path("eval.stabilising_threshold", 1.0)),
        on_fold=report,
    )

    pooled = result["pooled_oof"]
    protein = result["per_protein_oof"]
    print("\nout-of-fold, pooled")
    print(f"  rmse      {pooled['rmse']:.2f}  (constant baseline {result['baseline_oof']['rmse']:.2f})")
    print(f"  spearman  {pooled['spearman']}")
    print(f"\nout-of-fold, per protein ({protein['n_proteins']} proteins scored)")
    print(f"  median spearman  {protein['median_spearman']}")
    print(f"  IQR              {protein['iqr_spearman']}")
    print(
        "\nThe per-protein number is the one to quote. If it is far below the\n"
        "pooled number, the model is ranking proteins, not mutations."
    )

    # --- record ------------------------------------------------------
    metrics = {
        "pooled_oof": pooled,
        "per_protein_oof": {k: v for k, v in protein.items() if k != "per_protein"},
        "across_folds": result["across_folds"],
        "baseline_oof": result["baseline_oof"],
        "folds": result["folds"],
    }
    out = runs.save_run(
        cfg,
        metrics,
        split_file=split_spec.get("_path"),
        n_train=int(X.shape[0]),
        n_eval=int(result["scored_mask"].sum()),
        notes=args.notes,
        extra={"n_features": int(X.shape[1]), "used_embeddings": use_embeddings},
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

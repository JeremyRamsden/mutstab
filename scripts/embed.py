#!/usr/bin/env python
"""Fill the embedding cache. Resumable — rerun until it says nothing to do.

    python scripts/embed.py --config baseline

Needs a GPU runtime and the plm extra:

    pip install 'mutstab[plm,store]'

Sequences come from the dataset's `sequence` column where present. If your
export lacks sequences, fetch them from UniProt first and merge them in;
without the reference sequence, positions cannot be verified and every
position-indexed feature is a guess.
"""

from __future__ import annotations

import argparse

import pandas as pd

from mutstab import paths
from mutstab.config import load_config
from mutstab.features.embeddings import embed_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = paths.processed_dir() / cfg.get_path("data.processed_file")
    if not data_path.exists():
        raise SystemExit(f"no processed data at {data_path}; run prepare_data.py first")

    df = pd.read_parquet(data_path)
    if "sequence" not in df.columns or df["sequence"].isna().all():
        raise SystemExit(
            "the dataset has no sequences. Fetch them from UniProt by accession\n"
            "and merge into the processed file before embedding."
        )

    pairs = (
        df[["protein_id", "sequence"]]
        .dropna()
        .drop_duplicates(subset="protein_id")
        .astype({"protein_id": "string", "sequence": "string"})
    )
    sequences = {str(r.protein_id): str(r.sequence) for r in pairs.itertuples()}
    print(f"{len(sequences)} unique proteins in {data_path.name}")

    store_path = paths.cache_dir(create=True) / cfg.get_path("features.embedding_store")
    embed_sequences(
        sequences,
        store_path,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(f"store: {store_path}")


if __name__ == "__main__":
    main()

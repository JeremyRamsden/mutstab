"""Per-variant feature vectors.

The baseline feature set is intentionally plain: residue identities, a few
physicochemical deltas, and the language model embedding at the mutated
position. It exists to answer "does the complicated model beat this?", which
is a question worth being able to answer cheaply before investing in a
structure-aware architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from mutstab.features.embeddings import EmbeddingStore

__all__ = ["AA_ORDER", "AA_PROPERTIES", "one_hot_aa", "build_feature_matrix"]

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Kyte-Doolittle hydropathy, van der Waals volume (A^3), charge at pH 7,
# and a flexibility proxy. Enough to give a linear model something physical
# to work with without pulling in a feature library.
AA_PROPERTIES: dict[str, tuple[float, float, float, float]] = {
    "A": (1.8, 88.6, 0.0, 0.36),
    "C": (2.5, 108.5, 0.0, 0.35),
    "D": (-3.5, 111.1, -1.0, 0.51),
    "E": (-3.5, 138.4, -1.0, 0.50),
    "F": (2.8, 189.9, 0.0, 0.31),
    "G": (-0.4, 60.1, 0.0, 0.54),
    "H": (-3.2, 153.2, 0.1, 0.32),
    "I": (4.5, 166.7, 0.0, 0.46),
    "K": (-3.9, 168.6, 1.0, 0.47),
    "L": (3.8, 166.7, 0.0, 0.37),
    "M": (1.9, 162.9, 0.0, 0.30),
    "N": (-3.5, 114.1, 0.0, 0.46),
    "P": (-1.6, 112.7, 0.0, 0.51),
    "Q": (-3.5, 143.8, 0.0, 0.49),
    "R": (-4.5, 173.4, 1.0, 0.53),
    "S": (-0.8, 89.0, 0.0, 0.51),
    "T": (-0.7, 116.1, 0.0, 0.44),
    "V": (4.2, 140.0, 0.0, 0.39),
    "W": (-0.9, 227.8, 0.0, 0.31),
    "Y": (-1.3, 193.6, 0.0, 0.42),
}

_PROPERTY_NAMES = ("hydropathy", "volume", "charge", "flexibility")


def one_hot_aa(residues: Sequence[str]) -> np.ndarray:
    """(n, 20) one-hot encoding. Unknown residues become all-zero rows."""
    out = np.zeros((len(residues), len(AA_ORDER)), dtype=np.float32)
    for row, residue in enumerate(residues):
        index = _AA_INDEX.get(str(residue).strip().upper())
        if index is not None:
            out[row, index] = 1.0
    return out


def _property_matrix(residues: Sequence[str]) -> np.ndarray:
    out = np.zeros((len(residues), len(_PROPERTY_NAMES)), dtype=np.float32)
    for row, residue in enumerate(residues):
        values = AA_PROPERTIES.get(str(residue).strip().upper())
        if values is not None:
            out[row] = values
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    store: "EmbeddingStore | None" = None,
    use_embeddings: bool = True,
    include_mean_embedding: bool = False,
    extra_columns: Sequence[str] = (),
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build (X, feature_names, valid_mask) for the rows of `df`.

    `valid_mask` marks rows that could be featurised — a protein missing from
    the embedding store, or a position outside its sequence, yields False.
    Filter both X and your labels by it rather than letting zeros stand in for
    missing features.

    Returned X already has rows dropped, so `X.shape[0] == valid_mask.sum()`.
    """
    wt = df["wt_aa"].astype("string").fillna("").tolist()
    mut = df["mut_aa"].astype("string").fillna("").tolist()

    blocks: list[np.ndarray] = []
    names: list[str] = []

    blocks.append(one_hot_aa(wt))
    names += [f"wt_{aa}" for aa in AA_ORDER]
    blocks.append(one_hot_aa(mut))
    names += [f"mut_{aa}" for aa in AA_ORDER]

    wt_props = _property_matrix(wt)
    mut_props = _property_matrix(mut)
    blocks.append(wt_props)
    names += [f"wt_{p}" for p in _PROPERTY_NAMES]
    blocks.append(mut_props)
    names += [f"mut_{p}" for p in _PROPERTY_NAMES]
    # The delta is what a stability model mostly cares about, and giving it
    # directly saves a linear model from having to discover the subtraction.
    blocks.append(mut_props - wt_props)
    names += [f"delta_{p}" for p in _PROPERTY_NAMES]

    for column in extra_columns:
        if column not in df.columns:
            raise KeyError(f"extra column {column!r} not in the dataframe")
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float32)
        blocks.append(np.nan_to_num(values).reshape(-1, 1))
        names.append(column)

    valid = np.ones(len(df), dtype=bool)

    if use_embeddings:
        if store is None:
            raise ValueError("use_embeddings=True needs an EmbeddingStore")
        dim = store.dim
        if dim is None:
            raise ValueError(f"{store.path.name} has no 'dim' attribute; rebuild it")

        position_block = np.zeros((len(df), dim), dtype=np.float32)
        mean_block = np.zeros((len(df), dim), dtype=np.float32) if include_mean_embedding else None

        protein_ids = df["protein_id"].astype("string").tolist()
        positions = df["position"].tolist()

        # Group by protein so each dataset is read from the HDF5 file once.
        order: dict[str, list[int]] = {}
        for row, pid in enumerate(protein_ids):
            order.setdefault(str(pid), []).append(row)

        for pid, rows in order.items():
            if pid not in store:
                valid[rows] = False
                continue
            array = store.get(pid)
            mean_vector = array.mean(axis=0) if include_mean_embedding else None
            for row in rows:
                position = positions[row]
                if pd.isna(position):
                    valid[row] = False
                    continue
                index = int(position) - 1
                if not 0 <= index < array.shape[0]:
                    valid[row] = False
                    continue
                position_block[row] = array[index]
                if mean_block is not None and mean_vector is not None:
                    mean_block[row] = mean_vector

        blocks.append(position_block)
        names += [f"emb_pos_{i}" for i in range(dim)]
        if mean_block is not None:
            blocks.append(mean_block)
            names += [f"emb_mean_{i}" for i in range(dim)]

    X = np.hstack(blocks).astype(np.float32)
    return X[valid], names, valid

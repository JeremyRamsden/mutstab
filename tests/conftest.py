"""Shared fixtures. A synthetic dataset shaped like the real one.

Deliberately imbalanced — one protein holds most of the rows — because that is
the property the split logic has to cope with, and a tidy fixture would hide
bugs that only appear on skewed data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # (protein, n_rows) — the first protein dominates, as in real data.
    layout = [("P00001", 120), ("P00002", 40), ("P00003", 25), ("P00004", 10), ("P00005", 5)]
    for protein, count in layout:
        length = 150
        sequence = "".join(rng.choice(list(AA), size=length))
        for _ in range(count):
            position = int(rng.integers(1, length + 1))
            wt = sequence[position - 1]
            mut = str(rng.choice([a for a in AA if a != wt]))
            rows.append(
                {
                    "protein_id": protein,
                    "protein_name": f"protein-{protein}",
                    "pdb_id": None,
                    "chain": "A",
                    "position": position,
                    "wt_aa": wt,
                    "mut_aa": mut,
                    # Skewed destabilising, like the literature.
                    "dtm": float(rng.normal(-2.0, 3.0)),
                    "ddg": float(rng.normal(-0.8, 1.2)),
                    "tm_wt": 55.0,
                    "ph": 7.0,
                    "method": "DSC",
                    "sequence": sequence,
                    "source": "synthetic",
                }
            )
    df = pd.DataFrame(rows)
    df["variant"] = df["wt_aa"] + df["position"].astype(str) + df["mut_aa"]
    return df

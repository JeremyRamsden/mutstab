"""Embedding cache: compute once, reuse for every experiment.

Protein language model inference is the slowest step in this pipeline and the
one most likely to be interrupted by a Colab disconnect. So it is a separate,
resumable stage with its own output, not something a training script does on
the way past:

    1. Run `scripts/embed.py` until it reports nothing left to do.
    2. Treat the store as fixed input from then on.

Storage is a single HDF5 file keyed by protein id. This matters more than it
sounds: one file per protein means tens of thousands of small files on a FUSE
mount, and Drive will take hours to do what local disk does in seconds. One
file, opened once, staged to local disk for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

__all__ = ["EmbeddingStore", "embed_sequences"]


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "EmbeddingStore needs h5py. Install it with: pip install 'mutstab[store]'"
        ) from exc
    return h5py


class EmbeddingStore:
    """Per-residue embeddings for many proteins in one HDF5 file.

    Each protein is one dataset of shape (length, dim), float16 by default.
    float16 halves the file and the read time; the precision loss is far below
    the noise in any experimental dTm you will compare against.

    Usage::

        with EmbeddingStore(path, mode="a", dim=1280) as store:
            for pid, seq in todo:
                if pid in store:
                    continue
                store.put(pid, embed_one(seq), sequence=seq)
    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        *,
        dim: int | None = None,
        dtype: str = "float16",
        model_name: str = "",
    ):
        self._h5py = _require_h5py()
        self.path = Path(path)
        if mode in {"a", "w"}:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._h5py.File(self.path, mode)
        self.dtype = dtype
        if mode in {"a", "w"}:
            if dim is not None:
                self._file.attrs.setdefault("dim", int(dim))
            if model_name:
                self._file.attrs.setdefault("model_name", model_name)

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> EmbeddingStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    # -- introspection ----------------------------------------------------
    @property
    def dim(self) -> int | None:
        value = self._file.attrs.get("dim")
        return int(value) if value is not None else None

    @property
    def model_name(self) -> str:
        return str(self._file.attrs.get("model_name", ""))

    def __contains__(self, protein_id: str) -> bool:
        return str(protein_id) in self._file

    def __len__(self) -> int:
        return len(self._file)

    def keys(self) -> list[str]:
        return list(self._file.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self._file)

    def missing(self, protein_ids: Iterable[str]) -> list[str]:
        """Which ids are not yet stored — the resume list."""
        present = set(self._file.keys())
        seen: set[str] = set()
        out: list[str] = []
        for pid in protein_ids:
            key = str(pid)
            if key in present or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    # -- read / write -----------------------------------------------------
    def put(self, protein_id: str, array: np.ndarray, *, sequence: str = "") -> None:
        key = str(protein_id)
        if array.ndim != 2:
            raise ValueError(f"expected (length, dim) for {key}, got {array.shape}")
        if key in self._file:
            del self._file[key]
        dataset = self._file.create_dataset(
            key,
            data=array.astype(self.dtype, copy=False),
            compression="gzip",
            compression_opts=4,
        )
        if sequence:
            dataset.attrs["sequence"] = sequence
            if len(sequence) != array.shape[0]:
                # Not fatal — some models emit BOS/EOS tokens — but a silent
                # off-by-one here shifts every position-indexed feature.
                dataset.attrs["length_mismatch"] = True
        self._file.attrs.setdefault("dim", int(array.shape[1]))
        self._file.flush()

    def get(self, protein_id: str) -> np.ndarray:
        key = str(protein_id)
        if key not in self._file:
            raise KeyError(f"{key} not in {self.path.name}")
        return np.asarray(self._file[key][()], dtype=np.float32)

    def get_position(self, protein_id: str, position: int) -> np.ndarray:
        """Embedding of one residue. `position` is 1-based, as in the data."""
        array = self.get(protein_id)
        index = int(position) - 1
        if not 0 <= index < array.shape[0]:
            raise IndexError(
                f"position {position} outside {protein_id} of length {array.shape[0]}"
            )
        return array[index]

    def get_mean(self, protein_id: str) -> np.ndarray:
        """Whole-protein mean embedding."""
        return self.get(protein_id).mean(axis=0)

    def sequence(self, protein_id: str) -> str:
        return str(self._file[str(protein_id)].attrs.get("sequence", ""))


def embed_sequences(
    sequences: Mapping[str, str],
    store_path: str | Path,
    *,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 1,
    device: str | None = None,
    max_length: int = 1022,
    progress: bool = True,
) -> Path:
    """Compute per-residue embeddings for anything not already stored.

    Safe to re-run: finished proteins are skipped, so a disconnect costs only
    the protein in flight. Needs the `plm` extra (torch + fair-esm) and a GPU
    runtime to be anything but slow.

    Sequences longer than `max_length` are skipped rather than truncated — a
    truncated protein gives wrong positions for every variant past the cut,
    which is worse than missing data. Handle long proteins with a windowed
    pass if your dataset needs them.
    """
    try:
        import esm  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "embedding needs torch and fair-esm. In Colab:\n"
            "    pip install 'mutstab[plm]'\n"
            "and pick a GPU runtime."
        ) from exc

    store_path = Path(store_path)
    with EmbeddingStore(store_path, mode="a", model_name=model_name) as store:
        todo = [pid for pid in store.missing(sequences.keys()) if sequences[pid]]
        skipped = [pid for pid in todo if len(sequences[pid]) > max_length]
        todo = [pid for pid in todo if len(sequences[pid]) <= max_length]

        if skipped:
            print(f"skipping {len(skipped)} sequences longer than {max_length}")
        if not todo:
            print(f"nothing to do: {len(store)} proteins already in {store_path.name}")
            return store_path

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"embedding {len(todo)} proteins with {model_name} on {device}")

        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model = model.to(device).eval()
        batch_converter = alphabet.get_batch_converter()
        layer = model.num_layers

        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            data: Sequence[tuple[str, str]] = [(pid, sequences[pid]) for pid in chunk]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            with torch.inference_mode():
                out = model(tokens, repr_layers=[layer])["representations"][layer]
            for row, (pid, seq) in enumerate(data):
                # Strip the BOS token and trailing padding so row i is residue
                # i+1 of the sequence, matching 1-based positions in the data.
                array = out[row, 1 : len(seq) + 1].float().cpu().numpy()
                store.put(pid, array, sequence=seq)
            if progress:
                done = min(start + batch_size, len(todo))
                print(f"  {done}/{len(todo)}", flush=True)

    return store_path

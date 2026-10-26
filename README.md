# mutstab

Predicting the effect of point mutations on protein melting temperature (Tm).

The structure here is built around three constraints: code in git, data in
Drive, compute in Colab. Nothing important lives only in a notebook, and
nothing large lives in the repo.

## Layout

```
src/mutstab/        the real code — installed as a package
  paths.py          where data lives; the Colab/local switch
  config.py         YAML configs, hashed for traceability
  runs.py           one JSON per run: config + git SHA + metrics
  data/             loading, filtering, split definitions
  features/         embedding cache, per-variant feature vectors
  models/           models and the cross-validation loop
  eval/             metrics, including the per-protein view
configs/            one YAML per experiment
scripts/            prepare_data -> make_splits -> embed -> train
notebooks/          thin drivers that import from src/
splits/             split definitions, committed
results/            run records, committed
tests/              the split tests are the important ones
```

Notebooks stay thin on purpose. Logic in a notebook cannot be tested, diffed,
or reused, and a notebook that grows logic becomes the thing you are afraid to
change.

## Setup in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<you>/mutstab.git /content/mutstab
%pip install -qe /content/mutstab

from mutstab import paths
paths.mount_drive()
print(paths.describe())
```

Then run the pipeline in order:

```bash
python scripts/prepare_data.py --config baseline    # raw export -> parquet
python scripts/make_splits.py  --config baseline    # protein-level folds, once
python scripts/train.py --config properties_only    # no GPU needed
python scripts/embed.py --config baseline           # GPU; resumable
python scripts/train.py --config baseline
```

Locally, point `MUTSTAB_DATA_ROOT` at a folder and the same commands work:

```bash
export MUTSTAB_DATA_ROOT=~/data/mutstab
pip install -e '.[dev,store]'
pytest
```

## Where data goes

| What | Where | In git? |
|---|---|---|
| Raw exports | Drive `raw/` | no |
| Processed Parquet | Drive `processed/` | no |
| Embedding store | Drive `cache/` | no |
| Checkpoints | Drive `checkpoints/` | no |
| Split definitions | repo `splits/` | **yes** |
| Run records | repo `results/` | **yes** |

Splits and results are small, and they are what make a number mean something
later. Everything else is reproducible from them plus the code.

### Drive is slow

Drive is a FUSE mount: one big sequential read is fine, thousands of small
reads are not. Two rules follow.

Store data as few large files — Parquet for tables, a single HDF5 for
embeddings. One file per variant will cost you hours.

Stage before training. `paths.stage(path)` copies a file to Colab's local disk
once and returns the local path; read from that in the training loop and write
checkpoints back to Drive. Off Colab it is a no-op, so the same code runs
either way.

## Three things this project tries not to get wrong

**Splits are an artifact, not a function call.** `splits/*.json` is generated
once by `make_splits.py`, committed, and reused. No protein appears in two
folds. Splitting by mutation lets a model see other variants of the same
protein in training, and since a protein's mutations share a mean dTm, that
alone yields a decent-looking correlation. Grouping by `protein_id` is the
floor — homologs with different accessions still leak. With sequences in hand,
cluster at 30% identity (mmseqs2) and pass the cluster ids as `groups`.

**Per-protein metrics are the honest ones.** A pooled correlation rewards
ranking proteins against each other, which is easy and useless. What matters
is ranking mutations within a protein. `eval/metrics.py` reports both, plus the
constant baseline; `test_eval_and_model.py` has a case where pooled Spearman
exceeds 0.9 while the median per-protein Spearman is near zero. If your two
numbers diverge like that, the model has learned protein identity.

**dTm is not ddG.** They correlate but are not convertible — the relation
involves the unfolding enthalpy, which is usually unreported. The two stay in
separate columns and neither is derived from the other. dTm data is also much
scarcer, so training on ddG with dTm as an auxiliary or fine-tuning target is
worth trying; the config and loop are set up to make that a change of config
rather than a rewrite.

Also worth knowing: sign conventions vary between source papers. The
convention here is mutant minus wild type, so positive means stabilising.
Spot-check some entries against their original papers before trusting a model
trained on them — a flipped subset is the easiest way to get a quietly broken
model.

## Reproducibility, minimally

Each run writes `results/<name>-<confighash>.json` containing the full config,
the git SHA, whether the tree was dirty, the split file used, and the metrics.
That answers "which code and which data produced this number?" without any
service to log into. `runs.list_runs()` loads them all for comparison. Add
Weights & Biases on top if you want live curves — it does not replace the file.

## Current state

The pipeline is scaffolding, with real logic in the parts that are easy to get
wrong (filtering, splits, metrics) and deliberately plain baselines elsewhere.
Not yet present: structure features (RSA, contact number), conservation from an
MSA, sequence-identity clustering, and any torch model. `cross_validate` takes
fold indices and returns per-fold metrics — keep that signature when you add
one, so comparisons stay apples to apples.

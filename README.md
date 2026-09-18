# TP53 Mutation Subtype Prediction from Local Sequence Context

Code and analysis notebooks for a study asking whether the local nucleotide
context around a TP53 mutation predicts its substitution subtype (C>A, C>G,
C>T, T>A, T>C, T>G), using a compact CNN trained on COSMIC mutation data.

## Key methodological point

The 19 raw transcript records used here are TP53 splice isoforms, not
independent genes — COSMIC re-annotates the same genomic variant once per
affected isoform. Naively splitting by `(transcript, CDS position)` both
inflates/fragments per-locus instance counts and leaks identical sequence
context across train/test. The pipeline (`notebooks/data_pipeline.ipynb`)
fixes this by clustering positions into true genomic loci via exact sequence
identity at the smallest window size (11bp) before doing anything else —
this collapses 13,423 raw positions into 906 true loci and is what every
downstream split and result depends on. See `results/figures/fig_pipeline_flow.png`.

## Repository structure

```
notebooks/
  data_pipeline.ipynb          Notebook 00: raw COSMIC data -> locus-clustered,
                                leak-free train/val/test splits at 4 window sizes
  01_main_experiment.ipynb     Main CNN (21bp), hotspot vs. rare-variant loci,
                                baselines, locus-level tests (McNemar's kept for comparison)
  02_window_ablation.ipynb     Same CNN at 11/51/101bp
  03_position_analysis.ipynb   Entropy vs. per-position accuracy (H4)
  04_cpg_analysis.ipynb        CpG vs. non-CpG accuracy, cluster-level significance test
  04b_cpg_unweighted_check.ipynb    Rare-CpG result under unweighted training
  04c_cpg_hotspot_unweighted_check.ipynb  Same check for hotspot-CpG
  05_unweighted_robustness.ipynb    Main experiment retrained without class weighting
  07_repeated_splits.ipynb      Primary 21bp CNN on 5 additional locus-held-out
                                splits (split-robustness check)
  06_make_figures.ipynb         Final publication figures/tables (reads only
                                already-saved results, no new computation)

src/
  data.py       Schema-verified loading, one-hot encoding, CpG/position utilities
  model.py      SimpleCNN (7,206 params, window-size agnostic)
  train.py      Training loop (fixed hyperparameters, FP16, early stopping)
  metrics.py    Metrics, baselines, locus-resampled bootstrap / Wilcoxon tests, McNemar's test

data/
  raw/          COSMIC exports (not tracked -- see data/raw/MANIFEST.md)
  processed/    Derived, not tracked -- rebuilt by notebook 00
  splits/       Derived, not tracked -- rebuilt by notebook 00

results/        Aggregated metrics, summary tables, figures and model checkpoints
                from every notebook (tracked -- small; this is what the paper's
                tables/figures are drawn from). Instance-level and position-level
                files (predictions.parquet, per_position_table.csv) are NOT tracked;
                see "Data and licensing" below.
```

## Setup

```
pip install -r requirements.txt
```

Python 3.14. `requirements.txt` pins a CUDA 12.8 PyTorch build; substitute
the CPU wheel or your platform's CUDA index if different
([pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)).
Training does not require a GPU but is substantially faster with one.

## Reproducing the pipeline

1. Obtain the raw COSMIC data per `data/raw/MANIFEST.md` (exact transcripts
   and file layout listed there — not redistributed here due to COSMIC's
   academic license).
2. Run `notebooks/data_pipeline.ipynb` top to bottom. Rebuilds
   `data/processed/` and `data/splits/` from `data/raw/`.
3. Run the notebooks in this order (later notebooks read outputs saved by
   earlier ones rather than recomputing them; numeric order is *not* the
   dependency order):
   `01` -> `02` -> `03` -> `04` -> `05` -> `04b` -> `04c` -> `07` -> `06`.
   Each writes its outputs under `results/`. This regenerates the
   instance-level and position-level files that are not tracked in git.

All notebooks seed `random`/`numpy`/`torch` with `SEED = 42` in their first
cell.

## License

MIT — see `LICENSE`. Raw COSMIC data is subject to COSMIC's own license
terms (see `data/raw/MANIFEST.md`) and is not covered by this repository's
license.

## Data and licensing

No COSMIC data is redistributed here. `data/raw/`, `data/processed/` and
`data/splits/` are not tracked, and neither are the COSMIC-derived
instance-level and position-level result files (`results/**/predictions.parquet`,
`results/position_analysis/per_position_table.csv`), because COSMIC's licence
does not permit redistributing COSMIC-derived instance- or position-level
records. Everything else under `results/` is aggregated (metrics, summary
tables, figures, model checkpoints) and can be regenerated from a user's own
COSMIC download by following the steps above.

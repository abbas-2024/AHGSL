# AHGSL — Experiment Notes

This document describes the experiment scripts and the intended mapping between the current code and the AHGSL manuscript. It is a guide to the supplied implementation, not an independent guarantee that all manuscript experiments have been reproduced.

## Main files

```text
ahgsl/
  data_loading.py   Yeast FunCat ARFF and TCGA loaders
  graph.py          Heterogeneous graph construction and graph utilities
  model.py          AHGSL model, AEE/SER mechanisms, gates, losses, baselines
  train.py          Training routines and yeast metrics
  tcga.py           TCGA cross-validation and gene-selection evaluation
run_yeast_all.py    Yeast experiment runner
run_tcga_all.py     TCGA experiment runner
aggregate.py        Aggregate JSON outputs
make_figures.py     Plot-generation utility
make_workbook.py    Excel reporting utility
build_manuscript.py Manuscript assembly utility
```

## Current equation-to-code map

| Manuscript component | Current implementation location |
|---|---|
| Initial heterogeneous graph | `ahgsl/graph.py:build_hetero_graph` |
| Message passing | `ahgsl/model.py:HeteroRound` |
| Adaptive Edge Evolution (AEE) scoring | `ahgsl/model.py:AHGSL._aee_scores` |
| Edge-change and uncertainty measures / SER selection | `ahgsl/model.py:AHGSL.commit_graph` |
| Gene-selection gate | `ahgsl/model.py:AHGSL.gate` and `AHGSL.forward` |
| Classification and regularization losses | `ahgsl/model.py:AHGSL.losses` |
| Training loop | `ahgsl/train.py:train_hetero` |
| TCGA cross-validation | `ahgsl/tcga.py:run_model_tcga` |

This mapping should be checked against the final numbered equations and algorithm in the paper before publication.

## Current experiment settings visible in code

- Yeast main models use seeds 42, 1, and 7; ablations use seeds 42 and 1; the logistic-regression baseline uses seed 42.
- The yeast runner iterates over `cellcycle`, `church`, `derisi`, `eisen`, and `yeast14`.
- The TCGA runner uses five folds and evaluates AHGSL, static heterogeneous model, GCN, MLP, logistic regression, and four ablation variants.
- The TCGA data loader expects `data.csv` and `labels.csv` in the directory selected by `AHGSL_TCGA_DIR` (or its fallback path).

Consult the configuration dictionaries in `ahgsl/train.py`, `ahgsl/tcga.py`, and `ahgsl/model.py` for actual parameter defaults. Do not rely on settings copied into earlier manuscript drafts without comparing them to these files.

## Data required for a fresh run

The supplied archive does not contain the dataset files. The yeast suite requires 12 FunCat ARFF split files plus the two Mulan yeast ARFF files. The TCGA suite requires `data.csv` and `labels.csv` with the schema described in the main README. Use `AHGSL_YEAST_DIR`, `AHGSL_MULAN_DIR`, and `AHGSL_TCGA_DIR` to point to local data directories.

## Reproduction checklist

1. Record Python and third-party package versions.
2. Record dataset source/version, file checksums, and any preprocessing.
3. Confirm the data split and preprocessing behavior matches the paper.
4. Run the experiment scripts from the repository root and preserve logs.
5. Check every JSON result for an `error` field and investigate failed jobs.
6. Regenerate aggregate tables and figures from freshly generated results.
7. Compare regenerated metrics and tables against the manuscript; explain deviations.

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

## Datasat

Dataset	Data type	Task	Instances (train/valid/test)	Features	Labels	Cardinality	Density
Cellcycle	Yeast gene expression (FunCat)	Multi-label	1630 / 852 / 1284	77	17	2.07	0.122
Church	Yeast gene expression (FunCat)	Multi-label	1632 / 848 / 1284	26	17	2.07	0.122
Derisi	Yeast gene expression (FunCat)	Multi-label	1610 / 845 / 1278	63	17	2.07	0.122
Eisen	Yeast gene expression (FunCat)	Multi-label	1058 / 530 / 837	79	16	2.31	0.144
yeast-14 (Mulan)	Yeast gene expression (Mulan)	Multi-label	1275 / 225 / 917	103	14	4.24	0.303
TCGA Pan-Cancer	RNA-seq (human tumors)	Multiclass (5 types)	801 (5-fold CV)	20,531*	5	1.00	0.200


## Reproduction checklist

1. Record Python and third-party package versions.
2. Record dataset source/version, file checksums, and any preprocessing.
3. Confirm the data split and preprocessing behavior matches the paper.
4. Run the experiment scripts from the repository root and preserve logs.
5. Check every JSON result for an `error` field and investigate failed jobs.
6. Regenerate aggregate tables and figures from freshly generated results.
7. Compare regenerated metrics and tables against the manuscript; explain deviations.

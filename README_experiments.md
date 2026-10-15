# AHGSL — Experimental Setup and Reproducibility Notes

This document describes the AHGSL implementation, its experimental configuration, and the correspondence between the source code and the manuscript. It is intended to support reproducibility and help verify that the reported methodology, experimental settings, and results are consistent with the released implementation.

The source code defines the experimental procedures, while the manuscript reports the methodology and results supported by completed experiments. The presence of an experiment script does not, by itself, establish that the corresponding experiment has completed successfully or that its results match those reported in the manuscript.

## 1. Project Structure

```text
ahgsl/
  data_loading.py   Dataset loading and preprocessing utilities
  graph.py          Heterogeneous graph construction and utilities
  model.py          AHGSL architecture, AEE/SER mechanisms,
                    gene-selection gate, and loss functions
  train.py          Training routines and multi-label evaluation
  tcga.py           TCGA evaluation and gene-selection experiments

run_yeast_all.py    Yeast experiment runner
run_tcga_all.py     TCGA experiment runner
aggregate.py        Aggregation of experimental JSON outputs
make_figures.py     Generation of experimental figures
make_workbook.py    Generation of Excel reports
build_manuscript.py Manuscript assembly utility
```

The descriptions above indicate the intended responsibilities of the main files. Their exact behaviour should be confirmed against the source code before the final manuscript is submitted.

## 2. Mapping Between the Manuscript and the Implementation

| Manuscript component | Implementation location |
|---|---|
| Initial heterogeneous graph construction | `ahgsl/graph.py:build_hetero_graph` |
| Heterogeneous message passing | `ahgsl/model.py:HeteroRound` |
| Adaptive Edge Evolution (AEE) scoring | `ahgsl/model.py:AHGSL._aee_scores` |
| Edge-change and uncertainty measures and SER-based selection | `ahgsl/model.py:AHGSL.commit_graph` |
| Gene-selection gate | `ahgsl/model.py:AHGSL.gate` and `AHGSL.forward` |
| Classification and regularisation losses | `ahgsl/model.py:AHGSL.losses` |
| Model training | `ahgsl/train.py:train_hetero` |
| TCGA evaluation and cross-validation | `ahgsl/tcga.py:run_model_tcga` |

The mapping identifies the principal implementation locations; it does not imply that each mathematical expression in the manuscript has been verified against the code. Before publication, each component should be checked against the final numbered equations, algorithm pseudocode, and corresponding implementation.

In particular, the definitions of AEE and SER, the criteria used to modify graph edges, the gene-selection mechanism, and the loss terms must remain consistent across the manuscript, source code, and experimental description.

## 3. Experimental Configuration

### 3.1 Yeast experiments

The yeast experiment runner processes the following datasets:

- Cellcycle
- Church
- Derisi
- Eisen
- Yeast-14 (Mulan)

The yeast task is multi-label classification. Each instance may be associated with more than one functional label, so evaluation should preserve the original label structure.

The currently documented random seeds are:

- Main AHGSL experiments: 42, 1, and 7.
- Ablation experiments: 42 and 1.
- Logistic-regression baseline: 42.

These values describe the settings recorded for the current implementation and should be verified against the final experiment configuration before they are reported as the definitive settings of the study.

### 3.2 TCGA Pan-Cancer experiments

The TCGA experiment runner is configured to evaluate AHGSL, a static heterogeneous model, GCN, MLP, logistic regression, and four ablation variants.

The documented evaluation procedure uses five-fold cross-validation. The TCGA loader expects `data.csv` and `labels.csv` in the directory specified by `AHGSL_TCGA_DIR` or its configured fallback path.

The final description of this experiment must specify the exact prediction task implemented by the code, the target labels, the number of samples and features actually loaded, the preprocessing procedure, and the cross-validation protocol.

**Important:** The TCGA experiment must not be described as five-class classification or multi-label classification solely on the basis of a dataset summary table. The task definition must be confirmed from the label representation, model output, loss function, and evaluation metrics used in the implementation. Likewise, the reported sample and feature counts must correspond to the data actually used in the experiment.

### 3.3 Model parameters

The actual parameter values should be obtained from the configuration dictionaries and function defaults in:

- `ahgsl/train.py`
- `ahgsl/tcga.py`
- `ahgsl/model.py`

Earlier manuscript drafts, preliminary notes, and previously generated reports should not be treated as authoritative when their settings differ from the current source code.

For reproducibility, the final paper should report the parameters that were actually used in the successful experimental runs, including the number of folds, random seeds, training settings, model dimensions, optimisation settings, and any dataset-specific configuration relevant to the reported results.

## 4. Datasets and Experimental Tasks

The following table records the dataset configurations currently documented for the project. The numerical values are provisional until they have been checked against the dataset files and the preprocessing and splitting procedures used by the experiment scripts.

| Dataset | Data type | Task | Instances (train/validation/test) | Features | Labels | Label cardinality | Label density |
|---|---|---|---:|---:|---:|---:|---:|
| Cellcycle | Yeast gene expression (FunCat) | Multi-label classification | 1,630 / 852 / 1,284 | 77 | 17 | 2.07 | 0.122 |
| Church | Yeast gene expression (FunCat) | Multi-label classification | 1,632 / 848 / 1,284 | 26 | 17 | 2.07 | 0.122 |
| Derisi | Yeast gene expression (FunCat) | Multi-label classification | 1,610 / 845 / 1,278 | 63 | 17 | 2.07 | 0.122 |
| Eisen | Yeast gene expression (FunCat) | Multi-label classification | 1,058 / 530 / 837 | 79 | 16 | 2.31 | 0.144 |
| Yeast-14 (Mulan) | Yeast gene expression | Multi-label classification | 1,275 / 225 / 917 | 103 | 14 | 4.24 | 0.303 |
| TCGA Pan-Cancer | Human tumour RNA-seq | To be verified against the implementation | 801 reported; split to be verified | 20,531 reported; preprocessing to be verified | 5 reported; target definition to be verified | 1.00 reported; to be verified | 0.200 reported; to be verified |

**Dataset reporting requirements**

1. Confirm the original source, version, and provenance of each dataset.
2. Verify whether the yeast instance counts refer to fixed predefined partitions or partitions created by the current code.
3. Verify whether the TCGA instance count refers to the complete dataset or the subset retained after filtering and preprocessing.
4. Confirm whether the feature count represents the original features or the final features used for model training.
5. Calculate label cardinality and density from the actual label matrix, using the same definitions for every dataset.
6. Report missing values, filtering, normalisation, feature selection, and other preprocessing operations where applicable.

For a binary multi-label matrix \(Y \in \{0,1\}^{N \times L}\), label cardinality is the average number of positive labels per instance:

\[
\operatorname{Cardinality}(Y)
=\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{L}Y_{ij}.
\]

Label density is the average number of positive labels per instance divided by the number of labels:

\[
\operatorname{Density}(Y)
=\frac{\operatorname{Cardinality}(Y)}{L}.
\]

These definitions should only be applied where the target data are represented by a binary multi-label matrix. If the TCGA task is conventional single-label classification, its summary should use appropriate single-label statistics instead.

## 5. Experimental Evaluation and Reporting

All reported results should be generated from successful experimental runs using the documented implementation and configuration.

The evaluation procedure should distinguish between:

- **Main experiments:** Comparison of AHGSL with the specified baseline models.
- **Ablation experiments:** Evaluation of the effects of removing or modifying individual AHGSL components.
- **Gene-selection experiments:** Assessment of the selected gene subsets using the evaluation procedure defined in the manuscript.
- **Cross-validation:** Evaluation using the specified folds, with training and evaluation restricted to the appropriate partitions.
- **Aggregate reporting:** Generation of summary tables and figures from the corresponding experimental outputs.

The manuscript must specify the evaluation metrics used for each task and explain how results are aggregated across folds, random seeds, or independent runs. Metrics should not be compared directly when they were obtained using different data splits, preprocessing procedures, or evaluation protocols without an explicit explanation.

For gene-selection experiments, the procedure used to select genes must be distinguished from the procedure used to evaluate predictive performance. Where feature selection is part of model training, the experimental design must prevent information from the evaluation fold from influencing the selected features or model parameters.

## 6. Reproducibility Checklist

Before reporting the experimental results, complete the following checks:

1. **Software environment:** Record the Python version, operating system, and relevant third-party package versions.

2. **Dataset provenance:** Record dataset sources, versions, file names, checksums, and preprocessing steps.

3. **Data integrity:** Verify the number of samples, features, labels, class distributions, and label cardinality or density, where applicable.

4. **Data splitting:** Confirm that the partitions or cross-validation folds match the experimental protocol described in the manuscript.

5. **Configuration consistency:** Compare the parameters used in the successful runs with those reported in the paper.

6. **Execution:** Run the experiment scripts from the repository root and preserve the console output and execution logs.

7. **Error handling:** Inspect every generated JSON result for an `error` field or other indication of an unsuccessful run. Investigate failed jobs before including their outputs in aggregate results.

8. **Result aggregation:** Regenerate summary tables, figures, and Excel reports from the newly generated experimental outputs.

9. **Manuscript verification:** Compare the regenerated results with the corresponding tables, figures, and claims in the manuscript. Investigate and document any discrepancies.

10. **Final release:** Ensure that the released source code, dataset instructions, experimental configurations, and reported results refer to the same version of the implementation.

## 7. Scope and Limitations of This Document

This document describes the intended relationship between the AHGSL source code and the experimental methodology. It is not, by itself, proof that every experiment has been completed, that every reported result is reproducible, or that the implementation has been independently validated.

The final manuscript should report only results that can be traced to successful experimental runs. Any differences between the implementation and the published methodology should be resolved or explicitly documented before submission.

The objective is to maintain a transparent and verifiable relationship between the proposed method, its implementation, the experimental protocol, and the results reported in the research paper.

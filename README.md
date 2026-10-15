# AHGSL: Adaptive Heterogeneous Graph Structure Learning

Research code and experiment artifacts for **Learning Adaptive Heterogeneous Graph Structures for Joint Multi-Label Gene Selection and Classification (AHGSL)**.

> **Status and reproducibility.** This archive includes Python implementation files, previously generated JSON result files, and manuscript figures. The underlying datasets are not included. Syntax/import checks do not establish that the reported experiments can be reproduced. Before treating this repository as the final release, verify the code against the final manuscript, obtain the required datasets from their authorized sources, and rerun the experiments.

## Repository layout

```text
AHGSL/
├── ahgsl/
│   ├── model.py          # AHGSL, baselines, message passing, AEE/SER, losses
│   ├── graph.py          # Heterogeneous graph construction and graph utilities
│   ├── data_loading.py   # Yeast and TCGA loaders; preprocessing helpers
│   ├── train.py          # Training and evaluation routines for yeast experiments
│   └── tcga.py           # TCGA cross-validation and gene-selection experiments
├── data/                 # Create locally; datasets are not tracked
├── results/              # Existing JSON result artifacts
├── figures/              # Existing manuscript figures and equation images
├── run_yeast_all.py      # Yeast benchmark runner
├── run_tcga_all.py       # TCGA benchmark runner
├── aggregate.py          # Aggregate result JSON files
├── make_figures.py       # Generate plots from results and data
├── make_workbook.py      # Generate an Excel workbook from result artifacts
├── build_manuscript.py   # Manuscript assembly utility
├── README_experiments.md # Experiment notes and equation-to-code mapping
└── requirements.txt
```

## Requirements and installation

Use Python 3.10 or 3.11 as a practical starting point. Dependencies below are the direct third-party packages imported by the source and reporting utilities; versions are not pinned in this archive, so record the exact versions used for any published reproduction.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch using the command recommended for your operating system and CPU/GPU from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

## Dataset setup

Datasets are not bundled with this repository. Obtain them from their original or authorized sources and follow their licences/terms. Do not upload restricted, confidential, or patient-identifiable data.

### Four FunCat yeast datasets

Place the train, validation, and test ARFF files in `data/yeast/` using these exact filenames:

```text
data/yeast/
├── cellcycle_yeast_FUN.train.arff
├── cellcycle_yeast_FUN.valid.arff
├── cellcycle_yeast_FUN.test.arff
├── church_yeast_FUN.train.arff
├── church_yeast_FUN.valid.arff
├── church_yeast_FUN.test.arff
├── derisi_yeast_FUN.train.arff
├── derisi_yeast_FUN.valid.arff
├── derisi_yeast_FUN.test.arff
├── eisen_yeast_FUN.train.arff
├── eisen_yeast_FUN.valid.arff
└── eisen_yeast_FUN.test.arff
```

The loader also expects the Mulan yeast benchmark files here:

```text
data/mulan_yeast/yeast-train.arff
data/mulan_yeast/yeast-test.arff
```

You can override these default directories with environment variables `AHGSL_YEAST_DIR` and `AHGSL_MULAN_DIR` respectively.

### TCGA Pan-Cancer data

The loader expects `data.csv` and `labels.csv` in the same directory. Set `AHGSL_TCGA_DIR` to that directory. For example, on Windows PowerShell:

```powershell
$env:AHGSL_TCGA_DIR = "D:\datasets\TCGA-PANCAN-HiSeq-801x20531"
python run_tcga_all.py
```

On macOS/Linux:

```bash
export AHGSL_TCGA_DIR="/path/to/TCGA-PANCAN-HiSeq-801x20531"
python run_tcga_all.py
```

The expected files must match the schema used by `ahgsl.data_loading.load_tcga`: `data.csv` contains samples as rows and genes as columns, while `labels.csv` contains a `Class` column and matching sample index/order. Check the dataset documentation before use.

## Running experiments

Run commands from the repository root after installing dependencies and placing the datasets as described above.

```bash
# Full yeast benchmark suite; saves progress to results/yeast_results.json
python run_yeast_all.py

# Run selected yeast datasets and write to a separate output file
python run_yeast_all.py cellcycle,church results/yeast_subset.json

# TCGA five-fold suite; requires AHGSL_TCGA_DIR to be configured
python run_tcga_all.py

# Aggregate existing result files
python aggregate.py

# Generate plots/workbook (some plots also require dataset files)
python make_figures.py
python make_workbook.py
```

The runners save JSON incrementally and record exceptions as entries containing an `error` field. Inspect these entries and the logs before interpreting a run as successful. A complete reproduction should retain the exact dataset versions, package versions, command lines, seeds, fold/split definitions, and configuration values.

## Included results

JSON files in `results/` are artifacts already present in the supplied archive. They are not newly regenerated by installation and are not proof that a fresh run will reproduce the same values. Preserve their provenance and distinguish them from results generated in a new run.

## Method documentation

See `README_experiments.md` for the current equation-to-code mapping and experimental notes. Compare the implementation and configuration directly with the final manuscript before release; this repository has not been independently certified as an exact reproduction of every manuscript claim.

## Citation

Add the final bibliographic citation and DOI of the associated paper here once confirmed. Until then, do not cite this repository as a final archival release without an identifiable version/commit.

## License

No licence is included because reuse permissions have not been specified. Add a licence only after the rights holders have agreed to its terms.

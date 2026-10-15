"""Data loading for AHGSL experiments.

- Yeast FunCat (FUN) hierarchical multi-label ARFF datasets (Clus-HMC format,
  Vens et al. 2008): cellcycle, church, derisi, eisen.
  Instances = yeast genes described by expression measurements; labels = FunCat
  functional categories. For the flat multi-label formulation used in the paper
  we truncate FunCat annotations at the top level of the hierarchy (standard
  practice for flat multi-label benchmarks derived from FunCat).
- TCGA-PANCAN-HiSeq-801x20531 (UCI "gene expression cancer RNA-Seq"):
  801 samples x 20531 genes, 5 tumor types -> multiclass task.
"""
import os
import re
import numpy as np
import pandas as pd

YEAST_DIR = os.environ.get(
    "AHGSL_YEAST_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "yeast"),
)
TCGA_DIR = os.environ.get(
    "AHGSL_TCGA_DIR",
    os.path.join(
        os.path.dirname(__file__), "..",
        "src", "داتا سيت السرطانات",
        "المهمة هنا Multiclass Cancer Classification وليست Multi-label Classification",
        "TCGA-PANCAN-HiSeq-801x20531",
    ),
)


def _parse_clus_arff(path):
    """Parse a Clus-HMC ARFF file with a 'hierarchical' class attribute.

    Returns (X: float array with NaN for '?', labels: list[list[str]] of full
    hierarchical class paths per instance, feat_names: list[str]).
    """
    feat_names = []
    col_kinds = []  # per attribute in file order: 'num' | 'class' | 'other'
    data_started = False
    rows, labels = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if not data_started:
                if low.startswith("@attribute"):
                    toks = line.split()
                    a_type = toks[2].lower() if len(toks) >= 3 else ""
                    if a_type.startswith("hierarchical"):
                        col_kinds.append("class")
                    elif a_type == "numeric":
                        col_kinds.append("num")
                        feat_names.append(toks[1])
                    else:  # e.g. 'key', string, nominal — ignored
                        col_kinds.append("other")
                elif low.startswith("@data"):
                    data_started = True
                continue
            # data line; strip trailing % comment
            if "%" in line:
                line = line.split("%")[0].strip()
            if not line:
                continue
            parts = line.split(",")
            assert len(parts) == len(col_kinds), f"{path}: row has {len(parts)} cols, header {len(col_kinds)}"
            vals, cls = [], None
            for v, kind in zip(parts, col_kinds):
                v = v.strip()
                if kind == "num":
                    vals.append(np.nan if v == "?" else float(v))
                elif kind == "class":
                    cls = v
            rows.append(vals)
            labels.append([p.strip() for p in cls.split("@") if p.strip()])
    X = np.asarray(rows, dtype=np.float64)
    assert X.shape[1] == len(feat_names), f"{path}: {X.shape[1]} vs {len(feat_names)}"
    return X, labels, feat_names


def _top_level(path_str):
    """FunCat path '1/1/1/0' -> top-level class '1'."""
    return path_str.split("/")[0]


def load_yeast(name, label_level="top", min_label_freq=0):
    """Load one yeast FUN dataset with its predefined train/valid/test split.

    label_level='top': truncate FunCat paths at the root level (flat multi-label).
    Returns dict with X_*, Y_* (binary), label_names, feat_names.
    """
    out = {}
    all_paths = set()
    per_split = {}
    for split in ("train", "valid", "test"):
        p = os.path.join(YEAST_DIR, f"{name}_yeast_FUN.{split}.arff")
        X, labels, feat_names = _parse_clus_arff(p)
        per_split[split] = (X, labels)
        out["feat_names"] = feat_names
    # build label vocabulary from TRAIN split only (top-level classes)
    train_labels = per_split["train"][1]
    from collections import Counter
    cnt = Counter()
    for lst in train_labels:
        cnt.update({_top_level(s) for s in lst})
    label_names = sorted([c for c, n in cnt.items() if n >= min_label_freq],
                         key=lambda s: int(s))
    lab_idx = {c: i for i, c in enumerate(label_names)}
    for split in ("train", "valid", "test"):
        X, labels = per_split[split]
        Y = np.zeros((X.shape[0], len(label_names)), dtype=np.float32)
        for i, lst in enumerate(labels):
            for s in lst:
                t = _top_level(s)
                if t in lab_idx:
                    Y[i, lab_idx[t]] = 1.0
        # drop instances with no labels within vocabulary
        keep = Y.sum(1) > 0
        out[f"X_{split}"] = X[keep]
        out[f"Y_{split}"] = Y[keep]
    out["label_names"] = label_names
    return out


def load_tcga(top_var_genes=None):
    """Load TCGA-PANCAN-HiSeq-801x20531. Returns X (801 x G), y (int), classes,
    gene_names. Optionally keep only the top-variance genes."""
    data_p = os.path.join(TCGA_DIR, "data.csv")
    lab_p = os.path.join(TCGA_DIR, "labels.csv")
    df = pd.read_csv(data_p, index_col=0)
    lab = pd.read_csv(lab_p, index_col=0)
    assert (df.index == lab.index).all()
    X = df.values.astype(np.float32)
    gene_names = np.array(df.columns)
    classes = sorted(lab["Class"].unique())
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = lab["Class"].map(cls_idx).values.astype(np.int64)
    if top_var_genes is not None and top_var_genes < X.shape[1]:
        v = X.var(axis=0)
        keep = np.argsort(v)[::-1][:top_var_genes]
        keep = np.sort(keep)
        X = X[:, keep]
        gene_names = gene_names[keep]
    return X, y, classes, gene_names


def impute_standardize(X_train, *others):
    """Mean-impute NaNs and z-score using TRAIN statistics."""
    mu = np.nanmean(X_train, axis=0)
    Xt = np.where(np.isnan(X_train), mu, X_train)
    sd = Xt.std(axis=0)
    sd[sd == 0] = 1.0
    res = [(Xt - mu) / sd]
    for X in others:
        Xi = np.where(np.isnan(X), mu, X)
        res.append((Xi - mu) / sd)
    return res


MULAN_DIR = os.environ.get(
    "AHGSL_MULAN_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "mulan_yeast"),
)


def load_mulan_yeast(val_frac=0.15, seed=42):
    """Canonical Mulan 'yeast' multi-label benchmark (Elisseeff & Weston 2002):
    103 numeric features, 14 binary labels, official train/test split.
    A fixed random 15% of the training set forms the validation split
    (same split for every model; seed fixed)."""
    def parse(path):
        feats, kinds = [], []
        rows, data = [], False
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                low = line.lower()
                if not data:
                    if low.startswith("@attribute"):
                        toks = line.split()
                        if toks[1].lower().startswith("class"):
                            kinds.append("y")
                        else:
                            kinds.append("x")
                            feats.append(toks[1])
                    elif low.startswith("@data"):
                        data = True
                    continue
                vals = [v.strip() for v in line.split(",")]
                assert len(vals) == len(kinds)
                rows.append(vals)
        M = np.asarray(rows)
        xm = np.array([k == "x" for k in kinds])
        X = M[:, xm].astype(np.float64)
        Y = M[:, ~xm].astype(np.float32)
        return X, Y, feats

    Xtr, Ytr, feats = parse(os.path.join(MULAN_DIR, "yeast-train.arff"))
    Xte, Yte, _ = parse(os.path.join(MULAN_DIR, "yeast-test.arff"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    nv = int(round(val_frac * len(Xtr)))
    va, tr = idx[:nv], idx[nv:]
    return {"X_train": Xtr[tr], "Y_train": Ytr[tr],
            "X_valid": Xtr[va], "Y_valid": Ytr[va],
            "X_test": Xte, "Y_test": Yte,
            "feat_names": feats,
            "label_names": [f"Class{i+1}" for i in range(Ytr.shape[1])]}


if __name__ == "__main__":
    print("=" * 70)
    print("YEAST FUN DATASETS (multi-label, top-level FunCat)")
    print("=" * 70)
    for name in ("cellcycle", "church", "derisi", "eisen"):
        d = load_yeast(name)
        ntr, nva, nte = (d[f"X_{s}"].shape[0] for s in ("train", "valid", "test"))
        L = len(d["label_names"])
        Yall = np.vstack([d["Y_train"], d["Y_valid"], d["Y_test"]])
        card = Yall.sum(1).mean()
        dens = Yall.mean()
        miss = np.mean([np.isnan(d[f"X_{s}"]).mean() for s in ("train", "valid", "test")])
        print(f"{name:10s} feats={len(d['feat_names']):4d}  train={ntr}  valid={nva}  "
              f"test={nte}  total={ntr+nva+nte}  L={L}  card={card:.3f}  "
              f"density={dens:.4f}  missing={miss*100:.2f}%")
        print(f"           labels: {d['label_names']}")
    print("=" * 70)
    print("TCGA-PANCAN-HiSeq-801x20531 (multiclass)")
    print("=" * 70)
    X, y, classes, genes = load_tcga()
    print(f"X: {X.shape}, value range [{X.min():.2f}, {X.max():.2f}], "
          f"zero-variance genes: {(X.var(0)==0).sum()}")
    for i, c in enumerate(classes):
        print(f"  {c}: {(y==i).sum()}")

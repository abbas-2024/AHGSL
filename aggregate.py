"""Aggregate experiment JSONs into summary stats for tables (agg.json + xlsx)."""
import json
import numpy as np

YE = json.load(open("results/yeast_results.json"))
TC = json.load(open("results/tcga_results.json"))

DATASETS = ["cellcycle", "church", "derisi", "eisen", "yeast14"]
YE_METHODS = ["br_lr", "mlp", "gcn", "static_hgnn", "ahgsl"]
ABL = ["ahgsl", "ahgsl_no_aee", "ahgsl_no_ser", "ahgsl_no_gate", "ahgsl_no_stab"]
TC_METHODS = ["lr", "mlp", "gcn", "static_hgnn", "ahgsl"]
YE_METRICS = ["accuracy_ex", "precision_micro", "recall_micro", "f1_micro",
              "f1_macro", "auc_micro", "ranking_loss", "avg_precision_micro",
              "hamming", "train_time"]
TC_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
              "auc_macro_ovr", "train_time"]


def collect(store, prefix_fmt, keys, metrics):
    vals = {}
    for m in metrics:
        arr = []
        for k in keys:
            kk = prefix_fmt.format(k)
            if kk in store and m in store[kk]:
                arr.append(store[kk][m])
        if arr:
            vals[m] = (float(np.mean(arr)), float(np.std(arr)), len(arr))
    return vals


agg = {"yeast": {}, "yeast_abl": {}, "tcga": {}, "tcga_abl": {}}
for d in DATASETS:
    agg["yeast"][d] = {}
    for meth in YE_METHODS:
        seeds = [42] if meth == "br_lr" else [42, 1, 7]
        agg["yeast"][d][meth] = collect(YE, f"{d}|{meth}|{{}}", seeds, YE_METRICS)
    agg["yeast_abl"][d] = {}
    for meth in ABL:
        agg["yeast_abl"][d][meth] = collect(YE, f"{d}|{meth}|{{}}", [42, 1, 7],
                                            YE_METRICS + ["mean_refined_frac",
                                                          "mean_epoch_time",
                                                          "epochs_run"])
for meth in TC_METHODS + ABL[1:]:
    agg["tcga"][meth] = collect(TC, f"{meth}|f{{}}", range(5),
                                TC_METRICS + ["mean_refined_frac",
                                              "mean_epoch_time", "epochs_run"])
# confusion sum over folds for ahgsl
conf = None
for f in range(5):
    k = f"ahgsl|f{f}"
    if k in TC and "confusion" in TC[k]:
        c = np.array(TC[k]["confusion"])
        conf = c if conf is None else conf + c
agg["tcga_confusion"] = conf.tolist() if conf is not None else None

# dataset statistics for Table 2
from ahgsl.data_loading import load_yeast, load_tcga, load_mulan_yeast
import numpy as _np
stats = {}
for d in DATASETS:
    dd = load_mulan_yeast() if d == "yeast14" else load_yeast(d)
    Yall = _np.vstack([dd["Y_train"], dd["Y_valid"], dd["Y_test"]])
    stats[d] = dict(train=int(len(dd["X_train"])), valid=int(len(dd["X_valid"])),
                    test=int(len(dd["X_test"])), features=int(len(dd["feat_names"])),
                    labels=int(Yall.shape[1]), cardinality=float(Yall.sum(1).mean()),
                    density=float(Yall.mean()))
Xt, yt_, classes_, _g = load_tcga()
stats["tcga"] = dict(samples=int(Xt.shape[0]), genes=int(Xt.shape[1]),
                     classes={c: int((yt_ == i).sum()) for i, c in enumerate(classes_)})
agg["dataset_stats"] = stats

json.dump(agg, open("results/agg.json", "w"), indent=1)

# quick text summary
print("=== YEAST (micro-F1 / micro-AUC / AP, mean over seeds) ===")
for d in DATASETS:
    row = [f"{d:10s}"]
    for meth in YE_METHODS:
        v = agg["yeast"][d][meth]
        if "f1_micro" in v:
            row.append(f"{meth}:{v['f1_micro'][0]:.3f}/{v['auc_micro'][0]:.3f}/{v['avg_precision_micro'][0]:.3f}")
    print("  ".join(row))
print("=== YEAST ablation (AP mean) ===")
for d in DATASETS:
    row = [f"{d:10s}"]
    for meth in ABL:
        v = agg["yeast_abl"][d][meth]
        if "avg_precision_micro" in v:
            row.append(f"{meth.replace('ahgsl','A')}:{v['avg_precision_micro'][0]:.3f}")
    print("  ".join(row))
print("=== TCGA (acc / f1_macro / auc, mean over folds) ===")
for meth, v in agg["tcga"].items():
    if "accuracy" in v:
        print(f"{meth:15s} acc={v['accuracy'][0]:.4f}±{v['accuracy'][1]:.4f} "
              f"f1M={v['f1_macro'][0]:.4f} auc={v['auc_macro_ovr'][0]:.4f} "
              f"n={v['accuracy'][2]}")

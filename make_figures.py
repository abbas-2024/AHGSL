"""Publication figures for the AHGSL manuscript (300 dpi PNG, 6.5in wide).

Design: validated categorical palette (fixed slot order), thin marks, one axis
per panel, recessive grids, direct labels where helpful, no rainbow.
"""
import json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# validated categorical palette (light mode), fixed slot order
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
       "#008300", "#4a3aa7", "#e34948"]
GRID = dict(color="#d9d9d3", lw=0.6)
plt.rcParams.update({
    "font.size": 8.5, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#8a8a84", "axes.linewidth": 0.7,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.color": "#5f5f59", "ytick.color": "#5f5f59",
    "text.color": "#1a1a19", "axes.labelcolor": "#1a1a19",
    "legend.frameon": False, "figure.dpi": 120, "savefig.dpi": 300,
})
os.makedirs("figures", exist_ok=True)

agg = json.load(open("results/agg.json"))
DATASETS = ["cellcycle", "church", "derisi", "eisen", "yeast14"]
DLABEL = {"yeast14": "yeast-14"}
METH_LABEL = {"br_lr": "BR-LR", "mlp": "MLP", "gcn": "GCN",
              "static_hgnn": "Static-HGNN", "ahgsl": "AHGSL (proposed)",
              "lr": "LR (softmax)",
              "ahgsl_no_aee": "w/o AEE", "ahgsl_no_ser": "w/o SER",
              "ahgsl_no_gate": "w/o gene gate", "ahgsl_no_stab": "w/o stability"}
# fixed color per entity across ALL figures
MCOLOR = {"br_lr": PAL[3], "lr": PAL[3], "mlp": PAL[4], "gcn": PAL[2],
          "static_hgnn": PAL[1], "ahgsl": PAL[0],
          "ahgsl_no_aee": PAL[1], "ahgsl_no_ser": PAL[2],
          "ahgsl_no_gate": PAL[3], "ahgsl_no_stab": PAL[4]}


def style_ax(ax):
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- Fig 3: training dynamics & graph evolution ----------------
def fig3():
    cache_p = "results/hist_cache.pkl"
    if os.path.exists(cache_p):
        cache = pickle.load(open(cache_p, "rb"))
    else:
        cache = {}
        from ahgsl.train import run_model_yeast
        for d in DATASETS:
            print("history run:", d)
            res, model, hist = run_model_yeast(d, "ahgsl", 42, return_model=True)
            cache[d] = {"hist": hist,
                        "final_w": {r: model.a_state[r].numpy().tolist()
                                    for r in ("gg", "ss")},
                        "final_active": {r: model.active[r].numpy().tolist()
                                         for r in ("gg", "ss")},
                        "gate": model.gate().detach().numpy().tolist()}
        pickle.dump(cache, open(cache_p, "wb"))
    tc = json.load(open("results/tcga_results.json"))

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3))
    ax = axes[0]  # (a) validation AP convergence, one axis, all datasets
    for i, d in enumerate(DATASETS):
        v = cache[d]["hist"]["val"]
        ax.plot(np.arange(1, len(v) + 1), v, color=PAL[i], lw=1.4, label=DLABEL.get(d, d))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AP (micro)")
    ax.legend(fontsize=6.5, loc="lower right")
    style_ax(ax)
    ax.set_title("(a) Convergence", fontsize=8.5)

    ax = axes[1]  # (b) refined fraction over epochs
    for i, d in enumerate(DATASETS):
        rf = cache[d]["hist"]["refined_frac"]
        ax.plot(np.arange(1, len(rf) + 1), 100 * np.asarray(rf),
                color=PAL[i], lw=1.4, label=DLABEL.get(d, d))
    k = "ahgsl|f0"
    if k in tc and "refined_frac_hist" in tc[k]:
        rf = tc[k]["refined_frac_hist"]
        ax.plot(np.arange(1, len(rf) + 1), 100 * np.asarray(rf),
                color=PAL[6], lw=1.4, label="TCGA (fold 1)", ls='--')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Edges refined by SER (%)")
    ax.legend(fontsize=6.5, loc="upper right")
    style_ax(ax)
    ax.set_title("(b) Selective refinement", fontsize=8.5)

    ax = axes[2]  # (c) edge-weight distribution shift (cellcycle gg+ss)
    d = "cellcycle"
    from ahgsl.data_loading import load_yeast, impute_standardize
    from ahgsl.graph import build_hetero_graph
    dd = load_yeast(d)
    Xtr, Xva, Xte = impute_standardize(dd["X_train"], dd["X_valid"], dd["X_test"])
    X_all = np.vstack([Xtr, Xva, Xte]).astype(np.float32)
    Y_all = np.vstack([dd["Y_train"], dd["Y_valid"], dd["Y_test"]]).astype(np.float32)
    tr = np.arange(len(Xtr))
    np.random.seed(42)
    rel = build_hetero_graph(X_all, X_all[tr], Y_all[tr], tr, seed=42)
    w_init = np.concatenate([rel[r]["w"][rel[r]["active"]] for r in ("gg", "ss")])
    wf, af = cache[d]["final_w"], cache[d]["final_active"]
    w_fin = np.concatenate([np.asarray(wf[r])[np.asarray(af[r], bool)]
                            for r in ("gg", "ss")])
    bins = np.linspace(0, 1, 41)
    ax.hist(w_init, bins=bins, density=True, alpha=0.85, color=PAL[0],
            label="Initial $A^{(0)}$", histtype="stepfilled", lw=0)
    ax.hist(w_fin, bins=bins, density=True, alpha=0.6, color=PAL[1],
            label="Learned $A^{*}$", histtype="stepfilled", lw=0)
    ax.set_xlabel("Edge weight"); ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    style_ax(ax)
    ax.set_title("(c) Edge-weight evolution", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figures/fig3_training_dynamics.png", bbox_inches="tight")
    plt.close(fig)
    print("fig3 done")


# ---------------- Fig 4: yeast main comparison ----------------
def fig4():
    methods = ["br_lr", "mlp", "gcn", "static_hgnn", "ahgsl"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharex=True)
    for ax, metric, title in zip(
            axes, ["f1_micro", "avg_precision_micro"],
            ["(a) Micro-F1", "(b) Micro average precision"]):
        x = np.arange(len(DATASETS))
        w = 0.15
        for j, m in enumerate(methods):
            mu = [agg["yeast"][d][m].get(metric, (np.nan,))[0] for d in DATASETS]
            sd = [agg["yeast"][d][m].get(metric, (0, 0))[1] for d in DATASETS]
            off = (j - (len(methods) - 1) / 2) * w
            ax.bar(x + off, mu, width=w - 0.02, color=MCOLOR[m],
                   yerr=sd, error_kw=dict(lw=0.7, capsize=1.5, ecolor="#5f5f59"),
                   label=METH_LABEL[m])
        ax.set_xticks(x); ax.set_xticklabels([DLABEL.get(d, d) for d in DATASETS], fontsize=7.5, rotation=12)
        ax.set_ylabel(title.split(") ")[1])
        ax.set_ylim(0, None)
        style_ax(ax)
        ax.set_title(title, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.09), frameon=False)
    fig.tight_layout()
    fig.savefig("figures/fig4_yeast_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")


# ---------------- Fig 5: TCGA embedding + confusion ----------------
def fig5():
    import torch
    from ahgsl.tcga import run_model_tcga
    cache_p = "results/tcga_embed.pkl"
    if os.path.exists(cache_p):
        emb, y, classes = pickle.load(open(cache_p, "rb"))
    else:
        res, model, hist, ctx = run_model_tcga("ahgsl", 0, 42, return_model=True)
        import torch
        model.eval()
        from ahgsl.graph import gene_node_features
        Xg = torch.tensor(gene_node_features(ctx["Xz"][ctx["tr_in"]], 64))
        Xs = torch.tensor(ctx["Xz"])
        with torch.no_grad():
            _, H2, _ = model(Xg, Xs)
        emb = H2["s"].numpy()
        y, classes = ctx["y"], ctx["classes"]
        pickle.dump((emb, y, classes), open(cache_p, "wb"))
    from sklearn.manifold import TSNE
    Z = TSNE(n_components=2, random_state=42, perplexity=30,
             init="pca").fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for c, name in enumerate(classes):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=7, color=PAL[c], label=name,
                   linewidths=0.3, edgecolors="white")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=6.5, markerscale=1.6, handletextpad=0.2,
              borderaxespad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(a) AHGSL sample embeddings (t-SNE)", fontsize=8.5)
    ax = axes[1]
    conf = np.array(agg["tcga_confusion"])
    classes5 = ["BRCA", "COAD", "KIRC", "LUAD", "PRAD"]
    im = ax.imshow(conf, cmap="Blues", vmin=0)
    for i in range(5):
        for j in range(5):
            v = conf[i, j]
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=7,
                    color="white" if v > conf.max() * 0.6 else "#1a1a19")
    ax.set_xticks(range(5)); ax.set_xticklabels(classes5, rotation=45, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(classes5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("(b) Confusion (5-fold pooled)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig("figures/fig5_tcga.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5 done")


# ---------------- Fig 6: gene selection curves ----------------
def fig6():
    gs = json.load(open("results/tcga_gene_selection.json"))
    methods = ["AHGSL gate", "ANOVA F-test", "Mutual information",
               "L1-logistic", "Random"]
    colors = {"AHGSL gate": PAL[0], "ANOVA F-test": PAL[1],
              "Mutual information": PAL[2], "L1-logistic": PAL[3],
              "Random": "#8a8a84"}
    Ks = [25, 50, 100, 200, 500]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for m in methods:
        mu = []
        for K in Ks:
            vals = [gs[f]["curves"][m][str(K)]["accuracy"]
                    for f in gs if m in gs[f]["curves"]]
            mu.append(100 * np.mean(vals))
        ax.plot(Ks, mu, marker="o", ms=3.5, lw=1.4, color=colors[m], label=m)
    ax.set_xscale("log")
    ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("Number of selected genes (K)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=6.5)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig("figures/fig6_gene_selection.png", bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")


# ---------------- Fig 7: ablation ----------------
def fig7():
    variants = ["ahgsl", "ahgsl_no_aee", "ahgsl_no_ser", "ahgsl_no_gate",
                "ahgsl_no_stab"]
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    x = np.arange(len(DATASETS))
    w = 0.15
    for j, m in enumerate(variants):
        mu = np.array([agg["yeast_abl"][d][m].get("avg_precision_micro", (np.nan,))[0]
                       for d in DATASETS])
        sd = np.array([agg["yeast_abl"][d][m].get("avg_precision_micro", (0, 0))[1]
                       for d in DATASETS])
        off = (j - (len(variants) - 1) / 2) * w
        lbl = "AHGSL (full)" if m == "ahgsl" else METH_LABEL[m]
        col = PAL[0] if m == "ahgsl" else MCOLOR[m]
        ax.errorbar(x + off, mu, yerr=sd, fmt="o", ms=4.5, color=col,
                    ecolor="#8a8a84", elinewidth=0.8, capsize=2, label=lbl,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([DLABEL.get(d, d) for d in DATASETS], fontsize=7.5,
                       rotation=12)
    ax.set_ylabel("Micro average precision")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig("figures/fig7_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("fig7 done")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "3"): fig3()
    if which in ("all", "4"): fig4()
    if which in ("all", "5"): fig5()
    if which in ("all", "6"): fig6()
    if which in ("all", "7"): fig7()

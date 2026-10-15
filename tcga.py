"""TCGA-PANCAN-HiSeq-801x20531 experiments: 5-fold stratified CV, multiclass
cancer-type classification + gene (biomarker) selection via the AHGSL gate."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from .model import AHGSL, GCN, MLP, normalized_adj
from .graph import (build_hetero_graph, gene_node_features, corr_sim,
                    cosine_sim, _knn_from_sim, _minmax01)
from .train import multiclass_metrics, train_hetero, train_plain, DEFAULT_CFG
from .data_loading import load_tcga

TCGA_CFG = dict(DEFAULT_CFG, epochs=250, patience=25, eval_every=2,
                top_var_genes=3000, d_gene_in=64, k_sg=15)


def prepare_fold(X, y, tr, te, cfg, seed):
    """Fold-specific preprocessing: variance filter + z-score on fold-train."""
    tr_in, va_in = train_test_split(tr, test_size=0.15, stratify=y[tr],
                                    random_state=seed)
    v = X[tr_in].var(axis=0)
    keep = np.sort(np.argsort(v)[::-1][:cfg["top_var_genes"]])
    Xk = X[:, keep]
    mu, sd = Xk[tr_in].mean(0), Xk[tr_in].std(0)
    sd[sd == 0] = 1.0
    Xz = ((Xk - mu) / sd).astype(np.float32)
    return Xz, keep, tr_in, va_in


def hetero_graph_multiclass(Xz, y, tr_in, n_classes, cfg, seed):
    Y1h = np.eye(n_classes, dtype=np.float32)[y]
    rel = build_hetero_graph(Xz, Xz[tr_in], Y1h[tr_in], tr_in,
                             cfg["k_gg"], cfg["k_ss"], cfg["k_sg"], cfg["k_ll"],
                             cfg["k_gg_cand"], cfg["k_ss_cand"], seed=seed)
    # label-label similarity from class-centroid correlation (labels are
    # mutually exclusive, so co-occurrence is degenerate for multiclass)
    cent = np.stack([Xz[tr_in][y[tr_in] == c].mean(0) for c in range(n_classes)])
    S = corr_sim(cent.astype(np.float64))
    src, dst, w = _knn_from_sim(np.abs(S), min(cfg["k_ll"], n_classes - 1))
    rel["ll"] = dict(src=src, dst=dst, w=_minmax01(w),
                     active=np.ones(len(src), dtype=bool))
    return rel


def run_model_tcga(kind, fold, seed=42, cfg=None, return_model=False):
    cfg = dict(TCGA_CFG, **(cfg or {}))
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y, classes, gene_names = load_tcga()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr, te = list(skf.split(X, y))[fold]
    Xz, keep, tr_in, va_in = prepare_fold(X, y, tr, te, cfg, seed)
    n, C = len(y), len(classes)
    dev = "cpu"
    yt = torch.tensor(y, device=dev)
    t0 = time.time()

    if kind == "lr":
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(Xz[tr], y[tr])
        P = clf.predict_proba(Xz[te])
        res = multiclass_metrics(y[te], P)
        res["train_time"] = time.time() - t0
        return (res, None, None, None) if return_model else res

    Xs = torch.tensor(Xz, device=dev)
    masks = {nm: torch.zeros(n, dtype=torch.bool) for nm in ("tr", "va", "te")}
    masks["tr"][tr_in] = True
    masks["va"][va_in] = True
    masks["te"][te] = True

    def val_f1(logits, mask):
        pred = logits[mask].argmax(1).numpy()
        return f1_score(y[mask.numpy()], pred, average="macro")

    hist = None
    if kind == "mlp":
        model = MLP(Xz.shape[1], C, cfg["hidden"])
        train_plain(model, Xs, yt, masks["tr"], masks["va"], cfg, val_f1, True)
    elif kind == "gcn":
        src, dst, _ = _knn_from_sim(cosine_sim(Xz.astype(np.float64)), cfg["k_ss"])
        A = normalized_adj(src, dst, n, dev)
        model = GCN(Xz.shape[1], C, A, cfg["hidden"])
        train_plain(model, Xs, yt, masks["tr"], masks["va"], cfg, val_f1, True)
    else:
        rel = hetero_graph_multiclass(Xz, y, tr_in, C, cfg, seed)
        Xg = torch.tensor(gene_node_features(Xz[tr_in], cfg["d_gene_in"]), device=dev)
        use_gate = kind not in ("ahgsl_no_gate", "static_hgnn")
        use_aee = kind not in ("ahgsl_no_aee", "static_hgnn")
        run_cfg = dict(cfg)
        if kind == "ahgsl_no_ser":
            run_cfg["use_ser"] = False
        if kind == "ahgsl_no_stab":
            run_cfg["lambda_s"] = 0.0
        model = AHGSL(Xg.shape[0], Xg.shape[1], Xz.shape[1], C, rel,
                      cfg["hidden"], cfg["rounds"], use_gate, use_aee,
                      multiclass=True, device=dev)
        hist = train_hetero(model, Xg, Xs, yt, masks["tr"], masks["va"],
                            run_cfg, val_f1)
    model.eval()
    with torch.no_grad():
        if kind in ("mlp", "gcn"):
            logits = model(Xs)
        else:
            logits, _, _ = model(Xg, Xs)
        P = torch.softmax(logits[masks["te"]], dim=1).numpy()
    res = multiclass_metrics(y[te], P)
    res["train_time"] = time.time() - t0
    if hist is not None:
        res["epochs_run"] = len(hist["loss"])
        res["mean_epoch_time"] = float(np.mean(hist["epoch_time"]))
        res["mean_refined_frac"] = float(np.mean(hist["refined_frac"]))
    if return_model:
        return res, model, hist, dict(keep=keep, tr=tr, te=te, tr_in=tr_in,
                                      va_in=va_in, Xz=Xz, y=y, classes=classes,
                                      gene_names=gene_names[keep])
    return res


def gene_selection_eval(model, ctx, Ks=(25, 50, 100, 200, 500), seed=42):
    """Compare AHGSL gate ranking against classical filters: retrain LR on
    top-K genes, report test accuracy / macro-F1 per K."""
    from sklearn.feature_selection import f_classif, mutual_info_classif
    Xz, y, tr, te = ctx["Xz"], ctx["y"], ctx["tr"], ctx["te"]
    rng = np.random.default_rng(seed)
    G = Xz.shape[1]
    rankings = {}
    m = model.gate().detach().numpy()
    rankings["AHGSL gate"] = np.argsort(-m)
    F_stat, _ = f_classif(Xz[tr], y[tr])
    rankings["ANOVA F-test"] = np.argsort(-np.nan_to_num(F_stat))
    mi = mutual_info_classif(Xz[tr], y[tr], random_state=seed)
    rankings["Mutual information"] = np.argsort(-mi)
    l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=2000)
    l1.fit(Xz[tr], y[tr])
    rankings["L1-logistic"] = np.argsort(-np.abs(l1.coef_).sum(0))
    rankings["Random"] = rng.permutation(G)
    out = {}
    for name, order in rankings.items():
        out[name] = {}
        for K in Ks:
            idx = order[:K]
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(Xz[tr][:, idx], y[tr])
            pred = clf.predict(Xz[te][:, idx])
            out[name][K] = {
                "accuracy": float((pred == y[te]).mean()),
                "f1_macro": float(f1_score(y[te], pred, average="macro")),
            }
    return out, {k: v[:500].tolist() for k, v in rankings.items()}

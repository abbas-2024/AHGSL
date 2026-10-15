"""Training, evaluation metrics, and experiment runners for AHGSL."""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, hamming_loss, accuracy_score,
                             label_ranking_loss, average_precision_score,
                             confusion_matrix)

from .model import AHGSL, GCN, MLP, normalized_adj
from .graph import build_hetero_graph, gene_node_features
from .data_loading import load_yeast, load_tcga, impute_standardize


# --------------------------- metrics ---------------------------
def multilabel_metrics(Y, S, thr=0.5):
    """Y: true binary (n x L); S: scores in [0,1]."""
    P = (S >= thr).astype(int)
    inter = (Y * P).sum(1)
    union = ((Y + P) > 0).sum(1)
    ex_acc = float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))
    out = {
        "accuracy_ex": ex_acc,
        "precision_micro": precision_score(Y, P, average="micro", zero_division=0),
        "recall_micro": recall_score(Y, P, average="micro", zero_division=0),
        "f1_micro": f1_score(Y, P, average="micro", zero_division=0),
        "f1_macro": f1_score(Y, P, average="macro", zero_division=0),
        "hamming": hamming_loss(Y, P),
    }
    try:
        out["auc_micro"] = roc_auc_score(Y, S, average="micro")
        keep = (Y.sum(0) > 0) & (Y.sum(0) < len(Y))
        out["auc_macro"] = roc_auc_score(Y[:, keep], S[:, keep], average="macro")
    except ValueError:
        out["auc_micro"] = out["auc_macro"] = float("nan")
    out["ranking_loss"] = label_ranking_loss(Y, S)
    out["avg_precision_micro"] = average_precision_score(Y, S, average="micro")
    return out


def multiclass_metrics(y, P):
    """y: int labels; P: probability matrix (n x C)."""
    pred = P.argmax(1)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y, pred, average="micro", zero_division=0),
        "auc_macro_ovr": roc_auc_score(y, P, multi_class="ovr", average="macro"),
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def tune_threshold(Y, S):
    best, bt = -1, 0.5
    for t in np.arange(0.1, 0.91, 0.02):
        f = f1_score(Y, (S >= t).astype(int), average="micro", zero_division=0)
        if f > best:
            best, bt = f, t
    return bt


# --------------------------- generic trainer ---------------------------
def train_hetero(model, Xg, Xs, Y, train_mask, val_mask, cfg, val_metric_fn):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_val, best_state, patience = -np.inf, None, 0
    hist = {"loss": [], "l_cls": [], "l_stab": [], "val": [], "refined_frac": [],
            "activated": [], "pruned": [], "epoch_time": []}
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        logits, _, prov = model(Xg, Xs)
        loss, l_cls, l_sel, l_stab = model.losses(
            logits, Y, train_mask, prov, cfg["lambda_g"], cfg["lambda_s"],
            pos_weight=cfg.get("pos_weight"))
        loss.backward()
        opt.step()
        stats = model.commit_graph(Xg, Xs, tau=cfg["tau"], theta=cfg["theta"],
                                   tau_prune=cfg["tau_prune"],
                                   tau_activate=cfg["tau_activate"],
                                   use_ser=cfg.get("use_ser", True))
        ev = (ep % cfg.get("eval_every", 1) == 0) or ep == cfg["epochs"] - 1
        if ev:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(Xg, Xs)
                val = val_metric_fn(logits, val_mask)
        else:
            val = hist["val"][-1] if hist["val"] else -np.inf
        hist["loss"].append(loss.detach().item())
        hist["l_cls"].append(float(l_cls))
        hist["l_stab"].append(float(l_stab))
        hist["val"].append(float(val))
        hist["refined_frac"].append(stats["refined_frac"])
        hist["activated"].append(stats["activated"])
        hist["pruned"].append(stats["pruned"])
        hist["epoch_time"].append(time.time() - t0)
        if ev:
            if val > best_val + 1e-5:
                best_val, patience = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_graph = ({r: model.a_state[r].clone() for r in model.a_state},
                              {r: model.active[r].clone() for r in model.active})
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
        for r in model.a_state:
            model.a_state[r] = best_graph[0][r]
            model.active[r] = best_graph[1][r]
    return hist


def train_plain(model, X, Y, train_mask, val_mask, cfg, val_metric_fn, multiclass):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_val, best_state, patience = -np.inf, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(X)
        if multiclass:
            loss = F.cross_entropy(logits[train_mask], Y[train_mask])
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask], Y[train_mask], pos_weight=cfg.get("pos_weight"))
        loss.backward()
        opt.step()
        if (ep % cfg.get("eval_every", 1)) and ep != cfg["epochs"] - 1:
            continue
        model.eval()
        with torch.no_grad():
            val = val_metric_fn(model(X), val_mask)
        if val > best_val + 1e-5:
            best_val, patience = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return None


DEFAULT_CFG = dict(hidden=128, rounds=2, epochs=250, patience=25, eval_every=2, lr=1e-3, wd=5e-4,
                   lambda_g=1e-4, lambda_s=1e-3, tau=0.10, theta=0.60,
                   tau_prune=0.10, tau_activate=0.70,
                   k_gg=10, k_ss=10, k_sg=10, k_ll=5, k_gg_cand=20, k_ss_cand=20,
                   d_gene_in=64)


def _masks(n, tr, va, te, device):
    m = {}
    for name, idx in (("tr", tr), ("va", va), ("te", te)):
        mm = torch.zeros(n, dtype=torch.bool, device=device)
        mm[idx] = True
        m[name] = mm
    return m


def run_model_yeast(dname, kind, seed=42, cfg=None, return_model=False):
    """kind in: ahgsl, ahgsl_no_aee, ahgsl_no_ser, ahgsl_no_gate, ahgsl_no_stab,
    static_hgnn, gcn, mlp, br_lr."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dname == "yeast14":
        from .data_loading import load_mulan_yeast
        d = load_mulan_yeast()
    else:
        d = load_yeast(dname)
    Xtr, Xva, Xte = impute_standardize(d["X_train"], d["X_valid"], d["X_test"])
    Ytr, Yva, Yte = d["Y_train"], d["Y_valid"], d["Y_test"]
    ntr, nva, nte = len(Xtr), len(Xva), len(Xte)
    X_all = np.vstack([Xtr, Xva, Xte]).astype(np.float32)
    Y_all = np.vstack([Ytr, Yva, Yte]).astype(np.float32)
    tr = np.arange(ntr)
    va = np.arange(ntr, ntr + nva)
    te = np.arange(ntr + nva, ntr + nva + nte)
    L = Y_all.shape[1]
    dev = "cpu"
    t_build0 = time.time()
    if kind in ("br_lr",):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X_all[tr], Y_all[tr])
        S_va, S_te = clf.predict_proba(X_all[va]), clf.predict_proba(X_all[te])
        thr = tune_threshold(Y_all[va], S_va)
        res = multilabel_metrics(Y_all[te], S_te, thr)
        res["train_time"] = time.time() - t_build0
        res["thr"] = thr
        return (res, None, None) if return_model else res

    Xs = torch.tensor(X_all, device=dev)
    Y = torch.tensor(Y_all, device=dev)
    masks = _masks(len(X_all), tr, va, te, dev)
    p = Y_all[tr].mean(0).clip(1e-4, 1)
    cfg["pos_weight"] = torch.tensor(((1 - p) / p).clip(1.0, 10.0),
                                     dtype=torch.float32, device=dev)

    def val_f1(logits, mask):
        S = torch.sigmoid(logits[mask]).numpy()
        return average_precision_score(Y_all[mask.numpy()], S, average="micro")

    if kind == "mlp":
        model = MLP(X_all.shape[1], L, cfg["hidden"])
        t0 = time.time()
        train_plain(model, Xs, Y, masks["tr"], masks["va"], cfg, val_f1, False)
        tt = time.time() - t0
        hist = None
    elif kind == "gcn":
        from .graph import cosine_sim, _knn_from_sim
        src, dst, _ = _knn_from_sim(cosine_sim(X_all.astype(np.float64)), cfg["k_ss"])
        A = normalized_adj(src, dst, len(X_all), dev)
        model = GCN(X_all.shape[1], L, A, cfg["hidden"])
        t0 = time.time()
        train_plain(model, Xs, Y, masks["tr"], masks["va"], cfg, val_f1, False)
        tt = time.time() - t0
        hist = None
    else:
        rel = build_hetero_graph(X_all, X_all[tr], Y_all[tr], tr,
                                 cfg["k_gg"], cfg["k_ss"], cfg["k_sg"], cfg["k_ll"],
                                 cfg["k_gg_cand"], cfg["k_ss_cand"], seed=seed)
        Xg = torch.tensor(gene_node_features(X_all[tr], cfg["d_gene_in"]), device=dev)
        use_gate = kind not in ("ahgsl_no_gate", "static_hgnn")
        use_aee = kind not in ("ahgsl_no_aee", "static_hgnn")
        run_cfg = dict(cfg)
        if kind == "ahgsl_no_ser":
            run_cfg["use_ser"] = False
        if kind == "ahgsl_no_stab":
            run_cfg["lambda_s"] = 0.0
        model = AHGSL(Xg.shape[0], Xg.shape[1], X_all.shape[1], L, rel,
                      cfg["hidden"], cfg["rounds"], use_gate, use_aee,
                      multiclass=False, device=dev)
        p = Y_all[tr].mean(0).clip(1e-3, 1 - 1e-3)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))

        def val_f1_h(logits, mask):
            S = torch.sigmoid(logits[mask]).numpy()
            return average_precision_score(Y_all[mask.numpy()], S, average="micro")
        t0 = time.time()
        hist = train_hetero(model, Xg, Xs, Y, masks["tr"], masks["va"],
                            run_cfg, val_f1_h)
        tt = time.time() - t0
        Xs_in = Xs

    model.eval()
    with torch.no_grad():
        if kind in ("mlp", "gcn"):
            logits = model(Xs)
        else:
            logits, _, _ = model(Xg, Xs)
        S = torch.sigmoid(logits).numpy()
    thr = tune_threshold(Y_all[va], S[va])
    res = multilabel_metrics(Y_all[te], S[te], thr)
    res["train_time"] = tt
    res["thr"] = thr
    if hist is not None:
        res["epochs_run"] = len(hist["loss"])
        res["mean_epoch_time"] = float(np.mean(hist["epoch_time"]))
        res["mean_refined_frac"] = float(np.mean(hist["refined_frac"]))
    if return_model:
        return res, model, hist
    return res

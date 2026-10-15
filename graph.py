"""Heterogeneous graph construction for AHGSL (Eq. 1, initial weighting Eq. per
Section 4.1-4.2): gene(feature) nodes, sample nodes, label nodes; relations
GG, SS, SG, LL, SL with sparse kNN candidate edges."""
import numpy as np


def _knn_from_sim(S, k, sym=True):
    """Given similarity matrix S (n x n), return edge arrays (src, dst, w) of
    kNN edges (excluding self). Weights = similarity mapped to [0,1]."""
    n = S.shape[0]
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    k = min(k, n - 1)
    idx = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = idx.reshape(-1)
    w = S[src, dst]
    if sym:  # union of directed kNN -> undirected unique pairs
        a, b = np.minimum(src, dst), np.maximum(src, dst)
        key = a.astype(np.int64) * n + b
        _, uniq = np.unique(key, return_index=True)
        src, dst, w = a[uniq], b[uniq], w[uniq]
    return src.astype(np.int64), dst.astype(np.int64), w.astype(np.float32)


def _minmax01(w, lo=0.05, hi=1.0):
    w = np.asarray(w, dtype=np.float32)
    if w.size == 0:
        return w
    mn, mx = float(w.min()), float(w.max())
    if mx - mn < 1e-12:
        return np.full_like(w, (lo + hi) / 2)
    return lo + (hi - lo) * (w - mn) / (mx - mn)


def corr_sim(M):
    """Pearson correlation between rows of M (rows are entities)."""
    Mc = M - M.mean(axis=1, keepdims=True)
    sd = Mc.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Mn = Mc / sd
    return (Mn @ Mn.T) / M.shape[1]


def cosine_sim(M):
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    Mn = M / nrm
    return Mn @ Mn.T


def build_hetero_graph(X_all, X_train, Y_train, train_idx,
                       k_gg=10, k_ss=10, k_sg=10, k_ll=5,
                       k_gg_cand=20, k_ss_cand=20, seed=0):
    """Build initial heterogeneous graph.

    X_all: (N x G) standardized features for ALL samples (graph is built from
           features only -> no label leakage for valid/test samples).
    X_train, Y_train: training portion (used for gene-gene correlation and
           label-label co-occurrence, and S-L edges for train samples only).
    train_idx: indices (into X_all rows) of training samples.
    Returns dict of relations: each {src, dst, w, active_mask}; candidate pools
    for GG/SS include extra edges (inactive, weight ~ initial similarity).
    """
    rng = np.random.default_rng(seed)
    N, G = X_all.shape
    L = Y_train.shape[1]
    rel = {}

    # --- Gene-Gene: correlation across training samples (genes are columns) ---
    Sgg = np.abs(corr_sim(X_train.T.astype(np.float64)))
    src, dst, w = _knn_from_sim(Sgg, k_gg_cand)
    # active = within top k_gg per node (approx: rank by weight per src)
    act_src, act_dst, act_w = _knn_from_sim(Sgg, k_gg)
    act_set = set(zip(act_src.tolist(), act_dst.tolist()))
    active = np.array([(a, b) in act_set for a, b in zip(src, dst)])
    rel["gg"] = dict(src=src, dst=dst, w=_minmax01(w), active=active)

    # --- Sample-Sample: cosine on standardized features (all samples) ---
    Sss = cosine_sim(X_all.astype(np.float64))
    src, dst, w = _knn_from_sim(Sss, k_ss_cand)
    act_src, act_dst, _ = _knn_from_sim(Sss, k_ss)
    act_set = set(zip(act_src.tolist(), act_dst.tolist()))
    active = np.array([(a, b) in act_set for a, b in zip(src, dst)])
    rel["ss"] = dict(src=src, dst=dst, w=_minmax01(w), active=active)

    # --- Sample-Gene: top-|z| expressed genes per sample (X_all is z-scored) ---
    k_sg = min(k_sg, G)
    Az = np.abs(X_all)
    idx = np.argpartition(-Az, kth=k_sg - 1, axis=1)[:, :k_sg]
    s_src = np.repeat(np.arange(N), k_sg).astype(np.int64)
    g_dst = idx.reshape(-1).astype(np.int64)
    w = Az[s_src, g_dst]
    rel["sg"] = dict(src=s_src, dst=g_dst, w=_minmax01(w),
                     active=np.ones(len(s_src), dtype=bool))

    # --- Label-Label: co-occurrence cosine on training labels ---
    if L > 1:
        Sll = cosine_sim(Y_train.T.astype(np.float64))
        src, dst, w = _knn_from_sim(Sll, min(k_ll, L - 1))
        rel["ll"] = dict(src=src, dst=dst, w=_minmax01(w),
                         active=np.ones(len(src), dtype=bool))
    else:
        rel["ll"] = dict(src=np.zeros(0, np.int64), dst=np.zeros(0, np.int64),
                         w=np.zeros(0, np.float32), active=np.zeros(0, bool))

    # --- Sample-Label: observed train assignments only ---
    tr = np.asarray(train_idx)
    si, li = np.nonzero(Y_train)
    rel["sl"] = dict(src=tr[si].astype(np.int64), dst=li.astype(np.int64),
                     w=np.ones(len(si), dtype=np.float32),
                     active=np.ones(len(si), dtype=bool))
    return rel


def gene_node_features(X_train, dim=64, seed=0):
    """Initial gene-node features: PCA of the transposed training expression
    (each gene described by its profile across training samples)."""
    M = X_train.T.astype(np.float64)  # G x N_tr
    M = M - M.mean(0, keepdims=True)
    dim = min(dim, min(M.shape) - 1)
    # economical PCA via SVD on centered matrix
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Z = U[:, :dim] * S[:dim]
    Z = Z / (Z.std(0, keepdims=True) + 1e-8)
    return Z.astype(np.float32)

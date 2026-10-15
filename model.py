"""AHGSL model — faithful PyTorch implementation of the paper's Eq. (1)-(13).

Eq. (2)  heterogeneous message passing (relation-specific aggregation +
         relation attention)
Eq. (3)  Adaptive Edge Evolution: a~_ij = sigma(w_psi^T [h_i, h_j, a_ij] + b_psi)
Eq. (4)  Delta_ij = |a~ - a|
Eq. (5)  U_ij = binary entropy of a
Eq. (6)  SER refinement set S = {Delta > tau or U > theta}
Eq. (7)  gene gate m_g = sigma(omega_g)
Eq. (8)  gated gene representations
Eq. (9)  L_sel = lambda_g * sum_g m_g
Eq. (10) y_hat_il = sigma(z_i^S . h_l^L + b_l)
Eq. (11) BCE loss (CE for the multiclass configuration)
Eq. (12) L_stab = lambda_s * sum (a~ - a)^2
Eq. (13) L_total = L_cls + L_sel + L_stab

Round 1 propagates on the committed graph state A^(t); round 2 propagates on
the provisional AEE weights computed from round-1 embeddings, so the edge
scorers are trained end-to-end by the task loss.  After each epoch the
provisional weights are committed selectively through SER (Eq. 4-6) with
edge activation / pruning thresholds, yielding A^(t+1).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REL_TO_TYPES = {  # relation -> (src_type, dst_type)
    "gg": ("g", "g"), "ss": ("s", "s"), "sg": ("s", "g"),
    "ll": ("l", "l"), "sl": ("s", "l"),
}
# messages flow in both directions for every relation
DIRECTED = [(r, 0) for r in REL_TO_TYPES] + [(r, 1) for r in REL_TO_TYPES]


def scatter_wmean(msg, w, dst, n_dst):
    """Weighted mean aggregation: sum_j w_ij m_j / (sum_j w_ij + eps)."""
    d = msg.shape[1]
    num = torch.zeros(n_dst, d, device=msg.device)
    num.index_add_(0, dst, msg * w.unsqueeze(1))
    den = torch.zeros(n_dst, device=msg.device)
    den.index_add_(0, dst, w)
    return num / (den.unsqueeze(1) + 1e-8)


class HeteroRound(nn.Module):
    """One round of heterogeneous message passing (Eq. 2) with per-relation
    transforms and softmax relation attention per destination node type."""

    def __init__(self, dim):
        super().__init__()
        self.self_lin = nn.ModuleDict({t: nn.Linear(dim, dim) for t in "gsl"})
        self.rel_lin = nn.ModuleDict()
        self.rel_att = nn.ParameterDict()
        for rel, flip in DIRECTED:
            key = f"{rel}_{flip}"
            self.rel_lin[key] = nn.Linear(dim, dim)
            self.rel_att[key] = nn.Parameter(torch.zeros(1))

    def forward(self, H, edges, weights):
        """H: dict type->tensor; edges: dict rel->(src,dst) LongTensors (as
        stored, src/dst in the ORDER of REL_TO_TYPES types); weights: dict
        rel->FloatTensor of current edge weights."""
        agg = {t: [] for t in "gsl"}
        att = {t: [] for t in "gsl"}
        for rel, flip in DIRECTED:
            st, dt = REL_TO_TYPES[rel]
            src, dst = edges[rel]
            if flip:
                st, dt = dt, st
                src, dst = dst, src
            if src.numel() == 0:
                continue
            key = f"{rel}_{flip}"
            msg = self.rel_lin[key](H[st])[src]
            out = scatter_wmean(msg, weights[rel], dst, H[dt].shape[0])
            agg[dt].append(out)
            att[dt].append(self.rel_att[key])
        H_new = {}
        for t in "gsl":
            h = self.self_lin[t](H[t])
            if agg[t]:
                a = torch.softmax(torch.cat(att[t]), dim=0)
                h = h + sum(ai * oi for ai, oi in zip(a, agg[t]))
            H_new[t] = F.relu(h)
        return H_new


class AHGSL(nn.Module):
    def __init__(self, n_genes, d_gene_in, d_sample_in, n_labels, rel,
                 hidden=128, rounds=2, use_gate=True, use_aee=True,
                 multiclass=False, device="cpu"):
        super().__init__()
        self.n_genes, self.n_labels = n_genes, n_labels
        self.use_gate, self.use_aee = use_gate, use_aee
        self.multiclass = multiclass
        self.device = device
        self.proj = nn.ModuleDict({
            "g": nn.Linear(d_gene_in, hidden),
            "s": nn.Linear(d_sample_in, hidden),
        })
        self.label_emb = nn.Embedding(n_labels, hidden)
        self.rounds = nn.ModuleList([HeteroRound(hidden) for _ in range(rounds)])
        self.omega = nn.Parameter(torch.full((n_genes,), 2.0))  # Eq. (7) gate logits
        # AEE edge scorers (Eq. 3): per relation, w_psi over [h_i, h_j, a_ij]
        self.aee = nn.ModuleDict({r: nn.Linear(2 * hidden + 1, 1) for r in REL_TO_TYPES})
        for m in self.aee.values():  # neutral start: sigma(0) = 0.5
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.aee_temp = 0.5  # temperature sharpening -> decisive edge scores
        self.classifier = nn.Linear(2 * hidden, n_labels)  # Eq. (10): w_l^T z_i + b_l
        # persistent graph state
        self.edges, self.a_state, self.active = {}, {}, {}
        for r, e in rel.items():
            self.edges[r] = (torch.as_tensor(e["src"], device=device),
                             torch.as_tensor(e["dst"], device=device))
            self.a_state[r] = torch.as_tensor(e["w"], device=device).clone()
            self.active[r] = torch.as_tensor(e["active"], device=device).clone()

    # ---------------- forward ----------------
    def gate(self):
        return torch.sigmoid(self.omega)  # Eq. (7)

    def _active_w(self, rel):
        return self.a_state[rel] * self.active[rel].float()

    def _aee_scores(self, H, rel, mask=None):
        """Provisional edge scores a~ (Eq. 3) for edges of `rel` (or subset)."""
        st, dt = REL_TO_TYPES[rel]
        src, dst = self.edges[rel]
        a = self.a_state[rel]
        if mask is not None:
            src, dst, a = src[mask], dst[mask], a[mask]
        z = torch.cat([H[st][src], H[dt][dst], a.unsqueeze(1)], dim=1)
        a0 = a.clamp(1e-4, 1 - 1e-4)
        base = torch.log(a0 / (1 - a0))  # residual form: score starts at a_ij
        return torch.sigmoid(base + self.aee[rel](z).squeeze(1) / self.aee_temp)

    def forward(self, Xg, Xs):
        H = {"g": F.relu(self.proj["g"](Xg)),
             "s": F.relu(self.proj["s"](Xs)),
             "l": self.label_emb.weight}
        if self.use_gate:
            H = dict(H)
            H["g"] = H["g"] * self.gate().unsqueeze(1)  # Eq. (8)
        if self.training:
            H = {t: F.dropout(h, 0.2) for t, h in H.items()}
        w_r1 = {r: self._active_w(r) for r in self.edges}
        H1 = self.rounds[0](H, self.edges, w_r1)
        prov = {}
        if self.use_aee:
            # round 2 on provisional AEE weights (differentiable)
            w_r2 = {}
            for r in self.edges:
                if self.edges[r][0].numel() == 0:
                    w_r2[r] = w_r1[r]
                    continue
                a_tilde = self._aee_scores(H1, r)
                prov[r] = a_tilde
                w_r2[r] = a_tilde * self.active[r].float()
        else:
            w_r2 = w_r1
        H2 = self.rounds[1](H1, self.edges, w_r2) if len(self.rounds) > 1 else H1
        z_s = torch.cat([H2["s"], H["s"]], dim=1)  # message-passed + input projection
        logits = self.classifier(z_s)  # Eq. (10): sigma(w_l^T z_i^S + b_l)
        H2 = dict(H2); H2["s"] = z_s
        return logits, H2, prov

    # ---------------- losses ----------------
    def losses(self, logits, Y, mask, prov, lambda_g=1e-4, lambda_s=1e-3,
               pos_weight=None):
        if self.multiclass:
            l_cls = F.cross_entropy(logits[mask], Y[mask])  # Eq. (11) CE variant
        else:
            l_cls = F.binary_cross_entropy_with_logits(logits[mask], Y[mask],
                                                       pos_weight=pos_weight)
        l_sel = self.gate().mean() if self.use_gate else torch.zeros((), device=logits.device)
        if prov:
            l_stab = sum(((prov[r] - self.a_state[r]) ** 2 * self.active[r].float()).sum()
                         / (self.active[r].float().sum() + 1e-8) for r in prov) / len(prov)
        else:
            l_stab = torch.zeros((), device=logits.device)
        total = l_cls + lambda_g * l_sel * self.n_genes + lambda_s * l_stab  # Eq. (13)
        return total, l_cls.detach(), l_sel.detach(), l_stab.detach()

    # ---------------- AEE + SER commit (Eq. 3-6) ----------------
    @torch.no_grad()
    def commit_graph(self, Xg, Xs, tau=0.10, theta=0.50,
                     tau_prune=0.10, tau_activate=0.70, use_ser=True):
        """Periodic graph update A^(t) -> A^(t+1). Returns stats dict."""
        if not self.use_aee:
            return {"refined_frac": 0.0, "activated": 0, "pruned": 0,
                    "refined": 0, "candidates": 0}
        H = {"g": F.relu(self.proj["g"](Xg)),
             "s": F.relu(self.proj["s"](Xs)),
             "l": self.label_emb.weight}
        if self.use_gate:
            H["g"] = H["g"] * self.gate().unsqueeze(1)
        H1 = self.rounds[0](H, self.edges, {r: self._active_w(r) for r in self.edges})
        n_ref = n_cand = n_act = n_prn = 0
        for r in self.edges:
            if self.edges[r][0].numel() == 0 or r == "sl":
                continue  # observed sample-label training edges stay fixed
            a_tilde = self._aee_scores(H1, r)
            a_old = self.a_state[r]
            act = self.active[r]
            if use_ser:
                delta = (a_tilde - a_old).abs()                     # Eq. (4)
                p = a_old.clamp(1e-6, 1 - 1e-6)
                U = -(p * p.log() + (1 - p) * (1 - p).log())        # Eq. (5)
                sel = (delta > tau) | (U > theta)                   # Eq. (6)
            else:
                sel = torch.ones_like(a_old, dtype=torch.bool)
            refine = sel & act
            self.a_state[r][refine] = a_tilde[refine]
            # edge activation from candidate pool / pruning of weak edges
            new_edges = (~act) & (a_tilde > tau_activate)
            self.active[r][new_edges] = True
            self.a_state[r][new_edges] = a_tilde[new_edges]
            dead = self.active[r] & (self.a_state[r] < tau_prune)
            self.active[r][dead] = False
            n_ref += int(refine.sum())
            n_cand += int(act.sum())
            n_act += int(new_edges.sum())
            n_prn += int(dead.sum())
        return {"refined_frac": n_ref / max(n_cand, 1), "refined": n_ref,
                "candidates": n_cand, "activated": n_act, "pruned": n_prn}


class GCN(nn.Module):
    """Homogeneous 2-layer GCN baseline over the sample-sample graph."""

    def __init__(self, d_in, n_labels, A_hat, hidden=128):
        super().__init__()
        self.A = A_hat  # sparse normalized adjacency (torch.sparse)
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_labels)
        self.drop = nn.Dropout(0.3)

    def forward(self, X):
        h = F.relu(torch.sparse.mm(self.A, self.l1(X)))
        h = self.drop(h)
        h = F.relu(torch.sparse.mm(self.A, self.l2(h)))
        return self.out(h)


def normalized_adj(src, dst, n, device="cpu"):
    """Symmetric normalized adjacency with self-loops for GCN."""
    src2 = np.concatenate([src, dst, np.arange(n)])
    dst2 = np.concatenate([dst, src, np.arange(n)])
    w = np.ones(len(src2), dtype=np.float32)
    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, src2, w)
    wn = w / np.sqrt(deg[src2] * deg[dst2])
    idx = torch.tensor(np.vstack([src2, dst2]), dtype=torch.long, device=device)
    return torch.sparse_coo_tensor(idx, torch.tensor(wn, device=device),
                                   (n, n)).coalesce()


class MLP(nn.Module):
    def __init__(self, d_in, n_labels, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(hidden, n_labels))

    def forward(self, X):
        return self.net(X)

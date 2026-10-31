r"""
signed_rule_model.py
====================
Signed, weighted generalisation of the majority-rule diffusion model.

Edges may carry

    sign    in {+1, -1}      (default +1)
    weight  >= 0             (default 1)

Node v becomes active iff its NET active support strictly exceeds
theta_v * deg(v):

    sum_{u active, sign=+1} w(u,v)  -  sum_{u active, sign=-1} w(u,v)
                                                    >  theta_v * deg(v)     (1)

where deg(v) is the total incident ABSOLUTE weight.  For an all-positive
graph (A_neg = 0) this reduces exactly to MajorityRuleModel, so the signed
model is a faithful superset of the unsigned one.

Why this matters for the ABC study
----------------------------------
Negative edges break MONOTONICITY of sigma: activating a node that is an
"enemy" of v lowers v's net support and can push it back below its
threshold.  Greedy and CELF are hill climbers over marginal gains; under a
non-monotone objective their updates are not even self-consistent
(they assume sigma never decreases, and CELF's lazy bounds assume
submodularity on top).  ABC searches whole k-sets with swap operators that
can remove harmful nodes, so it is the natural fit -- exactly the regime
where the metaheuristic's advantage should be largest.

Reference points: Li et al. (2014) polarity-related IM on signed networks
(PLOS ONE 9(7)); Rawal & Khan (2019) contrasting opinions; Zahoor et al.
(2026) RASH, signed awareness model with gamma-weak submodularity.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import scipy.sparse as sp

from majority_rule_model import MajorityRuleModel, candidate_pool

__all__ = [
    "SignedMajorityRuleModel",
    "make_signed",
    "signed_star",
    "signed_cycle",
    "signed_sbm",
    "signed_ba",
    "signed_ws",
    "load_signed_edgelist",
    "signed_candidate_pool",
]


class SignedMajorityRuleModel(MajorityRuleModel):
    """
    Majority-rule diffusion on a signed, weighted undirected graph.

    Activation rule (1): positive active neighbours push v's net score up,
    negative (enemy) active neighbours push it down; v fires once the net
    score strictly exceeds theta_v * deg(v).  theta_v ~ same family as the
    unsigned model (dmr / smr / lt).
    """

    def _build_adjacency(self, graph: nx.Graph):
        n = self.n
        A_pos = sp.lil_matrix((n, n), dtype=self.dtype)
        A_neg = sp.lil_matrix((n, n), dtype=self.dtype)
        for u, v, d in graph.edges(data=True):
            w = float(abs(d.get("weight", 1.0)))
            s = int(d.get("sign", 1)) if d.get("sign") is not None else (
                1 if float(d.get("weight", 1.0)) >= 0 else -1)
            i, j = self.index[u], self.index[v]
            if s < 0:
                A_neg[i, j] = A_neg[j, i] = w
            else:
                A_pos[i, j] = A_pos[j, i] = w
        A_pos = A_pos.tocsr()
        A_neg = A_neg.tocsr()
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.n_neg_edges = int((A_neg > 0).nnz) // 2
        A = (A_pos - A_neg).tocsr()
        deg = np.asarray(A_pos.sum(axis=1)).ravel() + np.asarray(A_neg.sum(axis=1)).ravel()
        return A, deg

    def __init__(self, graph, variant="smr", delta=0.15, n_samples=200,
                 seed=0, dtype=np.float32, backend="auto"):
        super().__init__(graph, variant=variant, delta=delta,
                         n_samples=n_samples, seed=seed, dtype=dtype,
                         backend=backend)
        # positive support bias: used for sign-aware seed initialisation
        self.pool_weight = (np.asarray(self.A_pos.sum(axis=1)).ravel()
                            - np.asarray(self.A_neg.sum(axis=1)).ravel())
        self.pool_weight = np.maximum(self.pool_weight - self.pool_weight.min() + 1.0, 1e-6)

    def describe(self) -> str:
        return (super().describe().replace("MajorityRuleModel(", "SignedMajorityRuleModel(")
                + f", neg_edges={self.n_neg_edges})")


# --------------------------------------------------------------------------- #
#  Signed graph zoo / generators
# --------------------------------------------------------------------------- #
def make_signed(G: nx.Graph, p_neg: float = 0.2, weight: int | None = None,
                seed: int = 0) -> nx.Graph:
    """Assign random signs (and optional integer weights) to an unsigned graph."""
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v in G.edges():
        s = -1 if rng.random() < p_neg else 1
        w = int(rng.integers(1, weight + 1)) if weight else 1
        H.add_edge(u, v, sign=s, weight=w)
    return H


def signed_star(n_pos: int = 3, n_neg: int = 2) -> nx.Graph:
    """Hub with positive leaves 1..n_pos and enemy leaves -1..-n_neg."""
    G = nx.Graph()
    for i in range(1, n_pos + 1):
        G.add_edge(0, i, sign=1, weight=1.0)
    for i in range(1, n_neg + 1):
        G.add_edge(0, n_pos + i, sign=-1, weight=1.0)
    return G


def signed_hub_blocker() -> nx.Graph:
    """
    Minimal graph on which sigma is NOT monotone.

    Hub 0 has four positive leaves 1-4, one positive follower 6, and one
    enemy leaf 5.  Seeding {1,2,3,4} gives the hub net support 4 >= 4 = need,
    so hub and follower 6 activate (sigma = 6).  Adding enemy seed 5 drops
    the hub's net support to 3 < 4: hub and follower go dark.
    sigma({1,2,3,4,5}) = 5 < 6.  A strictly LARGER seed set with strictly
    SMALLER influence -- monotonicity fails, and with it greedy/CELF's only
    remaining rationale on signed graphs.
    """
    G = signed_star(4, 1)                    # nodes: hub 0, pos 1-4, enemy 5
    G.add_edge(0, 6, sign=1, weight=1.0)     # follower 6, activates only via hub
    return G


def signed_cycle(n: int, p_neg: float = 0.25, seed: int = 0) -> nx.Graph:
    return make_signed(nx.cycle_graph(n), p_neg=p_neg, seed=seed)


def signed_sbm(sizes, p_in: float, p_out: float, p_neg_in: float = 0.05,
               p_neg_out: float = 0.6, seed: int = 0) -> nx.Graph:
    """
    Community-structured signed graph: mostly friendly inside communities,
    mostly hostile across them (structural balance).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    offset, blocks = 0, []
    for s in sizes:
        blk = list(range(offset, offset + s))
        blocks.append(blk)
        offset += s
    G.add_nodes_from(range(offset))
    for a in range(len(blocks)):
        for b in range(a, len(blocks)):
            p, p_neg = (p_in, p_neg_in) if a == b else (p_out, p_neg_out)
            for u in blocks[a]:
                for v in blocks[b]:
                    if u == v:
                        continue
                    if rng.random() >= p:
                        continue
                    s = -1 if rng.random() < p_neg else 1
                    G.add_edge(u, v, sign=s, weight=1.0)
    return G


def signed_ba(n: int, m: int, p_neg: float = 0.2, seed: int = 0) -> nx.Graph:
    return make_signed(nx.barabasi_albert_graph(n, m, seed=seed),
                       p_neg=p_neg, seed=seed + 1)


def signed_ws(n: int, k: int, p: float, p_neg: float = 0.2, seed: int = 0) -> nx.Graph:
    return make_signed(nx.watts_strogatz_graph(n, k, p, seed=seed),
                       p_neg=p_neg, seed=seed + 1)


def load_signed_edgelist(path: str) -> nx.Graph:
    """Whitespace-separated lines:  u v sign [weight]."""
    G = nx.Graph()
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            u, v, s = int(parts[0]), int(parts[1]), float(parts[2])
            w = float(parts[3]) if len(parts) > 3 else 1.0
            G.add_edge(u, v, sign=int(s), weight=abs(w))
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, label_attribute="orig")


def find_nonmonotone_witness(model: SignedMajorityRuleModel, max_size: int = 3,
                             node_cap: int = 40, seed: int = 0) -> dict | None:
    """
    Targeted search for a non-monotonicity certificate: an (S, v) pair with
    sigma(S + v) < sigma(S) - tol, i.e. seeding v strictly HURTS.  Random
    S-subset-of-T sampling almost never sees this (the blocking configuration
    is a needle), so we search STRUCTURALLY -- for each node w, S = a subset
    of w's positive neighbours just below its threshold and v = a negative
    neighbour of w -- and then enumerate small S, v directly.

    Returns {S, v, sigma_S, sigma_Sv, drop} or None if none found.
    """
    rng = np.random.default_rng(seed)
    tol = 4.0 * float(np.mean(
        [model.sigma_std(rng.choice(model.n, 3, replace=False)) for _ in range(8)])) + 1e-9
    A_pos, A_neg = model.A_pos.tocsr(), model.A_neg.tocsr()
    pos_nb = {i: list(A_pos.indices[A_pos.indptr[i]:A_pos.indptr[i + 1]])
              for i in range(model.n)}
    neg_nb = {i: list(A_neg.indices[A_neg.indptr[i]:A_neg.indptr[i + 1]])
              for i in range(model.n)}
    need = model.need.mean(axis=1)

    # --- structured pass: positive coalition of w + an enemy of w --- #
    for w in range(model.n):
        if len(pos_nb[w]) < 2 or not neg_nb[w]:
            continue
        S = pos_nb[w][: int(min(len(pos_nb[w]), max(1, need[w] - 1)))]
        if len(S) < 1:
            continue
        s0 = model.sigma(S)
        for v in neg_nb[w]:
            sv = model.sigma(S + [v])
            if sv < s0 - tol:
                return {"S": S, "v": v, "sigma_S": s0,
                        "sigma_Sv": sv, "drop": s0 - sv, "tol": tol}

    # --- generic pass on a capped node set --- #
    from itertools import combinations
    nodes = list(range(model.n))
    if model.n > node_cap:
        idx = np.argsort(-(model.pool_weight + rng.random(model.n) * 1e-3))
        nodes = sorted(idx[:node_cap].tolist())
    for s_size in range(1, max_size + 1):
        for S in combinations(nodes, s_size):
            s0 = model.sigma(S)
            for v in nodes:
                if v in S:
                    continue
                sv = model.sigma(S + (v,))
                if sv < s0 - tol:
                    return {"S": list(S), "v": v, "sigma_S": s0,
                            "sigma_Sv": sv, "drop": s0 - sv, "tol": tol}
    return None


def signed_candidate_pool(model: SignedMajorityRuleModel, size: int,
                          seed: int = 0) -> list[int]:
    """
    Sign-aware pool: blend (a) nodes with large positive net support,
    (b) PageRank, (c) k-core, (d) low-threshold nodes, plus a random tail.
    """
    G, n = model.graph, model.n
    size = min(size, n)
    posw = np.asarray(model.A_pos.sum(axis=1)).ravel()
    negw = np.asarray(model.A_neg.sum(axis=1)).ravel()
    net_rank = sorted(range(n), key=lambda i: -(posw[i] - negw[i]))
    pr = nx.pagerank(G)
    pr_rank = sorted(range(n), key=lambda i: -pr[model.nodes[i]])
    core = nx.core_number(nx.Graph(G))
    core_rank = sorted(range(n), key=lambda i: -core[model.nodes[i]])
    easy_rank = sorted(range(n), key=lambda i: (model.need[i].mean(), -(posw[i] - negw[i])))

    pool, seen = [], set()
    ranked_slots = max(1, (3 * size) // 4)
    per = max(1, ranked_slots // 4)
    for ranking in (net_rank, pr_rank, core_rank, easy_rank):
        for v in ranking[: per + ranked_slots]:
            if v not in seen:
                seen.add(v)
                pool.append(v)
            if len(pool) >= ranked_slots:
                break
        if len(pool) >= ranked_slots:
            break
    if len(pool) < size:
        rng = np.random.default_rng(seed)
        rest = [v for v in range(n) if v not in seen]
        rng.shuffle(rest)
        pool.extend(rest[: size - len(pool)])
    return pool[:size]


if __name__ == "__main__":
    G = signed_hub_blocker()
    m = SignedMajorityRuleModel(G, variant="dmr", n_samples=1, seed=0)
    print(m.describe())
    print("  sigma({1,2,3,4})   =", m.sigma([1, 2, 3, 4]))
    print("  sigma({1,2,3,4,5}) =", m.sigma([1, 2, 3, 4, 5]), " <- enemy seed BLOCKS the hub")
    print("  non-monotone:", m.sigma([1, 2, 3, 4, 5]) < m.sigma([1, 2, 3, 4]))
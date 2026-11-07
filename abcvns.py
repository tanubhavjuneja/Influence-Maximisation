r"""
abcvns.py
=========
Faithful implementation of ABCVNS -- "Artificial bee colony meets variable
neighborhood search" (Tang, Dong, Wu, Chen, Wei; Swarm and Evolutionary
Computation 105 (2026) 102404, ABC.pdf) -- adapted to our code base.

What the paper does (and what this file reproduces)
---------------------------------------------------
* Search fitness: LIE (Local Influence Estimation, Eq. (2)) on the
  INDEPENDENT CASCADE model with constant edge probability p.  No Monte
  Carlo inside the search loop.
* Population init: half = top-k degree nodes, each node replaced w.p. 0.5;
  the other half = uniform random k-sets (Sec. 3.3.1).
* Employed bees: EGCNC -- elite-global cooperative neighborhood
  construction, Eqs. (5)-(7) (Sec. 3.3.2, Alg. 1).
* Onlooker bees: tournament selection (ts = 3) + EGCNC (Sec. 3.3.3, Alg. 2).
* Scout bees: bifurcated abandonment -- with prob. 0.5 keep S_i ^ Gbest and
  fill the rest randomly, else a fully random set (Sec. 3.3.4, Alg. 3).
* VNS on Gbest: three neighborhood tiers (Sec. 3.4.2)
      N1: Score-driven 1-swap        Score = 0.5*normDeg + 0.5*normPR
      N2: BC-oriented 2-hop swap     random removal, max-BC insertion
      N3: community-aware swap       retain top r% by BC, insert max-PR
                                     node of the uncovered community with
                                     the largest total PR
    + local search: for each seed (by descending Score) try the top-cn
    Score-ranked direct neighbours (Eq. (19)), first improvement,
    sequential vs cyclic neighborhood change (Alg. 4).
* Population management: when Gbest stagnates for tau_max iterations,
  keep the top r*NP individuals, reinitialise the rest via Score-guided
  sampling, then run VNS (cyclic) on P_best (Sec. 3.5, Alg. 5).

Paper parameter settings (Sec. 4.2, final values):
    gmax = 150, NP = 50, ts = 3, r = 0.2, cn = 10, L = 5, tau_max = 5,
    mmax = 3, alpha = 0.5 (Score), p = 0.05 (0.01 for the big networks),
    Louvain communities.

Documented deviations (kept as small as possible):
  * we use UNDIRECTED graphs (the project convention); the paper's graphs
    are undirected social networks as well.
  * betweenness centrality is exact for the bitcoin graphs, sampled
    (384-512 pivots) for Slashdot, instead of the paper's exact BC.
  * the paper runs 30 independent restarts; the runner uses a smaller
    restart count for wall-clock reasons (logged per run).
  * edge weights (|rating|, 1..10 on the bitcoin graphs) enter both the
    LIE adjacency and the IC oracle consistently, so LIE remains aligned
    with the IC model used for scoring.
  * FITNESS TWEAK (the one that matters for spread): LIE is an IC-specific
    surrogate and collapses on the majority-rule model (spread ~= k on the
    bitcoin graphs).  For majority-rule runs the runner builds the context
    with mode="mr", which replaces the fitness by a two-hop
    expected-activation surrogate of the stochastic majority rule
    (see _mr_fitness).  The entire search machinery (EGCNC, VNS tiers,
    scouts, population management, parameters) is unchanged; only the
    scalar fitness function differs.  IC runs keep the paper's LIE
    (mode="lie").
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

__all__ = ["LIEState", "lie_fitness", "abcnvs_search",
           "ABCVNS_PARAMS", "make_abcnvs_context"]


ABCVNS_PARAMS = dict(gmax=150, NP=50, ts=3, r=0.2, cn=10, L=5, tau_max=5,
                     mmax=3, alpha=0.5)


class LIEState:
    """Precomputed network data shared by every fitness evaluation."""

    def __init__(self, A: sp.csr_matrix, p: float, deg, pr, bc, score, comm,
                 mode: str = "lie", delta: float = 0.15):
        self.A = A
        self.p = float(p)
        self.n = A.shape[0]
        self.deg = np.asarray(deg, dtype=np.float64)
        self.pr = np.asarray(pr, dtype=np.float64)
        self.bc = np.asarray(bc, dtype=np.float64)
        self.score = np.asarray(score, dtype=np.float64)
        self.comm = np.asarray(comm, dtype=np.int64)
        self.mode = mode                      # "lie" (paper) or "mr" (tweak)
        self.delta = float(delta)
        # majority-rule threshold support: theta ~ U[lo, hi]
        self.lo = 0.5 - self.delta
        self.width = 2.0 * self.delta
        # community -> total PR (for the N3 tier)
        self.comm_pr = np.zeros(int(self.comm.max()) + 1, dtype=np.float64)
        np.add.at(self.comm_pr, self.comm, self.pr)

    def fitness(self, S) -> float:
        """Scalar fitness for a seed set (the only model-dependent piece)."""
        return _mr_fitness(self, S) if self.mode == "mr" else lie_fitness(self, S)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def make_abcnvs_context(G, p: float, metrics: dict, alpha: float = 0.5,
                        mode: str = "lie", delta: float = 0.15
                        ) -> LIEState:
    """Build the LIEState from cached structural metrics.

    mode="lie": the paper's IC fitness; mode="mr": the majority-rule
    surrogate (tweak, used for majority-rule runs).
    """
    import networkx as nx
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    pr = metrics["pr"]
    bc = metrics["bc"]
    comm = metrics["comm"]
    score = alpha * _norm(deg) + (1.0 - alpha) * _norm(pr)
    return LIEState(A, p, deg, pr, bc, score, comm, mode=mode, delta=delta)


def lie_fitness(st: LIEState, S) -> float:
    r"""
    Eq. (2) of the paper:

        LIE(S) = k + ( 1 + (1/|N1(S)\S|) * sum_{u in N2(S)\S} p*d*_u )
                     * sum_{i in N1(S)\S} ( 1 - prod_{j in S,(i,j) in E}(1-p) )

    where d*_u = number of edges from u into N1(S) u N2(S).
    With constant p the product term is (1 - (1-p)^{c_i}), c_i = # S-neighbours
    of i.  N2 excludes S and N1 (two-hop beyond the one-hop shell).
    """
    n, A, p = st.n, st.A, st.p
    k = len(set(S))
    s = np.zeros(n, dtype=np.int8)
    s[np.asarray(list(set(S)), dtype=np.int64)] = 1
    cnt = A.dot(s.astype(np.float32))                  # # neighbours in S
    n1 = (cnt > 0) & (s == 0)
    if not n1.any():
        return float(k)
    # sigma*1(S)
    sig1 = float((1.0 - (1.0 - p) ** cnt[n1]).sum())
    # N2: reachable in 2 hops, excluding S and N1
    n2 = (A.dot(n1.astype(np.float32)) > 0) & (s == 0) & (~n1)
    if not n2.any():
        return float(k + sig1)
    d_u = A.dot((n1 | n2).astype(np.float32))
    sum2 = float(d_u[n2].sum())
    return float(k + (1.0 + p * sum2 / max(1, int(n1.sum()))) * sig1)


def _mr_fitness(st: LIEState, S) -> float:
    r"""
    Two-hop expected-activation surrogate of the stochastic majority rule
    (the "minor tweak" that fixes ABCVNS's spread on the majority model).

    The majority model activates v when counts_v > theta_v * deg_v with
    theta_v ~ U[1/2-delta, 1/2+delta]; given an expected active mass c the
    activation probability is therefore

        P_v(c) = clamp( (c/deg_v - lo) / width, 0, 1 ),  lo = 1/2 - delta.

    Score(S) = |S|
             + sum_{i in N1(S)\S}  P_i( mass from S )
             + sum_{j in N2(S)\S}  P_j( mass from S + expected N1 ).

    This replaces the IC-specific two-hop-density reward of LIE with a
    reward for *low-degree neighbours that actually flip* -- the exact
    structure majority-rule cascades need.  Costs two sparse matvecs, i.e.
    the same order as LIE.
    """
    n, A = st.n, st.A
    k = len(set(S))
    s = np.zeros(n, dtype=np.float64)
    s[np.asarray(list(set(S)), dtype=np.int64)] = 1.0
    safe = np.maximum(st.deg, 1e-12)

    cnt1 = A.dot(s)                                   # mass from seeds only
    p1 = np.clip((cnt1 / safe - st.lo) / st.width, 0.0, 1.0)
    p1[s == 1] = 0.0
    n1 = cnt1 > 0
    n1[s == 1] = False

    e = s + p1                                        # expected active mass
    score = float(k) + float(p1.sum())

    cnt2 = A.dot(e)                                   # expected 2-hop mass
    p2 = np.clip((cnt2 / safe - st.lo) / st.width, 0.0, 1.0)
    p2[s == 1] = 0.0
    p2[n1] = 0.0
    return score + float(p2.sum())


# --------------------------------------------------------------------------- #
#  EGCNC candidate generation (Eqs. 5-7)
# --------------------------------------------------------------------------- #
def _egcnc_candidate(st: LIEState, S_i, P, fits, i, gbest, rng, k) -> list[int]:
    fit_i = fits[i]
    elite = [P[j] for j in range(len(P)) if j != i and fits[j] > fit_i]
    if elite:
        pool = set(S_i)
        for Sj in elite:
            pool |= set(Sj)
    else:
        pool = set(S_i) | set(gbest)
    pool = list(pool)
    if len(pool) < k:                       # safety: never starve the sample
        rest = [v for v in range(st.n) if v not in set(pool)]
        rng.shuffle(rest)
        pool += rest[: k - len(pool)]
    return rng.choice(pool, size=k, replace=False).tolist()


# --------------------------------------------------------------------------- #
#  VNS neighborhoods (Sec. 3.4.2) + VNS procedure (Alg. 4)
# --------------------------------------------------------------------------- #
def _neighbors_mask(st: LIEState, S, hops: int) -> np.ndarray:
    s = np.zeros(st.n, dtype=np.int8)
    s[np.asarray(S, dtype=np.int64)] = 1
    mask = np.zeros(st.n, dtype=bool)
    cur = s.astype(np.float32)
    for _ in range(hops):
        cur = (st.A.dot(cur) > 0).astype(np.float32)
        mask |= (cur > 0)
    mask[np.asarray(S, dtype=np.int64)] = False
    return mask


def _shake_n1(st: LIEState, S, rng, k) -> list[int]:
    r"""Score-driven one-hop swap: out = min-Score in S,
    in = max-Score in N1(S)\S."""
    S = list(S)
    Ss = np.asarray(S, dtype=np.int64)
    out = int(Ss[np.argmin(st.score[Ss])])
    cand = np.flatnonzero(_neighbors_mask(st, S, 1))
    if cand.size == 0:
        return S
    inn = int(cand[np.argmax(st.score[cand])])
    return [inn if v == out else v for v in S]


def _shake_n2(st: LIEState, S, rng, k) -> list[int]:
    r"""BC-oriented two-hop swap: out = random, in = max-BC in N2(S)\S."""
    S = list(S)
    out = S[int(rng.integers(k))]
    cand = np.flatnonzero(_neighbors_mask(st, S, 2))
    if cand.size == 0:
        return S
    inn = int(cand[np.argmax(st.bc[cand])])
    return [inn if v == out else v for v in S]


def _shake_n3(st: LIEState, S, rng, k, r: float) -> list[int]:
    """Community-aware swap: retain top r% by BC; evict min-Score from the
    rest; insert max-PR node of the uncovered community with max total PR."""
    S = list(S)
    Ss = np.asarray(S, dtype=np.int64)
    order = np.argsort(-st.bc[Ss])
    n_keep = max(0, int(np.ceil(r * k)))
    keep = set(int(x) for x in Ss[order[:n_keep]])
    removable = [v for v in S if v not in keep] or S
    out = min(removable, key=lambda v: st.score[v])
    covered = set(int(c) for c in st.comm[Ss])
    uncovered = [c for c in range(len(st.comm_pr)) if c not in covered]
    if not uncovered:
        cand = np.flatnonzero(_neighbors_mask(st, S, 3))
        if cand.size == 0:
            return S
        inn = int(cand[np.argmax(st.pr[cand])])
    else:
        c_star = int(uncovered[np.argmax(st.comm_pr[uncovered])])
        members = np.flatnonzero(st.comm == c_star)
        inn = int(members[np.argmax(st.pr[members])])
    return [inn if v == out else v for v in S]


def _vns(st: LIEState, S0, k, rng, mmax: int = 3, cn: int = 10,
         r: float = 0.2, cyclic: bool = False) -> list[int]:
    """Alg. 4: shaking (tier m) -> local search (Eq. 19) -> acceptance."""
    S = list(S0)
    best_lie = st.fitness(S)
    m = 1
    while m <= mmax:
        if m == 1:
            Sp = _shake_n1(st, S, rng, k)
        elif m == 2:
            Sp = _shake_n2(st, S, rng, k)
        else:
            Sp = _shake_n3(st, S, rng, k, r)
        Sp = list(dict.fromkeys(Sp))[:k]
        lie_p = st.fitness(Sp)
        # ---- local search (first improvement) ----
        found = False
        order = np.argsort(-st.score[np.asarray(Sp, dtype=np.int64)])
        for pos_i in order:
            v_i = int(Sp[pos_i])
            nbr = st.A.indices[st.A.indptr[v_i]: st.A.indptr[v_i + 1]]
            cand = [int(u) for u in nbr if u not in Sp]
            if not cand:
                continue
            sc = st.score[np.asarray(cand, dtype=np.int64)]
            top = [cand[j] for j in np.argsort(-sc)[:cn]]
            for u in top:
                Spp = list(Sp)
                Spp[pos_i] = u
                f_pp = st.fitness(Spp)
                if f_pp > lie_p:
                    Sp, lie_p = Spp, f_pp
                    found = True
                    break
            if found:
                break
        # ---- acceptance ----
        if lie_p > best_lie:
            S, best_lie = Sp, lie_p
            if not cyclic:
                break                    # sequential: exit on improvement
            m += 1
        else:
            m += 1
    return S


# --------------------------------------------------------------------------- #
#  Population management (Alg. 5)
# --------------------------------------------------------------------------- #
def _score_guided_sample(st: LIEState, k: int, rng) -> list[int]:
    w = st.score + 1e-9
    w = w / w.sum()
    return rng.choice(st.n, size=k, replace=False, p=w).tolist()


def _population_management(st: LIEState, P, fits, gbest, trials, rng, k,
                           r: float, mmax: int, cn: int) -> tuple:
    order = np.argsort(-np.asarray(fits))
    n_keep = max(1, int(np.ceil(r * len(P))))
    keep_idx = order[:n_keep]
    P_new = [list(P[i]) for i in keep_idx]
    fits_new = [fits[i] for i in keep_idx]
    trials_new = [0] * n_keep
    for _ in range(len(P) - n_keep):
        S = _score_guided_sample(st, k, rng)
        P_new.append(S)
        fits_new.append(st.fitness(S))
        trials_new.append(0)
    P = P_new
    fits = fits_new
    trials = trials_new
    i_best = int(np.argmax(fits))
    S_refined = _vns(st, P[i_best], k, rng, mmax=mmax, cn=cn, r=r, cyclic=True)
    f_refined = st.fitness(S_refined)
    if f_refined > fits[i_best]:
        P[i_best] = S_refined
        fits[i_best] = f_refined
    f_gbest = st.fitness(gbest)
    if fits[i_best] > f_gbest:
        gbest = list(P[i_best])
    return P, fits, gbest, trials


# --------------------------------------------------------------------------- #
#  Main search
# --------------------------------------------------------------------------- #
def abcnvs_search(st: LIEState, k: int, seed: int = 0, verbose: bool = True,
                  params: dict | None = None,
                  log=print) -> tuple[list[int], dict]:
    """Run ABCVNS; returns (best seed set by LIE, stats dict)."""
    p_ = dict(ABCVNS_PARAMS)
    if params:
        p_.update(params)
    gmax, NP = p_["gmax"], p_["NP"]
    ts, r, cn = p_["ts"], p_["r"], p_["cn"]
    L, tau_max, mmax = p_["L"], p_["tau_max"], p_["mmax"]

    rng = np.random.default_rng(seed)
    n = st.n
    t0 = time.perf_counter()

    # ---- population initialisation (Sec. 3.3.1) ----
    topk = np.argsort(-st.deg)[:k].tolist()
    P: list[list[int]] = []
    for i in range(NP):
        if i < NP // 2:
            S = list(topk)
            for pos in range(k):
                if rng.random() < 0.5:
                    v = int(rng.integers(n))
                    while v in S:
                        v = int(rng.integers(n))
                    S[pos] = v
        else:
            S = rng.choice(n, size=k, replace=False).tolist()
        P.append(S)
    fits = [st.fitness(S) for S in P]
    trials = [0] * NP
    i_gb = int(np.argmax(fits))
    gbest = list(P[i_gb])
    f_gb = fits[i_gb]
    tau = 0
    trace = [f_gb]

    for g in range(1, gmax + 1):
        f_before = f_gb
        # ---- employed bees (Alg. 1) ----
        for i in range(NP):
            cand = _egcnc_candidate(st, P[i], P, fits, i, gbest, rng, k)
            f_c = st.fitness(cand)
            if f_c > fits[i]:
                P[i], fits[i] = cand, f_c
                trials[i] = 0
                if f_c > f_gb:
                    gbest, f_gb = list(cand), f_c
            else:
                trials[i] += 1
        # ---- onlooker bees (Alg. 2) ----
        for _ in range(NP):
            tour = rng.choice(NP, size=ts, replace=False)
            tgt = int(tour[int(np.argmax(np.asarray(fits)[tour]))])
            cand = _egcnc_candidate(st, P[tgt], P, fits, tgt, gbest, rng, k)
            f_c = st.fitness(cand)
            if f_c > fits[tgt]:
                P[tgt], fits[tgt] = cand, f_c
                trials[tgt] = 0
                if f_c > f_gb:
                    gbest, f_gb = list(cand), f_c
            else:
                trials[tgt] += 1
        # ---- scout bees (Alg. 3) ----
        for i in range(NP):
            if trials[i] >= L:
                if rng.random() < 0.5:
                    inter = [v for v in P[i] if v in set(gbest)]
                    fill = rng.choice([v for v in range(n) if v not in set(inter)],
                                      size=k - len(inter), replace=False).tolist()
                    S_new = inter + fill
                else:
                    S_new = rng.choice(n, size=k, replace=False).tolist()
                P[i] = S_new
                fits[i] = st.fitness(S_new)
                trials[i] = 0
                if fits[i] > f_gb:
                    gbest, f_gb = list(S_new), fits[i]
        # ---- VNS on Gbest (Alg. 4, sequential) ----
        S_v = _vns(st, gbest, k, rng, mmax=mmax, cn=cn, r=r, cyclic=False)
        f_v = st.fitness(S_v)
        if f_v > f_gb:
            gbest, f_gb = S_v, f_v
        # ---- stagnation-aware population management (Alg. 5) ----
        if f_gb > f_before:
            tau = 0
        else:
            tau += 1
            if tau >= tau_max:
                P, fits, gbest, trials = _population_management(
                    st, P, fits, gbest, trials, rng, k, r=r, mmax=mmax, cn=cn)
                f_gb = st.fitness(gbest)
                tau = 0
        trace.append(f_gb)
        if verbose and (g % 25 == 0 or g == gmax):
            log(f"    [abcnvs] g={g:3d}  fit(gbest)={f_gb:12.3f}  "
                f"tau={tau}  t={time.perf_counter()-t0:7.1f}s")

    stats = {"LIE_best": f_gb, "fitness_mode": st.mode, "iterations": gmax,
             "wall_s": time.perf_counter() - t0, "trace": trace, "seed": seed,
             "params": dict(p_)}
    return gbest, stats


if __name__ == "__main__":
    # quick self-test on a small graph
    import networkx as nx
    from hybrid_engine import get_metrics
    G = nx.watts_strogatz_graph(300, 6, 0.1, seed=1)
    import numpy as np
    metrics = {
        "pr": np.ones(G.number_of_nodes()),
        "bc": np.ones(G.number_of_nodes()),
        "comm": np.arange(G.number_of_nodes()),
    }
    st = make_abcnvs_context(G, 0.05, metrics)
    S, stats = abcnvs_search(st, 10, seed=1, verbose=True)
    print("best:", S[:10], "LIE:", round(stats["LIE_best"], 3))
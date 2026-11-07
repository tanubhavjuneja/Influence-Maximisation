"""
abcvns_paper.py
===============
Self-contained port of the ABCVNS algorithm from ABC.pdf
(Tang, Dong, Wu, Chen, Wei; Swarm and Evolutionary Computation 105 (2026)
102404), re-written with numpy ONLY (+ optional cupy backend for the matrix
operations).  The algorithm is UNCHANGED -- same LIE fitness, EGCNC
neighbourhoods, three-tier VNS, scouts, population management and the
paper's parameter set (gmax=150, NP=50, ts=3, r=0.2, cn=10, L=5, tau_max=5,
mmax=3, alpha=0.5), using the IC configuration (p = 0.05 Bitcoin,
p = 0.01 Slashdot).  30 restarts, as the paper trains.

Port notes (documented deviations of the port itself):
  * LIE and the search use the weighted adjacency (|rating|), consistent
    with the IC oracle used everywhere in this project.
  * betweenness is exact (Brandes, metrics.py) for n < 20,000 and sampled
    (384 pivots) for larger graphs.
  * communities come from a hand-written Louvain local-moving phase
    (metrics.py) instead of python-louvain.
"""
from __future__ import annotations

import time

import numpy as np

import backend as be

ABCVNS_PARAMS = dict(gmax=150, NP=50, ts=3, r=0.2, cn=10, L=5, tau_max=5,
                     mmax=3, alpha=0.5)


def _xp(x):
    """numpy/device-agnostic conversion to a float32 device array."""
    if isinstance(x, np.ndarray):
        return be.asarray(x, be.f32)
    return be.astype(x, be.f32)                # already on device


def _np(x):
    return be.asnumpy(x)


class LIEState:
    """Precomputed network data shared by every fitness evaluation."""

    def __init__(self, G: dict, p: float, pr, bc, comm, deg, score,
                 mode: str = "lie", delta: float = 0.15):
        self.n = G["n"]
        self.p = float(p)
        self.pr = np.asarray(pr, dtype=np.float64)
        self.bc = np.asarray(bc, dtype=np.float64)
        self.comm = np.asarray(comm, dtype=np.int64)
        self.deg = np.asarray(deg, dtype=np.float64)
        self.score = np.asarray(score, dtype=np.float64)
        # fitness mode: "lie" = the paper's IC surrogate (unchanged);
        # "mr"  = the signed majority-rule surrogate (the fix)
        self.mode = mode
        self.delta = float(delta)
        self.lo = 0.5 - self.delta
        self.width = 2.0 * self.delta

        # weighted directed pairs for scatter matvecs (on device)
        src = np.concatenate([G["u"], G["v"]]).astype(np.int64)
        dst = np.concatenate([G["v"], G["u"]]).astype(np.int64)
        w = np.concatenate([G["w"], G["w"]]).astype(np.float32)
        sw = np.concatenate([(G["s"] * G["w"]).astype(np.float32),
                             (G["s"] * G["w"]).astype(np.float32)])
        self.src_dev = be.asarray(src, be.i64)
        self.dst_dev = be.asarray(dst, be.i64)
        self.w_dev = be.asarray(w, be.f32)
        self.sw_dev = be.asarray(sw, be.f32)

        # CPU adjacency for neighbour lookups (VNS local search)
        order = np.argsort(dst, kind="stable")
        self.adj_indices = src[order]
        self.adj_indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(self.adj_indptr, dst[order] + 1, 1)
        np.cumsum(self.adj_indptr, out=self.adj_indptr)

        # community -> total PR (N3 tier)
        self.comm_pr = np.zeros(int(self.comm.max()) + 1, dtype=np.float64)
        np.add.at(self.comm_pr, self.comm, self.pr)

    # -------------------------------------------------------------- #
    def dot(self, x):
        x = _xp(x)
        y = be.zeros(self.n, dtype=be.f32)
        be.add_at_flat(y, self.dst_dev, self.w_dev * x[self.src_dev])
        return y

    def dot_signed(self, x):
        """Signed scatter: y[dst] += sign*w * x[src] (net support)."""
        x = _xp(x)
        y = be.zeros(self.n, dtype=be.f32)
        be.add_at_flat(y, self.dst_dev, self.sw_dev * x[self.src_dev])
        return y

    def fitness(self, S) -> float:
        return _mr_fitness(self, S) if self.mode == "mr" else lie_fitness(self, S)


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def make_context(G: dict, p: float, metrics: dict, alpha: float = 0.5,
                 mode: str = "lie", delta: float = 0.15) -> LIEState:
    """Build the fitness state from cached structural metrics.

    mode="lie": the paper's IC LIE (unchanged).
    mode="mr" : the signed majority-rule surrogate (the fix).
    """
    deg = np.bincount(np.concatenate([G["u"], G["v"]]),
                      weights=np.concatenate([G["w"], G["w"]]),
                      minlength=G["n"]).astype(np.float64)
    score = alpha * _norm(deg) + (1.0 - alpha) * _norm(metrics["pr"])
    return LIEState(G, p, metrics["pr"], metrics["bc"], metrics["comm"],
                    deg, score, mode=mode, delta=delta)


def lie_fitness(st: LIEState, S) -> float:
    r"""
    Eq. (2) of the paper (IC configuration):

      LIE(S) = k + ( 1 + (1/|N1(S)\S|) * sum_{u in N2(S)\S} p*d*_u )
                   * sum_{i in N1(S)\S} ( 1 - prod_{j in S,(i,j) in E}(1-p) )

    with constant p the product is (1-(1-p)^{c_i}); d*_u = weighted edges
    from u into N1(S) u N2(S).  N2 excludes S and N1.
    """
    n, p = st.n, st.p
    k = len(set(S))
    s = np.zeros(n, dtype=np.float32)
    s[np.asarray(list(set(S)), dtype=np.int64)] = 1.0
    s_dev = _xp(s)
    cnt = st.dot(s_dev)
    n1 = (cnt > 0) & (s_dev == 0)
    if not bool(n1.any()):
        return float(k)
    sig1 = float(((1.0 - (1.0 - p) ** cnt[n1]).sum()))
    n2 = (st.dot(n1) > 0) & (s_dev == 0) & (~n1)
    if not bool(n2.any()):
        return float(k + sig1)
    d_u = st.dot(n1 | n2)
    sum2 = float(d_u[n2].sum())
    return float(k + (1.0 + p * sum2 / max(1, int(n1.sum()))) * sig1)


def _mr_fitness(st: LIEState, S) -> float:
    r"""
    Two-hop expected-activation surrogate of the SIGNED majority rule
    (the "fixed" fitness).

    The signed model activates v iff its net support strictly exceeds
    theta_v * deg_v, theta_v ~ U[lo, hi] with lo = 0.5 - delta.  Given an
    expected NET active mass c, the activation probability is

        P_v(c) = clamp( (c/deg_v - lo) / width, 0, 1 ),  width = 2*delta.

    Score(S) = |S|
             + sum_{i in N1(S)\S}  P_i( net mass from S )
             + sum_{j in N2(S)\S}  P_j( net mass from S + expected N1 ).

    Net mass uses the SIGNED scatter (positive edges push, negative edges
    pull), which is exactly the quantity the signed model thresholds.
    """
    n = st.n
    k = len(set(S))
    s = np.zeros(n, dtype=np.float32)
    s[np.asarray(list(set(S)), dtype=np.int64)] = 1.0
    s_dev = _xp(s)
    deg = be.asarray(st.deg.astype(np.float32), be.f32)
    safe = be.maximum(deg, 1e-12)

    cnt1 = st.dot_signed(s_dev)                    # net mass from seeds only
    p1 = be.clip((cnt1 / safe - st.lo) / st.width, 0.0, 1.0)
    p1[s_dev == 1] = 0.0
    n1 = cnt1 != 0
    n1[s_dev == 1] = False

    e = s_dev + p1                                 # expected activation prob.
    score = float(k) + float(p1.sum())

    cnt2 = st.dot_signed(e)                        # expected 2-hop net mass
    p2 = be.clip((cnt2 / safe - st.lo) / st.width, 0.0, 1.0)
    p2[s_dev == 1] = 0.0
    p2[n1] = 0.0
    return score + float(p2.sum())


# --------------------------------------------------------------------------- #
#  EGCNC candidate generation (Eqs. 5-7)
# --------------------------------------------------------------------------- #
def _egcnc_candidate(st, S_i, P, fits, i, gbest, rng, k):
    fit_i = fits[i]
    elite = [P[j] for j in range(len(P)) if j != i and fits[j] > fit_i]
    if elite:
        pool = set(S_i)
        for Sj in elite:
            pool |= set(Sj)
    else:
        pool = set(S_i) | set(gbest)
    pool = list(pool)
    if len(pool) < k:
        rest = [v for v in range(st.n) if v not in set(pool)]
        rng.shuffle(rest)
        pool += rest[: k - len(pool)]
    return rng.choice(pool, size=k, replace=False).tolist()


# --------------------------------------------------------------------------- #
#  VNS neighbourhoods (Sec. 3.4.2) + VNS procedure (Alg. 4)
# --------------------------------------------------------------------------- #
def _neighbors_mask(st, S, hops):
    s = np.zeros(st.n, dtype=np.float32)
    s[np.asarray(S, dtype=np.int64)] = 1.0
    cur = _xp(s)
    mask = be.zeros(st.n, dtype=be.bool_)
    for _ in range(hops):
        cur = st.dot(cur) > 0
        mask |= cur
    mask = _np(mask)
    mask[np.asarray(S, dtype=np.int64)] = False
    return mask


def _shake_n1(st, S, rng, k):
    S = list(S)
    Ss = np.asarray(S, dtype=np.int64)
    out = int(Ss[np.argmin(st.score[Ss])])
    cand = np.flatnonzero(_neighbors_mask(st, S, 1))
    if cand.size == 0:
        return S
    inn = int(cand[np.argmax(st.score[cand])])
    return [inn if v == out else v for v in S]


def _shake_n2(st, S, rng, k):
    S = list(S)
    out = S[int(rng.integers(k))]
    cand = np.flatnonzero(_neighbors_mask(st, S, 2))
    if cand.size == 0:
        return S
    inn = int(cand[np.argmax(st.bc[cand])])
    return [inn if v == out else v for v in S]


def _shake_n3(st, S, rng, k, r):
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


def _vns(st, S0, k, rng, mmax=3, cn=10, r=0.2, cyclic=False):
    S = list(S0)
    best_fit = st.fitness(S)
    m = 1
    while m <= mmax:
        if m == 1:
            Sp = _shake_n1(st, S, rng, k)
        elif m == 2:
            Sp = _shake_n2(st, S, rng, k)
        else:
            Sp = _shake_n3(st, S, rng, k, r)
        Sp = list(dict.fromkeys(Sp))[:k]
        fit_p = st.fitness(Sp)
        # local search (Eq. 19): first improvement over top-cn neighbours
        found = False
        order = np.argsort(-st.score[np.asarray(Sp, dtype=np.int64)])
        for pos_i in order:
            v_i = int(Sp[pos_i])
            nbr = st.adj_indices[st.adj_indptr[v_i]: st.adj_indptr[v_i + 1]]
            cand = [int(u) for u in nbr if u not in Sp]
            if not cand:
                continue
            sc = st.score[np.asarray(cand, dtype=np.int64)]
            top = [cand[j] for j in np.argsort(-sc)[:cn]]
            for u in top:
                Spp = list(Sp)
                Spp[pos_i] = u
                f_pp = st.fitness(Spp)
                if f_pp > fit_p:
                    Sp, fit_p = Spp, f_pp
                    found = True
                    break
            if found:
                break
        if fit_p > best_fit:
            S, best_fit = Sp, fit_p
            if not cyclic:
                break
            m += 1
        else:
            m += 1
    return S


# --------------------------------------------------------------------------- #
#  Population management (Alg. 5)
# --------------------------------------------------------------------------- #
def _score_guided_sample(st, k, rng):
    w = st.score + 1e-9
    w = w / w.sum()
    return rng.choice(st.n, size=k, replace=False, p=w).tolist()


def _population_management(st, P, fits, gbest, trials, rng, k, r, mmax, cn):
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
    P, fits, trials = P_new, fits_new, trials_new
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
def abcnvs_search(st, k, seed=0, verbose=True, params=None,
                  log=print) -> tuple[list[int], dict]:
    p_ = dict(ABCVNS_PARAMS)
    if params:
        p_.update(params)
    gmax, NP = p_["gmax"], p_["NP"]
    ts, r, cn = p_["ts"], p_["r"], p_["cn"]
    L, tau_max, mmax = p_["L"], p_["tau_max"], p_["mmax"]

    rng = np.random.default_rng(seed)
    n = st.n
    t0 = time.perf_counter()

    # ---- population init (Sec. 3.3.1) ----
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
        # employed bees (Alg. 1)
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
        # onlooker bees (Alg. 2)
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
        # scout bees (Alg. 3)
        for i in range(NP):
            if trials[i] >= L:
                if rng.random() < 0.5:
                    inter = [v for v in P[i] if v in set(gbest)]
                    fill = rng.choice(
                        [v for v in range(n) if v not in set(inter)],
                        size=k - len(inter), replace=False).tolist()
                    S_new = inter + fill
                else:
                    S_new = rng.choice(n, size=k, replace=False).tolist()
                P[i] = S_new
                fits[i] = st.fitness(S_new)
                trials[i] = 0
                if fits[i] > f_gb:
                    gbest, f_gb = list(S_new), fits[i]
        # VNS on Gbest (Alg. 4, sequential)
        S_v = _vns(st, gbest, k, rng, mmax=mmax, cn=cn, r=r, cyclic=False)
        f_v = st.fitness(S_v)
        if f_v > f_gb:
            gbest, f_gb = S_v, f_v
        # stagnation-aware population management (Alg. 5)
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
            log(f"    [abcvns:{st.mode}] g={g:3d}  fit(gbest)={f_gb:12.3f}  "
                f"tau={tau}  t={time.perf_counter() - t0:7.1f}s")

    stats = {"LIE_best": f_gb, "iterations": gmax,
             "wall_s": time.perf_counter() - t0,
             "trace": trace, "seed": seed, "params": dict(p_)}
    return gbest, stats
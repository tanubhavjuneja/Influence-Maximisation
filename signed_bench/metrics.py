"""
metrics.py
==========
Structural metrics computed BY HAND (numpy only, no networkx):

  * pagerank      - power iteration on the weighted undirected graph,
  * betweenness   - Brandes' algorithm (exact, or sampled pivots),
  * communities   - Louvain local-moving phase (Blondel et al. 2008),
  * kcore         - iterative peeling (binary degree),
  * weighted_deg  - sum of incident |weight|,
  * signed_pool   - sign-aware candidate pool (port of the main project's
                    signed_candidate_pool).

Each metric runs once per graph; results are cached to
data/metrics_<name>.npz so restarts never recompute them.
"""
from __future__ import annotations

from collections import deque

import numpy as np

import backend as be

from common import HERE

DATA = HERE / "data"


# --------------------------------------------------------------------------- #
def adjacency(G: dict):
    """Undirected CSR over |w| weights: (indptr, indices, wts) sorted by row."""
    src = np.concatenate([G["u"], G["v"]]).astype(np.int64)
    dst = np.concatenate([G["v"], G["u"]]).astype(np.int64)
    wts = np.concatenate([G["w"], G["w"]]).astype(np.float64)
    order = np.argsort(dst, kind="stable")
    src, dst, wts = src[order], dst[order], wts[order]
    indptr = np.zeros(G["n"] + 1, dtype=np.int64)
    np.add.at(indptr, dst + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, src, wts


# --------------------------------------------------------------------------- #
def pagerank(G: dict, alpha: float = 0.85, tol: float = 1e-8,
             max_iter: int = 200) -> np.ndarray:
    """Power iteration: r = (1-a)v + a * P^T r,  P = D^{-1} A (undirected)."""
    n, m = G["n"], G["m"]
    indptr, src, wts = adjacency(G)
    deg = np.bincount(src, weights=wts, minlength=n)
    deg_safe = np.maximum(deg, 1e-12)
    v = np.full(n, 1.0 / n)
    r = v.copy()
    for _ in range(max_iter):
        y = np.zeros(n)
        np.add.at(y, np.repeat(np.arange(n), np.diff(indptr)),
                  wts * r[src] / deg_safe[src])
        r_new = (1.0 - alpha) * v + alpha * y
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


# --------------------------------------------------------------------------- #
def betweenness(G: dict, pivots: int | None = None, seed: int = 7) -> np.ndarray:
    """Brandes' betweenness centrality.  Exact when pivots=None, otherwise
    sampled over `pivots` random sources (scaled to the full sum)."""
    n = G["n"]
    indptr, src, wts = adjacency(G)
    adj = [src[indptr[i]:indptr[i + 1]] for i in range(n)]
    rng = np.random.default_rng(seed)
    sources = (range(n) if pivots is None else
               rng.choice(n, size=min(int(pivots), n), replace=False))
    bc = np.zeros(n)
    n_src = 0
    for s in sources:
        n_src += 1
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1, dtype=np.int32)
        dist[s] = 0
        pred: list[tuple[int, int]] = []          # (child, parent) pairs
        Q = deque([s])
        while Q:
            v = Q.popleft()
            dv = dist[v] + 1
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dv
                    Q.append(w)
                if dist[w] == dv:
                    sigma[w] += sigma[v]
                    pred.append((w, v))
        delta = np.zeros(n)
        for w, v in reversed(pred):
            c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
            delta[v] += c
            bc[v] += c
    if pivots is None:
        norm = (n - 1) * (n - 2)
        if norm > 0:
            bc /= norm
    else:
        bc = bc * (n / n_src)
    return bc


# --------------------------------------------------------------------------- #
def communities(G: dict, max_passes: int = 10, seed: int = 7) -> np.ndarray:
    """Louvain local-moving phase (weighted modularity) on the |w| graph."""
    n = G["n"]
    indptr, src, wts = adjacency(G)
    adj = [src[indptr[i]:indptr[i + 1]] for i in range(n)]
    adjw = [wts[indptr[i]:indptr[i + 1]] for i in range(n)]
    deg = np.bincount(src, weights=wts, minlength=n)
    two_m = 2.0 * float(deg.sum())
    rng = np.random.default_rng(seed)

    comm = np.arange(n, dtype=np.int64)
    tot = deg.copy()

    for _ in range(max_passes):
        moved = 0
        for v in rng.permutation(n):
            nb = adj[v]
            if len(nb) == 0:
                continue
            w = adjw[v]
            c0 = int(comm[v])
            uniq, inv = np.unique(comm[nb], return_inverse=True)
            kin = np.bincount(inv, weights=w, minlength=len(uniq))
            idx0 = np.flatnonzero(uniq == c0)
            kin_c0 = float(kin[idx0[0]]) if len(idx0) else 0.0
            tot_c0_no_v = tot[c0] - deg[v]
            removal = kin_c0 - deg[v] * tot_c0_no_v / two_m
            best_c, best_g = c0, 0.0
            for i, c in enumerate(uniq):
                if int(c) == c0:
                    continue
                gain = kin[i] - deg[v] * tot[c] / two_m - removal
                if gain > best_g + 1e-12:
                    best_c, best_g = int(c), gain
            if best_c != c0:
                comm[v] = best_c
                tot[c0] -= deg[v]
                tot[best_c] += deg[v]
                moved += 1
        if moved == 0:
            break

    # renumber to 0..C-1
    _, comm = np.unique(comm, return_inverse=True)
    return comm.astype(np.int64)


# --------------------------------------------------------------------------- #
def kcore(G: dict) -> np.ndarray:
    """k-core numbers by bucket peeling -- O(n + m) with Python lists."""
    n = G["n"]
    indptr, src, wts = adjacency(G)
    adj = [src[indptr[i]:indptr[i + 1]] for i in range(n)]
    deg = np.array([len(a) for a in adj], dtype=np.int64)
    core = np.zeros(n, dtype=np.int64)
    bins: list[list[int]] = [[] for _ in range(int(deg.max()) + 1)]
    for v in range(n):
        bins[int(deg[v])].append(v)
    for d in range(len(bins)):
        while bins[d]:
            v = bins[d].pop()
            if deg[v] != d:                     # stale entry (already lowered)
                continue
            core[v] = d
            for w in adj[v]:
                dw = int(deg[w])
                if dw > d:
                    deg[w] -= 1
                    bins[dw - 1].append(w)
    return core


# --------------------------------------------------------------------------- #
def signed_pool(model, pr: np.ndarray, core: np.ndarray, size: int,
                seed: int = 0) -> list[int]:
    """Sign-aware candidate pool (port of the main project's
    signed_candidate_pool): net positive support, PageRank, k-core,
    low-threshold ranks, plus a random tail."""
    n = model.n
    size = min(size, n)
    posw = np.asarray(model.posw)
    negw = np.asarray(model.negw)
    net = posw - negw
    net_rank = sorted(range(n), key=lambda i: -net[i])
    pr_rank = sorted(range(n), key=lambda i: -pr[i])
    core_rank = sorted(range(n), key=lambda i: (-core[i], -model.deg[i]))
    need_mean = be.asnumpy(model.need).mean(axis=1)
    easy_rank = sorted(range(n), key=lambda i: (need_mean[i], -net[i]))

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


# --------------------------------------------------------------------------- #
def get_metrics(G: dict, recompute: bool = False) -> dict:
    """All cached structural metrics for a graph."""
    import os
    DATA.mkdir(exist_ok=True)
    out = DATA / f"metrics_{G['name']}.npz"
    if out.exists() and not recompute:
        z = np.load(out)
        return {"pr": z["pr"], "bc": z["bc"], "comm": z["comm"],
                "core": z["core"]}
    print(f"  computing metrics for {G['name']} (pr, bc, comm, core) ...")
    pr = pagerank(G)
    pivots = None if G["n"] < 20_000 else 384
    bc = betweenness(G, pivots=pivots)
    comm = communities(G)
    core = kcore(G)
    np.savez_compressed(out, pr=pr, bc=bc, comm=comm, core=core)
    print(f"  metrics cached -> {out.name}")
    return {"pr": pr, "bc": bc, "comm": comm, "core": core}
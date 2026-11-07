"""
diffusion.py
============
Signed majority-rule diffusion, written from scratch on numpy arrays
(optionally accelerated by CuPy on the GPU -- see backend.py).

Model (identical to the main project's SignedMajorityRuleModel)
----------------------------------------------------------------
Node v activates iff its NET active support strictly exceeds theta_v * deg(v):

    sum_{u active, sign=+1} w(u,v) - sum_{u active, sign=-1} w(u,v)
                                                    > theta_v * deg(v)

deg(v) = total incident ABSOLUTE weight; theta_v ~ U[1/2-delta, 1/2+delta]
with delta = 0.15, i.e. U[0.35, 0.65].  Active nodes stay active forever.

Implementation
--------------
All R draws are simulated in parallel as an (n, R) float32 active matrix.
Each round adds the signed weight of the FRONTIER (newly activated) nodes
to the net-support matrix with one chunked scatter-add over the edges --
`xp.add.at(net, idx, vals)`.  Thresholds are drawn on the CPU (numpy) once
so results are bit-reproducible across CPU/GPU backends.
"""
from __future__ import annotations

import numpy as np

import backend as be

INF = np.float32(np.inf)


class SignedDiffusion:
    """Signed majority-rule oracle with the interface the solvers need:
    sigma(S), sigma_potential(S) -> (sigma, Phi, sem), components(S),
    plus an evaluation cache and counters."""

    def __init__(self, G: dict, R: int, seed: int,
                 delta: float = 0.15, chunk: int = 40_000):
        self.n = int(G["n"])
        self.R = int(R)
        self.delta = float(delta)
        self.chunk = int(chunk)

        # both directions of every edge, signed weight
        self.src = np.concatenate([G["u"], G["v"]]).astype(np.int64)
        self.dst = np.concatenate([G["v"], G["u"]]).astype(np.int64)
        sw = (G["s"] * G["w"]).astype(np.float32)
        self.sw = np.concatenate([sw, sw]).astype(np.float32)
        aw = np.abs(self.sw)
        self.deg = np.bincount(self.dst, weights=aw,
                               minlength=self.n).astype(np.float64)

        # positive / negative incident mass -> sign-aware quantities
        self.posw = np.bincount(self.dst, weights=np.where(self.sw > 0, aw, 0.0),
                                minlength=self.n).astype(np.float64)
        self.negw = np.bincount(self.dst, weights=np.where(self.sw < 0, aw, 0.0),
                                minlength=self.n).astype(np.float64)
        net = self.posw - self.negw
        # node polarity: +1 if net incident weight >= 0, else -1
        self.polarity = np.where(net >= 0, np.int8(1), np.int8(-1))
        self.pol_pos = np.flatnonzero(self.polarity >= 0)
        self.pol_neg = np.flatnonzero(self.polarity < 0)

        # sign-aware pool weight (net positive support, shifted > 0)
        pw = net - net.min() + 1.0
        self.pool_weight = np.maximum(pw, 1e-6)

        # ---- thresholds (drawn on CPU for reproducibility) ----
        rng = np.random.default_rng(seed)
        lo, hi = 0.5 - delta, 0.5 + delta
        theta = rng.uniform(lo, hi, size=(self.n, R)).astype(np.float32)
        need = np.floor(theta * self.deg[:, None].astype(np.float32)) + 1.0
        need[self.deg == 0] = INF
        self.theta_dev = be.asarray(theta, be.f32)
        self.need = be.asarray(need, be.f32)
        self._deg_dev = be.asarray(self.deg.astype(np.float32), be.f32)

        # device copies of the edge arrays (once)
        self.src_dev = be.asarray(self.src, be.i64)
        self.dst_dev = be.asarray(self.dst, be.i64)
        self.sw_dev = be.asarray(self.sw, be.f32)

        # evaluation bookkeeping
        self.n_evals = 0
        self.cache: dict[frozenset, float] = {}
        self.cache_pot: dict[frozenset, tuple] = {}

        # adjacency (CPU numpy) for neighbour lookups (candidate sets etc.)
        order = np.argsort(self.dst, kind="stable")
        self.adj_indices = self.src[order]
        self.adj_indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(self.adj_indptr, self.dst[order] + 1, 1)
        np.cumsum(self.adj_indptr, out=self.adj_indptr)

    # ------------------------------------------------------------------ #
    def neighbours(self, v: int) -> np.ndarray:
        i, j = self.adj_indptr[v], self.adj_indptr[v + 1]
        return self.adj_indices[i:j]

    def reset_counters(self) -> None:
        self.n_evals = 0

    def clear_cache(self) -> None:
        self.cache.clear()
        self.cache_pot.clear()

    # ------------------------------------------------------------------ #
    def _run(self, S) -> tuple:
        """Simulate all R draws in parallel; return (active, net) on device."""
        n, R, chunk = self.n, self.R, self.chunk
        active = be.zeros((n, R), dtype=be.bool_)
        net = be.zeros((n, R), dtype=be.f32)
        if len(S):
            active[be.asarray(np.asarray(S, dtype=np.int64), be.i64), :] = True
        frontier = active.clone() if hasattr(active, "clone") else active.copy()
        rr = be.arange(R, dtype=be.i64)
        src, dst, sw = self.src_dev, self.dst_dev, self.sw_dev
        rounds = 0
        while frontier.any():
            rounds += 1
            if rounds > 4 * n // max(1, len(S)) + 100:
                break                          # safety (diameter bound)
            e = be.any_axis(frontier[src], 1)  # edges out of the frontier
            if not e.any():
                break
            s_sel = src[e]
            d_sel = dst[e]
            w_sel = sw[e]
            fr = be.astype(frontier[s_sel], be.f32)
            net_flat = net.reshape(-1)
            for c0 in range(0, int(s_sel.shape[0]), chunk):
                s_ = s_sel[c0:c0 + chunk]
                d_ = d_sel[c0:c0 + chunk]
                w_ = w_sel[c0:c0 + chunk]
                f_ = fr[c0:c0 + chunk]
                idx = d_[:, None] * R + rr[None, :]
                be.add_at_flat(net_flat, idx.reshape(-1),
                               (w_[:, None] * f_).reshape(-1))
            newly = (net >= self.need) & (~active)
            active |= newly
            frontier = newly
        return active, net

    # ------------------------------------------------------------------ #
    def sigma(self, S) -> float:
        key = frozenset(S)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        active, _ = self._run(S)
        val = float(active.sum(dtype=be.f64) / self.R)
        self.cache[key] = val
        self.n_evals += 1
        return val

    def sigma_potential(self, S) -> tuple:
        """(sigma, Phi, sem): expected spread, plateau potential, std error."""
        key = frozenset(S)
        hit = self.cache_pot.get(key)
        if hit is not None:
            return hit
        active, net = self._run(S)
        counts = be.asnumpy(be.sum_axis(active, 0, be.f64))
        pot = be.minimum(1.0, net / self.need)
        phi = float(pot.sum(dtype=be.f64) / self.R)
        mu = float(counts.mean())
        se = float(counts.std(ddof=1) / np.sqrt(self.R)) if self.R > 1 else 0.0
        self.cache_pot[key] = (mu, phi, se)
        self.n_evals += 1
        return mu, phi, se

    def components(self, S):
        """Held-out decomposition by NODE POLARITY (net incident weight):
        (positive_spread, negative_spread, total_spread), expected values."""
        key = ("comp", frozenset(S))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        active, _ = self._run(S)
        pos = int(active[be.asarray(self.pol_pos, be.i64)].sum()) / self.R
        neg = int(active[be.asarray(self.pol_neg, be.i64)].sum()) / self.R
        tot = pos + neg
        self.cache[key] = (pos, neg, tot)
        self.n_evals += 1
        return pos, neg, tot

    def diagnostics(self, S, cap: int = 60):
        """Border nodes for cascade-aware refinement: inactive nodes whose
        mean slack (need - net) at convergence is <= 1."""
        active, net = self._run(S)
        active_np = be.asnumpy(active)
        net_np = be.asnumpy(net)
        need_np = be.asnumpy(self.need)
        inactive = active_np.mean(axis=1) < 0.999
        slack = (need_np - net_np).mean(axis=1)
        score = np.where(inactive, slack, np.inf)
        order = np.argsort(score)
        out = [int(v) for v in order[:cap] if np.isfinite(score[v])]
        return out
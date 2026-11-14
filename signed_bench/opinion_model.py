"""
opinion_model.py
================
CORRECTED signed majority rule: nodes carry an OPINION state, not just a
binary active flag.

    state s_v in {-1, 0, +1}   (-1 = negative-active, 0 = inactive, +1 = positive-active)

Influence of an active neighbour u on v:

    contribution = sign(edge_uv) * s_u * weight_uv
    support_v    = sum of contributions from all active neighbours

v activates when |support_v| > theta_v * deg(v), and takes the SIGN of the
support as its opinion.  Progressive: once active, a node never changes.

Seeds are planted with opinion +1 (we are maximising positive influence).

Reported per seed set (held-out draws):
    positive spread  = mean # nodes ending +1
    negative spread  = mean # nodes ending -1
    net spread       = positive - negative
    diversity        = normalised Shannon entropy of the final opinions
                       among active nodes (0 = one opinion only, 1 = 50/50)
    fairness         = 1 - (max - min) community activation rate, over
                       communities with at least MIN_COMM_SIZE nodes
                       (1 = every community reached equally, 0 = very uneven)

The class mirrors the interface of diffusion.SignedDiffusion so the same
ABC / ABCVNS code can drive it.
"""
from __future__ import annotations

import numpy as np

import backend as be

INF = np.float32(np.inf)
MIN_COMM_SIZE = 10


class SignedOpinionModel:
    def __init__(self, G: dict, R: int, seed: int, comm: np.ndarray,
                 delta: float = 0.15, chunk: int = 40_000):
        self.n = int(G["n"])
        self.R = int(R)
        self.delta = float(delta)
        self.chunk = int(chunk)

        self.src = np.concatenate([G["u"], G["v"]]).astype(np.int64)
        self.dst = np.concatenate([G["v"], G["u"]]).astype(np.int64)
        sw = (G["s"] * G["w"]).astype(np.float32)
        self.sw = np.concatenate([sw, sw]).astype(np.float32)
        aw = np.abs(self.sw)
        self.deg = np.bincount(self.dst, weights=aw,
                               minlength=self.n).astype(np.float64)
        self.posw = np.bincount(self.dst,
                                weights=np.where(self.sw > 0, aw, 0.0),
                                minlength=self.n).astype(np.float64)
        self.negw = np.bincount(self.dst,
                                weights=np.where(self.sw < 0, aw, 0.0),
                                minlength=self.n).astype(np.float64)
        self.pool_weight = np.maximum(self.deg, 1e-6)

        # communities for the fairness metric (ignore tiny ones)
        comm = np.asarray(comm, dtype=np.int64)
        ids, counts = np.unique(comm, return_counts=True)
        self.comm_masks = [np.flatnonzero(comm == c)
                           for c, cnt in zip(ids, counts)
                           if cnt >= MIN_COMM_SIZE]
        self.comm_sizes = np.array([len(m) for m in self.comm_masks],
                                   dtype=np.float64)

        # thresholds (CPU for reproducibility)
        rng = np.random.default_rng(seed)
        lo, hi = 0.5 - delta, 0.5 + delta
        theta = rng.uniform(lo, hi, size=(self.n, R)).astype(np.float32)
        need = np.floor(theta * self.deg[:, None].astype(np.float32)) + 1.0
        need[self.deg == 0] = INF
        self.need_np = need
        self.need = be.asarray(need, be.f32)

        self.src_dev = be.asarray(self.src, be.i64)
        self.dst_dev = be.asarray(self.dst, be.i64)
        self.sw_dev = be.asarray(self.sw, be.f32)

        self.n_evals = 0
        self.cache: dict = {}
        self.cache_pot: dict = {}
        self._gpu_ok = be.GPU

        order = np.argsort(self.dst, kind="stable")
        self.adj_indices = self.src[order]
        self.adj_indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(self.adj_indptr, self.dst[order] + 1, 1)
        np.cumsum(self.adj_indptr, out=self.adj_indptr)

    # ------------------------------------------------------------------ #
    def neighbours(self, v: int) -> np.ndarray:
        i, j = self.adj_indptr[v], self.adj_indptr[v + 1]
        return self.adj_indices[i:j]

    def reset_counters(self):
        self.n_evals = 0

    def clear_cache(self):
        self.cache.clear()
        self.cache_pot.clear()

    # ------------------------------------------------------------------ #
    def _run(self, S):
        """Return (states (n,R) int8, support (n,R) float32) as numpy."""
        if be.GPU and self._gpu_ok:
            try:
                return self._run_dev(S)
            except Exception as exc:
                msg = str(exc)
                if ("CUDA" in msg or "illegal instruction" in msg
                        or "Accelerator" in msg):
                    self._gpu_ok = False
                    print(f"  [opinion] CUDA fault -> CPU fallback "
                          f"({msg.splitlines()[0][:80]})", flush=True)
                    return self._run_cpu(S)
                raise
        return self._run_cpu(S)

    def _run_dev(self, S):
        n, R, chunk = self.n, self.R, self.chunk
        active = be.zeros((n, R), dtype=be.i8)
        if len(S):
            active[be.asarray(np.asarray(S, dtype=np.int64), be.i64), :] = 1
        support = be.zeros((n, R), dtype=be.f32)
        frontier = active.clone() if hasattr(active, "clone") else active.copy()
        rr = be.arange(R, dtype=be.i64)
        src, dst, sw = self.src_dev, self.dst_dev, self.sw_dev
        rounds = 0
        while frontier.any():
            rounds += 1
            if rounds > 200:
                break
            e = be.any_axis(frontier[src] != 0, 1)
            if not e.any():
                break
            s_sel, d_sel, w_sel = src[e], dst[e], sw[e]
            fr = be.astype(frontier[s_sel], be.f32)
            sup_flat = support.reshape(-1)
            for c0 in range(0, int(s_sel.shape[0]), chunk):
                d_ = d_sel[c0:c0 + chunk]
                w_ = w_sel[c0:c0 + chunk]
                f_ = fr[c0:c0 + chunk]
                idx = d_[:, None] * R + rr[None, :]
                be.add_at_flat(sup_flat, idx.reshape(-1),
                               (w_[:, None] * f_).reshape(-1))
            newly = (be.abs(support) >= self.need) & (active == 0)
            if not newly.any():
                break
            sgn = be.astype(support > 0, be.i8) - be.astype(support < 0, be.i8)
            active = be.where(newly, sgn, active)
            frontier = newly
        return be.asnumpy(active), be.asnumpy(support)

    def _run_cpu(self, S):
        n, R, chunk = self.n, self.R, self.chunk
        active = np.zeros((n, R), dtype=np.int8)
        if len(S):
            active[np.asarray(S, dtype=np.int64), :] = 1
        support = np.zeros((n, R), dtype=np.float32)
        frontier = active.copy()
        rr = np.arange(R, dtype=np.int64)
        src, dst, sw = self.src, self.dst, self.sw
        rounds = 0
        while frontier.any():
            rounds += 1
            if rounds > 200:
                break
            e = frontier[src].any(axis=1)
            if not e.any():
                break
            s_sel, d_sel, w_sel = src[e], dst[e], sw[e]
            fr = frontier[s_sel].astype(np.float32)
            for c0 in range(0, int(s_sel.shape[0]), chunk):
                d_ = d_sel[c0:c0 + chunk]
                w_ = w_sel[c0:c0 + chunk]
                f_ = fr[c0:c0 + chunk]
                idx = d_[:, None] * R + rr[None, :]
                np.add.at(support.ravel(), idx.ravel(),
                          (w_[:, None] * f_).ravel())
            newly = (np.abs(support) >= self.need_np) & (active == 0)
            if not newly.any():
                break
            sgn = (support > 0).astype(np.int8) - (support < 0).astype(np.int8)
            active = np.where(newly, sgn, active)
            frontier = newly
        return active, support

    # ------------------------------------------------------------------ #
    def sigma_pos(self, S) -> float:
        key = frozenset(S)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        active, _ = self._run(S)
        val = float((active == 1).sum() / self.R)
        self.cache[key] = val
        self.n_evals += 1
        return val

    # the ABC solver calls model.sigma(...) — optimise POSITIVE spread
    def sigma(self, S) -> float:
        return self.sigma_pos(S)

    def sigma_potential(self, S):
        """(positive spread, Phi, sem) where Phi = mean sum of positive
        activation potential min(1, support/need)."""
        key = frozenset(S)
        hit = self.cache_pot.get(key)
        if hit is not None:
            return hit
        active, support = self._run(S)
        pos_per_draw = (active == 1).sum(axis=0).astype(np.float64)
        pot = np.clip(support / self.need_np, 0.0, 1.0)
        phi = float(pot.sum(axis=0).mean())
        mu = float(pos_per_draw.mean())
        se = float(pos_per_draw.std(ddof=1) / np.sqrt(self.R)) if self.R > 1 else 0.0
        self.cache_pot[key] = (mu, phi, se)
        self.n_evals += 1
        return mu, phi, se

    def components(self, S):
        """(pos, neg, net, diversity, fairness) held-out, means over draws."""
        key = ("comp", frozenset(S))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        active, _ = self._run(S)
        pos_d = (active == 1).sum(axis=0).astype(np.float64)
        neg_d = (active == -1).sum(axis=0).astype(np.float64)
        pos, neg = float(pos_d.mean()), float(neg_d.mean())
        net = pos - neg

        # opinion entropy among active nodes, per draw
        tot = pos_d + neg_d
        p = np.divide(pos_d, tot, out=np.zeros_like(pos_d), where=tot > 0)
        ent = np.zeros_like(p)
        mask = (p > 0) & (p < 1)
        ent[mask] = -(p[mask] * np.log2(p[mask])
                      + (1 - p[mask]) * np.log2(1 - p[mask]))
        diversity = float(ent.mean())

        # community fairness: 1 - (max rate - min rate), per draw
        if self.comm_masks:
            rates = np.stack([
                (active[m] != 0).sum(axis=0) / float(len(m))
                for m in self.comm_masks
            ], axis=0)                       # (C, R)
            fairness = float((1.0 - (rates.max(axis=0) - rates.min(axis=0))).mean())
        else:
            fairness = 0.0

        out = (pos, neg, net, diversity, fairness)
        self.cache[key] = out
        self.n_evals += 1
        return out

    def diagnostics(self, S, cap: int = 60):
        active, support = self._run(S)
        inactive = (active == 0).mean(axis=1) > 0.001
        slack = (self.need_np - np.abs(support)).mean(axis=1)
        score = np.where(inactive, slack, np.inf)
        order = np.argsort(score)
        return [int(v) for v in order[:cap] if np.isfinite(score[v])]
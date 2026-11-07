r"""
ic_model.py
===========
Independent Cascade (IC) diffusion model for the cross-model comparison
against the ABCVNS paper (Tang et al., Swarm and Evolutionary Computation
105 (2026) 102404, "Artificial bee colony meets variable neighborhood
search: Fresh insights into influence maximization in social networks").

The paper's protocol
--------------------
* diffusion: IC, constant edge probability p (p = 0.05 for small networks,
  p = 0.01 for the large ones: Soc-slashdot, Soc-delicious),
* final spread: 1000 Monte-Carlo simulations, averaged over 30 runs,
* search fitness: LIE (two-hop local influence estimation), no MC inside
  the search loop (see abcvns.py).

This class
----------
* sigma(S): mean IC spread over R live-edge draws, EXACT and vectorised:
  in round t, a node with cnt newly-active neighbours activates with
  probability 1 - (1-p)^cnt (edge independence on a simple graph), so one
  sparse matmul + one random tensor per round simulates the process.
* GPU path (torch CUDA): the R draws are stacked into one (n x R) batch
  per round -- same trick as the majority model, so IC also runs on the
  GPU.  Common random numbers: the per-round random tensors are drawn from
  a generator seeded by (model seed + round), so every candidate set sees
  the IDENTICAL draws (paired comparison, exactly reproducible).
* CPU path (fallback, and for the held-out parallel scoring): the same
  algorithm per draw with numpy/scipy; a ProcessPoolExecutor splits draws
  across CPU workers when `parallel=True` -- "CPU workers generate the
  live-edge subgraphs and run the IC numbers".
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp


def _ic_draw_worker(args):
    """One live-edge draw of IC, exact and vectorised (CPU path)."""
    seed, S_idx, indptr, indices, p = args
    n = len(indptr) - 1
    A = sp.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(n, n),
    )
    rng = np.random.default_rng(seed)
    active = np.zeros(n, dtype=bool)
    S = np.asarray(S_idx, dtype=np.int64)
    if len(S):
        active[S] = True
    new = active.copy()
    q = 1.0 - float(p)
    while True:
        cnt = A.dot(new.astype(np.float32))          # newly-active neighbours
        cand = (cnt > 0) & (~active)
        if not cand.any():
            break
        thr = 1.0 - q ** cnt[cand]
        fire = rng.random(int(cand.sum())) < thr
        newly = np.flatnonzero(cand)[fire]
        active[newly] = True
        new = np.zeros(n, dtype=bool)
        new[newly] = True
    return int(active.sum())


def _ic_draws_chunk(args):
    """Simulate a contiguous chunk of draws serially (used by the CPU pool)."""
    seed0, start, count, S_idx, indptr, indices, p = args
    return [_ic_draw_worker((seed0 + d, S_idx, indptr, indices, p))
            for d in range(start, start + count)]


class IndependentCascadeModel:
    """Monte-Carlo IC oracle with the same interface the ABC/SA/CELF/Greedy
    solvers expect (sigma / sigma_potential / n_evals / reset_counters)."""

    def __init__(self, graph: nx.Graph, p: float = 0.05, n_samples: int = 150,
                 seed: int = 0, dtype=np.float32, backend: str = "auto",
                 chunk: int = 256):
        self.p = float(p)
        self.q = 1.0 - float(p)
        self.R = int(n_samples)
        self.dtype = dtype
        self.seed = seed
        self.chunk = int(chunk)
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.index = {u: i for i, u in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.A = nx.to_scipy_sparse_array(
            graph, nodelist=self.nodes, format="csr", dtype=np.float32
        )
        self.deg = np.asarray(self.A.sum(axis=1)).ravel()
        self._indptr = self.A.indptr
        self._indices = self.A.indices
        self.n_evals = 0
        self._cache: dict[frozenset, float] = {}
        self._cache_pot: dict[frozenset, tuple] = {}
        self.backend = self._resolve_backend(backend)
        self._dev = None
        if self.backend == "torch_cuda":
            self._init_torch()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "auto":
            try:
                import torch
                return "torch_cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        if backend not in ("cpu", "torch_cuda"):
            raise ValueError(f"unknown backend '{backend}'")
        return backend

    def _init_torch(self) -> None:
        import torch
        self._torch = torch
        coo = self.A.tocoo()
        idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        vals = torch.from_numpy(coo.data.astype(np.float32))
        self._dev = {
            "A": torch.sparse_coo_tensor(idx, vals, size=(self.n, self.n),
                                         device="cuda").coalesce().to_sparse_csr(),
            "dev": "cuda",
            "gen": torch.Generator(device="cuda"),
        }

    # ---------------- simulation ---------------- #
    def _simulate(self, seeds, parallel: bool = False,
                  n_workers: int | None = None) -> np.ndarray:
        S_idx = sorted({int(x) for x in seeds})
        if self._dev is not None:
            return self._simulate_torch(S_idx)
        if not parallel or self.R <= 32:
            return np.asarray([_ic_draw_worker(
                (self.seed + d, S_idx, self._indptr, self._indices, self.p))
                for d in range(self.R)], dtype=np.float32)
        n_workers = n_workers or min(8, self.R)
        chunk = max(1, self.R // n_workers)
        jobs = [(self.seed, c, min(chunk, self.R - c), S_idx,
                 self._indptr, self._indices, self.p)
                for c in range(0, self.R, chunk)]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            out = list(ex.map(_ic_draws_chunk, jobs))
        vals = [v for chunk_vals in out for v in chunk_vals]
        return np.asarray(vals, dtype=np.float32)

    def _simulate_torch(self, S_idx) -> np.ndarray:
        """GPU: all R draws batched per round; chunked over the draws."""
        torch = self._torch
        dev = self._dev
        A = dev["A"]
        gen = dev["gen"]
        gen.manual_seed(self.seed)                # CRN: same draws every eval
        q, n, R = self.q, self.n, self.R
        out = []
        for c in range(0, R, self.chunk):
            B = min(self.chunk, R - c)
            active = torch.zeros((n, B), dtype=torch.float32, device="cuda")
            if S_idx:
                active[torch.from_numpy(np.asarray(S_idx, dtype=np.int64)),
                       :] = 1.0
            new = active.clone()
            while True:
                cnt = torch.sparse.mm(A, new)
                cand = (cnt > 0) & (active == 0)
                if not bool(cand.any()):
                    break
                r = torch.rand((n, B), device="cuda", generator=gen)
                # activation probability 1 - (1-p)^cnt, computed only on
                # candidate rows (sparse activation = cheap elementwise work)
                cc = cnt[cand]
                fire = r[cand] < (1.0 - q ** cc)
                newly = cand.clone()
                newly[cand] = fire
                active = active.masked_fill(newly, 1.0)
                new = newly.float()
            out.append(active.sum(dim=0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(R, dtype=np.float32)

    def sigma(self, seeds, use_cache: bool = True, parallel: bool = False,
              n_workers: int | None = None) -> float:
        key = frozenset(seeds)
        if use_cache and key in self._cache:
            return self._cache[key]
        self.n_evals += 1
        val = float(self._simulate(sorted(key), parallel=parallel,
                                   n_workers=n_workers).mean())
        if use_cache:
            self._cache[key] = val
        return val

    def sigma_potential(self, seeds, use_cache: bool = True):
        """(sigma, Phi, se); IC has no plateau potential -> Phi = 0."""
        key = frozenset(seeds)
        if use_cache and key in self._cache_pot:
            return self._cache_pot[key]
        self.n_evals += 1
        vals = self._simulate(sorted(key))
        out = (float(vals.mean()), 0.0,
               float(vals.std(ddof=1) / np.sqrt(self.R)) if self.R > 1 else 0.0)
        if use_cache:
            self._cache_pot[key] = out
        return out

    def sigma_std(self, seeds) -> float:
        vals = self._simulate(sorted(set(seeds)))
        return float(vals.std(ddof=1) / np.sqrt(self.R))

    def reset_counters(self) -> None:
        self.n_evals = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_pot.clear()

    def describe(self) -> str:
        return (f"IndependentCascadeModel(p={self.p}, R={self.R}, "
                f"n={self.n}, m={self.graph.number_of_edges()}, "
                f"backend={self.backend})")


def ic_heldout_model(model: IndependentCascadeModel, seed: int | None = None
                     ) -> IndependentCascadeModel:
    """Clone with FRESH, independent draws (held-out scoring)."""
    s = model.seed + 10_000 if seed is None else seed
    return IndependentCascadeModel(model.graph, p=model.p,
                                   n_samples=model.R, seed=s,
                                   chunk=model.chunk)


if __name__ == "__main__":
    import networkx as nx
    G = nx.watts_strogatz_graph(300, 6, 0.1, seed=1)
    m = IndependentCascadeModel(G, p=0.05, n_samples=200, seed=7)
    t0 = time.perf_counter()
    print("sigma:", m.sigma([0, 1, 2]), "time", time.perf_counter() - t0,
          "backend", m.backend)
    print("sigma_potential:", m.sigma_potential([0, 1, 2]))
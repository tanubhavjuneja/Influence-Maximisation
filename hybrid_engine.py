r"""
hybrid_engine.py
================
CPU + GPU parallel pipeline for the real-network runs.

Split of work (the "CPU generates subgraphs, GPU runs the numbers" design)
--------------------------------------------------------------------------
CPU workers (ProcessPoolExecutor)
  * dataset parsing / relabelling                 (real_datasets.py)
  * candidate-pool construction + heavy structural metrics: PageRank,
    k-core, approximate betweenness centrality, Louvain communities.
    These are cached to data/metrics_<name>.npz so a restart of the run
    never recomputes them.
  * Independent-Cascade live-edge draws           (ic_model.py)

GPU worker (torch CUDA, RTX 3060; falls back to scipy/numpy on CPU)
  * majority-rule cascade batches: one sparse matmul per diffusion round
    over R threshold draws.  `ChunkedMajorityRuleModel` processes the R
    draws in column-chunks so held-out scoring with R = 1000 fits in
    GPU memory on the big graphs.

CpuGpuPipeline
  * submits the CPU metric jobs FIRST (futures), then runs the GPU
    solvers in the main thread, and only blocks on the futures when a
    solver actually needs them -- concrete CPU/GPU overlap.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from majority_rule_model import MajorityRuleModel

DATA = Path("data")

# --------------------------------------------------------------------------- #
#  Chunked majority model (memory-safe big-graph scoring)
# --------------------------------------------------------------------------- #
class ChunkedMajorityRuleModel(MajorityRuleModel):
    """
    MajorityRuleModel that evaluates the R threshold draws in column
    chunks (default 256).  Same results as the parent (the draws are
    independent), but the working set stays at n x chunk instead of
    n x R, which is what lets R = 1000 held-out scoring run on an 82k
    node graph inside 6 GB of GPU memory.
    """

    def __init__(self, *args, chunk: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk = int(chunk)

    def _simulate_full(self, seeds):
        if self.R <= self.chunk:
            return super()._simulate_full(seeds)
        if self._dev is not None:
            return self._simulate_full_torch_chunked(seeds)
        sizes, pots = [], []
        for c in range(0, self.R, self.chunk):
            sl = slice(c, min(c + self.chunk, self.R))
            sizes_c, pots_c = self._simulate_full_slice(seeds, sl)
            sizes.append(sizes_c)
            pots.append(pots_c)
        return (np.concatenate(sizes), np.concatenate(pots))

    def _simulate_full_slice(self, seeds, sl):
        """CPU path over the R-column slice `sl`."""
        need = self.need[:, sl]
        active = np.zeros((self.n, need.shape[1]), dtype=self.dtype)
        idx = np.fromiter(seeds, dtype=np.int64)
        if idx.size:
            active[idx, :] = 1.0
        counts = self.A.dot(active)
        while True:
            newly = (counts >= need) & (active == 0)
            if not newly.any():
                break
            active[newly] = 1.0
            counts = self.A.dot(active)
        with np.errstate(divide="ignore", invalid="ignore"):
            pot = np.minimum(1.0, counts / need)
        pot[np.isnan(pot)] = 0.0
        return active.sum(axis=0), pot.sum(axis=0)

    def _simulate_full_torch_chunked(self, seeds):
        torch = self._torch
        dev = self._dev
        A = dev["A"]
        need_full = dev["need"]                     # (n, R) f32 on device
        sizes, pots = [], []
        for c in range(0, self.R, self.chunk):
            sl = slice(c, min(c + self.chunk, self.R))
            need = need_full[:, sl]
            active = torch.zeros((self.n, need.shape[1]),
                                 dtype=torch.float32, device=dev["dev"])
            idx = np.fromiter(seeds, dtype=np.int64)
            if idx.size:
                active[torch.from_numpy(idx).to(dev["dev"]), :] = 1.0
            counts = torch.sparse.mm(A, active)
            while True:
                newly = (counts >= need) & (active == 0)
                if not bool(newly.any()):
                    break
                active[newly] = 1.0
                counts = torch.sparse.mm(A, active)
            pot = torch.clamp(counts / need, 0.0, 1.0)
            pot = torch.nan_to_num(pot, nan=0.0)
            sizes.append(active.sum(dim=0).cpu().numpy())
            pots.append(pot.sum(dim=0).cpu().numpy())
        return (np.concatenate(sizes), np.concatenate(pots))


# --------------------------------------------------------------------------- #
#  Structural metrics (CPU workers, cached)
# --------------------------------------------------------------------------- #
def _pagerank_worker(args):
    name, pkl = args
    import networkx as nx
    import pickle as _pkl
    t0 = time.perf_counter()
    with open(pkl, "rb") as f:
        G = _pkl.load(f)
    pr = nx.pagerank(G, tol=1e-6)
    vals = np.asarray([pr[i] for i in range(G.number_of_nodes())], dtype=np.float64)
    print(f"    [worker] {name} pagerank {time.perf_counter()-t0:.1f}s", flush=True)
    return vals


def _kcore_worker(args):
    name, pkl = args
    import networkx as nx
    import pickle as _pkl
    t0 = time.perf_counter()
    with open(pkl, "rb") as f:
        G = _pkl.load(f)
    core = nx.core_number(G)
    vals = np.asarray([core[i] for i in range(G.number_of_nodes())],
                      dtype=np.float64)
    print(f"    [worker] {name} k-core {time.perf_counter()-t0:.1f}s", flush=True)
    return vals


def _betweenness_worker(args):
    name, pkl, pivots, seed = args
    import networkx as nx
    import pickle as _pkl
    t0 = time.perf_counter()
    with open(pkl, "rb") as f:
        G = _pkl.load(f)
    if pivots and pivots < G.number_of_nodes():
        bc = nx.betweenness_centrality(G, k=int(pivots), seed=int(seed),
                                       normalized=True)
    else:
        bc = nx.betweenness_centrality(G, normalized=True)
    vals = np.asarray([bc[i] for i in range(G.number_of_nodes())],
                      dtype=np.float64)
    print(f"    [worker] {name} betweenness {time.perf_counter()-t0:.1f}s",
          flush=True)
    return vals


def _louvain_worker(args):
    name, pkl, seed = args
    import networkx as nx
    import community as community_louvain
    import pickle as _pkl
    t0 = time.perf_counter()
    with open(pkl, "rb") as f:
        G = _pkl.load(f)
    part = community_louvain.best_partition(G, random_state=int(seed))
    vals = np.asarray([part[i] for i in range(G.number_of_nodes())],
                      dtype=np.int64)
    print(f"    [worker] {name} louvain {time.perf_counter()-t0:.1f}s "
          f"({int(vals.max()) + 1} communities)", flush=True)
    return vals


def _bc_pivots_for(n: int) -> int:
    """Paper reports exact BC for their nets; for big graphs we sample
    pivots to stay tractable (documented deviation)."""
    if n <= 6_000:
        return 0                       # exact
    if n <= 20_000:
        return 512
    return 384


def get_metrics(name: str, G, workers: int | None = None,
                recompute: bool = False) -> dict:
    """
    PageRank / k-core / betweenness / Louvain communities for a graph,
    computed in CPU worker processes and cached to
    data/metrics_<name>.npz (save-as-you-generate across restarts).
    """
    out = DATA / f"metrics_{name}.npz"
    if out.exists() and not recompute:
        z = np.load(out)
        return {"pr": z["pr"], "core": z["core"], "bc": z["bc"],
                "comm": z["comm"], "n": int(z["n"])}
    pkl = DATA / f"cache_{name}.pkl"
    pivots = _bc_pivots_for(G.number_of_nodes())
    jobs = [
        ("pr", (_pagerank_worker, (name, str(pkl)))),
        ("core", (_kcore_worker, (name, str(pkl)))),
        ("bc", (_betweenness_worker, (name, str(pkl), pivots, 7))),
        ("comm", (_louvain_worker, (name, str(pkl), 7))),
    ]
    results = {}
    # run the four jobs with a bounded pool (CPU side of the pipeline)
    with ProcessPoolExecutor(max_workers=workers or min(4, os.cpu_count() or 2)) as ex:
        futures = {key: ex.submit(fn, args) for key, (fn, args) in jobs}
        for key, fut in futures.items():
            results[key] = fut.result()
    np.savez_compressed(out, pr=results["pr"], core=results["core"],
                        bc=results["bc"], comm=results["comm"],
                        n=G.number_of_nodes())
    return {**results, "n": G.number_of_nodes()}


def signed_pagerank(G, alpha: float = 0.85, tol: float = 1e-10,
                    max_iter: int = 500) -> np.ndarray:
    """PageRank on the signed adjacency (negative edges = negative votes).

    A_ij = sign_ij * |weight_ij|, P = D_abs^{-1} A, and

        r <- alpha * P^T r + alpha * dangling_mass * v + (1-alpha) * v.

    Scores may be negative; ranking descending gives the most "trusted"
    nodes.  Runs in well under a second on the Bitcoin graphs and a few
    seconds on Slashdot.
    """
    import scipy.sparse as sp

    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    rows, cols, vals = [], [], []
    for u, v, d in G.edges(data=True):
        w = float(d.get("sign", 1)) * float(d.get("weight", 1))
        i, j = idx[u], idx[v]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    d_abs = np.asarray(abs(A).sum(axis=1)).ravel()
    inv = np.where(d_abs > 0, 1.0 / d_abs, 0.0)
    P = sp.diags(inv) @ A
    v = np.full(n, 1.0 / n)
    r = v.copy()
    dangling = d_abs == 0
    for _ in range(max_iter):
        r_new = (alpha * (P.T @ r)
                 + alpha * r[dangling].sum() * v
                 + (1.0 - alpha) * v)
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def get_signed_metrics(name: str, G, recompute: bool = False) -> dict:
    """Signed PageRank + original directed out-degree, cached per graph.

    Kept separate from `metrics_<name>.npz` so adding these two baselines
    never invalidates the expensive PageRank/k-core/betweenness/Louvain
    cache.  Output: `data/signed_metrics_<name>.npz`.
    """
    out = DATA / f"signed_metrics_{name}.npz"
    if out.exists() and not recompute:
        z = np.load(out)
        return {"spr": z["spr"], "outdeg": z["outdeg"]}
    from real_datasets import load_outdegree
    spr = signed_pagerank(G)
    outdeg = load_outdegree(name)
    np.savez_compressed(out, spr=spr, outdeg=outdeg)
    return {"spr": spr, "outdeg": outdeg}


# --------------------------------------------------------------------------- #
#  Pipeline (CPU futures -> GPU solvers)
# --------------------------------------------------------------------------- #
@dataclass
class CpuGpuPipeline:
    """Holds the CPU process pool so metric jobs can run while the GPU
    evaluates cascades in the main thread."""
    name: str
    graph: object
    metrics: dict | None = None

    def start_metrics(self, workers: int | None = None):
        """Submit missing structural-metric jobs; cached ones are reused."""
        self.metrics = {}
        out = DATA / f"metrics_{self.name}.npz"
        if out.exists():
            z = np.load(out)
            self.metrics = {"pr": z["pr"], "core": z["core"],
                            "bc": z["bc"], "comm": z["comm"]}
        missing = [k for k in ("pr", "core", "bc", "comm")
                   if k not in self.metrics]
        self._futures = {}
        if not missing:
            self._pool = None
            return self
        self._pool = ProcessPoolExecutor(
            max_workers=workers or min(4, os.cpu_count() or 2))
        pkl = DATA / f"cache_{self.name}.pkl"
        pivots = _bc_pivots_for(self.graph.number_of_nodes())
        jobs = {
            "pr": (_pagerank_worker, (self.name, str(pkl))),
            "core": (_kcore_worker, (self.name, str(pkl))),
            "bc": (_betweenness_worker, (self.name, str(pkl), pivots, 7)),
            "comm": (_louvain_worker, (self.name, str(pkl), 7)),
        }
        for key in missing:
            fn, args = jobs[key]
            self._futures[key] = self._pool.submit(fn, args)
        return self

    def get_metric(self, key: str):
        """Block until one metric future is done (cache it)."""
        if self.metrics is None:
            self.metrics = {}
        if key not in self.metrics:
            self.metrics[key] = self._futures[key].result()
        return self.metrics[key]

    def shutdown(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.shutdown(wait=False)


def gpu_info() -> dict:
    try:
        import torch
        return {"torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"}
    except Exception as e:                                     # pragma: no cover
        return {"torch": None, "cuda": False, "device": "cpu", "error": str(e)}


if __name__ == "__main__":
    print("GPU:", gpu_info())
    # smoke test chunked scoring on a small graph
    import networkx as nx
    from majority_rule_model import candidate_pool
    G = nx.watts_strogatz_graph(400, 6, 0.1, seed=1)
    m = ChunkedMajorityRuleModel(G, "smr", 0.15, 600, seed=7, chunk=200)
    pool = candidate_pool(m, 100)
    t0 = time.perf_counter()
    print("chunked sigma:", round(m.sigma(pool[:10]), 3),
          "time", round(time.perf_counter() - t0, 2), "backend", m.backend)
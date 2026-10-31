"""
majority_rule_model.py
======================
Baseline ("existing") implementation for influence maximisation under the
MAJORITY RULE diffusion model.

Model definition
----------------
Progressive (once active, always active) threshold diffusion on an undirected
graph G = (V, E).

Every node v independently draws a *susceptibility threshold*

        theta_v ~ D          (D supported on [0, 1])

At every time step, an inactive node v becomes active iff

        |{ active neighbours of v }|  >  theta_v * deg(v)                (1)

i.e. strictly more than a theta_v-fraction of its neighbourhood is active.
Because thresholds are drawn once and then frozen, the process is
deterministic conditional on theta, and the influence spread is

        sigma(S) = E_theta [ |A_infinity(S, theta)| ]                    (2)

Three members of this family are used throughout the study:

  * DETERMINISTIC MAJORITY  ("dmr")  : theta_v = 1/2 with probability 1.
        Node fires once a *strict majority* of its neighbours is active.
        need(v) = floor(deg(v)/2) + 1.
  * STOCHASTIC MAJORITY     ("smr")  : theta_v ~ U[1/2 - delta, 1/2 + delta].
        A "soft" majority rule with heterogeneous stubbornness. delta -> 0
        recovers the deterministic rule; delta = 1/2 recovers LT.
  * LINEAR THRESHOLD        ("lt")   : theta_v ~ U[0, 1]  (Kempe et al. 2003).
        Included ONLY as a submodular control condition.

Implementation note
-------------------
sigma(S) is estimated by Monte-Carlo over R pre-sampled threshold vectors that
are *reused for every evaluation* (common random numbers).  This is essential:
it removes the sampling noise that otherwise swamps the marginal-gain
comparisons that greedy / CELF / ABC all depend on, and it makes the whole
experiment exactly reproducible.

The R cascades are simulated simultaneously with one sparse matrix product per
round, so a single sigma(S) call costs O(rounds * nnz(A) * R) flops in BLAS
rather than R separate Python loops.

Author: (project skeleton)
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

__all__ = [
    "MajorityRuleModel",
    "random_seeds",
    "top_degree",
    "top_pagerank",
    "top_kcore",
    "mixed_heuristic",
    "greedy_mc",
    "celf",
    "SelectionResult",
    "candidate_pool",
    "load_graph",
    "load_edgelist",
    "heldout_model",
    "bfs_ball",
    "dense_cluster",
]


# --------------------------------------------------------------------------- #
#  Diffusion model
# --------------------------------------------------------------------------- #
class MajorityRuleModel:
    """Vectorised Monte-Carlo estimator of sigma(S) for the majority rule family."""

    VARIANTS = ("dmr", "smr", "lt", "slt")

    def __init__(
        self,
        graph: nx.Graph,
        variant: str = "smr",
        delta: float = 0.15,
        n_samples: int = 200,
        seed: int = 0,
        dtype=np.float32,
        backend: str = "auto",
    ):
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")

        self.variant = variant
        self.delta = float(delta)
        self.R = int(n_samples)
        self.dtype = dtype
        self.seed = seed
        self.backend = self._resolve_backend(backend)

        # stable node <-> index mapping
        self.nodes = list(graph.nodes())
        self.index = {u: i for i, u in enumerate(self.nodes)}
        self.n = len(self.nodes)
        self.graph = graph

        self.A, self.deg = self._build_adjacency(graph)

        self.rng = np.random.default_rng(seed)
        self.theta = self._draw_thresholds()          # (n, R)
        self.need = self._thresholds_to_counts()      # (n, R) float
        self.isolated = self.deg == 0                 # never activated by anyone

        # bookkeeping
        self.n_evals = 0
        self._cache: dict[frozenset, float] = {}
        self._cache_pot: dict[frozenset, tuple[float, float]] = {}

        self._dev = None
        if self.backend == "torch_cuda":
            self._init_torch()

    # ---------------- adjacency / backend machinery ---------------- #
    def _build_adjacency(self, graph: nx.Graph):
        """Return (A, deg). Subclasses (e.g. signed) override A's content."""
        A = nx.to_scipy_sparse_array(
            graph, nodelist=self.nodes, format="csr", dtype=self.dtype
        )
        deg = np.asarray(A.sum(axis=1)).ravel()
        return A, deg

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
        dev = torch.device("cuda")
        coo = self.A.tocoo()
        idx = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        vals = torch.from_numpy(coo.data.astype(np.float32))
        self._dev = {
            "A": torch.sparse_coo_tensor(idx, vals, size=(self.n, self.n),
                                         device=dev).coalesce(),
            "need": torch.from_numpy(self.need.astype(np.float32)).to(dev),
            "dev": dev,
        }

    # ---------------- threshold machinery ---------------- #
    def _draw_thresholds(self) -> np.ndarray:
        if self.variant == "dmr":
            return np.full((self.n, self.R), 0.5, dtype=self.dtype)
        if self.variant == "smr":
            lo, hi = 0.5 - self.delta, 0.5 + self.delta
            return self.rng.uniform(lo, hi, size=(self.n, self.R)).astype(self.dtype)
        if self.variant == "slt":
            # SANDWICH upper bound (Lu, Chen & Lakshmanan 2015 style): the
            # concave envelope of the majority CDF on [0,1] is
            #     F_bar(x) = min(x / (0.5 + delta), 1)
            # i.e. theta ~ U[0, 0.5 + delta].  F_bar >= F_majority pointwise,
            # so with a common-uniform coupling sigma_bar >= sigma pathwise,
            # and F_bar concave => sigma_bar is submodular (Mossel & Roch
            # sufficiency).  CELF is therefore VALID on this model.
            return self.rng.uniform(0.0, 0.5 + self.delta,
                                    size=(self.n, self.R)).astype(self.dtype)
        return self.rng.uniform(0.0, 1.0, size=(self.n, self.R)).astype(self.dtype)

    def _thresholds_to_counts(self) -> np.ndarray:
        """need(v, r) = smallest integer strictly greater than theta * deg."""
        raw = self.theta * self.deg[:, None]
        need = np.floor(raw) + 1.0
        # a node with degree 0 can never be activated externally
        need[self.deg == 0] = np.inf
        return need.astype(np.float64)

    # ---------------- spread ---------------- #
    def _simulate(self, seeds: Sequence[int]) -> np.ndarray:
        """Return the (R,) vector of final active-set sizes."""
        sizes, _ = self._simulate_full(seeds)
        return sizes

    def _simulate_full(self, seeds: Sequence[int]):
        """
        Return (sizes, pots): final active-set sizes and the PLATEAU POTENTIAL
            Phi_r(S) = sum_v min(1, net_v(r) / need_v(r))           (0/0 := 0)
        evaluated at convergence.  Active nodes contribute 1; inactive nodes
        contribute how close their neighbourhood is to flipping them.  Phi is
        the smooth gradient over the zero-gain plateau that sigma cannot see:
        two seed sets with identical spread are ordered by how many nodes they
        have sitting one neighbour short of activation.
        """
        if self._dev is not None:
            return self._simulate_full_torch(seeds)
        active = np.zeros((self.n, self.R), dtype=self.dtype)
        idx = np.fromiter(seeds, dtype=np.int64)
        if idx.size:
            active[idx, :] = 1.0
        counts = self.A.dot(active)
        while True:
            newly = (counts >= self.need) & (active == 0)
            if not newly.any():
                break
            active[newly] = 1.0
            counts = self.A.dot(active)
        with np.errstate(divide="ignore", invalid="ignore"):
            pot = np.minimum(1.0, counts / self.need)
        pot[np.isnan(pot)] = 0.0
        return active.sum(axis=0), pot.sum(axis=0)

    def _simulate_full_torch(self, seeds: Sequence[int]):
        torch = self._torch
        dev = self._dev
        active = torch.zeros((self.n, self.R), dtype=torch.float32, device=dev["dev"])
        idx = np.fromiter(seeds, dtype=np.int64)
        if idx.size:
            active[torch.from_numpy(idx).to(dev["dev"]), :] = 1.0
        A, need = dev["A"], dev["need"]
        counts = torch.sparse.mm(A, active)
        while True:
            newly = (counts >= need) & (active == 0)
            if not bool(newly.any()):
                break
            active[newly] = 1.0
            counts = torch.sparse.mm(A, active)
        pot = torch.clamp(counts / need, 0.0, 1.0)
        pot = torch.nan_to_num(pot, nan=0.0)
        return (active.sum(dim=0).cpu().numpy(),
                pot.sum(dim=0).cpu().numpy())

    def _border_nodes(self, seeds: Sequence[int], cap: int = 40) -> list[int]:
        """
        Nodes one neighbour short of activation at convergence (slack <= 1 on
        the majority of the R draws) plus the seeds themselves.  These are the
        cascade-relevant places for local moves.
        """
        active = np.zeros((self.n, self.R), dtype=self.dtype)
        idx = np.fromiter(seeds, dtype=np.int64)
        if idx.size:
            active[idx, :] = 1.0
        counts = self.A.dot(active)
        while True:
            newly = (counts >= self.need) & (active == 0)
            if not newly.any():
                break
            active[newly] = 1.0
            counts = self.A.dot(active)
        inactive = active.mean(axis=1) < 0.999
        slack = (self.need - counts).mean(axis=1)
        score = np.where(inactive, slack, np.inf)
        border = [int(i) for i in np.argsort(score) if 0 < score[i] <= 1.0 + 1e-9]
        return border[:cap]

    def sigma_potential(self, seeds: Iterable[int], use_cache: bool = True) -> tuple[float, float, float]:
        """(sigma, Phi, sigma standard error) for `seeds` from ONE cascade
        pass.  Phi = plateau potential; the standard error lets acceptance
        rules treat sigma differences within Monte-Carlo noise as ties."""
        key = frozenset(seeds)
        if use_cache and key in self._cache_pot:
            return self._cache_pot[key]
        self.n_evals += 1
        sizes, pots = self._simulate_full(sorted(key))
        val = (float(sizes.mean()), float(pots.mean()),
               float(sizes.std(ddof=1) / np.sqrt(self.R)) if self.R > 1 else 0.0)
        if use_cache:
            self._cache_pot[key] = val
        return val

    def sigma(self, seeds: Iterable[int], use_cache: bool = True) -> float:
        """Expected number of finally-active nodes for seed set `seeds`."""
        key = frozenset(seeds)
        if use_cache and key in self._cache:
            return self._cache[key]
        self.n_evals += 1
        val = float(self._simulate(sorted(key)).mean())
        if use_cache:
            self._cache[key] = val
        return val

    def sigma_std(self, seeds: Iterable[int]) -> float:
        """Monte-Carlo standard error of sigma (for confidence intervals)."""
        vals = self._simulate(sorted(set(seeds)))
        return float(vals.std(ddof=1) / np.sqrt(self.R))

    def marginal(self, seeds: Iterable[int], v: int) -> float:
        s = frozenset(seeds)
        return self.sigma(s | {v}) - self.sigma(s)

    # ---------------- utilities ---------------- #
    def reset_counters(self) -> None:
        self.n_evals = 0

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_pot.clear()

    def describe(self) -> str:
        return (
            f"MajorityRuleModel(variant={self.variant}, delta={self.delta}, "
            f"R={self.R}, n={self.n}, m={self.graph.number_of_edges()})"
        )


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class SelectionResult:
    algorithm: str
    seeds: list[int]
    spread: float
    runtime: float
    evaluations: int
    trace: list[float] = field(default_factory=list)   # spread after each addition
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "k": len(self.seeds),
            "spread": self.spread,
            "runtime_s": self.runtime,
            "evaluations": self.evaluations,
            **{f"x_{a}": b for a, b in self.extra.items()},
        }


# --------------------------------------------------------------------------- #
#  Candidate pool
# --------------------------------------------------------------------------- #
def heldout_model(
    model: MajorityRuleModel,
    seed: int | None = None,
    backend: str | None = None,
) -> MajorityRuleModel:
    """
    Clone of `model` with FRESH, independent threshold draws.

    Used to re-score final seed sets out-of-sample so that optimisation on one
    set of thresholds cannot be rewarded for overfitting them (common random
    numbers are used *within* each model, never across fit and score).
    """
    s = model.seed + 10_000 if seed is None else seed
    return type(model)(
        model.graph,
        variant=model.variant,
        delta=model.delta,
        n_samples=model.R,
        seed=s,
        dtype=model.dtype,
        backend=backend or model.backend,
    )


def candidate_pool(model: MajorityRuleModel, size: int, seed: int = 0) -> list[int]:
    """
    Diverse pool of plausible seeds.

    Under the majority rule high-degree hubs are *hard to activate* but *good
    activators*, while low-degree nodes are easy to activate.  A pool built on
    degree alone therefore biases the search, so we blend four rankings plus a
    random tail.
    """
    G, idx = model.graph, model.index
    n = model.n
    size = min(size, n)

    deg_rank = sorted(range(n), key=lambda i: -model.deg[i])
    pr = nx.pagerank(G)
    pr_rank = sorted(range(n), key=lambda i: -pr[model.nodes[i]])
    core = nx.core_number(nx.Graph(G))
    core_rank = sorted(range(n), key=lambda i: -core[model.nodes[i]])
    # "easily-flippable" nodes: small degree => small majority requirement
    easy_rank = sorted(range(n), key=lambda i: (model.need[i].mean(), -model.deg[i]))

    pool: list[int] = []
    seen: set[int] = set()
    ranked_slots = max(1, (3 * size) // 4)      # rankings fill 75% of the pool
    per = max(1, ranked_slots // 4)
    for ranking in (deg_rank, pr_rank, core_rank, easy_rank):
        for v in ranking[: per + ranked_slots]:
            if v not in seen:
                seen.add(v)
                pool.append(v)
            if len(pool) >= ranked_slots:
                break
        if len(pool) >= ranked_slots:
            break

    # guaranteed random tail: keeps every basin of the search space reachable
    # (pure ranking-based pools can exclude entire regions, e.g. the corner
    # diagonals of a grid, which caps every solver at a pool-internal optimum)
    if len(pool) < size:
        rng = np.random.default_rng(seed)
        rest = [v for v in range(n) if v not in seen]
        rng.shuffle(rest)
        pool.extend(rest[: size - len(pool)])
    return pool[:size]


# --------------------------------------------------------------------------- #
#  Baseline heuristics
# --------------------------------------------------------------------------- #
def _finish(name, model, seeds, t0, e0, trace=None, **extra) -> SelectionResult:
    return SelectionResult(
        algorithm=name,
        seeds=list(seeds),
        spread=model.sigma(seeds),
        runtime=time.perf_counter() - t0,
        evaluations=model.n_evals - e0,
        trace=trace or [],
        extra=extra,
    )


def random_seeds(model: MajorityRuleModel, k: int, seed: int = 0) -> SelectionResult:
    t0, e0 = time.perf_counter(), model.n_evals
    rng = np.random.default_rng(seed)
    s = rng.choice(model.n, size=k, replace=False).tolist()
    return _finish("Random", model, s, t0, e0)


def top_degree(model: MajorityRuleModel, k: int, **_) -> SelectionResult:
    t0, e0 = time.perf_counter(), model.n_evals
    s = sorted(range(model.n), key=lambda i: -model.deg[i])[:k]
    return _finish("Degree", model, s, t0, e0)


def top_pagerank(model: MajorityRuleModel, k: int, **_) -> SelectionResult:
    t0, e0 = time.perf_counter(), model.n_evals
    pr = nx.pagerank(model.graph)
    s = sorted(range(model.n), key=lambda i: -pr[model.nodes[i]])[:k]
    return _finish("PageRank", model, s, t0, e0)


def top_kcore(model: MajorityRuleModel, k: int, **_) -> SelectionResult:
    t0, e0 = time.perf_counter(), model.n_evals
    core = nx.core_number(nx.Graph(model.graph))
    s = sorted(range(model.n), key=lambda i: (-core[model.nodes[i]], -model.deg[i]))[:k]
    return _finish("k-Core", model, s, t0, e0)


def mixed_heuristic(model: MajorityRuleModel, k: int, **_) -> SelectionResult:
    """
    Majority-aware degree discount: score a node by how many of its neighbours
    it moves closest to their majority threshold, discounted by overlap with
    already-chosen seeds.  Cheap, model-aware, no Monte-Carlo at all.
    """
    t0, e0 = time.perf_counter(), model.n_evals
    need = model.need.mean(axis=1)
    A = model.A.tocsr()
    score = np.zeros(model.n)
    for v in range(model.n):
        nb = A.indices[A.indptr[v]: A.indptr[v + 1]]
        if nb.size:
            score[v] = np.sum(1.0 / np.maximum(need[nb], 1.0))
    chosen: list[int] = []
    covered = np.zeros(model.n)
    for _ in range(k):
        adj = score - 0.5 * covered
        adj[chosen] = -np.inf
        v = int(np.argmax(adj))
        chosen.append(v)
        nb = A.indices[A.indptr[v]: A.indptr[v + 1]]
        covered[nb] += 1.0
    return _finish("MajorityDiscount", model, chosen, t0, e0)


# --------------------------------------------------------------------------- #
#  Zero-simulation constructive baselines (target-set-selection style)
# --------------------------------------------------------------------------- #
def bfs_ball(model: MajorityRuleModel, k: int, seed: int = 0, **_) -> SelectionResult:
    """
    A BFS ball grown from the highest-degree root (bootstrap-percolation /
    target-set-selection style).  Costs ZERO cascade evaluations: seeds a
    connected, centrally located coalition directly.
    """
    t0, e0 = time.perf_counter(), model.n_evals
    A = model.A.tocsr()
    root = int(np.argmax(model.deg))
    chosen: list[int] = []
    seen: set[int] = set()
    frontier = [root]
    while frontier and len(chosen) < k:
        v = frontier.pop(0)
        if v in seen:
            continue
        seen.add(v)
        chosen.append(v)
        nb = A.indices[A.indptr[v]: A.indptr[v + 1]].tolist()
        frontier.extend(x for x in nb if x not in seen)
    return _finish("BFSBall", model, chosen, t0, e0)


def dense_cluster(model: MajorityRuleModel, k: int, seed: int = 0, **_) -> SelectionResult:
    """
    Zero-simulation constructive baseline: iteratively add the node with the
    most already-chosen neighbours -- a densely connected coalition, which is
    exactly the structure a majority cascade needs.
    """
    t0, e0 = time.perf_counter(), model.n_evals
    A = model.A.tocsr()
    chosen: list[int] = [int(np.argmax(model.deg))]
    covered = np.zeros(model.n, dtype=np.int32)
    while len(chosen) < k:
        nb = A.indices[A.indptr[chosen[-1]]: A.indptr[chosen[-1] + 1]]
        covered[nb] += 1
        mask = np.ones(model.n, dtype=bool)
        mask[chosen] = False
        v = int(np.argmax(np.where(mask, covered, -1)))
        chosen.append(v)
    return _finish("DenseCluster", model, chosen, t0, e0)


# --------------------------------------------------------------------------- #
#  Exhaustive-evaluation greedy (Kempe et al. hill climbing)
# --------------------------------------------------------------------------- #
def greedy_mc(
    model: MajorityRuleModel,
    k: int,
    pool: Sequence[int] | None = None,
    verbose: bool = False,
) -> SelectionResult:
    """Vanilla greedy: every candidate is re-evaluated at every one of the k rounds."""
    t0, e0 = time.perf_counter(), model.n_evals
    cand = list(pool) if pool is not None else list(range(model.n))
    S: list[int] = []
    trace: list[float] = []
    cur = model.sigma(S)
    for _ in range(k):
        best_v, best_gain = None, -np.inf
        for v in cand:
            if v in S:
                continue
            g = model.sigma(S + [v]) - cur
            if g > best_gain:
                best_gain, best_v = g, v
        S.append(best_v)
        cur += best_gain
        trace.append(cur)
        if verbose:
            print(f"  greedy round {len(S)}: +{best_v} gain={best_gain:.3f} sigma={cur:.3f}")
    return _finish("Greedy", model, S, t0, e0, trace=trace)


def greedy_phi(
    model: MajorityRuleModel,
    k: int,
    pool: Sequence[int] | None = None,
    tol_frac: float = 1.0,
    verbose: bool = False,
) -> SelectionResult:
    """
    Greedy with the plateau-potential TIE-BREAKER: among candidates whose
    sigma gain is within `tol_frac` Monte-Carlo standard errors of the best,
    pick the one with the largest Phi gain.  This gives greedy a fairer shot
    at the zero-gain plateau (the same gradient ABC's fitness uses) without
    changing the objective: it never accepts a worse sigma gain.

    Evaluation cost stays exactly k * |pool| (sigma and Phi come from the
    same cascade pass).
    """
    t0, e0 = time.perf_counter(), model.n_evals
    cand = list(pool) if pool is not None else list(range(model.n))
    S: list[int] = []
    trace: list[float] = []
    cur_s, cur_p, _ = model.sigma_potential(S)
    cur = cur_s
    for _ in range(k):
        best_v, best_gain, best_pg, best_tol = None, -np.inf, -np.inf, 0.0
        for v in cand:
            if v in S:
                continue
            sp = model.sigma_potential(S + [v])
            g = sp[0] - cur_s
            tol = tol_frac * sp[2]
            if g > best_gain + best_tol:
                best_v, best_gain, best_pg, best_tol = v, g, sp[1] - cur_p, tol
            elif g >= best_gain - best_tol:
                pg = sp[1] - cur_p
                if pg > best_pg:
                    best_v, best_gain, best_pg, best_tol = v, g, pg, tol
        S.append(best_v)
        cur_s, cur_p, _ = model.sigma_potential(S)
        cur = cur_s
        trace.append(cur)
        if verbose:
            print(f"  greedy_phi round {len(S)}: +{best_v} gain={best_gain:.3f} sigma={cur:.3f}")
    return _finish("Greedy+Phi", model, S, t0, e0, trace=trace)


# --------------------------------------------------------------------------- #
#  CELF  (Leskovec et al. 2007 "Cost-Effective Lazy Forward")
# --------------------------------------------------------------------------- #
def celf(
    model: MajorityRuleModel,
    k: int,
    pool: Sequence[int] | None = None,
    verbose: bool = False,
    audit_lazy_bound: bool = True,
) -> SelectionResult:
    """
    Lazy-forward greedy.

    CELF's correctness rests on submodularity: a stale marginal gain is an
    UPPER bound on the current one, so a candidate whose stale gain already
    loses to a freshly evaluated one can be skipped.

    Under the majority rule that guarantee does NOT hold.  We therefore run
    CELF as a heuristic accelerator and, when `audit_lazy_bound` is on, we
    count how often the bound is actually violated (a re-evaluated gain that
    turns out LARGER than its stale value).  This number is reported in
    `extra["lazy_bound_violations"]` and is one of the key empirical results
    of the study.
    """
    t0, e0 = time.perf_counter(), model.n_evals
    cand = list(pool) if pool is not None else list(range(model.n))

    base = model.sigma([])
    heap: list[tuple[float, int, int]] = []
    stale: dict[int, float] = {}
    for v in cand:
        g = model.sigma([v]) - base
        stale[v] = g
        heapq.heappush(heap, (-g, v, 0))

    S: list[int] = []
    cur = base
    trace: list[float] = []
    violations = 0
    checks = 0

    while len(S) < k and heap:
        neg_g, v, last_round = heapq.heappop(heap)
        if last_round == len(S):
            S.append(v)
            cur += -neg_g
            trace.append(cur)
            if verbose:
                print(f"  celf round {len(S)}: +{v} gain={-neg_g:.3f} sigma={cur:.3f}")
            continue
        new_g = model.sigma(S + [v]) - cur
        if audit_lazy_bound:
            checks += 1
            # stale[v] was computed on a SUBSET of S; submodularity would force
            # new_g <= stale[v] + tol
            if new_g > stale[v] + 1e-9:
                violations += 1
        stale[v] = new_g
        heapq.heappush(heap, (-new_g, v, len(S)))

    return _finish(
        "CELF",
        model,
        S,
        t0,
        e0,
        trace=trace,
        lazy_bound_checks=checks,
        lazy_bound_violations=violations,
        lazy_violation_rate=(violations / checks if checks else 0.0),
    )


# --------------------------------------------------------------------------- #
#  Graph zoo
# --------------------------------------------------------------------------- #
def load_graph(name: str, seed: int = 42) -> nx.Graph:
    """Small, self-contained benchmark graphs (no network access required)."""
    name = name.lower()
    if name == "karate":
        G = nx.karate_club_graph()
    elif name == "lesmis":
        G = nx.les_miserables_graph()
    elif name == "florentine":
        G = nx.florentine_families_graph()
    elif name == "davis":
        G = nx.davis_southern_women_graph()
    elif name == "ba500":
        G = nx.barabasi_albert_graph(500, 3, seed=seed)
    elif name == "ba300":
        G = nx.barabasi_albert_graph(300, 3, seed=seed)
    elif name == "ba1000":
        G = nx.barabasi_albert_graph(1000, 4, seed=seed)
    elif name == "ba2000":
        G = nx.barabasi_albert_graph(2000, 4, seed=seed)
    elif name == "ws300":
        G = nx.watts_strogatz_graph(300, 8, 0.1, seed=seed)
    elif name == "ws1000":
        G = nx.watts_strogatz_graph(1000, 8, 0.1, seed=seed)
    elif name == "ws2000":
        G = nx.watts_strogatz_graph(2000, 8, 0.1, seed=seed)
    elif name == "powerlaw300":
        G = nx.powerlaw_cluster_graph(300, 4, 0.3, seed=seed)
    elif name == "powerlaw1000":
        G = nx.powerlaw_cluster_graph(1000, 4, 0.3, seed=seed)
    elif name == "powerlaw2000":
        G = nx.powerlaw_cluster_graph(2000, 4, 0.3, seed=seed)
    elif name == "sbm450":
        sizes = [150, 150, 150]
        p = [[0.04, 0.004, 0.003], [0.004, 0.04, 0.004], [0.003, 0.004, 0.04]]
        G = nx.stochastic_block_model(sizes, p, seed=seed)
    elif name == "sbm900":
        sizes = [300, 300, 300]
        p = [[0.04, 0.004, 0.003], [0.004, 0.04, 0.004], [0.003, 0.004, 0.04]]
        G = nx.stochastic_block_model(sizes, p, seed=seed)
    elif name == "star60":
        G = nx.star_graph(60)
    elif name == "grid100":
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(10, 10))
    elif name == "cycle60":
        G = nx.cycle_graph(60)
    elif name == "tree63":
        G = nx.balanced_tree(2, 5)
    elif name == "barbell50":
        G = nx.barbell_graph(25, 0)
    elif name == "caveman60":
        G = nx.caveman_graph(6, 10)
    elif name == "windmill46":
        G = nx.windmill_graph(5, 10)
    else:
        raise ValueError(f"unknown graph '{name}'")
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, label_attribute="orig")


def load_edgelist(path: str, delimiter: str | None = None) -> nx.Graph:
    """Hook for real SNAP-style datasets: two whitespace-separated columns."""
    G = nx.read_edgelist(path, delimiter=delimiter, nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G, label_attribute="orig")


if __name__ == "__main__":
    G = load_graph("ba500")
    m = MajorityRuleModel(G, variant="smr", delta=0.15, n_samples=200, seed=1)
    print(m.describe())
    pool = candidate_pool(m, 120)
    for fn in (random_seeds, top_degree, mixed_heuristic):
        r = fn(m, 10)
        print(f"{r.algorithm:20s} sigma={r.spread:8.3f}  t={r.runtime:6.3f}s")
    r = celf(m, 10, pool=pool, verbose=False)
    print(f"{r.algorithm:20s} sigma={r.spread:8.3f}  t={r.runtime:6.3f}s "
          f"evals={r.evaluations}  lazy-violations={r.extra['lazy_bound_violations']}")

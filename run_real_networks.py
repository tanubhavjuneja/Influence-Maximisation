r"""
run_real_networks.py
====================
Real-network comparison (supervisor task): Bitcoin Alpha / Bitcoin OTC /
Slashdot, comparing spread AND wall-clock of Random, Degree, PageRank,
k-Core, MajorityDiscount, BFSBall, Greedy, CELF, CELF+LS, SA, GRASP,
ABC (our optimised version) and ABCVNS (the ABC.pdf paper's algorithm).

Stages (each resumable, save-as-you-go):
  majority   majority-rule suite      (batched cascades on the GPU)
  ic         independent-cascade suite (ABCVNS home model; CPU-parallel draws)
  signed     signed majority-rule suite on the bitcoin graphs (bonus)
  report     summary JSON + figures (numbers for the final report)

Resumability / logging / saving
-------------------------------
* every completed (graph, k, algorithm, restart) cell is APPENDED to
  results/real_<stage>.csv IMMEDIATELY and skipped on re-run;
* verbose logging goes to console AND results/real_run.log;
* results/real_progress.json tracks cells done/total per stage;
* heavy CPU metrics (PageRank / k-core / betweenness / Louvain) are
  cached in data/metrics_<graph>.npz -- recomputation never happens
  twice even across process restarts.

CPU + GPU parallelism
---------------------
CPU workers (ProcessPoolExecutor): structural metrics + IC live-edge draws.
GPU (torch CUDA, RTX 3060): majority-rule cascade batches, chunked so
R=1000 held-out scoring fits in memory (hybrid_engine.ChunkedMajorityRuleModel).

Usage:
  python run_real_networks.py majority [--graphs alpha otc slashdot] [--quick]
  python run_real_networks.py ic       [--graphs alpha otc slashdot] [--quick]
  python run_real_networks.py signed   [--graphs alpha otc]          [--quick]
  python run_real_networks.py report
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from majority_rule_model import (
    MajorityRuleModel, SelectionResult, greedy_mc, celf,
    random_seeds, top_degree, mixed_heuristic, bfs_ball,
)
from abc_celf import ABCConfig, abc_optimize, celf_exchange_refine
from baselines import simulated_annealing, grasp
from real_datasets import load_real_graph, dataset_stats, ensure_datasets
from hybrid_engine import (ChunkedMajorityRuleModel, get_metrics,
                           get_signed_metrics, CpuGpuPipeline, gpu_info)
from ic_model import IndependentCascadeModel, ic_heldout_model
from abcvns import make_abcnvs_context, abcnvs_search, ABCVNS_PARAMS

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
DATA = Path("data"); DATA.mkdir(exist_ok=True)

HELDOUT_SEED = 999_999
GRAPH_ALIAS = {"alpha": "bitcoin_alpha", "otc": "bitcoin_otc",
               "slashdot": "slashdot"}

# canonical CSV schema: every row carries ALL columns, extras empty when
# absent -- the append-only file stays readable by pandas at all times
BASE_COLS = ["stage", "graph", "n", "m", "k", "algorithm", "restart",
             "spread", "spread_heldout", "runtime_s", "evaluations", "budget"]
EXTRA_COLS = [
    "x_lazy_bound_checks", "x_lazy_bound_violations", "x_lazy_violation_rate",
    "x_temp_start", "x_restart",
    "x_cycles_run", "x_scout_resets", "x_dup_replacements",
    "x_refine_swaps", "x_refine_pair_swaps",
    "x_celf_seed_spread", "x_lie_best", "x_abcnvs_wall_s",
]
CANON_COLS = BASE_COLS + EXTRA_COLS

# protocol --------------------------------------------------------------------
GRAPH_PROTOCOL = {
    "bitcoin_alpha": dict(ks=(10, 20, 30), p_ic=0.05, restarts=3),
    "bitcoin_otc":   dict(ks=(10, 20, 30), p_ic=0.05, restarts=3),
    "slashdot":      dict(ks=(10, 20),     p_ic=0.01, restarts=2),
}
MAJORITY_PROTOCOL = dict(variant="smr", delta=0.15, R_fit=150,
                         R_ho=1000, pool_size=200)
IC_PROTOCOL = dict(R_fit=100, R_ho=1000, pool_size=200,
                   R_ho_parallel_workers=4)
QUICK = dict(ks=(5,), restarts=1, R_fit=40, R_ho=200, pool_size=60,
             abcnvs=dict(gmax=20, NP=20))

MAJORITY_SOLVERS = [
    "Random", "Degree", "PageRank", "SignedPageRank", "OutDegree", "k-Core",
    "MajorityDiscount", "BFSBall", "Greedy", "CELF", "CELF + LS", "SA",
    "GRASP", "ABC (proposed)", "ABCVNS (paper)",
]
IC_SOLVERS = [
    "Random", "Degree", "PageRank", "SignedPageRank", "OutDegree", "k-Core",
    "Greedy", "CELF", "CELF + LS", "SA", "ABC (proposed)", "ABCVNS (paper)",
]
STOCHASTIC = {"SA", "GRASP", "ABC (proposed)", "ABCVNS (paper)"}


# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
def setup_logging() -> logging.Logger:
    logger = logging.getLogger("realnet")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(RESULTS / "real_run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


log = setup_logging()


# --------------------------------------------------------------------------- #
#  Candidate pool (uses the cached CPU metrics -- no recomputation)
# --------------------------------------------------------------------------- #
def pool_from_metrics(model, metrics, size, seed=0) -> list[int]:
    """Same four-way blend as majority_rule_model.candidate_pool but using
    the cached CPU metrics (PageRank / k-core) instead of recomputing them."""
    n = model.n
    size = min(size, n)
    deg = np.asarray(model.deg, dtype=float)
    pr = np.asarray(metrics["pr"], dtype=float)
    core = np.asarray(metrics["core"], dtype=float)
    need_mean = getattr(model, "need", None)
    if need_mean is not None:
        easy = np.asarray(need_mean.mean(axis=1), dtype=float)
    else:
        easy = deg                            # IC: fall back to degree
    deg_rank = np.argsort(-deg)
    pr_rank = np.argsort(-pr)
    core_rank = np.argsort(-core)
    easy_rank = np.argsort(easy)
    pool: list[int] = []
    seen: set[int] = set()
    ranked_slots = max(1, (3 * size) // 4)
    per = max(1, ranked_slots // 4)
    for ranking in (deg_rank, pr_rank, core_rank, easy_rank):
        for v in ranking[: per + ranked_slots]:
            if int(v) not in seen:
                seen.add(int(v))
                pool.append(int(v))
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
#  Row bookkeeping (resumable, save-as-you-go)
# --------------------------------------------------------------------------- #
class ResultStore:
    """Append-only CSV store; done cells are skipped on re-run."""

    def __init__(self, stage: str):
        self.stage = stage
        self.path = RESULTS / f"real_{stage}.csv"
        self.df = self._load()
        self.done = set()
        if self.df is not None and len(self.df):
            for _, r in self.df.iterrows():
                self.done.add((str(r["graph"]), int(r["k"]),
                               str(r["algorithm"]), int(r["restart"])))
        self.progress = (RESULTS / "real_progress.json")
        self._prog = self._load_progress()

    def _load(self):
        if not self.path.exists():
            return None
        try:
            return pd.read_csv(self.path)
        except Exception:
            # legacy ragged file from the pre-canonical writer: repair once
            from repair_real_csv import repair as _repair, write_clean as _wc
            rows = _repair(self.path)
            if rows:
                _wc(self.path, rows)
                return pd.read_csv(self.path)
            raise

    def _load_progress(self):
        if self.progress.exists():
            return json.loads(self.progress.read_text())
        return {}

    def _save_progress(self, **kw):
        self._prog.setdefault(self.stage, {}).update(kw)
        self.progress.write_text(json.dumps(self._prog, indent=2))

    def is_done(self, graph, k, alg, restart) -> bool:
        return (graph, int(k), alg, int(restart)) in self.done

    def add(self, row: dict):
        aligned = {c: row.get(c, "") for c in CANON_COLS}
        # first write: full canonical header
        if not self.path.exists():
            self.path.write_text(",".join(CANON_COLS) + "\n",
                                 encoding="utf-8")
        else:
            # guard against any header drift: rewrite from memory if needed
            with open(self.path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
            if header.split(",") != CANON_COLS:
                lines = [",".join(CANON_COLS)]
                for r in (self._rows_list() or []):
                    lines.append(",".join(str(r.get(c, "")) for c in CANON_COLS))
                self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(",".join(str(aligned[c]) for c in CANON_COLS) + "\n")
        self.done.add((str(row["graph"]), int(row["k"]), str(row["algorithm"]),
                       int(row["restart"])))

    def _rows_list(self) -> list[dict] | None:
        if self.df is None:
            return None
        return self.df.to_dict("records")

    def flush(self):
        pass                                        # already append-per-cell


def _cell_row(stage, graph, n, m, k, alg, restart, res: SelectionResult,
              heldout: float, budget, **extra) -> dict:
    row = {"stage": stage, "graph": graph, "n": n, "m": m, "k": k,
           "algorithm": alg, "restart": restart,
           "spread": float(res.spread), "spread_heldout": float(heldout),
           "runtime_s": float(res.runtime), "evaluations": int(res.evaluations),
           "budget": budget}
    # canonical extras: always present in the row (empty when absent) so the
    # append-only CSV keeps one fixed schema
    for c in EXTRA_COLS:
        row[c] = ""
    for key, val in (res.extra or {}).items():
        if isinstance(val, (int, float, bool, str)):
            row[f"x_{key}"] = val
    for key, val in extra.items():
        if isinstance(val, (int, float, bool, str)):
            row[f"x_{key}"] = val
    return row


# --------------------------------------------------------------------------- #
#  IC local refinement (CELF + LS under IC)
# --------------------------------------------------------------------------- #
def ic_local_refine(model, seeds, pool, k, eval_budget) -> SelectionResult:
    t0, e0 = time.perf_counter(), model.n_evals
    S = list(seeds)
    best = model.sigma(S)
    improved = True
    while improved and (model.n_evals - e0) < eval_budget:
        improved = False
        for pos in range(k):
            if (model.n_evals - e0) >= eval_budget:
                break
            for v in pool:
                if v in S:
                    continue
                if (model.n_evals - e0) >= eval_budget:
                    break
                cand = list(S)
                cand[pos] = v
                f = model.sigma(cand)
                if f > best + 1e-9:
                    S, best = cand, f
                    improved = True
                    break
            if improved:
                break
    return SelectionResult(algorithm="CELF + LS", seeds=S, spread=best,
                           runtime=time.perf_counter() - t0,
                           evaluations=model.n_evals - e0)


# --------------------------------------------------------------------------- #
#  Majority-rule cell solver
# --------------------------------------------------------------------------- #
def solve_majority(model, ho, metrics, alg, k, pool, budget, rs,
                   st_lie=None, abcnvs_params=None, log=log):
    """Return a SelectionResult for one (algorithm, restart) cell."""
    if alg == "Random":
        return random_seeds(model, k, seed=rs)
    if alg == "Degree":
        return top_degree(model, k)
    if alg == "PageRank":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["pr"])[:k].tolist()
        return SelectionResult("PageRank", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "SignedPageRank":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["spr"])[:k].tolist()
        return SelectionResult("SignedPageRank", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "OutDegree":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["outdeg"])[:k].tolist()
        return SelectionResult("OutDegree", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "k-Core":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["core"])[:k].tolist()
        return SelectionResult("k-Core", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "MajorityDiscount":
        return mixed_heuristic(model, k)
    if alg == "BFSBall":
        return bfs_ball(model, k)
    if alg == "Greedy":
        return greedy_mc(model, k, pool=pool)
    if alg == "CELF":
        return celf(model, k, pool=pool)
    if alg == "CELF + LS":
        r_c = celf(model, k, pool=pool)
        t0 = time.perf_counter()
        S2, f2, _ = celf_exchange_refine(
            model, r_c.seeds, pool, max_passes=4,
            eval_budget=max(0, budget - r_c.evaluations))
        return SelectionResult("CELF + LS", S2, f2,
                               time.perf_counter() - t0 + r_c.runtime,
                               model.n_evals)
    if alg == "SA":
        return simulated_annealing(model, k, pool, eval_budget=budget,
                                   restart=True, seed=rs)
    if alg == "GRASP":
        return grasp(model, k, pool, eval_budget=budget, seed=rs)
    if alg == "ABC (proposed)":
        cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                        seed_with_celf=True, eval_budget=budget,
                        random_state=600 + rs)
        return abc_optimize(model, k, pool, cfg, name="ABC (proposed)")
    if alg == "ABCVNS (paper)":
        # paper search: LIE fitness, no MC in the loop; scored on sigma here
        params = dict(abcnvs_params or ABCVNS_PARAMS)
        S, stats = abcnvs_search(st_lie, k, seed=rs, verbose=True,
                                 params=params, log=log.info)
        t0 = time.perf_counter()
        spread = model.sigma(S)
        return SelectionResult("ABCVNS (paper)", S, spread,
                               time.perf_counter() - t0 + stats["wall_s"],
                               model.n_evals,
                               extra={"lie_best": stats["LIE_best"],
                                      "abcnvs_wall_s": stats["wall_s"]})
    raise ValueError(alg)


def solve_ic(model, metrics, alg, k, pool, budget, rs,
             st_lie=None, abcnvs_params=None, log=log):
    """SelectionResult for one (algorithm, restart) cell under IC."""
    if alg == "Random":
        return random_seeds(model, k, seed=rs)
    if alg == "Degree":
        return top_degree(model, k)
    if alg == "PageRank":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["pr"])[:k].tolist()
        return SelectionResult("PageRank", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "SignedPageRank":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["spr"])[:k].tolist()
        return SelectionResult("SignedPageRank", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "OutDegree":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["outdeg"])[:k].tolist()
        return SelectionResult("OutDegree", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "k-Core":
        t0 = time.perf_counter()
        s = np.argsort(-metrics["core"])[:k].tolist()
        return SelectionResult("k-Core", s, model.sigma(s),
                               time.perf_counter() - t0, model.n_evals)
    if alg == "Greedy":
        return greedy_mc(model, k, pool=pool)
    if alg == "CELF":
        return celf(model, k, pool=pool)
    if alg == "CELF + LS":
        r_c = celf(model, k, pool=pool)
        return ic_local_refine(model, r_c.seeds, pool, k,
                               eval_budget=max(0, budget - r_c.evaluations))
    if alg == "SA":
        return simulated_annealing(model, k, pool, eval_budget=budget,
                                   restart=True, seed=rs)
    if alg == "ABC (proposed)":
        cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                        seed_with_celf=True, eval_budget=budget,
                        random_state=600 + rs)
        return abc_optimize(model, k, pool, cfg, name="ABC (proposed)")
    if alg == "ABCVNS (paper)":
        params = dict(abcnvs_params or ABCVNS_PARAMS)
        S, stats = abcnvs_search(st_lie, k, seed=rs, verbose=True,
                                 params=params, log=log.info)
        t0 = time.perf_counter()
        spread = model.sigma(S)
        return SelectionResult("ABCVNS (paper)", S, spread,
                               time.perf_counter() - t0 + stats["wall_s"],
                               model.n_evals,
                               extra={"lie_best": stats["LIE_best"],
                                      "abcnvs_wall_s": stats["wall_s"]})
    raise ValueError(alg)


# --------------------------------------------------------------------------- #
#  Stages
# --------------------------------------------------------------------------- #
def run_majority(graphs, quick: bool = False):
    stage = "majority"
    store = ResultStore(stage)
    log.info(f"== majority stage == gpu={gpu_info()} graphs={graphs} "
             f"quick={quick}")
    for name in graphs:
        G = load_real_graph(name)
        stats = dataset_stats(name)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        proto = dict(GRAPH_PROTOCOL[name])
        if quick:
            proto["ks"] = QUICK["ks"]
            proto["restarts"] = QUICK["restarts"]
        R_fit = QUICK["R_fit"] if quick else MAJORITY_PROTOCOL["R_fit"]
        R_ho = QUICK["R_ho"] if quick else MAJORITY_PROTOCOL["R_ho"]
        pool_size = QUICK["pool_size"] if quick else MAJORITY_PROTOCOL["pool_size"]
        abcnvs_params = dict(QUICK["abcnvs"]) if quick else None

        log.info(f"[{name}] n={n} m={m}  loading metrics (CPU workers) ...")
        pipe = CpuGpuPipeline(name, G).start_metrics()
        metrics = None

        log.info(f"[{name}] building majority model R_fit={R_fit} ...")
        t0 = time.perf_counter()
        model = ChunkedMajorityRuleModel(
            G, MAJORITY_PROTOCOL["variant"], MAJORITY_PROTOCOL["delta"],
            R_fit, seed=7, chunk=256)
        log.info(f"[{name}] model built ({time.perf_counter()-t0:.1f}s, "
                 f"backend={model.backend})")

        # pool needs pr+core metrics -> join those two CPU futures
        metrics = {"pr": pipe.get_metric("pr"), "core": pipe.get_metric("core")}
        pool = pool_from_metrics(model, metrics, pool_size, seed=11)
        metrics.update(get_signed_metrics(name, G))
        log.info(f"[{name}] pool={len(pool)}")

        # held-out model (fresh draws, chunked on the GPU)
        t0 = time.perf_counter()
        ho = ChunkedMajorityRuleModel(
            G, MAJORITY_PROTOCOL["variant"], MAJORITY_PROTOCOL["delta"],
            R_ho, seed=HELDOUT_SEED, chunk=256)
        log.info(f"[{name}] held-out model built ({time.perf_counter()-t0:.1f}s)")

        st_lie = None
        for k in proto["ks"]:
            budget = k * len(pool)
            restarts = proto["restarts"]
            log.info(f"[{name}] k={k}  budget={budget}  solvers="
                     f"{len(MAJORITY_SOLVERS)}  restarts(stoch)={restarts}")
            for alg in MAJORITY_SOLVERS:
                n_rest = restarts if alg in STOCHASTIC else 1
                for rs in range(n_rest):
                    if store.is_done(name, k, alg, rs):
                        continue
                    if alg == "ABCVNS (paper)":
                        if st_lie is None:            # bc + comm futures
                            bc = pipe.get_metric("bc")
                            comm = pipe.get_metric("comm")
                            st_lie = make_abcnvs_context(
                                G, GRAPH_PROTOCOL[name]["p_ic"],
                                {**metrics, "bc": bc, "comm": comm},
                                mode="mr",
                                delta=MAJORITY_PROTOCOL["delta"])
                    t0 = time.perf_counter()
                    model.reset_counters(); model.clear_cache()
                    res = solve_majority(model, ho, metrics, alg, k, pool,
                                         budget, rs, st_lie=st_lie,
                                         abcnvs_params=abcnvs_params)
                    heldout = ho.sigma(res.seeds)
                    row = _cell_row(stage, name, n, m, k, alg, rs, res,
                                    heldout, budget)
                    if res.evaluations > budget + max(2, budget // 20) \
                            and alg not in ("ABCVNS (paper)", "CELF + LS",
                                            "Random", "Degree", "PageRank",
                                            "k-Core", "MajorityDiscount",
                                            "BFSBall"):
                        log.warning(f"  budget violation {name} k={k} {alg}: "
                                    f"{res.evaluations} > {budget}")
                    store.add(row)
                    log.info(f"  done {name:14s} k={k:2d} {alg:16s} rs={rs} "
                             f"ho={heldout:10.3f} t={res.runtime:8.2f}s "
                             f"evals={res.evaluations}")
            store._save_progress(graph=name, k=k,
                                 cells=len(store.done), t=time.strftime("%H:%M:%S"))
        pipe.shutdown()
    log.info(f"majority stage finished: {len(store.done)} cells -> {store.path}")


def run_ic(graphs, quick: bool = False):
    stage = "ic"
    store = ResultStore(stage)
    log.info(f"== ic stage == gpu={gpu_info()} graphs={graphs} quick={quick}")
    for name in graphs:
        G = load_real_graph(name)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        proto = dict(GRAPH_PROTOCOL[name])
        if quick:
            proto["ks"] = QUICK["ks"]
            proto["restarts"] = QUICK["restarts"]
        R_fit = QUICK["R_fit"] if quick else IC_PROTOCOL["R_fit"]
        R_ho = QUICK["R_ho"] if quick else IC_PROTOCOL["R_ho"]
        pool_size = QUICK["pool_size"] if quick else IC_PROTOCOL["pool_size"]
        abcnvs_params = dict(QUICK["abcnvs"]) if quick else None
        p = proto["p_ic"]

        log.info(f"[{name}] p={p} R_fit={R_fit} R_ho={R_ho}  metrics (CPU) ...")
        pipe = CpuGpuPipeline(name, G).start_metrics()

        model = IndependentCascadeModel(G, p=p, n_samples=R_fit, seed=7)
        metrics = {"pr": pipe.get_metric("pr"), "core": pipe.get_metric("core")}
        pool = pool_from_metrics(model, metrics, pool_size, seed=11)
        metrics.update(get_signed_metrics(name, G))
        ho = IndependentCascadeModel(G, p=p, n_samples=R_ho, seed=HELDOUT_SEED)
        log.info(f"[{name}] ic model n={n} m={m} pool={len(pool)}")

        st_lie = None
        for k in proto["ks"]:
            budget = k * len(pool)
            restarts = proto["restarts"]
            log.info(f"[{name}] k={k}  budget={budget}  solvers="
                     f"{len(IC_SOLVERS)}  restarts(stoch)={restarts}")
            for alg in IC_SOLVERS:
                n_rest = restarts if alg in STOCHASTIC else 1
                for rs in range(n_rest):
                    if store.is_done(name, k, alg, rs):
                        continue
                    if alg == "ABCVNS (paper)":
                        if st_lie is None:
                            bc = pipe.get_metric("bc")
                            comm = pipe.get_metric("comm")
                            st_lie = make_abcnvs_context(
                                G, p, {**metrics, "bc": bc, "comm": comm})
                    t0 = time.perf_counter()
                    model.reset_counters(); model.clear_cache()
                    res = solve_ic(model, metrics, alg, k, pool, budget, rs,
                                   st_lie=st_lie, abcnvs_params=abcnvs_params)
                    # held-out scoring: CPU-parallel live-edge draws on the
                    # big graph (CPU side of the hybrid pipeline)
                    heldout = ho.sigma(res.seeds,
                                       parallel=(name == "slashdot"),
                                       n_workers=4)
                    row = _cell_row(stage, name, n, m, k, alg, rs, res,
                                    heldout, budget)
                    store.add(row)
                    log.info(f"  done {name:14s} k={k:2d} {alg:16s} rs={rs} "
                             f"ho={heldout:10.3f} t={res.runtime:8.2f}s "
                             f"evals={res.evaluations}")
            store._save_progress(graph=name, k=k,
                                 cells=len(store.done), t=time.strftime("%H:%M:%S"))
        pipe.shutdown()
    log.info(f"ic stage finished: {len(store.done)} cells -> {store.path}")


def run_signed(graphs, quick: bool = False):
    """Bonus stage: signed majority rule on the (genuinely signed) bitcoin
    trust graphs.  Solver set mirrors the majority suite minus ABCVNS."""
    from signed_rule_model import SignedMajorityRuleModel, signed_candidate_pool
    stage = "signed"
    store = ResultStore(stage)
    log.info(f"== signed stage == graphs={graphs} quick={quick}")
    signed_solvers = [a for a in MAJORITY_SOLVERS if a != "ABCVNS (paper)"]
    for name in graphs:
        if name == "slashdot":
            log.warning("signed stage: skipping slashdot (huge); bitcoin only")
            continue
        G = load_real_graph(name)
        n, m = G.number_of_nodes(), G.number_of_edges()
        proto = dict(GRAPH_PROTOCOL[name])
        if quick:
            proto["ks"] = QUICK["ks"]
            proto["restarts"] = QUICK["restarts"]
        R_fit = QUICK["R_fit"] if quick else MAJORITY_PROTOCOL["R_fit"]
        R_ho = QUICK["R_ho"] if quick else MAJORITY_PROTOCOL["R_ho"]
        pool_size = QUICK["pool_size"] if quick else MAJORITY_PROTOCOL["pool_size"]

        metrics = get_metrics(name, G)              # cached CPU metrics
        metrics.update(get_signed_metrics(name, G))
        model = SignedMajorityRuleModel(
            G, MAJORITY_PROTOCOL["variant"], MAJORITY_PROTOCOL["delta"],
            R_fit, seed=7)
        ho = SignedMajorityRuleModel(
            G, MAJORITY_PROTOCOL["variant"], MAJORITY_PROTOCOL["delta"],
            R_ho, seed=HELDOUT_SEED)
        pool = signed_candidate_pool(model, pool_size)
        log.info(f"[{name}] signed n={n} m={m} neg={model.n_neg_edges} "
                 f"pool={len(pool)} backend={model.backend}")
        for k in proto["ks"]:
            budget = k * len(pool)
            for alg in signed_solvers:
                n_rest = proto["restarts"] if alg in STOCHASTIC else 1
                for rs in range(n_rest):
                    if store.is_done(name, k, alg, rs):
                        continue
                    model.reset_counters(); model.clear_cache()
                    res = solve_majority(model, ho, metrics, alg, k, pool,
                                         budget, rs)
                    heldout = ho.sigma(res.seeds)
                    store.add(_cell_row(stage, name, n, m, k, alg, rs, res,
                                        heldout, budget))
                    log.info(f"  done {name:14s} k={k:2d} {alg:16s} rs={rs} "
                             f"ho={heldout:10.3f} t={res.runtime:8.2f}s")
    log.info(f"signed stage finished: {len(store.done)} cells -> {store.path}")


# --------------------------------------------------------------------------- #
#  Report stage (summary JSON + figures)
# --------------------------------------------------------------------------- #
def run_report():
    log.info("== report stage ==")
    for stage in ("majority", "ic", "signed"):
        path = RESULTS / f"real_{stage}.csv"
        if not path.exists():
            log.warning(f"no data for {stage}")
            continue
        df = pd.read_csv(path)
        out = {
            "meta": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stage": stage, "n_rows": len(df),
                "gpu": gpu_info(),
                "graphs": sorted(df.graph.unique().tolist()),
                "algorithms": sorted(df.algorithm.unique().tolist()),
            },
            "groupby_algorithm": {},
            "by_graph_k": {},
        }
        g = df.groupby("algorithm").agg(
            spread=("spread_heldout", "mean"),
            spread_in=("spread", "mean"),
            runtime_s=("runtime_s", "mean"),
            runtime_sum_s=("runtime_s", "sum"),
            evaluations=("evaluations", "mean"),
            n_cells=("spread_heldout", "size"),
        ).round(4)
        out["groupby_algorithm"] = g.reset_index().to_dict("records")
        for (gn, k), sub in df.groupby(["graph", "k"]):
            tab = sub.groupby("algorithm").agg(
                spread_ho=("spread_heldout", "mean"),
                spread=("spread", "mean"),
                runtime_s=("runtime_s", "mean"),
                runtime_sum_s=("runtime_s", "sum"),
                evaluations=("evaluations", "mean"),
                restarts=("restart", "max"),
            ).round(4).reset_index().to_dict("records")
            out["by_graph_k"][f"{gn}|k={k}"] = tab
        (RESULTS / f"real_summary_{stage}.json").write_text(
            json.dumps(out, indent=2))
        log.info(f"wrote real_summary_{stage}.json ({len(df)} rows)")
        # quick console table
        print(f"\n=== {stage}: mean held-out spread / mean time ===")
        for rec in sorted(out["groupby_algorithm"], key=lambda r: -r["spread"]):
            print(f"  {rec['algorithm']:18s}  ho={rec['spread']:10.3f}  "
                  f"t={rec['runtime_s']:9.2f}s  sum_t={rec['runtime_sum_s']:10.1f}s  "
                  f"evals={rec['evaluations']:9.0f}  cells={rec['n_cells']}")
    build_real_figures()
    log.info("report stage done")


def build_real_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for stage in ("majority", "ic"):
        path = RESULTS / f"real_{stage}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        graphs = sorted(df.graph.unique())
        fig, axes = plt.subplots(1, len(graphs), figsize=(6 * len(graphs), 5),
                                 squeeze=False)
        for ax, gn in zip(axes[0], graphs):
            sub = df[df.graph == gn]
            piv = sub.pivot_table(index="algorithm", columns="k",
                                  values="spread_heldout", aggfunc="mean")
            piv = piv.loc[sorted(piv.index, key=lambda a: piv.loc[a].mean()
                                 if len(piv.loc[a].dropna()) else 0, reverse=True)]
            piv.plot.bar(ax=ax, rot=45)
            ax.set_title(f"{gn} (n={sub.n.iloc[0]})")
            ax.set_ylabel("held-out spread")
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_real_{stage}_spread.png", dpi=150)
        plt.close(fig)

        # wall-clock
        fig, ax = plt.subplots(figsize=(10, 5))
        time_ = df.groupby(["graph", "algorithm"])["runtime_s"].sum() \
            .reset_index().pivot_table(index="algorithm", columns="graph",
                                       values="runtime_s", aggfunc="sum")
        time_.loc[time_.sum(axis=1).sort_values().index].plot.barh(ax=ax)
        ax.set_xlabel("total wall-clock (s)")
        ax.set_title(f"{stage}: wall-clock by algorithm (sum over graph × k)")
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_real_{stage}_time.png", dpi=150)
        plt.close(fig)
    log.info("figures written")


def purge_algorithm(stage: str, alg: str) -> int:
    """Delete all rows of one algorithm from a stage CSV (for --rerun)."""
    path = RESULTS / f"real_{stage}.csv"
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    keep = df["algorithm"].astype(str).str.strip() != alg
    n = int((~keep).sum())
    if n:
        df[keep].to_csv(path, index=False)
    return n


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["majority", "ic", "signed", "report",
                                      "graphs", "all"])
    ap.add_argument("--graphs", nargs="*", default=None,
                    help="alpha otc slashdot (aliases)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke protocol (validates the pipeline)")
    ap.add_argument("--rerun", action="append", default=[],
                    help="algorithm label to purge and re-run (repeatable); "
                         "e.g. --rerun \"ABCVNS (paper)\"")
    args = ap.parse_args()

    ensure_datasets()
    if args.graphs:
        graphs = [GRAPH_ALIAS.get(g, g) for g in args.graphs]
    else:
        graphs = ["bitcoin_alpha", "bitcoin_otc", "slashdot"]

    for alg in args.rerun:
        for stage in ("majority", "ic", "signed"):
            n = purge_algorithm(stage, alg)
            if n:
                log.info(f"--rerun: purged {n} '{alg}' rows from "
                         f"real_{stage}.csv")

    if args.stage in ("majority", "all"):
        run_majority(graphs, quick=args.quick)
    if args.stage in ("ic", "all"):
        run_ic(graphs, quick=args.quick)
    if args.stage in ("signed", "all"):
        run_signed([g for g in graphs if g != "slashdot"], quick=args.quick)
    if args.stage in ("report", "all"):
        run_report()
    if args.stage in ("graphs", "all"):
        from make_four_graphs import main as make_graphs
        make_graphs()


if __name__ == "__main__":
    main()
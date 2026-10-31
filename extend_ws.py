r"""
extend_ws.py -- one-off: extend the S10 multi-instance study from 8 to 16 WS
instances (primary pre-specified comparison: ABC vs Greedy on WS), appending
to results/s10_multi.csv with the cascade-fraction metric computed from the
held-out threshold draws.
"""

import time

import networkx as nx
import numpy as np
import pandas as pd

from majority_rule_model import (MajorityRuleModel, candidate_pool, greedy_mc,
                                 celf, heldout_model)
from abc_celf import ABCConfig, abc_optimize
from baselines import simulated_annealing
from run_experiments import log, rescore, HELDOUT_SEED, HELDOUT_R

RESULTS = pd.DataFrame()


def main():
    out_path = __import__("pathlib").Path("results/s10_multi.csv")
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.graph, old.k, old.algorithm))

    solvers = ("Greedy", "CELF", "SA", "ABC (proposed)")
    for i in range(8, 16):          # instances 8..15 -> 16 WS graphs total
        deg = 6 + 2 * (i % 3)
        gname = f"ws_{deg}_{i}"
        G = nx.watts_strogatz_graph(1000, deg, 0.1, seed=300 + i)
        m = MajorityRuleModel(G, "smr", 0.15, n_samples=100, seed=7)
        ho = MajorityRuleModel(G, "smr", 0.15, n_samples=HELDOUT_R,
                               seed=HELDOUT_SEED, backend=m.backend)
        pool = candidate_pool(m, min(m.n, 200))
        for k in (5, 10, 20):
            budget = k * len(pool)
            for alg in solvers:
                if (gname, k, alg) in done:
                    continue
                t0 = time.perf_counter()
                m.reset_counters(); m.clear_cache()
                if alg == "Greedy":
                    r = greedy_mc(m, k, pool=pool)
                elif alg == "CELF":
                    r = celf(m, k, pool=pool)
                elif alg == "SA":
                    r = simulated_annealing(m, k, pool, eval_budget=budget,
                                            restart=True, seed=1)
                else:
                    cfg = ABCConfig(colony_size=30, max_cycles=150,
                                    refine_every=0, seed_with_celf=True,
                                    eval_budget=budget, random_state=600)
                    r = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                wall = time.perf_counter() - t0
                sizes = ho._simulate_full(r.seeds)[0]
                row = {"family": "WS", "graph": gname, "k": k,
                       "algorithm": alg, "budget": budget,
                       "spread": float(r.spread), "heldout": float(sizes.mean()),
                       "wall_s": float(wall), "evaluations": int(r.evaluations),
                       "frac_cascade_k5": float((sizes >= k + 5).mean()),
                       "frac_cascade_k10": float((sizes >= k + 10).mean())}
                pd.DataFrame([row]).to_csv(out_path, mode="a",
                                           header=not out_path.exists(), index=False)
                log(f"    {gname:12s} k={k:2d} {alg:16s} ho={row['heldout']:7.2f} "
                    f"frac10={row['frac_cascade_k10']:.2f} ({wall:.0f}s)  [saved]")
    log("WS extension done (16 instances total)")


if __name__ == "__main__":
    main()
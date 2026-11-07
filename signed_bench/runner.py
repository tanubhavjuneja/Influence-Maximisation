"""
runner.py
=========
Orchestrates the signed-network benchmark and writes results.csv.

  graphs   x k in {10,20,30,40,50} x algorithms
  -------------------------------------------------------------------
  Random          1 run   (fixed seed per k)
  Degree          1 run   (deterministic)
  ABC (proposed)  5 restarts
  ABCVNS (paper)  30 restarts (the paper's training setting), UNCHANGED

Recorded per cell (results.csv):
  graph, k, algorithm, restart, time_s,
  positive_spread, negative_spread, total_spread, fitness

  * time_s              wall-clock of the search (incl. in-sample scoring)
  * positive_spread     expected # final active nodes with net incident
                        weight >= 0          (held-out, R_ho = 300)
  * negative_spread     expected # final active nodes with net < 0
  * total_spread        positive + negative
  * fitness             the solver's terminal objective
                        (ABCVNS: LIE value; others: in-sample sigma)

Resumable: every finished (graph, k, algorithm, restart) cell is appended
to results.csv immediately and skipped on re-run.

Protocol: R_fit = 30 in-sample draws, R_ho = 300 held-out draws,
budget = k x 200 cascade evaluations per ABC restart, pool = 200 nodes,
ABCVNS uses the paper's IC configuration (p = 0.05 Bitcoin / 0.01 Slashdot)
with the paper's parameters and 30 restarts.
"""
import argparse
import time

import numpy as np

from abc_proposed import ABCConfig, abc_optimize
from abcvns_paper import make_context, abcnvs_search
from common import append_cell, load_done, log
from degree_baseline import run_degree
from diffusion import SignedDiffusion
from graphio import load_graph
from metrics import get_metrics, signed_pool
from random_baseline import run_random

# --------------------------------------------------------------------------- #
GRAPHS = [
    "bitcoin_alpha",
    "bitcoin_otc",
    # "slashdot",     # <-- uncomment to include Slashdot.
    #                   est. +66 h (CPU) / ~15 h (GPU) at 30 ABCVNS restarts
]
KS = [10, 20, 30, 40, 50]
RESTARTS = {"Random": 1, "Degree": 1,
            "ABC (proposed)": 30, "ABCVNS (paper)": 30,
            "ABCVNS (fixed)": 30}
ALGORITHMS = ["Random", "Degree", "ABC (proposed)", "ABCVNS (paper)",
              "ABCVNS (fixed)"]
R_FIT, R_HO = 30, 300
POOL_SIZE = 200
P_IC = {"bitcoin_alpha": 0.05, "bitcoin_otc": 0.05, "slashdot": 0.01}
ABC_RS_SEED = 600
# --------------------------------------------------------------------------- #

QUICK = dict(ks=[10], restarts=1, abcnvs=dict(gmax=20, NP=20),
             abc=dict(colony_size=8, max_cycles=5))


def one_cell(name, G, k, alg, rs, model, ho, metrics, pool, st_lie, st_mr,
             quick):
    """Run one (algorithm, restart) cell; returns the CSV row dict."""
    budget = k * len(pool)
    model.reset_counters()
    model.clear_cache()
    t0 = time.perf_counter()

    if alg == "Random":
        res = run_random(model, k)
    elif alg == "Degree":
        res = run_degree(model, k)
    elif alg == "ABC (proposed)":
        if quick:
            cfg = ABCConfig(eval_budget=budget, random_state=ABC_RS_SEED + rs,
                            **QUICK["abc"])
        else:
            cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                            seed_with_celf=True, eval_budget=budget,
                            random_state=ABC_RS_SEED + rs)
        r = abc_optimize(model, k, pool, cfg, name="ABC (proposed)")
        res = {"seeds": r.seeds, "fitness": float(r.spread),
               "time_s": r.runtime}
    elif alg in ("ABCVNS (paper)", "ABCVNS (fixed)"):
        params = dict(QUICK["abcnvs"]) if quick else None
        st = st_lie if alg == "ABCVNS (paper)" else st_mr
        S, stats = abcnvs_search(st, k, seed=rs, verbose=True,
                                 params=params, log=log)
        res = {"seeds": S, "fitness": float(stats["LIE_best"]),
               "time_s": stats["wall_s"]}
    else:
        raise ValueError(alg)

    time_s = time.perf_counter() - t0
    pos, neg, tot = ho.components(res["seeds"])
    return {"graph": name, "k": k, "algorithm": alg, "restart": rs,
            "time_s": round(time_s, 4),
            "positive_spread": round(pos, 4),
            "negative_spread": round(neg, 4),
            "total_spread": round(tot, 4),
            "fitness": round(res["fitness"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke protocol (k=[10], 1 restart, tiny gmax)")
    ap.add_argument("--graphs", nargs="*", default=None,
                    help="subset of bitcoin_alpha bitcoin_otc slashdot")
    ap.add_argument("--algs", nargs="*", default=None,
                    help="subset of algorithms to run")
    ap.add_argument("--restarts", nargs="*", default=None, metavar="ALG=N",
                    help='override restart counts, e.g. "ABC (proposed)=5"')
    args = ap.parse_args()
    quick = args.quick
    ks = QUICK["ks"] if quick else KS
    restarts = {a: (1 if quick else RESTARTS[a]) for a in RESTARTS}
    if args.restarts:
        for spec in args.restarts:
            alg, n = spec.rsplit("=", 1)
            restarts[alg] = int(n)
    graphs = args.graphs if args.graphs else GRAPHS
    algs = args.algs if args.algs else ALGORITHMS

    done = load_done()
    log(f"== signed benchmark == graphs={graphs} ks={ks} "
        f"restarts={restarts} R_fit={R_FIT} R_ho={R_HO} quick={quick}")
    total_cells = 0
    for name in graphs:
        G = load_graph(name)
        metrics = get_metrics(G)
        model = SignedDiffusion(G, R_FIT, seed=7)
        ho = SignedDiffusion(G, R_HO, seed=999_999)
        pool = signed_pool(model, metrics["pr"], metrics["core"],
                           POOL_SIZE, seed=11)
        st_lie = make_context(G, P_IC[name], metrics)               # paper: LIE
        st_mr = make_context(G, P_IC[name], metrics, mode="mr",
                             delta=0.15)                            # fixed surrogate
        log(f"[{name}] n={G['n']} m={G['m']} pool={len(pool)} "
            f"R_fit={R_FIT} R_ho={R_HO}")

        for k in ks:
            budget = k * len(pool)
            for alg in algs:
                for rs in range(restarts[alg]):
                    key = (name, k, alg, rs)
                    if key in done:
                        continue
                    log(f"[{name}] k={k} {alg} rs={rs} "
                        f"budget={budget} ...")
                    row = one_cell(name, G, k, alg, rs, model, ho, metrics,
                                   pool, st_lie, st_mr, quick)
                    append_cell(row)
                    done.add(key)
                    total_cells += 1
                    log(f"  done {name} k={k:2d} {alg:16s} rs={rs:2d} "
                        f"t={row['time_s']:9.2f}s "
                        f"pos={row['positive_spread']:9.2f} "
                        f"neg={row['negative_spread']:9.2f} "
                        f"tot={row['total_spread']:9.2f} "
                        f"fit={row['fitness']:12.3f}")
    log(f"benchmark finished: {total_cells} new cells "
        f"({len(load_done())} total) -> results.csv")


if __name__ == "__main__":
    main()
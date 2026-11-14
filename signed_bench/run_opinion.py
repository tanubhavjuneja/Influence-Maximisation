"""
run_opinion.py
==============
Reduced test suite for the CORRECTED two-opinion signed majority rule.

Algorithms (5):
    ABC (proposed)   5 restarts   (CELF-seeded, gbest/tournament/scouts/Φ)
    ABC (plain)      5 restarts   (canonical ABC: roulette, random partners,
                                   random scouts, no dedupe, no Φ, no CELF)
    ABCVNS (fixed)  10 restarts   (paper machinery + majority-rule surrogate)
    SignedPageRank   1 run
    Degree           1 run

Scope: Bitcoin Alpha, Bitcoin OTC, Slashdot  x  k in {10,20,30}
Draws: R_fit = 30 (search), R_ho = 300 (held-out)
Budget: k x 200 evaluations per ABC restart.

CSV columns (results_opinion.csv):
    graph, k, algorithm, restart, time_s,
    pos_spread, neg_spread, net_spread, diversity, fairness, fitness

Resumable: every finished cell is appended immediately and skipped on re-run.
"""
import argparse
import time

import numpy as np

from abc_proposed import ABCConfig, abc_optimize
from abcvns_paper import make_context, abcnvs_search
from common import HERE
from graphio import load_graph
from metrics import get_metrics, signed_pool, signed_pagerank
from opinion_model import SignedOpinionModel

import csv
from pathlib import Path

RESULTS = HERE / "results_opinion.csv"
COLS = ["graph", "k", "algorithm", "restart", "time_s",
        "pos_spread", "neg_spread", "net_spread", "diversity",
        "fairness", "fitness"]

GRAPHS = ["bitcoin_alpha", "bitcoin_otc", "slashdot"]
KS = [10, 20, 30]
ALGORITHMS = ["ABC (proposed)", "ABC (plain)", "ABCVNS (fixed)",
              "SignedPageRank", "Degree"]
RESTARTS = {"ABC (proposed)": 3, "ABC (plain)": 3, "ABCVNS (fixed)": 5,
            "SignedPageRank": 1, "Degree": 1}
R_FIT, R_HO = 30, 300
POOL_SIZE = 200
P_IC = {"bitcoin_alpha": 0.05, "bitcoin_otc": 0.05, "slashdot": 0.01}


def log(msg):
    from common import log as _log
    _log(msg)


def load_done():
    done = set()
    if RESULTS.exists():
        with open(RESULTS, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["graph"], int(row["k"]),
                          row["algorithm"], int(row["restart"])))
    return done


def append_cell(row):
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})


def one_cell(name, G, k, alg, rs, model, ho, metrics, pool, st_mr):
    budget = k * len(pool)
    model.reset_counters()
    model.clear_cache()
    t0 = time.perf_counter()

    if alg == "Degree":
        S = sorted(range(model.n), key=lambda i: -model.deg[i])[:k]
        fit = model.sigma_pos(S)
    elif alg == "SignedPageRank":
        S = np.argsort(-metrics["spr"])[:k].tolist()
        fit = model.sigma_pos(S)
    elif alg == "ABC (proposed)":
        cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                        seed_with_celf=True, eval_budget=budget,
                        random_state=600 + rs)
        r = abc_optimize(model, k, pool, cfg, name=alg)
        S, fit = r.seeds, float(r.spread)
    elif alg == "ABC (plain)":
        cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                        seed_with_celf=False, gbest_prob=0.0,
                        selection="roulette", directed_scout=False,
                        dedupe=False, potential_weight=0.0,
                        explore_prob=0.0, accept_equal=False,
                        eval_budget=budget, random_state=600 + rs)
        r = abc_optimize(model, k, pool, cfg, name=alg)
        S, fit = r.seeds, float(r.spread)
    elif alg == "ABCVNS (fixed)":
        S, stats = abcnvs_search(st_mr, k, seed=rs, verbose=False, log=log)
        fit = float(stats["LIE_best"])
    else:
        raise ValueError(alg)

    time_s = time.perf_counter() - t0
    pos, neg, net, div, fair = ho.components(S)
    return {"graph": name, "k": k, "algorithm": alg, "restart": rs,
            "time_s": round(time_s, 4),
            "pos_spread": round(pos, 4), "neg_spread": round(neg, 4),
            "net_spread": round(net, 4), "diversity": round(div, 5),
            "fairness": round(fair, 5), "fitness": round(fit, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="*", default=None)
    ap.add_argument("--algs", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    graphs = args.graphs or GRAPHS
    algs = args.algs or ALGORITHMS
    ks = [10] if args.quick else KS
    restarts = {a: (1 if args.quick else RESTARTS[a]) for a in ALGORITHMS}

    done = load_done()
    log(f"== opinion benchmark == graphs={graphs} ks={ks} algs={algs} "
        f"restarts={restarts} R_fit={R_FIT} R_ho={R_HO}")
    for name in graphs:
        G = load_graph(name)
        metrics = get_metrics(G)
        metrics["spr"] = signed_pagerank(G)
        model = SignedOpinionModel(G, R_FIT, seed=7, comm=metrics["comm"])
        ho = SignedOpinionModel(G, R_HO, seed=999_999, comm=metrics["comm"])
        pool = signed_pool(model, metrics["pr"], metrics["core"],
                           POOL_SIZE, seed=11)
        st_mr = make_context(G, P_IC[name], metrics, mode="mr", delta=0.15)
        log(f"[{name}] n={G['n']} m={G['m']} pool={len(pool)}")

        for k in ks:
            for alg in algs:
                for rs in range(restarts[alg]):
                    key = (name, k, alg, rs)
                    if key in done:
                        continue
                    log(f"[{name}] k={k} {alg} rs={rs} ...")
                    row = one_cell(name, G, k, alg, rs, model, ho, metrics,
                                   pool, st_mr)
                    append_cell(row)
                    done.add(key)
                    log(f"  done {name} k={k:2d} {alg:16s} rs={rs:2d} "
                        f"t={row['time_s']:8.1f}s pos={row['pos_spread']:8.1f} "
                        f"neg={row['neg_spread']:7.1f} net={row['net_spread']:8.1f} "
                        f"div={row['diversity']:.3f} fair={row['fairness']:.3f}")
    log(f"opinion benchmark finished: {len(load_done())} cells total")


if __name__ == "__main__":
    main()
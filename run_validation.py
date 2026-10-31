r"""
run_validation.py
=================
Small/unique-graph validation of the upgraded ABC before the final big runs.

1. OPT checks: brute-force optimum on small unique graphs, held-out ratios.
2. LEAVE-ONE-OUT ablations: start from the FULL config and remove ONE
   operator at a time (not cumulative), 20 seeds, on tuning graphs DISJOINT
   from the final test set -- the reconciled replacement for the earlier
   cumulative table.
3. Decision-relevant CELF audit: how often CELF's chosen node differs from
   exact Greedy's, and how often CELF's final set scores below Greedy's.
4. Signed graphs: monotonicity witnesses + ABC-CELF vs greedy/CELF/SA.

Outputs: results/val_*.csv
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from majority_rule_model import (
    MajorityRuleModel, candidate_pool, load_graph, heldout_model,
    greedy_mc, celf,
)
from abc_celf import ABCConfig, abc_optimize, celf_exchange_refine
from baselines import simulated_annealing
from signed_rule_model import (
    SignedMajorityRuleModel, signed_candidate_pool, find_nonmonotone_witness,
    signed_hub_blocker, signed_star, signed_cycle, signed_sbm, signed_ws, signed_ba,
)
from theory_checks import check_monotonicity

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
HELDOUT_SEED = 999999


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rescore(model, seeds, ho=None):
    if ho is None:
        ho = heldout_model(model, seed=HELDOUT_SEED)
    return ho.sigma(seeds)


# --------------------------------------------------------------------------- #
# 1. OPT on unique small graphs
# --------------------------------------------------------------------------- #
def study_opt(graphs=(
        ("florentine", 4), ("davis", 3), ("star60", 3), ("cycle60", 3),
        ("grid100", 3), ("tree63", 3), ("barbell50", 3), ("caveman60", 3)),
        variant="smr", delta=0.15, R=200):
    log("OPT study on unique small graphs")
    rows = []
    for gname, k in graphs:
        G = load_graph(gname)
        m = MajorityRuleModel(G, variant, delta, n_samples=R, seed=7)
        nodes = list(range(m.n))
        best, best_set = -1.0, None
        t0 = time.perf_counter()
        for S in itertools.combinations(nodes, k):
            v = float(m._simulate(S).mean())
            if v > best:
                best, best_set = v, S
        opt_ho = rescore(m, best_set)
        log(f"  {gname:10s} k={k} OPT={best:.3f} (ho {opt_ho:.3f})  "
            f"({time.perf_counter()-t0:.1f}s)")

        m.reset_counters(); m.clear_cache()
        pool = candidate_pool(m, min(m.n, 60))
        budget = k * len(pool)

        r_celf = celf(m, k, pool=pool)
        S2, f2, _ = celf_exchange_refine(m, r_celf.seeds, pool, max_passes=4,
                                         eval_budget=max(0, budget - r_celf.evaluations))
        best_abc = -1
        for rs in range(3):
            m.reset_counters(); m.clear_cache()
            cfg = ABCConfig(colony_size=30, max_cycles=100, refine_every=0,
                            seed_with_celf=True, eval_budget=budget,
                            random_state=500 + rs)
            r = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
            best_abc = max(best_abc, rescore(m, r.seeds))
        m.reset_counters(); m.clear_cache()
        r_g = greedy_mc(m, k, pool=pool)
        g_ho = rescore(m, r_g.seeds)
        m.reset_counters(); m.clear_cache()
        r_sa = simulated_annealing(m, k, pool, eval_budget=budget, restart=True, seed=1)
        sa_ho = rescore(m, r_sa.seeds)

        rows.append({"graph": gname, "k": k, "OPT": best, "OPT_heldout": opt_ho,
                     "greedy_ho_ratio": g_ho / opt_ho,
                     "celf_ho_ratio": rescore(m, r_celf.seeds) / opt_ho,
                     "celf_ls_ho_ratio": rescore(m, S2) / opt_ho,
                     "sa_ho_ratio": sa_ho / opt_ho,
                     "abc_ho_ratio": best_abc / opt_ho})
        log(f"    ratio(ho): Greedy={g_ho/opt_ho:.3f}  CELF={rescore(m, r_celf.seeds)/opt_ho:.3f}  "
            f"CELF+LS={rescore(m, S2)/opt_ho:.3f}  SA={sa_ho/opt_ho:.3f}  ABC={best_abc/opt_ho:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "val_opt.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. LEAVE-ONE-OUT ablations on disjoint tuning graphs
# --------------------------------------------------------------------------- #
TUNE_GRAPHS = ("florentine", "davis", "star60", "cycle60", "grid100",
               "tree63", "barbell50", "caveman60", "windmill46",
               "ba300", "ws300", "powerlaw300", "sbm450")
TUNE_KS = (3, 5)

FULL = dict(gbest_prob=0.6, selection="tournament", directed_scout=True,
            dedupe=True, accept_equal=True, explore_prob=0.15,
            potential_weight=0.05, reserve_fraction=0.15,
            two_swap=True, cascade_aware=True, n_mod=1, mutation_rate=0.2)

ABLATE = {
    "full": {},
    "-gbest": dict(gbest_prob=0.0),
    "-tournament": dict(selection="roulette"),
    "-directed_scout": dict(directed_scout=False),
    "-dedupe": dict(dedupe=False),
    "-accept_equal": dict(accept_equal=False),
    "-explore": dict(explore_prob=0.0),
    "-potential": dict(potential_weight=0.0),
    "-reserve": dict(reserve_fraction=0.0),
    "-two_swap": dict(two_swap=False),
    "-cascade_aware": dict(cascade_aware=False),
}


def study_loo(graphs=TUNE_GRAPHS, ks=TUNE_KS, seeds=20, variant="smr",
              delta=0.15, R=150):
    log("leave-one-out ablations (20 seeds, disjoint tuning graphs)")
    rows = []
    out_path = RESULTS / "val_loo.csv"
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        done = set(zip(old.graph, old.k, old.variant))
        rows = old.to_dict("records")
        log(f"  resuming: {len(done)} (graph,k,variant) cells already done")
    for gname in graphs:
        G = load_graph(gname)
        m = MajorityRuleModel(G, variant, delta, n_samples=R, seed=7)
        ho = heldout_model(m, seed=HELDOUT_SEED)
        pool = candidate_pool(m, min(m.n, 60))
        for k in ks:
            if k >= m.n // 2:
                continue
            budget = k * len(pool)
            for vname, kw in ABLATE.items():
                if (gname, k, vname) in done:
                    continue
                vals = []
                for sd in range(seeds):
                    m.reset_counters(); m.clear_cache()
                    cfg = ABCConfig(colony_size=30, max_cycles=100,
                                    refine_every=10, seed_with_celf=True,
                                    eval_budget=budget, random_state=1000 + sd,
                                    **kw)
                    r = abc_optimize(m, k, pool, cfg, name=vname)
                    vals.append(rescore(m, r.seeds, ho))
                row = {"graph": gname, "k": k, "variant": vname,
                       "mean_ho": float(np.mean(vals)),
                       "std_ho": float(np.std(vals)),
                       "median_ho": float(np.median(vals))}
                rows.append(row)
                pd.DataFrame([row]).to_csv(out_path, mode="a",
                                           header=not out_path.exists(), index=False)
                log(f"    {gname:12s} k={k} {vname:16s} "
                    f"mean_ho={row['mean_ho']:8.3f}  [saved {len(rows)} rows]")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "val_loo.csv", index=False)

    piv = df.pivot_table(index=["graph", "k"], columns="variant",
                         values="mean_ho").reindex(ABLATE.keys(), axis=1)
    base = piv["full"]
    print("\nleave-one-out: mean held-out per config, and delta vs full:")
    for c in piv.columns:
        d = (piv[c] - base)
        print(f"  {c:16s}  mean {piv[c].mean():8.3f}   delta {d.mean():+8.3f}")
    print("\npaired (config-level) differences vs full:")
    for c in [x for x in piv.columns if x != "full"]:
        d = (piv[c] - base).dropna()
        w = sps.wilcoxon(d, zero_method="zsplit")
        print(f"  {c:16s}  median {d.median():+.3f}  mean {d.mean():+.3f}  "
              f"wins {(d > 0).sum()}/{len(d)}  p={w.pvalue:.3g}")
    return df


# --------------------------------------------------------------------------- #
# 3. Decision-relevant CELF audit
# --------------------------------------------------------------------------- #
def study_celf_audit(graphs=("karate", "lesmis", "ba500", "ws1000", "powerlaw1000",
                             "sbm900"), ks=(5, 10, 20), variant="smr", delta=0.15, R=150):
    log("CELF decision audit: CELF picks vs exact Greedy picks")
    rows = []
    for gname in graphs:
        G = load_graph(gname)
        m = MajorityRuleModel(G, variant, delta, n_samples=R, seed=7)
        pool = candidate_pool(m, min(m.n, 200))
        for k in ks:
            if k >= m.n // 2:
                continue
            m.reset_counters(); m.clear_cache()
            r_g = greedy_mc(m, k, pool=pool)
            m.reset_counters(); m.clear_cache()
            r_c = celf(m, k, pool=pool)
            gs, cs = set(r_g.seeds), set(r_c.seeds)
            overlap = len(gs & cs)
            gh, ch = rescore(m, r_g.seeds), rescore(m, r_c.seeds)
            rows.append({
                "graph": gname, "k": k,
                "node_overlap": overlap / k,
                "celf_differs": overlap < k,
                "greedy_ho": gh, "celf_ho": ch,
                "celf_below_greedy": ch < gh - 1e-9,
                "gap_pct": 100 * (gh - ch) / max(gh, 1e-9),
                "celf_lazy_checks": r_c.extra["lazy_bound_checks"],
                "celf_lazy_violations": r_c.extra["lazy_bound_violations"],
            })
            log(f"  {gname:12s} k={k:2d}  overlap={overlap}/{k}  "
                f"Greedy={gh:7.2f}  CELF={ch:7.2f}  "
                f"celf_below={rows[-1]['celf_below_greedy']}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "val_celf_audit.csv", index=False)
    n = len(df)
    log(f"  CELF picks differ from Greedy in "
        f"{(df.celf_differs).sum()}/{n} configs; final set scores strictly below "
        f"Greedy in {(df.celf_below_greedy).sum()}/{n} configs")
    return df


# --------------------------------------------------------------------------- #
# 4. signed graphs
# --------------------------------------------------------------------------- #
def _signed_graphs():
    return {
        "signed_hub": signed_hub_blocker(),
        "signed_star7": signed_star(5, 2),
        "signed_cycle40": signed_cycle(40, 0.25),
        "signed_sbm120": signed_sbm((60, 60), p_in=0.10, p_out=0.02, seed=11),
        "signed_sbm450": signed_sbm((150, 150, 150), p_in=0.08, p_out=0.015,
                                    p_neg_in=0.08, p_neg_out=0.7, seed=11),
        "signed_ws300": signed_ws(300, 8, 0.1, p_neg=0.3, seed=12),
        "signed_ba500": signed_ba(500, 4, p_neg=0.35, seed=13),
    }


def study_signed_ablation(ks=(5, 10), seeds=(0, 1, 2), variant="smr",
                          delta=0.15, R=150):
    log("signed-graph validation")
    rows = []
    for gname, G in _signed_graphs().items():
        m = SignedMajorityRuleModel(G, variant=variant, delta=delta,
                                    n_samples=R, seed=7)
        ho = heldout_model(m, seed=HELDOUT_SEED)
        mono = check_monotonicity(m, trials=400, seed=3)
        wit = find_nonmonotone_witness(m, max_size=2, node_cap=30, seed=3)
        log(f"  {gname:16s} n={m.n} neg_edges={m.n_neg_edges}  "
            f"monotone={mono['holds']} ({mono['violations']}/{mono['trials']})  "
            f"witness={'drop %.2f' % wit['drop'] if wit else 'none'}")
        pool = signed_candidate_pool(m, min(m.n, 60))
        for k in ks:
            if k >= m.n // 2:
                continue
            budget = k * len(pool)
            m.reset_counters(); m.clear_cache()
            r_g = greedy_mc(m, k, pool=pool)
            gh = rescore(m, r_g.seeds, ho)
            m.reset_counters(); m.clear_cache()
            r_c = celf(m, k, pool=pool)
            ch = rescore(m, r_c.seeds, ho)
            m.reset_counters(); m.clear_cache()
            r_sa = simulated_annealing(m, k, pool, eval_budget=budget,
                                       restart=True, seed=1)
            sah = rescore(m, r_sa.seeds, ho)
            ah = []
            for sd in seeds:
                m.reset_counters(); m.clear_cache()
                cfg = ABCConfig(colony_size=30, max_cycles=80, refine_every=0,
                                seed_with_celf=True, eval_budget=budget,
                                random_state=900 + sd)
                r = abc_optimize(m, k, pool, cfg, name="ABC (proposed)")
                ah.append(rescore(m, r.seeds, ho))
            rows.append({"graph": gname, "k": k,
                         "greedy_ho": gh, "celf_ho": ch, "sa_ho": sah,
                         "abc_ho_mean": float(np.mean(ah))})
            log(f"    k={k}  Greedy={gh:7.2f}  CELF={ch:7.2f}  SA={sah:7.2f}  "
                f"ABC-CELF={np.mean(ah):7.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "val_signed.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
def main():
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("opt", "all"):
        study_opt()
    if stage in ("loo", "all"):
        study_loo()
    if stage in ("audit", "all"):
        study_celf_audit()
    if stage in ("signed", "all"):
        study_signed_ablation()


if __name__ == "__main__":
    main()